/**
 * `Material` — a compiled, drawable shader.
 *
 * A material is a `Drawable`. It owns exactly three GPU-side things of its own:
 * a material uniform buffer, a texture bind group, and a reference to the
 * shared frame and object uniforms. Everything else it uses is shared and
 * cached per device, because a scene with 500 materials should not own 500
 * identical `GPUBindGroupLayout` objects.
 *
 * # Why construction is a promise
 *
 * ```ts
 * const m = await Material.create(device, spec);   // correct
 * const m = new Material(device, spec);            // does not exist, on purpose
 * ```
 *
 * Building a render pipeline compiles and links a shader module, and the
 * synchronous `createRenderPipeline` blocks the calling thread until that
 * finishes. On a cold shader cache that is routinely **two to five seconds**,
 * and because it is synchronous it happens on the main thread, so the page
 * does not merely render late — it stops responding, and a long task that
 * overlaps a user gesture is how input gets dropped. It is the single most
 * common cause of a slow first frame in WebGPU, and it is entirely avoidable:
 * `createRenderPipelineAsync` returns as soon as the descriptor is validated
 * and the compile proceeds on the GPU process.
 *
 * So the constructor is private and {@link Material.create} is the only way in.
 * Do not be tempted to reach for a synchronous path; there is none.
 *
 * # What is shared, and why
 *
 * | object                        | owner          | count for N materials |
 * |-------------------------------|----------------|-----------------------|
 * | `GPURenderPipeline`           | device cache   | ≤ N, usually ≪ N      |
 * | `GPUPipelineLayout`           | device cache   | ≤ N                   |
 * | `GPUBindGroupLayout` (0-2)    | device cache   | ~3 total              |
 * | `GPUSampler`                  | device cache   | 1 per distinct config |
 * | scene uniform buffer + group  | device cache   | **1**                 |
 * | material uniform buffer       | the material   | N                     |
 * | texture bind group            | the material   | ≤ N                   |
 *
 * The frame and the object share one buffer and one bind group, so both live in
 * the same row. {@link FrameUniforms} and {@link ObjectUniforms} are two *faces*
 * of that one allocation, and that they are the same allocation is the invariant
 * this module is most careful about.
 *
 * The scene uniform is identical for every material in a frame, and the object
 * block is identical for every material too, so both are shared. Allocating either
 * per material would mean 500 writes of the same 256 bytes and 500 bind group
 * creations to save one uniform load in the shader — the shader load happens
 * either way. It is exported so the renderer can own the one instance that
 * matters.
 *
 * The buffer is one buffer because binding it is the cost. A recorded census at
 * 1000 boxes found 921 `setBindGroup` calls in a frame, of which 920 were the
 * object group and 2 the frame group, and `setBindGroup` is the most expensive
 * call in the WebGPU API — per-draw encode is 85-100% of the frame. Three.js packs
 * per-object data into one uniform buffer in one bind group and pays one
 * `setBindGroup` per draw; apse paid two. Now it pays one.
 *
 * # Disposal
 *
 * {@link Material.onDispose} destroys the material uniform buffer and nothing
 * else. The pipeline is *not* destroyed, because it is shared with every other
 * material that has the same cache key, and destroying it would silently break
 * those. The practical consequence is that a disposed material's pipeline stays
 * resident until the device is destroyed or the application drops every
 * reference to every material that shares it — bounded by the number of
 * *distinct* materials, not their instances, which is the whole point of the
 * cache. The texture bind group is not destroyed either, and could not be:
 * `GPUBindGroup` has no `destroy()`. It is reclaimed when the JS wrapper and
 * its backing reference are collected, and the sampler and texture *views* it
 * refers to are owned by whoever created them.
 */

import { fail } from '../core/error.ts';
import { FRAME_BLOCK, OBJECT_BLOCK, SCENE_BLOCK } from '../core/slot.ts';
import type { SceneBlockSpec, SlotType } from '../core/slot.ts';
import { UniformBlock } from '../core/uniform.ts';
import type { UniformBlockSpec } from '../core/uniform.ts';
import { Resource } from '../core/resource.ts';
import type { Disposable } from '../core/resource.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import { textureViewDimension } from './texture-slot.ts';
import {
  describeMaterial,
  generateScaffold,
  validateGeneratedWGSL,
} from './scaffold.ts';
import type {
  GeneratedShader,
  MaterialDescription,
  MaterialSpec,
  ResolvedMaterialSpec,
  ResolvedSampler,
} from './scaffold.ts';
import type { Drawable, DrawPhase } from '../render/types.ts';
import type {
  BlendSpec,
  CullMode,
  DepthSpec,
  FrontFace,
  PrimitiveTopology,
} from '../render/pipeline-state.ts';

/**
 * `GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT`.
 *
 * Spelled numerically so this module can be imported in any host, not just a
 * browser. The WebGPU enum is fixed: VERTEX = 0x1, FRAGMENT = 0x2,
 * COMPUTE = 0x4.
 */
const SHADER_STAGES = 0x3 as GPUShaderStageFlags;

/** The entry-point names the scaffold always generates. */
export const VERTEX_ENTRY = 'vs';
export const FRAGMENT_ENTRY = 'fs';

/**
 * `OBJECT_BLOCK` field offsets, resolved once.
 *
 * Written as a literal rather than a `find` because `ObjectUniforms.pack` runs
 * once per object per frame; at 2000 objects a linear search per field per
 * object is 20 000 string comparisons a frame for no reason.
 */
const MODEL_FIELD = objectField('model');
const NORMAL_MATRIX_FIELD = objectField('normalMatrix');
const OBJECT_ID_FIELD = objectField('objectId');
const VISIBILITY_FIELD = objectField('visibility');

function objectField(name: string): { offset: number; size: number; components: number } {
  const f = OBJECT_BLOCK.fields.find((x) => x.name === name);
  if (f === undefined) {
    // Unreachable unless OBJECT_BLOCK changes, which is a compile-time-visible
    // edit to src/core/slot.ts. Failing loudly beats a silent wrong offset.
    fail('INTERNAL_INVARIANT', `OBJECT_BLOCK has no field "${name}".`, {
      why: 'The object packing code is generated from a hand-written list of field names that must match OBJECT_FIELDS in src/core/slot.ts.',
      fix: 'Add the field to the offset table in this file, or remove it from OBJECT_FIELDS.',
    });
  }
  return f;
}

/** Byte offset to float index, given the byte offset of the record's field. */
function base32(byteBase: number, fieldOffset: number): number {
  return (byteBase + fieldOffset) >> 2;
}

// ---------------------------------------------------------------------------
// Device-scoped caches
// ---------------------------------------------------------------------------

/**
 * Everything apse caches per device.
 *
 * A `GPUBindGroupLayout` per material would defeat the entire point of the
 * scaffold: a scene with 500 materials would create 2000 layout objects, and
 * the driver would deduplicate the underlying state anyway, so the cost would
 * be pure JS and pure waste. All of it is keyed by a hash and looked up in a
 * `Map`, so creation is a numeric compare.
 */
export class DeviceCache {
  readonly device: GPUDevice;

  #sceneBGL: GPUBindGroupLayout | null = null;
  readonly #materialBGLs = new Map<number, GPUBindGroupLayout>();
  readonly #textureBGLs = new Map<string, GPUBindGroupLayout>();
  readonly #pipelineLayouts = new Map<string, GPUPipelineLayout>();
  readonly #pipelines = new Map<number, Promise<GPURenderPipeline>>();
  readonly #samplers = new Map<string, GPUSampler>();
  #sceneUniforms: SceneUniformBuffer | null = null;
  readonly #frameUniforms = new Map<string, FrameUniforms>();
  readonly #objectUniforms = new Map<string, ObjectUniforms>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * `@group(0)`: both reserved uniform regions, in one group.
   *
   * Two entries, because a WGSL module may declare exactly one variable per
   * `(group, binding)` and the frame and the object are two different structs in
   * one buffer. One of them is dynamic and the other is not:
   *
   *  - `@binding(0)` `obj` has `hasDynamicOffset`, and its range is a whole
   *    {@link SCENE_BLOCK}.stride. A draw supplies the object's offset; the
   *    stride is what a driver may round the range up to, so binding a whole
   *    stride is what keeps the *last* object in a full buffer from asking for
   *    bytes past its end.
   *  - `@binding(1)` `frame` is static, at byte 0, and every draw in every pass
   *    sees the same 432 bytes. `minBindingSize` is the struct's own size, not
   *    the reserved region: the reserved region is padded so the *object* side
   *    can be stride-addressed, and declaring the padding as a minimum would
   *    make the binding larger than the frame ever is.
   *
   * Both together cover the frame region and one object stride, which is
   * {@link SceneBlockSpec.minimumByteLength} — the smallest buffer a scene with
   * any object at all can be.
   */
  sceneBindGroupLayout(): GPUBindGroupLayout {
    if (this.#sceneBGL !== null) return this.#sceneBGL;
    this.#sceneBGL = this.device.createBindGroupLayout({
      label: 'apse:scene',
      entries: [
        {
          binding: SCENE_BLOCK.object.binding,
          visibility: SHADER_STAGES,
          buffer: {
            type: 'uniform',
            hasDynamicOffset: true,
            minBindingSize: OBJECT_BLOCK.size,
          },
        },
        {
          binding: SCENE_BLOCK.frame.binding,
          visibility: SHADER_STAGES,
          buffer: { type: 'uniform', minBindingSize: FRAME_BLOCK.size },
        },
      ],
    });
    return this.#sceneBGL;
  }

  /**
   * @group(1). One per distinct block *size* — the generated `MaterialData`
   * differs between materials, and `minBindingSize` is worth keeping, so the
   * size rather than the field list is the cache key. In practice an
   * application has a handful of distinct material block sizes.
   */
  materialBindGroupLayout(size: number): GPUBindGroupLayout {
    const hit = this.#materialBGLs.get(size);
    if (hit !== undefined) return hit;
    const created = this.device.createBindGroupLayout({
      label: `apse:material:${size}`,
      entries: [{
        binding: 0,
        visibility: SHADER_STAGES,
        buffer: { type: 'uniform', minBindingSize: size },
      }],
    });
    this.#materialBGLs.set(size, created);
    return created;
  }

  /** @group(2). Keyed by the declared texture and sampler signature. */
  textureBindGroupLayout(resolved: ResolvedMaterialSpec): GPUBindGroupLayout {
    if (resolved.textures.length === 0) {
      fail('INTERNAL_INVARIANT', 'Asked for a texture bind group layout for an untextured material.', {
        why: 'A material with no texture slots emits no @group(3) declarations, so there is no layout to build.',
        fix: 'This is a bug in apse. Report it with the material spec.',
      });
    }
    const key = textureLayoutKey(resolved);
    const hit = this.#textureBGLs.get(key);
    if (hit !== undefined) return hit;
    const created = this.device.createBindGroupLayout({
      label: `apse:texture:${key}`,
      entries: [
        ...resolved.textures.map((t) => ({
          binding: t.bindingIndex,
          visibility: SHADER_STAGES,
          texture: {
            sampleType: t.sampleType,
            viewDimension: textureViewDimension(t.kind),
            multisampled: false,
          },
        })),
        ...resolved.samplers.map((s) => ({
          binding: s.bindingIndex,
          visibility: SHADER_STAGES,
          sampler: { type: s.layoutType },
        })),
      ],
    });
    this.#textureBGLs.set(key, created);
    return created;
  }

  /**
   * The pipeline layout.
   *
   * A `null` entry reserves a group index without declaring anything in it,
   * which is what lets an untextured or slotless material keep `@group(2)` at
   * 2 while omitting `@group(1)`. The renderer then never has to bind it. The
   * order is `BIND_GROUP` — scene, material, texture — and it is positional
   * because that is what a pipeline layout is.
   */
  pipelineLayout(resolved: ResolvedMaterialSpec): GPUPipelineLayout {
    const key = `${resolved.materialBlock?.size ?? 0}:${textureLayoutKey(resolved)}`;
    const hit = this.#pipelineLayouts.get(key);
    if (hit !== undefined) return hit;
    const created = this.device.createPipelineLayout({
      label: `apse:layout:${key}`,
      bindGroupLayouts: [
        this.sceneBindGroupLayout(),
        resolved.materialBlock === null ? undefined : this.materialBindGroupLayout(resolved.materialBlock.size),
        resolved.textures.length === 0 ? undefined : this.textureBindGroupLayout(resolved),
      ],
    });
    this.#pipelineLayouts.set(key, created);
    return created;
  }

  /**
   * The render pipeline, keyed by the full generated program.
   *
   * Stores the *promise*, not the result, so two materials created in the same
   * tick with the same key issue one compile rather than two. See
   * {@link GeneratedShader.pipelineKey} for why the shader body is part of the
   * key while the bind group layouts are keyed without it.
   */
  renderPipeline(
    generated: GeneratedShader,
    depthFormat: GPUTextureFormat,
    stripIndexFormat: GPUIndexFormat | undefined,
  ): Promise<GPURenderPipeline> {
    const key = generated.pipelineKey;
    const hit = this.#pipelines.get(key);
    if (hit !== undefined) return hit;
    const promise = compilePipeline(this.device, generated, this.pipelineLayout(generated.resolved), depthFormat, stripIndexFormat);
    this.#pipelines.set(key, promise);
    // A failed compile must not poison the cache for every later material.
    promise.catch(() => this.#pipelines.delete(key));
    return promise;
  }

  /** One `GPUSampler` per distinct configuration, shared by every material. */
  sampler(s: ResolvedSampler): GPUSampler {
    const hit = this.#samplers.get(s.key);
    if (hit !== undefined) return hit;
    const created = this.device.createSampler({
      label: `apse:sampler:${s.key}`,
      addressModeU: s.addressMode,
      addressModeV: s.addressMode,
      addressModeW: s.addressMode,
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: s.mipmapFilter ? 'linear' : 'nearest',
      ...(s.compare ? { compare: 'less' } : {}),
    });
    this.#samplers.set(s.key, created);
    return created;
  }

  /**
   * The one scene uniform buffer for this device, and the one true instance.
   *
   * Not keyed by anything. Every route to a scene uniform — the renderer, a
   * material with no renderer, a second renderer, a test — resolves through
   * here, and they all get the same `GPUBuffer` and the same `GPUBindGroup`.
   * That is not tidiness. If the renderer and a material each constructed their
   * own, the renderer would write the camera matrices and the transforms into a
   * buffer the material's bind group never reads, and every object would draw
   * with an identity transform: no validation error, no console line, and a
   * scene that is simply wrong. apse shipped that bug once. A single
   * unkeyed instance makes it unrepresentable rather than merely discouraged —
   * there is no second place to get a scene buffer from.
   */
  sceneUniforms(): SceneUniformBuffer {
    if (this.#sceneUniforms === null) this.#sceneUniforms = new SceneUniformBuffer(this);
    return this.#sceneUniforms;
  }

  /**
   * The frame face of {@link sceneUniforms}.
   *
   * Memoised per label so `cache.frameUniforms(x) === cache.frameUniforms(x)`,
   * but the label selects nothing about the *buffer*: two labels are two faces
   * of one allocation. `DEFAULT_FRAME_LABEL` and `DEFAULT_OBJECT_LABEL` are
   * different strings and resolve to the same buffer, which is exactly the point.
   */
  frameUniforms(label: string): FrameUniforms {
    const hit = this.#frameUniforms.get(label);
    if (hit !== undefined) return hit;
    const created = new FrameUniforms(this.device, label);
    this.#frameUniforms.set(label, created);
    return created;
  }

  /**
   * The object face of {@link sceneUniforms}. `maxObjects` is a floor on the
   * shared buffer's capacity, not a private one: two callers asking for
   * different sizes still write into the same allocation.
   */
  objectUniforms(label: string, maxObjects: number): ObjectUniforms {
    const hit = this.#objectUniforms.get(label);
    if (hit !== undefined) return hit;
    const created = new ObjectUniforms(this.device, label, maxObjects);
    this.#objectUniforms.set(label, created);
    return created;
  }

  /**
   * Destroys the shared scene uniform buffer this cache created.
   *
   * The bind group layouts, pipeline layouts, pipelines, and samplers are
   * *not* released: none of them has a `destroy()`, and all of them become
   * collectable when the device is. Call this from the renderer's teardown,
   * before the device is dropped, so the uniform buffer is freed
   * deterministically rather than whenever the GC gets round to it.
   */
  disposeSharedUniforms(): void {
    this.#sceneUniforms?.dispose();
    this.#sceneUniforms = null;
    this.#frameUniforms.clear();
    this.#objectUniforms.clear();
  }

  /** Cache sizes, for tests and for a debug overlay. */
  get stats(): { readonly pipelines: number; readonly pipelineLayouts: number; readonly bindGroupLayouts: number; readonly samplers: number } {
    return {
      pipelines: this.#pipelines.size,
      pipelineLayouts: this.#pipelineLayouts.size,
      bindGroupLayouts: 1 + this.#materialBGLs.size + this.#textureBGLs.size,
      samplers: this.#samplers.size,
    };
  }
}

function textureLayoutKey(resolved: ResolvedMaterialSpec): string {
  if (resolved.textures.length === 0) return '';
  return [
    ...resolved.textures.map((t) => `${t.bindingIndex}:${t.kind}:${t.sampleType}`),
    ...resolved.samplers.map((s) => `${s.bindingIndex}:${s.layoutType}`),
  ].join('|');
}

const deviceCaches = new WeakMap<GPUDevice, DeviceCache>();

/** The per-device cache. One per device, for the life of the device. */
export function deviceCache(device: GPUDevice): DeviceCache {
  let cache = deviceCaches.get(device);
  if (cache === undefined) {
    cache = new DeviceCache(device);
    deviceCaches.set(device, cache);
  }
  return cache;
}


// ---------------------------------------------------------------------------
// Pipeline compilation
// ---------------------------------------------------------------------------

async function compilePipeline(
  device: GPUDevice,
  generated: GeneratedShader,
  layout: GPUPipelineLayout,
  depthFormat: GPUTextureFormat,
  stripIndexFormat: GPUIndexFormat | undefined,
): Promise<GPURenderPipeline> {
  const r = generated.resolved;
  device.pushErrorScope('validation');
  // `!` rather than a nullable: every path that can reach `return` assigns it,
  // and the two failure paths below end in `never` as far as the caller is
  // concerned. Tracking nullability across the two error routes buys nothing.
  let pipeline!: GPURenderPipeline;
  // Hoisted so a failure can ask the module for its real diagnostics rather than
  // reporting only the validation wrapper.
  let shaderModule: GPUShaderModule | null = null;
  try {
    // Named `shaderModule`, not `module`: Bun's types declare a *global*
    // `module: NodeModule`, so a local named `module` shadows a real global and
    // every later reference to the shader module silently resolves to it. That
    // is not a stylistic preference — it is the bug this rename fixes.
    shaderModule = device.createShaderModule({
      label: `apse:shader:${r.name}`,
      code: generated.code,
    });
    pipeline = await device.createRenderPipelineAsync({
      label: `apse:pipeline:${r.name}`,
      layout,
      vertex: {
        module: shaderModule,
        entryPoint: VERTEX_ENTRY,
        // `gpuLayouts()`, not `gpuLayout()`. An instanced layout has two slots
        // — vertex data and the per-instance transform — and WGSL reads both
        // through one `VertexIn` struct. Passing only the first produces a
        // pipeline that fails at draw time with "vertex attribute slot 6 used
        // in (vs) is not present in the VertexState", which names a slot number
        // rather than the mistake.
        buffers: r.layout.gpuLayouts(),
      },
      fragment: {
        module: shaderModule,
        entryPoint: FRAGMENT_ENTRY,
        targets: r.targets.map((t) => ({
          format: t.format,
          blend: toGPUBlend(t.blend),
          writeMask: t.writeMask,
        })),
      },
      primitive: {
        topology: r.topology,
        cullMode: r.cull,
        frontFace: r.frontFace,
        ...(stripIndexFormat !== undefined ? { stripIndexFormat } : {}),
      },
      depthStencil: {
        format: depthFormat,
        depthWriteEnabled: r.depth.write,
        depthCompare: r.depth.compare,
      },
      multisample: { count: r.sampleCount },
    });
  } catch (cause) {
    await device.popErrorScope();
    await pipelineFailure(r, generated, shaderModule, cause);
  }

  // The async path can also fail without rejecting, by setting a device error.
  // Both routes are real, and both get the same typed error.
  const scopeError = await device.popErrorScope();
  if (scopeError !== null) await pipelineFailure(r, generated, shaderModule, scopeError.message);
  return pipeline;
}

/**
 * Where in the generated program a diagnostic points.
 *
 * `vertex` and `fragment` are the two a material author can act on, and they are
 * the two the error scope gets *wrong* often enough to matter: Dawn's wrapper
 * text frequently says "while validating vertex stage" for a mistake in the
 * fragment body, and a person who believes it looks in the wrong stage. The
 * other two exist because a real error can point at neither — the prelude is
 * user text, and the declarations above it are generated.
 */
export type ShaderStage = 'vertex' | 'fragment' | 'prelude' | 'generated';

/**
 * Which section of `code` line `lineNum` falls in.
 *
 * Read from the section banners `generateScaffold` emits rather than from a
 * regex over `fn vs` / `fn fs`, because the banners are *the same generator's*
 * description of the file: a change to the template moves both together, and
 * there is no second place to forget.
 *
 * **A line outside the program is `generated`,** and that is a decision rather
 * than a leftover. Walking the banners and returning whatever the last one was
 * would report a diagnostic from an included file — or a `lineNum` of 0 from an
 * implementation that does not track it — as `fragment`, because the fragment
 * banner is last. Reporting the *last* stage is the same class of mistake as
 * reporting the vertex stage by default, and it is worse because it looks
 * deliberate.
 */
export function stageAtLine(code: string, lineNum: number): ShaderStage {
  const lines = code.split('\n');
  if (!Number.isInteger(lineNum) || lineNum < 1 || lineNum > lines.length) return 'generated';
  let stage: ShaderStage = 'generated';
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('// ---- prelude')) stage = 'prelude';
    else if (line.startsWith('// ---- generated by apse: vertex stage')) stage = 'vertex';
    else if (line.startsWith('// ---- generated by apse: fragment stage')) stage = 'fragment';
    if (i + 1 === lineNum) return stage;
  }
  return stage;
}

/** What the headline says. `null` means "the compiler told us nothing usable". */
interface CompilerDiagnostic {
  readonly stage: ShaderStage;
  readonly lineNum: number;
  readonly linePos: number;
  readonly message: string;
  /** The source line the diagnostic points at, trimmed. */
  readonly excerpt: string;
}

/** How many compiler errors to quote. Four is the point past which nobody reads. */
const MAX_DIAGNOSTICS = 4;

/**
 * Turns a pipeline failure into a typed error carrying the *compiler's* message.
 *
 * The error scope's text is a wrapper: it names a stage and the validation rule
 * and points at a line of the generated program, and its stage attribution is
 * frequently wrong. `getCompilationInfo()` has the real diagnostic — file, line,
 * column, and the message itself — and it is right there. Reading it is the
 * difference between "expected ';' for variable declaration" and "while
 * validating vertex stage", which is a *lie* when the mistake was in the
 * fragment body.
 *
 * Three properties this has to keep, each of which has been broken here before:
 *
 * 1. **The stage is derived from the line, not assumed.** Every diagnostic is
 *    attributed with `stageAtLine`, and the headline names the *first* error's
 *    stage. There is no path through here that reports "vertex" by default.
 * 2. **Nothing from the compiler is discarded.** `lineNum`, `linePos`, and
 *    `message` all reach the thrown error, for every diagnostic, and none of them
 *    is used only to build a string that is then dropped. In particular the
 *    *whole* list goes into `why` — truncating the detail to fit a headline and
 *    losing the rest was the earlier bug, and the excerpt of the offending line
 *    is what makes the line number usable at all in a 3 KB program that is mostly
 *    generated.
 * 3. **It degrades rather than throws.** `getCompilationInfo` is optional in
 *    practice — some implementations do not have it, some reject it, and some
 *    hand back a message list with no `errors` in it. Each of those falls back
 *    to the error-scope text *and says so*, instead of failing to report the
 *    failure or reporting a stage it invented.
 *
 * Source excerpts are included because a line number in a generated program is
 * not something a person can act on, and because that program is mostly noise
 * around the handful of lines the author wrote.
 */
async function pipelineFailure(
  r: ResolvedMaterialSpec,
  generated: GeneratedShader,
  shaderModule: GPUShaderModule | null,
  raw: unknown,
): Promise<never> {
  const scopeText = raw instanceof Error ? raw.message : String(raw);
  const diagnostics = await compilerDiagnostics(generated.code, shaderModule);

  const headline = diagnostics.length > 0
    ? `${diagnostics[0]!.stage} stage, line ${diagnostics[0]!.lineNum} column ${diagnostics[0]!.linePos}: ${diagnostics[0]!.message}`
    : scopeText;

  const detail = diagnostics.length > 0
    ? diagnostics
      .map((d) => `  [${d.stage}] ${d.lineNum}:${d.linePos} ${d.message}\n    ${d.excerpt}`)
      .join('\n')
    : `  The shader compiler's own diagnostics were unavailable, so this is the validation error and it names a stage that may be wrong:\n  ${scopeText}`;

  fail('SHADER_COMPILE_FAILED',
    `Material "${r.name}" did not compile: ${headline}`, {
    why: `The WGSL was rejected by the shader compiler. apse generates the bindings, the uniform structs, the varying struct, and the entry-point signatures, so the fault is almost always in the \`vertex\`/\`fragment\` statements rather than in generated code.\n${detail}`,
    fix: `The line and column are in the generated program; the quoted line is the source. Fix the statement it points at. If the stage is \`prelude\`, check a helper's name and arity — that text is yours, and it is the one part of the program the generator did not write. describeMaterial() lists every field that exists, and \`scaffold: true\` prints the full ${generated.byteLength}-byte program.`,
    cause: raw instanceof Error ? raw : undefined,
  });
}

/**
 * The compiler's own errors, or an empty list.
 *
 * Never throws: every route that can fail here — no `getCompilationInfo`, a
 * rejected promise, a message list with no `messages` on it — returns `[]`, and
 * the caller falls back to the validation error. A diagnostic path that can
 * itself throw replaces a useful message with an unhelpful one, which is the
 * failure mode this whole function exists to prevent.
 */
async function compilerDiagnostics(
  code: string,
  shaderModule: GPUShaderModule | null,
): Promise<CompilerDiagnostic[]> {
  if (shaderModule === null) return [];
  let info: GPUCompilationInfo;
  try {
    info = await shaderModule.getCompilationInfo();
  } catch {
    return [];
  }
  const messages = info?.messages;
  if (!Array.isArray(messages)) return [];
  const lines = code.split('\n');
  const out: CompilerDiagnostic[] = [];
  for (const m of messages) {
    if (m.type !== 'error') continue;
    out.push({
      stage: stageAtLine(code, m.lineNum),
      lineNum: m.lineNum,
      linePos: m.linePos,
      message: m.message,
      excerpt: (lines[m.lineNum - 1] ?? '').trim(),
    });
    if (out.length === MAX_DIAGNOSTICS) break;
  }
  return out;
}

function toGPUBlend(blend: BlendSpec | null): GPUBlendState | undefined {
  if (blend === null) return undefined;
  return {
    color: { srcFactor: blend.color.srcFactor, dstFactor: blend.color.dstFactor, operation: blend.color.operation },
    alpha: { srcFactor: blend.alpha.srcFactor, dstFactor: blend.alpha.dstFactor, operation: blend.alpha.operation },
  };
}

function deviceLimits(device: GPUDevice): { maxInterStageShaderVariables: number } {
  // `maxInterStageComponents` is not declared in the current @webgpu/types, so
  // it is read defensively: present on every real device, absent in some
  // releases of the type package. The scaffold falls back to the core-profile
  // default of 60 when it is missing, which is the correct value anyway.
  const l = device.limits as GPUSupportedLimits & { maxInterStageComponents?: number };
  return {
    maxInterStageShaderVariables: device.limits.maxInterStageShaderVariables,
    ...(l.maxInterStageComponents !== undefined ? { maxInterStageComponents: l.maxInterStageComponents } : {}),
  };
}

// ---------------------------------------------------------------------------
// The shared scene uniform
//
// One `GPUBuffer` and one `GPUBindGroup` hold the frame at byte 0 and every
// object's transform after it. Two classes below, `FrameUniforms` and
// `ObjectUniforms`, are faces of it, because the frame is written whole once a
// frame and the objects are written a slot at a time, and one class owning two
// write disciplines would be a worse API than two names over one buffer.
// ---------------------------------------------------------------------------

/**
 * The scene uniform buffer: frame at byte 0, one object slot per stride after.
 *
 * This is the *only* place a `GPUBuffer` or a `GPUBindGroup` for the reserved
 * uniforms is created. `DeviceCache.sceneUniforms` hands out one instance per
 * device, `FrameUniforms` and `ObjectUniforms` hold a reference to it, and
 * nothing else can make a second one — which is the property that makes the
 * class of bug apse shipped (the renderer writing uniforms into a buffer no bind
 * group reads) impossible rather than merely unlikely.
 *
 * # Allocation is lazy, and growth throws the mirror away
 *
 * The buffer is created on first *use*, not at material creation: a scene that is
 * built but never rendered should not pay for a megabyte, and most materials in a
 * typical project are never drawn. Growth destroys the old buffer and allocates a
 * new one, because a `GPUBindGroup` captures the buffer it was created with and
 * a new buffer is the only correct thing to bind. The CPU mirror is recreated
 * with it, which is why {@link generation} exists: every object write made
 * against the old mirror is gone, and any caller caching "this slot is already
 * packed" has to be told.
 *
 * # Two mirrors, one buffer
 *
 * The frame keeps its own `ArrayBuffer` and the object array keeps another. They
 * are concatenated by `writeBuffer`, which takes a byte offset into the source as
 * well as a byte offset into the destination, so there is no reason to make them
 * one allocation — and a reason not to: the frame's mirror then survives a
 * growth of the object array, so a realloc cannot silently zero the camera.
 */
export class SceneUniformBuffer implements Disposable {
  /** Where each region sits. Resolved once, from the built blocks. */
  readonly spec: SceneBlockSpec = SCENE_BLOCK;
  /** CPU mirror of the frame region. Written through `FrameUniforms.set`. */
  readonly frameBlock = new UniformBlock(FRAME_BLOCK);
  /** The device this buffer belongs to. A `GPUBuffer` is not shareable. */
  readonly device: GPUDevice;
  /** Object slots the buffer holds. Grows by doubling. */
  capacity: number;
  readonly label: string;

  readonly #cache: DeviceCache;
  #buffer: GPUBuffer | null = null;
  #bindGroup: GPUBindGroup | null = null;
  #data: ArrayBuffer | null = null;
  #f32: Float32Array | null = null;
  #u32: Uint32Array | null = null;
  #frameDirty = false;
  /** Bumped by every reallocation. See {@link generation}. */
  #generation = 1;
  #disposed = false;

  constructor(cache: DeviceCache, label = DEFAULT_SCENE_LABEL) {
    this.#cache = cache;
    this.device = cache.device;
    this.label = label;
    // One object is the true minimum: a zero-slot buffer cannot hold a legal
    // object binding, and doubling from zero would never terminate.
    this.capacity = 1;
  }

  /** True once the buffer has been created by a draw. */
  get allocated(): boolean {
    return this.#buffer !== null;
  }

  /**
   * A token that changes exactly when the object mirror is replaced.
   *
   * A caller that skips packing a slot because the node's transform has not
   * changed must also skip it because the *mirror* is new — the bytes it is
   * comparing against no longer exist. This is the number to put in that cache's
   * key, and it is why it is public.
   */
  get generation(): number {
    return this.#generation;
  }

  get buffer(): GPUBuffer {
    this.assertLive(`SceneUniformBuffer "${this.label}"`);
    this.allocate(this.capacity);
    return this.#buffer as GPUBuffer;
  }

  /**
   * `@group(0)`. One bind, both regions. Rebuilt transparently after a growth.
   */
  get bindGroup(): GPUBindGroup {
    this.assertLive(`SceneUniformBuffer "${this.label}"`);
    this.allocate(this.capacity);
    if (this.#bindGroup === null) {
      this.#bindGroup = this.#cache.device.createBindGroup({
        label: `apse:${this.label}:group`,
        layout: this.#cache.sceneBindGroupLayout(),
        entries: [
          {
            binding: SCENE_BLOCK.object.binding,
            resource: {
              buffer: this.#buffer as GPUBuffer,
              offset: 0,
              // A whole stride rather than OBJECT_BLOCK.size. Both satisfy the
              // spec — `setBindGroup` range-checks `offset + dynamicOffset +
              // minBindingSize`, and `minBindingSize` is the struct — but the
              // stride is the alignment a dynamic offset is measured in, so this
              // is the range the shader's view is rounded to. It also makes the
              // last slot's range end exactly at the end of the buffer instead of
              // stopping 128 bytes short of it.
              size: SCENE_BLOCK.object.byteLength,
            },
          },
          {
            binding: SCENE_BLOCK.frame.binding,
            resource: {
              buffer: this.#buffer as GPUBuffer,
              offset: SCENE_BLOCK.frame.byteOffset,
              size: FRAME_BLOCK.size,
            },
          },
        ],
      });
    }
    return this.#bindGroup;
  }

  /**
   * Byte offset of object `index`, growing the buffer by doubling if needed.
   *
   * This is the value a draw passes as its dynamic offset, and it is the same
   * value the packer writes at and `fieldOffset` reports, so the three cannot
   * disagree.
   */
  objectOffsetFor(index: number): number {
    this.assertLive(`SceneUniformBuffer "${this.label}"`);
    // Two ways to be unallocated: never allocated at all, or allocated smaller
    // than `index` needs. Only handling the second is a null-dereference
    // waiting for the first frame of a scene that fits inside the default
    // capacity — which is most scenes. The first allocation takes the reserved
    // capacity rather than `index + 1`, so packing slot 0 does not shrink the
    // buffer below what the caller asked to reserve and then reallocate twice.
    if (this.#buffer === null) this.allocate(Math.max(1, this.capacity, index + 1));
    else if (index >= this.capacity) this.allocate(index + 1);
    return SCENE_BLOCK.object.byteOffset + index * SCENE_BLOCK.stride;
  }

  /** Asks for room for `count` objects without allocating it. */
  reserve(count: number): void {
    if (this.#disposed) return;
    if (count > this.capacity && this.#buffer === null) this.capacity = Math.max(1, count | 0);
  }

  /** Grows the buffer to hold `count` object slots. Safe to call repeatedly. */
  allocate(count: number): void {
    if (this.#disposed) return;
    if (this.#buffer !== null && count <= this.capacity) return;
    // Grow by doubling, so a scene that keeps adding objects reallocates a
    // logarithmic number of times rather than once per object. The renderer also
    // rounds, and rounding twice is harmless; rounding nowhere is not.
    this.capacity = nextPowerOfTwo(Math.max(1, count | 0));
    this.#buffer?.destroy();
    this.#buffer = this.#cache.device.createBuffer({
      // The object array is what sizes the buffer, so the count in the label is
      // the object count — and the label names both regions, because a recorded
      // write is all the frame debugger has to go on.
      label: `apse:${this.label}:${this.capacity}`,
      size: SCENE_BLOCK.object.byteOffset + this.capacity * SCENE_BLOCK.stride,
      // STORAGE would allow read_write; both regions are written by the CPU and
      // read by the GPU, which is exactly UNIFORM | COPY_DST.
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // A bind group captures its buffer, so growth invalidates it.
    this.#bindGroup = null;
    this.#data = new ArrayBuffer(this.capacity * SCENE_BLOCK.stride);
    this.#f32 = new Float32Array(this.#data);
    this.#u32 = new Uint32Array(this.#data);
    // A new mirror means every previous write is gone, and the caller's cache of
    // what it has already packed is now wrong. This is the line that tells it.
    this.#generation++;
    // The frame's *mirror* survived, but its bytes on the GPU did not: a new
    // buffer is zeroed, and a zeroed `Frame` is a zeroed `viewProj` — degenerate
    // triangles, successful draws, empty screen. A block that was clean before
    // the growth would otherwise never be flushed again, so it is marked dirty
    // here. One 432-byte write, once, on the frame that grew the buffer.
    this.#frameDirty = true;
  }

  /**
   * Packs one object's fields at `index`, into the CPU mirror.
   *
   * Nothing is uploaded here: the renderer packs every object during the
   * traversal and then calls {@link uploadObjects} once, so a 2000-object frame
   * is one `writeBuffer` rather than 2000.
   *
   * The field offsets come from `OBJECT_BLOCK` — the same generated struct the
   * WGSL declares — so a transform cannot be written at the wrong byte. The
   * offsets are resolved once, at module load, because `find` in a per-object
   * loop is exactly the kind of cost that makes a scene stutter.
   */
  pack(
    index: number,
    model: ArrayLike<number>,
    normalMatrix?: ArrayLike<number>,
    objectId: number = index,
    instanceId = 0,
    visibility = 1,
  ): void {
    this.objectOffsetFor(index);
    // The mirror is the object array alone, so the frame's base offset does not
    // appear here. `writeBuffer` takes the destination offset separately, which
    // is what keeps the two regions from having to share an allocation.
    const base = index * SCENE_BLOCK.stride;
    const f32 = this.#f32 as Float32Array;

    // model: mat4x4f at the start of the record.
    for (let i = 0; i < 16; i++) f32[base32(base, MODEL_FIELD.offset) + i] = model[i];

    // normalMatrix: mat3x3f is three 16-byte-aligned columns of 4 floats each,
    // so column j starts at offset + 16j and only its first 3 floats are data.
    if (normalMatrix !== undefined) {
      const at = base32(base, NORMAL_MATRIX_FIELD.offset);
      for (let j = 0; j < 3; j++) {
        const col = j * 4;
        f32[at + col] = normalMatrix[col];
        f32[at + col + 1] = normalMatrix[col + 1];
        f32[at + col + 2] = normalMatrix[col + 2];
      }
    }

    // objectId and instanceId are adjacent u32s in ObjectData.
    const ids = base32(base, OBJECT_ID_FIELD.offset);
    (this.#u32 as Uint32Array)[ids] = objectId >>> 0;
    (this.#u32 as Uint32Array)[ids + 1] = instanceId >>> 0;
    f32[base32(base, VISIBILITY_FIELD.offset)] = visibility;
  }

  /** Uploads object slots `[0, count)` in one write. */
  uploadFrom(count: number): void {
    const slots = Math.max(0, Math.min(count, this.capacity));
    if (slots === 0) return;
    const bytes = slots * SCENE_BLOCK.stride;
    this.write(SCENE_BLOCK.object.byteOffset, 0, bytes);
  }

  /**
   * Uploads one object's own bytes — `OBJECT_BLOCK.size` of them, and nothing
   * else.
   *
   * The narrowest upload apse can make for an object: it cannot reach the frame,
   * and it cannot reach either neighbour. `OBJECT_BLOCK.size` is 128, a multiple
   * of 4, so it is a legal `writeBuffer` size; a range that were not would
   * invalidate the whole command buffer with nothing to catch it.
   */
  uploadObject(index: number): void {
    if (index < 0 || index >= this.capacity) return;
    this.write(
      SCENE_BLOCK.object.byteOffset + index * SCENE_BLOCK.stride,
      index * SCENE_BLOCK.stride,
      OBJECT_BLOCK.size,
    );
  }

  /**
   * Uploads object slots `[lo, hi]`, both ends inclusive, in one write.
   *
   * Whole strides, so one `writeBuffer` covers the range. Slots, not bytes: byte
   * 0 of this buffer is the *frame*, so a byte-offset API here is an API that can
   * overwrite the camera, and the type of the argument is the cheapest guard
   * against it.
   */
  uploadObjects(lo: number, hi: number): void {
    const a = Math.max(0, lo);
    const b = Math.min(hi, this.capacity - 1);
    if (b < a) return;
    this.write(
      SCENE_BLOCK.object.byteOffset + a * SCENE_BLOCK.stride,
      a * SCENE_BLOCK.stride,
      (b - a + 1) * SCENE_BLOCK.stride,
    );
  }

  /** Bytes the buffer occupies for its current capacity. */
  get byteLength(): number {
    return SCENE_BLOCK.object.byteOffset + this.capacity * SCENE_BLOCK.stride;
  }

  // --- the frame region ------------------------------------------------------

  /** Marks the frame block dirty. `flush` is a no-op without it. */
  markFrameDirty(): void {
    this.#frameDirty = true;
  }

  /** Copies the frame mirror to byte 0. A no-op when nothing marked it dirty. */
  flushFrame(): void {
    if (!this.#frameDirty || this.#disposed) return;
    this.uploadFrame(this.frameBlock.data);
  }

  /**
   * Copies an already-packed frame block onto byte 0.
   *
   * `bytes` must be exactly {@link UniformBlockSpec}.size — it is the block the
   * renderer already packed, so re-packing it here would only add a copy. The
   * source is the frame's own mirror, not the object array's: the two are
   * separate allocations precisely so a growth of the objects cannot take the
   * camera with it.
   *
   * This allocates if nothing has yet. The frame is at byte 0 of a buffer the
   * objects also live in, so the first frame of a scene with nothing packed yet
   * is still a first write to that buffer.
   */
  uploadFrame(bytes: ArrayBuffer): void {
    if (this.#disposed) return;
    if (bytes.byteLength !== FRAME_BLOCK.size) {
      fail('INTERNAL_INVARIANT',
        `Uploading a frame block of ${bytes.byteLength} bytes, but the frame block is ${FRAME_BLOCK.size}.`, {
        why: 'The frame block layout is generated by apse. A differently sized buffer means the caller packed against a different struct, and the GPU would read the fields at the wrong offsets.',
        fix: `Pack against FRAME_BLOCK, or use \`uniforms.block\` and set fields by name.`,
      });
    }
    // Cleared *after* the write, not before: reading `this.buffer` can allocate,
    // and an allocation marks the frame dirty because the new buffer is zeroed.
    // The bytes are on the GPU by the time this line runs, so clean is the truth
    // either way.
    this.#cache.device.queue.writeBuffer(this.buffer, SCENE_BLOCK.frame.byteOffset, bytes);
    this.#frameDirty = false;
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#buffer?.destroy();
    this.#buffer = null;
    this.#bindGroup = null;
    this.#data = null;
    this.#f32 = null;
    this.#u32 = null;
  }

  /**
   * The one `writeBuffer` call, with both offsets kept straight.
   *
   * `bufferOffset` is where the bytes land in the GPU buffer and `dataOffset` is
   * where they start in the mirror, and they differ by the frame region's size
   * for every object write. Writing one number where the other belongs is a
   * silent corruption of somebody else's data, so they are separate parameters.
   */
  write(bufferOffset: number, dataOffset: number, size: number): void {
    if (this.#disposed || this.#buffer === null || this.#data === null) return;
    this.#cache.device.queue.writeBuffer(this.#buffer, bufferOffset, this.#data, dataOffset, size);
  }

  private assertLive(what: string): void {
    if (!this.#disposed) return;
    fail('INTERNAL_INVARIANT', `${what} was used after it was disposed.`, {
      why: 'The buffer it binds is destroyed, so every object in the scene would draw with an undefined transform.',
      fix: 'Hold the shared uniforms for as long as any object can still be drawn.',
    });
  }
}

/**
 * Smallest power of two that is ≥ `n`, and 1 for anything below it.
 *
 * Local rather than imported: `material` sits below `render` in the layer graph
 * and a power-of-two helper is four lines of arithmetic, not a dependency across
 * a boundary the size gate exists to police.
 */
function nextPowerOfTwo(n: number): number {
  let out = 1;
  while (out < n) out *= 2;
  return out;
}

// ---------------------------------------------------------------------------
// Shared frame uniforms
// ---------------------------------------------------------------------------

/**
 * The frame region of {@link SceneUniformBuffer}, as its own name.
 *
 * 500 materials sharing one 432-byte region instead of 500 is the difference
 * between a scene that uploads 216 KB of identical frame state per frame and one
 * that uploads 432 bytes.
 *
 * A `Drawable.writeFrameUniform` is idempotent and forwards here, so a renderer
 * that walks materials and calls it per material behaves correctly — it just
 * performs redundant writes, which is why the renderer should call
 * {@link FrameUniforms.flush} once and skip the per-material path.
 *
 * It is a face, not an owner: the buffer and the bind group are
 * {@link DeviceCache.sceneUniforms}' and are shared with {@link ObjectUniforms}.
 */
export class FrameUniforms implements Disposable {
  readonly spec: UniformBlockSpec = FRAME_BLOCK;
  readonly label: string;

  readonly #scene: SceneUniformBuffer;

  constructor(device: GPUDevice, label = DEFAULT_FRAME_LABEL) {
    this.#scene = deviceCache(device).sceneUniforms();
    this.label = label;
  }

  /** CPU-side mirror of the block. Packed by the same code as every other block. */
  get block(): UniformBlock {
    return this.#scene.frameBlock;
  }

  /** The shared buffer. Byte 0 through {@link UniformBlockSpec}.size is this. */
  get buffer(): GPUBuffer {
    return this.#scene.buffer;
  }

  /**
   * `@group(0)`, covering the frame *and* the object. One bind per draw.
   *
   * The frame's own bytes are at a fixed offset inside it, so a pass binds this
   * once and never touches it again.
   */
  get sceneBindGroup(): GPUBindGroup {
    return this.#scene.bindGroup;
  }

  /**
   * @deprecated Use {@link sceneBindGroup}. Kept so `Drawable`, which still names
   * the frame group, keeps compiling; it is the same object, and the object
   * region is the only part a draw varies.
   */
  get bindGroup(): GPUBindGroup {
    return this.#scene.bindGroup;
  }

  /** The buffer both faces share. Identity here *is* the invariant. */
  get sceneBuffer(): SceneUniformBuffer {
    return this.#scene;
  }

  /** Changes when the object array is reallocated. See {@link SceneUniformBuffer.generation}. */
  get generation(): number {
    return this.#scene.generation;
  }

  /**
   * Copies a fully packed frame block to byte 0.
   *
   * `bytes` must be exactly {@link UniformBlockSpec}.size — it is the block the
   * renderer already packed, so re-packing it here would only add a copy.
   */
  upload(bytes: ArrayBuffer): void {
    this.#scene.uploadFrame(bytes);
  }

  /**
   * Copies the internal mirror to the GPU, if anything marked it dirty.
   *
   * Writing frame state through {@link set} rather than through {@link block} is
   * not a style preference: `set` is what marks the block dirty, and `flush` on a
   * clean block does nothing. Write the block behind its back and the GPU keeps
   * a zeroed uniform buffer — every `frame.viewProj` is zero, every triangle is
   * degenerate, every draw succeeds, and the screen is empty. There is no error
   * to find.
   */
  flush(): void {
    this.#scene.flushFrame();
  }

  /** Packs one field and marks the block for {@link flush}. */
  set(name: string, value: number | ArrayLike<number>): void {
    this.#scene.frameBlock.set(name, value);
    this.#scene.markFrameDirty();
  }

  /** Bytes this region occupies, padding included. */
  get size(): number {
    return FRAME_BLOCK.size;
  }

  /**
   * Releases this face's handle to the shared buffer.
   *
   * **It does not free the buffer.** The scene buffer is one allocation per
   * device, and its lifetime belongs to whoever allocated it -- the renderer.
   * A face disposing it would mean `PresentPass.dispose()` destroyed the
   * renderer's uniform storage, so every other material on the device lost its
   * camera: no validation error, every object at identity, on the next frame.
   */
  dispose(): void {}
}

// ---------------------------------------------------------------------------
// Shared object uniforms
// ---------------------------------------------------------------------------

/**
 * The object array of {@link SceneUniformBuffer}, as its own name.
 *
 * `ObjectData` is the same struct for every material, so this is shared with
 * {@link FrameUniforms} for the same reason the frame is. Each object occupies
 * {@link UniformBlockSpec}.stride bytes — the struct size rounded up to the
 * 256-byte `minUniformBufferOffsetAlignment` the API requires of a dynamic
 * offset — and the renderer writes a draw item's transform at
 * {@link SCENE_BLOCK}.object.byteOffset + `objectId` * stride.
 *
 * The buffer is **allocated lazily on first draw**, not at material creation:
 * a scene that is built but never rendered should not pay for a 1 MB
 * allocation, and most materials in a typical project are never drawn. When an
 * object index exceeds the capacity, the buffer **doubles** and the bind group
 * is rebuilt, because a bind group captures the buffer it was created with.
 */
export class ObjectUniforms implements Disposable {
  readonly spec: UniformBlockSpec = OBJECT_BLOCK;
  readonly label: string;

  readonly #scene: SceneUniformBuffer;

  constructor(device: GPUDevice, label = DEFAULT_OBJECT_LABEL, maxObjects = DEFAULT_MAX_OBJECTS) {
    this.#scene = deviceCache(device).sceneUniforms();
    this.label = label;
    // A floor on the shared capacity, not a private one: a second caller asking
    // for more raises the same buffer's capacity rather than making another.
    this.#scene.reserve(maxObjects);
  }

  /** Object slots the shared buffer holds. Grows by doubling. */
  get capacity(): number {
    return this.#scene.capacity;
  }

  /** True once the buffer has been created by a draw. */
  get allocated(): boolean {
    return this.#scene.allocated;
  }

  get buffer(): GPUBuffer {
    return this.#scene.buffer;
  }

  /** `@group(0)`. The same bind group {@link FrameUniforms} hands out. */
  get sceneBindGroup(): GPUBindGroup {
    return this.#scene.bindGroup;
  }

  /**
   * @deprecated Use {@link sceneBindGroup}. Kept so `Drawable`, which still
   * names the object group, keeps compiling. It is the same object the frame
   * returns, which is the point.
   */
  get bindGroup(): GPUBindGroup {
    return this.#scene.bindGroup;
  }

  /** The buffer both faces share. Identity here *is* the invariant. */
  get sceneBuffer(): SceneUniformBuffer {
    return this.#scene;
  }

  /** Changes when the object array is reallocated. See {@link SceneUniformBuffer.generation}. */
  get generation(): number {
    return this.#scene.generation;
  }

  /**
   * The dynamic offset for object `index`, growing the buffer by doubling if
   * needed. This is what a draw passes to `setBindGroup`.
   */
  offsetFor(index: number): number {
    return this.#scene.objectOffsetFor(index);
  }

  /** Grows the shared buffer to hold `count` object slots. Safe to call repeatedly. */
  allocate(count: number): void {
    this.#scene.allocate(count);
  }

  /**
   * Packs one object's fields at `index`, into the CPU mirror. Nothing is
   * uploaded; call {@link uploadObjects} once the frame's objects are packed.
   */
  pack(
    index: number,
    model: ArrayLike<number>,
    normalMatrix?: ArrayLike<number>,
    objectId: number = index,
    instanceId = 0,
    visibility = 1,
  ): void {
    this.#scene.pack(index, model, normalMatrix, objectId, instanceId, visibility);
  }

  /** Byte offset of a named field within object `index`, in the GPU buffer. */
  fieldOffset(index: number, field: string): number {
    const f = OBJECT_BLOCK.fields.find((x) => x.name === field);
    if (f === undefined) {
      fail('INTERNAL_INVARIANT', `ObjectUniforms has no field "${field}".`, {
        why: 'ObjectData is generated by apse; writing an unknown field means the writer and the struct disagree.',
        fix: 'Report this with the material that triggered it.',
      });
    }
    return SCENE_BLOCK.object.byteOffset + index * SCENE_BLOCK.stride + f.offset;
  }

  /** Uploads object slots `[0, count)`. */
  uploadFrom(count: number): void {
    this.#scene.uploadFrom(count);
  }

  /** Uploads one object's own bytes and nothing else. */
  uploadObject(index: number): void {
    this.#scene.uploadObject(index);
  }

  /** Uploads object slots `[lo, hi]`, both ends inclusive, in one write. */
  uploadObjects(lo: number, hi: number): void {
    this.#scene.uploadObjects(lo, hi);
  }

  /** Bytes the shared buffer occupies, frame region included. */
  get byteLength(): number {
    return this.#scene.byteLength;
  }

  /**
   * Releases this face's handle to the shared buffer. It does **not** free it --
   * see {@link FrameUniforms.dispose}. The buffer is one allocation per device
   * and the renderer owns it.
   */
  dispose(): void {}
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

export interface MaterialOptions {
  /**
   * The shared frame uniforms. Omit and the device's own is used — which is the
   * same instance, so omitting is correct and passing a foreign one is not.
   * Pass the renderer's instance so the whole app writes one buffer.
   */
  readonly frame?: FrameUniforms;
  /** The shared object uniforms. Same reasoning, and the same buffer as `frame`. */
  readonly object?: ObjectUniforms;
  /** Object slots to reserve. Default 4096. Only used when `object` is omitted. */
  readonly maxObjects?: number;
}

/**
 * Canonical cache keys for the shared uniform blocks. Exported so a second
 * consumer — the renderer, an editor, a test — resolves the *same* faces
 * rather than constructing parallel ones.
 *
 * The two labels are deliberately different strings over one buffer. They name
 * the two ways the buffer is written, and either one resolves to the same
 * `GPUBuffer` and the same `GPUBindGroup`.
 */
export const DEFAULT_FRAME_LABEL = 'apse.frame';
export const DEFAULT_OBJECT_LABEL = 'apse.object';
/** Debug label for the merged buffer. Names both regions, because it holds both. */
export const DEFAULT_SCENE_LABEL = 'apse:scene:frame+object';
export const DEFAULT_MAX_OBJECTS = 4096;

export class Material extends Resource implements Drawable {
  readonly name: string;
  readonly layout: VertexLayout;
  readonly renderPipeline: GPURenderPipeline;

  readonly phase: DrawPhase;
  readonly depth: DepthSpec;
  readonly blend: BlendSpec | null;
  readonly topology: PrimitiveTopology;
  readonly cull: CullMode;
  readonly frontFace: FrontFace;

  /**
   * The colour formats this pipeline was compiled for, read straight off the
   * resolved spec. The renderer checks these against the target before
   * `setPipeline`; see `Drawable.targetFormats` for why it has to.
   */
  get targetFormats(): readonly GPUTextureFormat[] {
    return this.#targetFormats ??= this.generated.resolved.targets.map((t) => t.format);
  }
  #targetFormats: readonly GPUTextureFormat[] | null = null;

  /** The full generated program and everything resolved from the spec. */
  readonly generated: GeneratedShader;
  readonly wgsl: string;
  readonly slotTypes: Readonly<Record<string, SlotType>>;
  readonly textureNames: readonly string[];
  readonly textured: boolean;
  readonly spec: MaterialSpec;

  readonly #device: GPUDevice;
  readonly #cache: DeviceCache;
  readonly #frame: FrameUniforms;
  readonly #object: ObjectUniforms;
  readonly #materialBlock: UniformBlockSpec | null;
  readonly #materialBuffer: GPUBuffer | null;
  readonly #materialData: UniformBlock;
  readonly #slotFields: ReadonlyMap<string, { offset: number; size: number; components: number }>;

  #materialBindGroup: GPUBindGroup | null = null;
  #views = new Map<string, GPUTextureView>();
  #textureBindGroup: GPUBindGroup | null = null;
  #textureBindGroupValid = false;

  #dirty = false;
  #dirtyLo = Number.POSITIVE_INFINITY;
  #dirtyHi = -1;

  private constructor(
    device: GPUDevice,
    spec: MaterialSpec,
    generated: GeneratedShader,
    pipeline: GPURenderPipeline,
    opts: MaterialOptions,
  ) {
    super('MATERIAL_DISPOSED');
    const r = generated.resolved;
    this.#device = device;
    this.#cache = deviceCache(device);
    this.spec = spec;
    this.generated = generated;
    this.wgsl = generated.code;
    this.name = r.name;
    this.layout = r.layout;
    this.renderPipeline = pipeline;
    this.phase = r.phase;
    this.depth = r.depth;
    this.blend = r.blend;
    this.topology = r.topology;
    this.cull = r.cull;
    this.frontFace = r.frontFace;
    this.slotTypes = r.slotTypes;
    this.textureNames = Object.freeze(r.textures.map((t) => t.slotName));
    this.textured = r.textures.length > 0;

    this.#frame = opts.frame ?? this.#cache.frameUniforms(DEFAULT_FRAME_LABEL);
    this.#object = opts.object ?? this.#cache.objectUniforms(DEFAULT_OBJECT_LABEL, opts.maxObjects ?? DEFAULT_MAX_OBJECTS);
    // Both faces have to be the same buffer on the *same device*, or the
    // pipeline layout this material was compiled against is satisfied by a bind
    // group whose frame region is not the one the renderer writes. Nothing
    // downstream can detect that, so it is checked here, where both values are
    // in hand.
    const scene = this.#frame.sceneBuffer;
    if (scene !== this.#object.sceneBuffer || scene.device !== device) {
      fail('INVALID_USAGE',
        'A Material was given frame or object uniforms that are not this device\'s shared scene buffer.', {
        why: 'The frame state and the per-object transforms live in one buffer in one bind group, so a material\'s pipeline layout is satisfied by exactly one of them. A different buffer — or the same shape of buffer on another device — means the renderer writes one and this material reads the other: no validation error, no warning, and every object drawn with an identity transform.',
        fix: 'Pass the renderer\'s `frameUniforms` and `objectUniforms`, or omit both and let the device cache supply them.',
      });
    }

    this.#materialBlock = r.materialBlock;
    if (this.#materialBlock === null) {
      this.#materialBuffer = null;
      this.#materialData = new UniformBlock(generated.materialBlock);
    } else {
      this.#materialBuffer = device.createBuffer({
        label: `apse:${r.name}:material`,
        size: this.#materialBlock.size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.#materialData = new UniformBlock(this.#materialBlock);
    }

    this.#slotFields = new Map(
      (this.#materialBlock?.fields ?? []).map((f) => [f.name, { offset: f.offset, size: f.size, components: f.components }]),
    );

    for (const [name, value] of r.slotDefaults) {
      this.#materialData.set(name, value);
    }
    if (r.slotDefaults.size > 0) this.markSlotsDirty();
  }

  /**
   * Compiles a material.
   *
   * Returns a promise because pipeline compilation is the only expensive step
   * and it must not block the main thread. See the module comment.
   */
  static async create(device: GPUDevice, spec: MaterialSpec, opts: MaterialOptions = {}): Promise<Material> {
    const generated = generateScaffold(spec, deviceLimits(device));
    validateGeneratedWGSL(generated.code);

    if (generated.resolved.scaffoldLog) {
      // eslint-disable-next-line no-console
      console.info(`[apse] generated WGSL for material "${generated.resolved.name}" (${generated.byteLength} bytes)\n${generated.code}`);
    }

    const cache = deviceCache(device);
    const depthFormat = spec.depthFormat ?? 'depth24plus';
    const pipeline = await cache.renderPipeline(generated, depthFormat, spec.stripIndexFormat);
    return new Material(device, spec, generated, pipeline, opts);
  }

  // --- Drawable: GPU objects -------------------------------------------------

  /**
   * `@group(0)`. The one bind a draw pays for, carrying the frame at a fixed
   * offset and the object at the dynamic offset in the same buffer.
   */
  get sceneBindGroup(): GPUBindGroup {
    return this.#object.sceneBindGroup;
  }

  /**
   * @deprecated Use {@link sceneBindGroup}. `Drawable` still names the frame
   * group, so this stays to keep it compiling; it returns the *same* bind group,
   * and the frame's bytes are a fixed region inside it.
   */
  get frameBindGroup(): GPUBindGroup {
    return this.#object.sceneBindGroup;
  }

  /**
   * @deprecated Use {@link sceneBindGroup}. Returns the same bind group, bound
   * with the object's dynamic offset.
   */
  get objectBindGroup(): GPUBindGroup {
    return this.#object.sceneBindGroup;
  }

  /** @group(1). Null when the material declares no slots. */
  get materialBindGroup(): GPUBindGroup | null {
    if (this.#materialBlock === null || this.#materialBuffer === null) return null;
    if (this.#materialBindGroup === null) {
      this.#materialBindGroup = this.#device.createBindGroup({
        label: `apse:${this.name}:material:group`,
        layout: this.#cache.materialBindGroupLayout(this.#materialBlock.size),
        entries: [{
          binding: 0,
          resource: { buffer: this.#materialBuffer, offset: 0, size: this.#materialBlock.size },
        }],
      });
    }
    return this.#materialBindGroup;
  }

  /**
   * @group(2). Null when the material declares no textures.
   *
   * Rebuilt lazily whenever a view is assigned, and shared between materials
   * that were given the same views, because the bind group is fully determined
   * by the view identities plus the (cached) layout.
   */
  get textureBindGroup(): GPUBindGroup | null {
    if (this.generated.textureSlots.length === 0) return null;
    if (this.#textureBindGroupValid && this.#textureBindGroup !== null) return this.#textureBindGroup;

    const entries: GPUBindGroupEntry[] = [];
    for (const t of this.generated.textureSlots) {
      const view = this.#views.get(t.slotName);
      if (view === undefined) {
        fail('TEXTURE_SLOT_MISSING',
          `Material "${this.name}" has no texture assigned for slot "${t.slotName}".`, {
          why: 'The generated WGSL declares this binding and the generated bind group layout requires a view, so the bind group cannot be created without one.',
          fix: `Call material.setTexture("${t.slotName}", view) before drawing, or declare the material without that texture. Assigned: ${[...this.#views.keys()].join(', ') || 'nothing'}.`,
        });
      }
      entries.push({ binding: t.bindingIndex, resource: view });
    }
    for (const s of this.generated.resolved.samplers) {
      entries.push({ binding: s.bindingIndex, resource: this.#cache.sampler(s) });
    }

    this.#textureBindGroup = this.#device.createBindGroup({
      label: `apse:${this.name}:texture:group`,
      layout: this.#cache.textureBindGroupLayout(this.generated.resolved),
      entries,
    });
    this.#textureBindGroupValid = true;
    return this.#textureBindGroup;
  }

  // --- Drawable: per-frame mutation -----------------------------------------

  /**
   * Copies a packed frame block. Forwards to the shared {@link FrameUniforms}.
   *
   * Idempotent, and correct on any material, because there is only one frame
   * block. A renderer should call `FrameUniforms.upload` once and skip this.
   */
  writeFrameUniform(bytes: ArrayBuffer): void {
    this.#frame.upload(bytes);
  }

  /**
   * Writes one material slot.
   *
   * Validation is the point: an unknown name and a wrong component count both
   * fail here, with a typed error, rather than silently writing over the
   * neighbouring field or being ignored.
   *
   * Error codes: an unknown slot name is **`OPTION_UNKNOWN`**. That code is
   * chosen over `SLOT_TYPE_UNKNOWN` (which is about a declared type being
   * unrecognised) and over `INTERNAL_INVARIANT` (whose guidance explicitly
   * says the fault is in apse, which is not true when the caller misspelled a
   * slot). The `why` and `fix` are overridden with slot-specific text, and the
   * message lists the slots that do exist. A wrong component count keeps
   * `SLOT_VALUE_WRONG_LENGTH`, which `UniformBlock` raises with the right
   * wording already.
   */
  setSlot(name: string, value: number | ArrayLike<number>): void {
    this.assertLive(`Material "${this.name}"`);
    const field = this.#slotFields.get(name);
    if (field === undefined) {
      fail('OPTION_UNKNOWN', `Material "${this.name}" has no slot "${name}".`, {
        why: 'apse packs material slots by name into a fixed uniform block. Writing a name that was not declared would either be a typo or a field the material does not have.',
        fix: this.#slotFields.size === 0
          ? `This material declares no slots. Add them to the spec: \`slots: { ${name}: 'f32' }\`.`
          : `Use one of: ${[...this.#slotFields.keys()].join(', ')}. call describeMaterial() for the full inventory including offsets.`,
      });
    }
    this.#materialData.set(name, value);
    const end = field.offset + field.size;
    if (field.offset < this.#dirtyLo) this.#dirtyLo = field.offset;
    if (end > this.#dirtyHi) this.#dirtyHi = end;
    this.#dirty = true;
  }

  /** Marks the whole block for upload, without touching the values. */
  markSlotsDirty(): void {
    this.assertLive(`Material "${this.name}"`);
    if (this.#materialBlock === null) return;
    this.#dirtyLo = 0;
    this.#dirtyHi = this.#materialBlock.size;
    this.#dirty = true;
  }

  get slotsDirty(): boolean {
    return this.#dirty;
  }

  /**
   * Uploads pending slot writes.
   *
   * A no-op when clean. When dirty it writes only the byte range that actually
   * changed, which is tracked exactly because `setSlot` is the only thing that
   * marks it — the alternative, writing the whole block, is at most 64 bytes
   * and usually saves nothing, so this is a small win that costs one integer
   * min and one max per write.
   */
  flushSlots(): void {
    if (!this.#dirty) return;
    const lo = this.#dirtyLo;
    const hi = this.#dirtyHi;
    this.#dirty = false;
    this.#dirtyLo = Number.POSITIVE_INFINITY;
    this.#dirtyHi = -1;
    if (this.#materialBuffer === null || lo > hi) return;
    // `data` is an ArrayBuffer, so dataOffset and size are in bytes.
    this.#device.queue.writeBuffer(this.#materialBuffer, lo, this.#materialData.data, lo, hi - lo);
  }

  // --- Textures --------------------------------------------------------------

  /** Assigns a view to a declared texture slot. Rebuilds the bind group. */
  setTexture(slotName: string, view: GPUTextureView): void {
    this.assertLive(`Material "${this.name}"`);
    if (!this.textureNames.includes(slotName)) {
      fail('TEXTURE_SLOT_MISSING', `Material "${this.name}" declares no texture slot "${slotName}".`, {
        why: 'apse generates one WGSL binding per declared texture, and a bind group entry per binding, so it can only bind a slot that exists in the spec.',
        fix: `Declare it: \`textures: { ${slotName}: { kind: '2d' } }\`. Declared: ${this.textureNames.join(', ') || 'none'}.`,
      });
    }
    this.#views.set(slotName, view);
    this.#textureBindGroupValid = false;
  }

  /** The view currently assigned to a slot, if any. */
  getTexture(slotName: string): GPUTextureView | undefined {
    return this.#views.get(slotName);
  }

  // --- Layout compatibility --------------------------------------------------

  /**
   * Checks a mesh against this material's vertex layout at bind time.
   *
   * Without this the mismatch surfaces from Dawn as a validation error during
   * `setVertexBuffer` or `draw`, naming an attribute shader location and a
   * vertex format and nothing else. With it, the error names the attribute, the
   * layout, and the fix, at the call site that caused it.
   */
  updateMesh(mesh: { readonly layout: VertexLayout }): void {
    this.layout.assertCompatible(mesh.layout, 'mesh');
  }

  // --- Introspection ---------------------------------------------------------

  /** The machine-readable inventory. Memoised; safe to call every frame. */
  describe(): MaterialDescription {
    return describeMaterial(this.spec);
  }

  /** The shared frame uniforms this material draws with. */
  get frameUniforms(): FrameUniforms {
    return this.#frame;
  }

  /** The shared object uniforms this material draws with. */
  get objectUniforms(): ObjectUniforms {
    return this.#object;
  }

  /**
   * The one buffer and bind group both of the above are faces of.
   *
   * Exposed so a caller — the renderer, a test, an inspector — can assert that
   * they really are the same allocation rather than two that happen to have
   * equal contents.
   */
  get sceneUniforms(): SceneUniformBuffer {
    return this.#object.sceneBuffer;
  }

  /** CPU-side mirror of the material block. Read it in tests and tooling. */
  get materialData(): UniformBlock {
    return this.#materialData;
  }

  protected onDispose(): void {
    // The material uniform buffer is the only GPU object this material owns
    // exclusively, so it is the only thing destroyed.
    //
    // The render pipeline is deliberately NOT destroyed. It is shared with
    // every other material whose generated WGSL and fixed-function state match,
    // so destroying it here would leave those materials holding an invalid
    // pipeline — a failure that would appear far from its cause. The cost is
    // that a disposed material's pipeline stays resident until the device goes
    // away; that is bounded by the number of *distinct* materials in the
    // application, which is the entire reason the cache exists.
    //
    // The texture bind group is not destroyed either, and could not be:
    // `GPUBindGroup` has no destroy() in WebGPU. It becomes collectable once
    // this material drops the last JS reference to it, at which point the
    // implementation releases the binding. Its views and the shared samplers
    // are owned by the caller and the device cache respectively, and outlive it
    // either way.
    this.#materialBuffer?.destroy();
    this.#materialBindGroup = null;
    this.#textureBindGroup = null;
    this.#textureBindGroupValid = false;
    this.#views.clear();
  }
}

export { generateScaffold, describeMaterial, resolveSpec } from './scaffold.ts';
export type {
  GeneratedShader,
  MaterialDescription,
  MaterialSpec,
  VaryingDefs,
  VaryingType,
  DescribedSlot,
  DescribedTexture,
  DescribedVarying,
} from './scaffold.ts';
export type { ResolvedTextureSlot } from './texture-slot.ts';
export type { TargetSpec, DepthSpec, BlendSpec, PrimitiveTopology } from '../render/pipeline-state.ts';
