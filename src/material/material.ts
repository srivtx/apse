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
 * | `GPUBindGroupLayout` (0-3)    | device cache   | ~4 total              |
 * | `GPUSampler`                  | device cache   | 1 per distinct config |
 * | frame uniform buffer + group  | `FrameUniforms`| **1**                 |
 * | object uniform buffer + group | `ObjectUniforms`| **1**                 |
 * | material uniform buffer       | the material   | N                     |
 * | texture bind group            | the material   | ≤ N                   |
 *
 * The frame uniform is identical for every material in a frame. Allocating one
 * per material would mean 500 writes of 256 bytes of identical data and 500
 * bind group creations to save one uniform load in the shader — the shader
 * load happens either way. `FrameUniforms` is the fix, and it is exported so
 * the renderer can own the one instance that matters.
 *
 * The object buffer is shared for the same reason: `ObjectData` is the same
 * struct for every material, so there is nothing material-specific about it.
 * It is a single large buffer addressed with dynamic offsets, allocated
 * lazily on first draw and grown by doubling.
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
import { FRAME_BLOCK, OBJECT_BLOCK } from '../core/slot.ts';
import type { SlotType } from '../core/slot.ts';
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

  #frameBGL: GPUBindGroupLayout | null = null;
  #objectBGL: GPUBindGroupLayout | null = null;
  readonly #materialBGLs = new Map<number, GPUBindGroupLayout>();
  readonly #textureBGLs = new Map<string, GPUBindGroupLayout>();
  readonly #pipelineLayouts = new Map<string, GPUPipelineLayout>();
  readonly #pipelines = new Map<number, Promise<GPURenderPipeline>>();
  readonly #samplers = new Map<string, GPUSampler>();
  readonly #frameUniforms = new Map<string, FrameUniforms>();
  readonly #objectUniforms = new Map<string, ObjectUniforms>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /** @group(0). Identical for every material, so built once. */
  frameBindGroupLayout(): GPUBindGroupLayout {
    if (this.#frameBGL !== null) return this.#frameBGL;
    this.#frameBGL = this.device.createBindGroupLayout({
      label: 'apse:frame',
      entries: [{
        binding: 0,
        visibility: SHADER_STAGES,
        buffer: { type: 'uniform', minBindingSize: FRAME_BLOCK.size },
      }],
    });
    return this.#frameBGL;
  }

  /** @group(1). Identical for every material. Dynamic offset. */
  objectBindGroupLayout(): GPUBindGroupLayout {
    if (this.#objectBGL !== null) return this.#objectBGL;
    this.#objectBGL = this.device.createBindGroupLayout({
      label: 'apse:object',
      entries: [{
        binding: 0,
        visibility: SHADER_STAGES,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: OBJECT_BLOCK.size },
      }],
    });
    return this.#objectBGL;
  }

  /**
   * @group(2). One per distinct block *size* — the generated `MaterialData`
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

  /** @group(3). Keyed by the declared texture and sampler signature. */
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
   * which is what lets an untextured or slotless material keep `@group(3)` at
   * 3 while omitting `@group(2)`. The renderer then never has to bind it.
   */
  pipelineLayout(resolved: ResolvedMaterialSpec): GPUPipelineLayout {
    const key = `${resolved.materialBlock?.size ?? 0}:${textureLayoutKey(resolved)}`;
    const hit = this.#pipelineLayouts.get(key);
    if (hit !== undefined) return hit;
    const created = this.device.createPipelineLayout({
      label: `apse:layout:${key}`,
      bindGroupLayouts: [
        this.frameBindGroupLayout(),
        this.objectBindGroupLayout(),
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
   * The shared frame uniforms, created on first use.
   *
   * The renderer normally creates one explicitly and hands it to every
   * material; this is the fallback for a material built without a renderer, and
   * it is keyed by label so an explicitly created instance wins.
   */
  /**
   * The shared frame uniform for a device, and the one true instance of it.
   *
   * Both the renderer and every material resolve their frame and object uniforms
   * through this cache, keyed by these labels. That indirection is the whole
   * point: if the renderer and a material each constructed their own, the
   * renderer would write the camera matrices into a buffer the material's bind
   * group never reads, and every object would be drawn with an identity
   * transform. Nothing would error. The scene would just be wrong.
   */
  frameUniforms(label: string): FrameUniforms {
    const hit = this.#frameUniforms.get(label);
    if (hit !== undefined) return hit;
    const created = new FrameUniforms(this.device, label);
    this.#frameUniforms.set(label, created);
    return created;
  }

  objectUniforms(label: string, maxObjects: number): ObjectUniforms {
    const hit = this.#objectUniforms.get(label);
    if (hit !== undefined) return hit;
    const created = new ObjectUniforms(this.device, label, maxObjects);
    this.#objectUniforms.set(label, created);
    return created;
  }

  /**
   * Disposes the shared uniform blocks this cache created.
   *
   * The bind group layouts, pipeline layouts, pipelines, and samplers are
   * *not* released: none of them has a `destroy()`, and all of them become
   * collectable when the device is. Call this from the renderer's teardown,
   * before the device is dropped, so the two uniform buffers are freed
   * deterministically rather than whenever the GC gets round to them.
   */
  disposeSharedUniforms(): void {
    for (const u of this.#frameUniforms.values()) u.dispose();
    for (const u of this.#objectUniforms.values()) u.dispose();
    this.#frameUniforms.clear();
    this.#objectUniforms.clear();
  }

  /** Cache sizes, for tests and for a debug overlay. */
  get stats(): { readonly pipelines: number; readonly pipelineLayouts: number; readonly bindGroupLayouts: number; readonly samplers: number } {
    return {
      pipelines: this.#pipelines.size,
      pipelineLayouts: this.#pipelineLayouts.size,
      bindGroupLayouts: 1 + 1 + this.#materialBGLs.size + this.#textureBGLs.size,
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
// Shared frame uniforms
// ---------------------------------------------------------------------------

/**
 * The frame uniform: one buffer, one bind group, for the whole frame.
 *
 * Owned by the renderer, referenced by every material. 500 materials sharing
 * one 256-byte uniform instead of 500 is the difference between a scene that
 * uploads 128 KB of frame state per frame and one that uploads 256 bytes.
 *
 * A `Drawable.writeFrameUniform` is idempotent and forwards here, so a renderer
 * that walks materials and calls it per material behaves correctly — it just
 * performs redundant writes, which is why the renderer should call
 * {@link FrameUniforms.upload} once and skip the per-material path.
 */
export class FrameUniforms implements Disposable {
  readonly spec: UniformBlockSpec = FRAME_BLOCK;
  /** CPU-side mirror of the block. Packed by the same code as every other block. */
  readonly block = new UniformBlock(FRAME_BLOCK);
  readonly buffer: GPUBuffer;
  readonly label: string;

  readonly #cache: DeviceCache;
  #bindGroup: GPUBindGroup | null = null;
  #dirty = false;
  #disposed = false;

  constructor(device: GPUDevice, label = 'frame') {
    this.#cache = deviceCache(device);
    this.label = label;
    this.buffer = device.createBuffer({
      label: `apse:${label}`,
      size: FRAME_BLOCK.size,
      // STORAGE would allow read_write; a frame block is written by the CPU and
      // read by the GPU, which is exactly UNIFORM | COPY_DST.
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** @group(0). Created once, on first access. */
  get bindGroup(): GPUBindGroup {
    if (this.#disposed) {
      fail('INTERNAL_INVARIANT', `FrameUniforms "${this.label}" was used after it was disposed.`, {
        why: 'The buffer it binds is destroyed, so the draw would read freed memory.',
        fix: 'Hold the FrameUniforms for as long as any material can still be drawn.',
      });
    }
    if (this.#bindGroup === null) {
      this.#bindGroup = this.#cache.device.createBindGroup({
        label: `apse:${this.label}:group`,
        layout: this.#cache.frameBindGroupLayout(),
        entries: [{ binding: 0, resource: { buffer: this.buffer, offset: 0, size: FRAME_BLOCK.size } }],
      });
    }
    return this.#bindGroup;
  }

  /**
   * Copies a fully packed frame block onto the GPU.
   *
   * `bytes` must be exactly {@link UniformBlockSpec}.size — it is the block the
   * renderer already packed, so re-packing it here would only add a copy.
   */
  upload(bytes: ArrayBuffer): void {
    if (this.#disposed) return;
    if (bytes.byteLength !== FRAME_BLOCK.size) {
      fail('INTERNAL_INVARIANT',
        `FrameUniforms.upload() got ${bytes.byteLength} bytes but the frame block is ${FRAME_BLOCK.size}.`, {
        why: 'The frame block layout is generated by apse. A differently sized buffer means the caller packed against a different struct, and the GPU would read the fields at the wrong offsets.',
        fix: `Pack against FRAME_BLOCK, or use \`uniforms.block\` and set fields by name.`,
      });
    }
    this.#cache.device.queue.writeBuffer(this.buffer, 0, bytes);
  }

  /** Copies the internal mirror to the GPU. */
  flush(): void {
    if (this.#disposed || !this.#dirty) return;
    this.#dirty = false;
    this.#cache.device.queue.writeBuffer(this.buffer, 0, this.block.data);
  }

  /** Packs one field and marks the block for {@link flush}. */
  set(name: string, value: number | ArrayLike<number>): void {
    this.block.set(name, value);
    this.#dirty = true;
  }

  /** @group(0) of `this.buffer`, for a manual bind. */
  get size(): number {
    return FRAME_BLOCK.size;
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#bindGroup = null;
    this.buffer.destroy();
  }
}


// ---------------------------------------------------------------------------
// Shared object uniforms
// ---------------------------------------------------------------------------

/**
 * The per-object transform block: one large buffer, addressed by dynamic
 * offset.
 *
 * `ObjectData` is the same struct for every material, so this is shared for the
 * same reason the frame block is. Each object occupies `OBJECT_BLOCK.stride`
 * bytes — the struct size rounded up to the 256-byte
 * `minUniformBufferOffsetAlignment` the API requires of a dynamic offset — and
 * the renderer writes a draw item's transform at `objectId * stride`.
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
  /** Object slots the buffer can hold. Grows by doubling. */
  capacity: number;

  readonly #cache: DeviceCache;
  #buffer: GPUBuffer | null = null;
  #bindGroup: GPUBindGroup | null = null;
  #data: ArrayBuffer | null = null;
  #f32: Float32Array | null = null;
  #u32: Uint32Array | null = null;
  #disposed = false;

  constructor(device: GPUDevice, label = 'object', maxObjects = 4096) {
    this.#cache = deviceCache(device);
    this.label = label;
    // One object is the true minimum: a zero-object buffer is not a legal
    // allocation, and growing from zero would never terminate.
    this.capacity = Math.max(1, maxObjects | 0);
  }

  /** True once the buffer has been created by a draw. */
  get allocated(): boolean {
    return this.#buffer !== null;
  }

  get buffer(): GPUBuffer {
    if (this.#disposed) {
      fail('INTERNAL_INVARIANT', `ObjectUniforms "${this.label}" was used after it was disposed.`, {
        why: 'The transform buffer is gone, so every object in the scene would draw with an undefined transform.',
        fix: 'Hold the ObjectUniforms for as long as any object can still be drawn.',
      });
    }
    this.allocate(this.capacity);
    return this.#buffer as GPUBuffer;
  }

  /** @group(1). Rebuilt transparently after a growth. */
  get bindGroup(): GPUBindGroup {
    if (this.#disposed) {
      fail('INTERNAL_INVARIANT', `ObjectUniforms "${this.label}" was used after it was disposed.`, {
        why: 'The buffer it binds is destroyed.',
        fix: 'Hold the ObjectUniforms for as long as any object can still be drawn.',
      });
    }
    if (this.#buffer === null) this.allocate(this.capacity);
    if (this.#bindGroup === null) {
      this.#bindGroup = this.#cache.device.createBindGroup({
        label: `apse:${this.label}:group`,
        layout: this.#cache.objectBindGroupLayout(),
        entries: [{
          binding: 0,
          resource: { buffer: this.#buffer!, offset: 0, size: OBJECT_BLOCK.size },
        }],
      });
    }
    return this.#bindGroup;
  }

  /** Byte offset for object `index`, growing the buffer by doubling if needed. */
  offsetFor(index: number): number {
    // Two ways to be unallocated: never allocated at all, or allocated smaller
    // than `index` needs. Only handling the second is a null-dereference
    // waiting for the first frame of a scene that fits inside the default
    // capacity — which is most scenes.
    if (this.#buffer === null) {
      this.allocate(Math.max(1, this.capacity, index + 1));
    } else if (index >= this.capacity) {
      let next = Math.max(1, this.capacity);
      while (next <= index) next *= 2;
      this.allocate(next);
    }
    return index * OBJECT_BLOCK.stride;
  }

  /** Creates the buffer for `count` objects. Safe to call repeatedly. */
  allocate(count: number): void {
    if (this.#disposed) return;
    if (this.#buffer !== null && count <= this.capacity) return;
    this.capacity = count;
    this.#buffer?.destroy();
    // Grow by doubling so a scene that keeps adding objects reallocates a
    // logarithmic number of times rather than once per object.
    const bytes = count * OBJECT_BLOCK.stride;
    this.#buffer = this.#cache.device.createBuffer({
      label: `apse:${this.label}:${count}`,
      size: bytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // A bind group captures its buffer, so growth invalidates it.
    this.#bindGroup = null;
    this.#data = new ArrayBuffer(bytes);
    this.#f32 = new Float32Array(this.#data);
    this.#u32 = new Uint32Array(this.#data);
  }

  /**
   * Packs one object's fields at `index`, into the CPU mirror.
   *
   * Nothing is uploaded here: the renderer packs every object during the
   * traversal and then calls {@link uploadFrom} once, so a 2000-object frame is
   * one `writeBuffer` rather than 2000.
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
    if (this.#f32 === null) this.offsetFor(index);
    const base = index * OBJECT_BLOCK.stride;
    const f32 = this.#f32!;

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
    this.#u32![ids] = objectId >>> 0;
    this.#u32![ids + 1] = instanceId >>> 0;
    f32[base32(base, VISIBILITY_FIELD.offset)] = visibility;
  }

  /** Byte offset of a named field within object `index`. */
  fieldOffset(index: number, field: string): number {
    const f = OBJECT_BLOCK.fields.find((x) => x.name === field);
    if (f === undefined) {
      fail('INTERNAL_INVARIANT', `ObjectUniforms has no field "${field}".`, {
        why: 'ObjectData is generated by apse; writing an unknown field means the writer and the struct disagree.',
        fix: 'Report this with the material that triggered it.',
      });
    }
    return index * OBJECT_BLOCK.stride + f.offset;
  }

  /** Uploads `count` objects from the CPU mirror. */
  uploadFrom(count: number): void {
    if (this.#disposed || this.#buffer === null || this.#data === null) return;
    const bytes = Math.min(count, this.capacity) * OBJECT_BLOCK.stride;
    if (bytes === 0) return;
    this.#cache.device.queue.writeBuffer(this.#buffer, 0, this.#data, 0, bytes);
  }

  /**
   * Uploads only object slots `[lo, hi]`.
   *
   * The whole buffer is `count * 256` bytes — 1.28 MB at 5000 objects, of which
   * only 124 bytes per slot is data. Uploading all of it every frame is the
   * single largest avoidable cost in the frame, and a static scene pays it for
   * nothing.
   *
   * `lo` and `hi` are **byte** offsets into the buffer, rounded to the 256-byte
   * slot stride by the caller. `writeBuffer`'s data offset is in bytes too,
   * because the source is an `ArrayBuffer`.
   */
  uploadRange(lo: number, hi: number): void {
    if (this.#disposed || this.#buffer === null || this.#data === null) return;
    const capacityBytes = this.capacity * OBJECT_BLOCK.stride;
    const a = Math.max(0, lo);
    const b = Math.min(capacityBytes, hi);
    if (b <= a) return;
    this.#cache.device.queue.writeBuffer(this.#buffer, a, this.#data, a, b - a);
  }

  get byteLength(): number {
    return this.capacity * OBJECT_BLOCK.stride;
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
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

export interface MaterialOptions {
  /**
   * The shared frame uniforms. Omit and one is created per device on demand.
   * Pass the renderer's instance so the whole app writes one buffer.
   */
  readonly frame?: FrameUniforms;
  /** The shared object uniforms. Same reasoning. */
  readonly object?: ObjectUniforms;
  /** Object slots to reserve. Default 4096. Only used when `object` is omitted. */
  readonly maxObjects?: number;
}

/**
 * Canonical cache keys for the shared uniform blocks. Exported so a second
 * consumer — the renderer, an editor, a test — resolves the *same* instances
 * rather than constructing parallel ones.
 */
export const DEFAULT_FRAME_LABEL = 'apse.frame';
export const DEFAULT_OBJECT_LABEL = 'apse.object';
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
    this.#object = opts.object ?? this.#cache.objectUniforms(DEFAULT_OBJECT_LABEL, opts.maxObjects ?? 4096);

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

  /** @group(0). The shared frame bind group, not a per-material one. */
  get frameBindGroup(): GPUBindGroup {
    return this.#frame.bindGroup;
  }

  /** @group(1). The shared object bind group, bound with a dynamic offset. */
  get objectBindGroup(): GPUBindGroup {
    return this.#object.bindGroup;
  }

  /** @group(2). Null when the material declares no slots. */
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
   * @group(3). Null when the material declares no textures.
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
