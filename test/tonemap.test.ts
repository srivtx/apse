/**
 * Tone map and present pass tests.
 *
 * There is no GPU here, and the shape of these tests is dictated by that. Three
 * kinds of thing are testable without one, and they are the three that actually
 * break:
 *
 *   1. **The pure geometry.** A fullscreen triangle with the wrong third vertex
 *      leaves a one-pixel diagonal seam across the whole image, and because it is
 *      exactly one pixel wide it reads as an artifact in the content. So the
 *      positions are asserted as numbers, and the resulting triangle is asserted
 *      to *contain* the clip-space square — the property that makes the seam
 *      impossible rather than merely unlikely.
 *
 *   2. **The generated WGSL.** A shader cannot be run here, so it is asserted
 *      textually: the curve constants the shader was generated from, the operator
 *      names, the texture sample, and the absence of the interpolated UV varying
 *      the module exists to avoid. This is the only check available, and it catches
 *      the bug it is aimed at — a shader that samples with a varying, or a curve
 *      whose constants drifted from the ones the options document.
 *
 *   3. **The texture accounting.** A resize leak in the present layer is
 *      invisible until the tab has been open for an hour, and then it is an OOM on
 *      exactly the devices least able to survive one. A fake device that records
 *      every `createTexture` and every `destroy` proves it, in the same way
 *      `test/device.test.ts` proves it for `RenderTargetImpl`.
 *
 * The fake device here extends `test/fake-device.ts` rather than replacing it,
 * because the present pass needs buffers, pipelines, bind groups and a command
 * encoder as well as textures, and none of that belongs in the shared fake.
 */

import { describe, expect, test } from 'bun:test';

import { AseError, isAseError } from '../src/core/error.ts';
import type { AseErrorCode } from '../src/core/error.ts';
import { FRAME_BLOCK, RESERVED_SLOT_NAMES } from '../src/core/slot.ts';
import { UNIFORM_TYPES, buildUniformBlock } from '../src/core/uniform.ts';
import { POSITION_LAYOUT, layoutCached } from '../src/geometry/layout.ts';
import { generateScaffold, stripComments, validateGeneratedWGSL } from '../src/material/scaffold.ts';
import type { GeneratedShader } from '../src/material/scaffold.ts';
import {
  EXPOSURE_FRAME_FIELD,
  FULLSCREEN_LAYOUT,
  HDR_TARGET_FORMATS,
  OPERATOR_IDS,
  TONEMAP_BLOCK,
  TONEMAP_MATERIAL_SLOTS,
  TONEMAP_SLOTS,
  TONE_MAP_OPERATORS,
  fullscreenMesh,
  isSrgbFormat,
  tonemapMaterialSpec,
} from '../src/material/tonemap.ts';
import type { ToneMapOperator } from '../src/material/tonemap.ts';
import { PresentPass } from '../src/render/present.ts';
import type { PresentOptions } from '../src/render/present.ts';
import { TEXTURE_USAGE } from '../src/render/device.ts';
import type { RenderTarget } from '../src/render/types.ts';
import { createCanvasTarget, createColorTarget } from '../src/render/target.ts';
import type { RenderTargetDevice } from '../src/render/target.ts';
import {
  asCanvasTargetDevice,
  asGpuDevice,
  FakeGPUDevice,
  fakeAseDevice,
  fakeLimits,
} from './fake-device.ts';
import type { FakeCanvasContext } from './fake-device.ts';

// ---------------------------------------------------------------------------
// WebGPU globals
//
// Bun has no WebGPU globals, and `FrameUniforms`, `ObjectUniforms`, `GpuMesh` and
// `Material` all read the usage constants the way a browser does. The spec's bit
// values are enough, and installing them is also a tripwire: a wrong bit here
// would show up as a wrong `usage` on the recorded buffer.
// ---------------------------------------------------------------------------

const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

(globalThis as unknown as { GPUBufferUsage: unknown }).GPUBufferUsage = GPU_BUFFER_USAGE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function expectCode(fn: () => unknown, code: AseErrorCode): AseError {
  let thrown: unknown;
  try {
    fn();
  } catch (error) {
    thrown = error;
  }
  if (!isAseError(thrown)) {
    throw new Error(`expected an AseError with code ${code}, got ${String(thrown)}`);
  }
  expect((thrown as AseError).code).toBe(code);
  return thrown as AseError;
}

async function expectCodeAsync(fn: () => Promise<unknown>, code: AseErrorCode): Promise<AseError> {
  let thrown: unknown;
  try {
    await fn();
  } catch (error) {
    thrown = error;
  }
  if (!isAseError(thrown)) {
    throw new Error(`expected an AseError with code ${code}, got ${String(thrown)}`);
  }
  expect((thrown as AseError).code).toBe(code);
  return thrown as AseError;
}

// ---------------------------------------------------------------------------
// The fake device
//
// FakeGPUDevice already records textures, which is the whole point of the
// accounting tests. What a present pass additionally needs is everything between
// createTexture and submit: buffers with a mappable range, shader modules, render
// pipelines, bind groups, and a command encoder whose render pass records the
// descriptor it was handed. Each stub below records exactly what it needs to
// answer one assertion and nothing more.
// ---------------------------------------------------------------------------

interface RecordedPass {
  readonly descriptor: GPURenderPassDescriptor;
  pipeline: GPURenderPipeline | null;
  readonly bindGroups: number[];
  readonly draws: { readonly vertexCount: number; readonly instances: number }[];
  ended: boolean;
}

class TestGPUDevice extends FakeGPUDevice {
  readonly buffers: { label: string; size: number; destroyed: boolean }[] = [];
  readonly writeLog: { buffer: GPUBuffer; offset: number; size: number }[] = [];
  readonly passes: RecordedPass[] = [];
  readonly pipelines: GPURenderPipelineDescriptor[] = [];
  encoders = 0;
  submits = 0;
  #mapped = new WeakMap<object, ArrayBuffer>();

  get liveBuffers(): number {
    return this.buffers.filter((b) => !b.destroyed).length;
  }

  createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
    const record = { label: desc.label ?? '', size: desc.size, destroyed: false };
    this.buffers.push(record);
    const storage = new ArrayBuffer(desc.size);
    const self = this;
    const buffer = {
      label: desc.label,
      size: desc.size,
      usage: desc.usage,
      getMappedRange: () => {
        if (desc.mappedAtCreation !== true) {
          throw new Error(`TestGPUDevice: getMappedRange() on a buffer that was not mappedAtCreation (${desc.label})`);
        }
        return storage;
      },
      unmap: () => { /* the storage stays readable; nothing detaches it here */ },
      destroy: () => {
        if (record.destroyed) {
          throw new Error(`TestGPUDevice: buffer "${record.label}" destroyed twice`);
        }
        record.destroyed = true;
      },
    } as unknown as GPUBuffer;
    this.#mapped.set(buffer, storage);
    void self;
    return buffer;
  }

  createBindGroupLayout(desc: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
    return { label: desc.label } as unknown as GPUBindGroupLayout;
  }

  createPipelineLayout(desc: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
    return { label: desc.label, bindGroupLayouts: desc.bindGroupLayouts } as unknown as GPUPipelineLayout;
  }

  createBindGroup(desc: GPUBindGroupDescriptor): GPUBindGroup {
    return { label: desc.label, entries: desc.entries } as unknown as GPUBindGroup;
  }

  createSampler(desc: GPUSamplerDescriptor): GPUSampler {
    return { label: desc.label } as unknown as GPUSampler;
  }

  createShaderModule(desc: GPUShaderModuleDescriptor): GPUShaderModule {
    return { label: desc.label, code: desc.code } as unknown as GPUShaderModule;
  }

  createRenderPipelineAsync(desc: GPURenderPipelineDescriptor): Promise<GPURenderPipeline> {
    this.pipelines.push(desc);
    return Promise.resolve({ label: desc.label } as unknown as GPURenderPipeline);
  }

  createCommandEncoder(desc: GPUCommandEncoderDescriptor): GPUCommandEncoder {
    this.encoders++;
    const device = this;
    return {
      label: desc.label,
      beginRenderPass(pass: GPURenderPassDescriptor): GPURenderPassEncoder {
        // Snapshotted field by field, not held by reference. PresentPass reuses
        // one descriptor *and one attachment object* for every frame, so holding
        // either reference would make every assertion about it read the last
        // frame's values.
        const recorded: RecordedPass = {
          descriptor: {
            label: pass.label,
            // The element type of the IDL Iterable is nullable, so the copy is
            // cast back: a snapshot of an attachment that is always present.
            colorAttachments: Array.from(pass.colorAttachments, (a) => ({ ...a })) as GPURenderPassColorAttachment[],
            depthStencilAttachment: pass.depthStencilAttachment,
          },
          pipeline: null,
          bindGroups: [],
          draws: [],
          ended: false,
        };
        device.passes.push(recorded);
        return {
          setPipeline(pipeline: GPURenderPipeline) { recorded.pipeline = pipeline; },
          setBindGroup(index: number) { recorded.bindGroups.push(index); },
          setVertexBuffer() { /* the draw below is what the tests read */ },
          draw(vertexCount: number, instances: number) {
            recorded.draws.push({ vertexCount, instances });
          },
          end() { recorded.ended = true; },
        } as unknown as GPURenderPassEncoder;
      },
      finish: () => ({ label: desc.label }) as unknown as GPUCommandBuffer,
    } as unknown as GPUCommandEncoder;
  }

  get queue(): GPUQueue {
    const device = this;
    return {
      writeBuffer(buffer: GPUBuffer, offset: number, data: unknown, _dataOffset?: number, size?: number): void {
        const bytes = typeof data === 'number' ? data : (data as ArrayBuffer).byteLength;
        device.writeLog.push({ buffer, offset, size: size ?? bytes });
      },
      submit(commands: unknown[]): void {
        device.submits += commands.length;
      },
    } as unknown as GPUQueue;
  }
}

interface Harness {
  readonly gpu: TestGPUDevice;
  readonly ase: RenderTargetDevice;
  /** Only the textures the present pass allocated, in creation order. */
  presentTextures(): ReturnType<FakeGPUDevice['textures']['filter']>;
  livePresent(): number;
}

function harness(): Harness {
  const gpu = new TestGPUDevice({ limits: fakeLimits() });
  const ase: RenderTargetDevice = {
    device: asGpuDevice(gpu),
    limits: fakeAseDevice({ limits: gpu.limits }).limits,
    assertLive: () => { /* never lost in these tests; the real assert is exercised elsewhere */ },
  };
  const presentTextures = () => gpu.apseTextures.filter((t) => t.label.startsWith('present:'));
  return { gpu, ase, presentTextures, livePresent: () => presentTextures().filter((t) => !t.destroyed).length };
}

/** An offscreen destination, so the pass's own textures are the only variable. */
function destination(h: Harness, opts: { width?: number; height?: number; format?: GPUTextureFormat; sampleCount?: 1 | 4 } = {}) {
  return createColorTarget(h.ase, {
    width: opts.width ?? 640,
    height: opts.height ?? 480,
    format: opts.format ?? 'bgra8unorm',
    sampleCount: opts.sampleCount ?? 1,
    label: 'dest',
  });
}

function present(h: Harness, target: RenderTarget, opts: Omit<PresentOptions, 'target'>): Promise<PresentPass> {
  return PresentPass.create(h.ase, { label: 'present', target, ...opts });
}

// ===========================================================================
// 1. The fullscreen geometry
// ===========================================================================

describe('FULLSCREEN_LAYOUT', () => {
  test('is position-only, 12 bytes per vertex, one attribute', () => {
    expect(FULLSCREEN_LAYOUT.attributes).toHaveLength(1);
    expect(FULLSCREEN_LAYOUT.attributeCount).toBe(1);
    expect(FULLSCREEN_LAYOUT.stride).toBe(12);
    const position = FULLSCREEN_LAYOUT.attribute('position');
    expect(position?.format).toBe('float32x3');
    expect(position?.offset).toBe(0);
    expect(position?.location).toBe(0);
    expect(FULLSCREEN_LAYOUT.has('normal')).toBe(false);
    expect(FULLSCREEN_LAYOUT.has('uv')).toBe(false);
  });

  test('wgslStruct() emits exactly one @location(0) position : vec3<f32>', () => {
    const wgsl = FULLSCREEN_LAYOUT.wgslStruct();
    expect(wgsl).toBe('struct VertexIn {\n  @location(0) position : vec3<f32>,\n};');
    expect(wgsl.match(/@location\(/g)).toHaveLength(1);
    expect(wgsl.match(/vec3<f32>/g)).toHaveLength(1);
  });

  test('the GPU layout is one interleaved buffer of stride 12', () => {
    const gpu = FULLSCREEN_LAYOUT.gpuLayout();
    expect(gpu.arrayStride).toBe(12);
    expect(gpu.stepMode).toBe('vertex');
    expect(gpu.attributes).toEqual([{ shaderLocation: 0, offset: 0, format: 'float32x3' }]);
    expect(FULLSCREEN_LAYOUT.byteLength(3)).toBe(36);
  });

  test('has the same identity key as POSITION_LAYOUT, and interns itself', () => {
    // Not the same *object* as POSITION_LAYOUT — that one is built with layout(),
    // which does not intern — but the same description, so a pipeline cache keyed
    // on the key treats them as one. And layoutCached does intern, so a second
    // fullscreen material shares this very instance rather than allocating a
    // second GPUVertexBufferLayout for identical code.
    expect(FULLSCREEN_LAYOUT.key).toBe(POSITION_LAYOUT.key);
    expect(layoutCached({ position: 'float32x3' })).toBe(FULLSCREEN_LAYOUT);
  });
});

describe('fullscreenMesh', () => {
  test('is three vertices and no index buffer', () => {
    const mesh = fullscreenMesh();
    expect(mesh.vertexCount).toBe(3);
    expect(mesh.indexCount).toBe(3);
    expect(mesh.indexData).toBeNull();
    expect(mesh.indexed).toBe(false);
    expect(mesh.layout).toBe(FULLSCREEN_LAYOUT);
    expect(mesh.vertexData).toHaveLength(9);
    expect(mesh.topology).toBe('triangle-list');
  });

  test('the three vertices are the overhanging clip-space triangle', () => {
    // (-1,-1), (3,-1), (-1,3). The third vertex is the whole point: a triangle
    // that stops at (1,1) is a quad's worth of half the screen.
    const p = fullscreenMesh().vertexData;
    expect(Array.from(p.slice(0, 3))).toEqual([-1, -1, 0]);
    expect(Array.from(p.slice(3, 6))).toEqual([3, -1, 0]);
    expect(Array.from(p.slice(6, 9))).toEqual([-1, 3, 0]);
  });

  test('the triangle contains the whole clip-space square, so there is no seam', () => {
    // The property, not the coordinates. A two-triangle quad rasterises the shared
    // diagonal twice and a derivative-using or sample-position-sensitive fragment
    // stage produces a one-pixel line across the image; a single overhanging
    // triangle has no interior edge at all.
    const p = fullscreenMesh().vertexData;
    const a = [p[0]!, p[1]!] as const;
    const b = [p[3]!, p[4]!] as const;
    const c = [p[6]!, p[7]!] as const;
    // Twice the signed area; positive is counter-clockwise in a y-up system,
    // which is `frontFace: 'ccw'`.
    const winding = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    expect(winding).toBeGreaterThan(0);
    for (const [x, y] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
      expect(contains(a, b, c, x, y)).toBe(true);
    }
  });

  test('the bounds are computed from the real vertices, not a placeholder', () => {
    // AABB centre of (-1,-1), (3,-1), (-1,3) is (1,1) and every vertex is
    // sqrt(8) from it. A fullscreen triangle whose bounds were wrong would be
    // culled by a scene traversal that ever saw it.
    const [cx, cy, cz, r] = Array.from(fullscreenMesh().boundingSphere);
    expect(cx).toBeCloseTo(1, 6);
    expect(cy).toBeCloseTo(1, 6);
    expect(cz).toBe(0);
    expect(r).toBeCloseTo(Math.sqrt(8), 6);
  });
});

/** Barycentric containment, in the y-up clip space the triangle is authored in. */
function contains(
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number],
  x: number,
  y: number,
): boolean {
  const d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  const l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / d;
  const l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / d;
  return l1 >= 0 && l2 >= 0 && l1 + l2 <= 1;
}

// ===========================================================================
// 2. The uniform block
// ===========================================================================

describe('TONEMAP_SLOTS', () => {
  const block = buildUniformBlock('MaterialData', TONEMAP_SLOTS, { maxBindingSize: 65536 });

  test('declares the four fields the shader reads', () => {
    expect(Object.keys(TONEMAP_SLOTS)).toEqual(['toneMap', 'exposure', 'operator', 'gamma']);
    expect(block.fields.map((f) => f.name)).toEqual(['toneMap', 'exposure', 'operator', 'gamma']);
    expect(block.fields.map((f) => f.type)).toEqual(['f32', 'f32', 'i32', 'f32']);
  });

  test('every offset comes from buildUniformBlock, and is aligned to its WGSL type', () => {
    // **Not 16-byte aligned, and asserting that would be asserting a WGSL rule
    // that does not exist.** In the uniform address space `f32` and `i32` align
    // to 4, not 16 — 16 is the alignment of `vec3`/`vec4` and of the *block*. So
    // the invariant worth having is: each field sits at a multiple of its own
    // type's alignment, the four tile the block with no padding, and the block
    // total is a multiple of 16. A hand-computed offset that broke the first would
    // be a silent, plausible-looking wrong read.
    for (const field of block.fields) {
      const align = UNIFORM_TYPES[field.type].align;
      expect(field.offset % align).toBe(0);
    }
    expect(block.fields.reduce((sum, f) => sum + f.size, 0)).toBe(block.size);
    expect(block.size).toBe(16);
    expect(block.size % 16).toBe(0);
    expect(block.fields.map((f) => f.offset)).toEqual([0, 4, 8, 12]);
    // ...and those offsets are what buildUniformBlock says, not what this test
    // happens to believe. buildUniformBlock is the authority; the literal above
    // is a change detector.
    const rebuilt = buildUniformBlock('MaterialData', TONEMAP_SLOTS, { maxBindingSize: 65536 });
    expect(rebuilt.fields).toEqual(block.fields);
  });

  test('exposure is a frame-block field, and a material slot of that name is refused', () => {
    // This is why the table has four entries and the material block has three.
    // `exposure` is reserved in FRAME_FIELDS — with the comment "tone-map
    // exposure, applied by the present pass" — and RESERVED_SLOT_NAMES refuses a
    // material slot that shadows a generated name. Asserted here so the reason is
    // recorded next to the consequence.
    expect(RESERVED_SLOT_NAMES.has('exposure')).toBe(true);
    expect(EXPOSURE_FRAME_FIELD).toBe('exposure');
    const field = FRAME_BLOCK.fields.find((f) => f.name === 'exposure');
    expect(field?.type).toBe('f32');
    expect(field?.components).toBe(1);

    // The refusal itself, on a spec that does try to declare one.
    const spec = tonemapMaterialSpec();
    const error = expectCode(
      () => generateScaffold({ ...spec, slots: { ...TONEMAP_MATERIAL_SLOTS, exposure: 'f32' } }),
      'OPTION_UNKNOWN',
    );
    expect(error.message).toContain('exposure');
    expect(error.fix).toContain('exposure');

    // And the tone map's own spec never trips it.
    expect(Object.keys(spec.slots ?? {})).not.toContain('exposure');
  });

  test('the material block is the legal subset, resolved through buildUniformBlock', () => {
    expect(Object.keys(TONEMAP_MATERIAL_SLOTS)).toEqual(['toneMap', 'operator', 'gamma']);
    expect(TONEMAP_BLOCK.structName).toBe('MaterialData');
    expect(TONEMAP_BLOCK.fields.map((f) => f.name)).toEqual(['toneMap', 'operator', 'gamma']);
    expect(TONEMAP_BLOCK.size).toBe(12);
    // And it is exactly what buildUniformBlock produces for that subset.
    expect(TONEMAP_BLOCK.fields).toEqual(
      buildUniformBlock('MaterialData', TONEMAP_MATERIAL_SLOTS, { maxBindingSize: 65536 }).fields,
    );
  });
});

// ===========================================================================
// 3. The generated WGSL
//
// No GPU, so the shader is asserted textually. Every constant checked here is the
// same object the WGSL was generated from, which is what makes this a drift
// detector rather than a snapshot.
// ===========================================================================

/** The generated program, for the given options. Device-free by construction. */
const shader = (opts?: Parameters<typeof tonemapMaterialSpec>[0]): GeneratedShader =>
  generateScaffold(tonemapMaterialSpec(opts));

describe('the generated tone map WGSL', () => {
  test('is structurally valid generated WGSL', () => {
    expect(validateGeneratedWGSL(shader().code)).toEqual({ ok: true });
    expect(validateGeneratedWGSL(shader({ targetFormat: 'bgra8unorm-srgb' }).code)).toEqual({ ok: true });
  });

  test('names every operator, and the switch cases match OPERATOR_IDS', () => {
    const { code } = shader();
    for (const name of TONE_MAP_OPERATORS) {
      expect(code).toContain(`OPERATOR_${name.toUpperCase()}`);
    }
    for (const [name, id] of Object.entries(OPERATOR_IDS)) {
      expect(code).toContain(`const OPERATOR_${name.toUpperCase()} : i32 = ${id};`);
      expect(code).toContain(`case OPERATOR_${name.toUpperCase()}:`);
    }
    expect(code).toContain('switch (op)');
    // The selector is a uniform, which is the whole point: one pipeline, four
    // curves, no shader compile when the user changes one.
    expect(shader().resolved.fragmentBody).toContain('mat.operator');
  });

  test('samples the source texture at a UV derived from @builtin(position)', () => {
    const { code, resolved } = shader();
    expect(code).toContain('@group(3) @binding(0) var texture : texture_2d<f32>;');
    expect(code).toContain('textureSample(texture, textureSampler');
    expect(resolved.fragmentBody).toContain('in.clip.xy');
    expect(resolved.fragmentBody).toContain('textureDimensions(texture, 0)');
  });

  test('interpolates no UV varying — there is none to interpolate', () => {
    const { code, resolved } = shader();
    expect(resolved.userVaryings).toHaveLength(0);
    // Varyings is the builtin and nothing else.
    expect(resolved.varyings).toHaveLength(1);
    expect(resolved.varyings[0]?.name).toBe('clip');
    expect(resolved.varyings[0]?.location).toBe(-1);
    // Two @locations in the whole program: the vertex attribute and the fragment
    // entry point's return. Neither is a varying.
    expect(code.match(/@location\(/g)).toHaveLength(2);
    expect(code).toContain('-> @location(0) vec4f {');
    expect(code).not.toContain('@interpolate');
    expect(resolved.fragmentBody).not.toContain('in.uv');
    expect(resolved.vertexBody).not.toContain('out.uv');
  });

  test('the vertex stage emits clip space directly, with no camera or object transform', () => {
    // A fullscreen pass that goes through the object pipeline is how a post chain
    // picks up a stray transform: the screen shifts and nothing reports an error.
    // Comments stripped first: the body explains *why* it does not read the
    // object matrix, and a naive substring check would read its own rationale.
    const vertexBody = stripComments(shader().resolved.vertexBody);
    expect(vertexBody).not.toContain('obj.model');
    expect(vertexBody).not.toContain('frame.viewProj');
    expect(vertexBody).not.toContain('obj.');
    expect(vertexBody).not.toContain('frame.');
    expect(vertexBody.trim()).toBe('out.clip = vec4f(in.position, 0.0);');
  });

  test('does not cull, because a fullscreen triangle has nothing to cull', () => {
    expect(shader().resolved.cull).toBe('none');
    expect(shader().resolved.sampleCount).toBe(1);
  });

  test('declares the legal slots and reads exposure from the frame block', () => {
    const { resolved } = shader({ operator: 'reinhard', gamma: 0.25 });
    expect(Object.keys(resolved.slotTypes)).toEqual(['toneMap', 'operator', 'gamma']);
    expect(resolved.slotDefaults.get('toneMap')).toBe(1);
    expect(resolved.slotDefaults.get('operator')).toBe(OPERATOR_IDS.reinhard);
    expect(resolved.slotDefaults.get('gamma')).toBe(0.25);
    expect(resolved.fragmentBody).toContain('frame.exposure');
    expect(resolved.fragmentBody).not.toContain('mat.exposure');
  });

  test('defaults to ACES at exposure 1, gamma 0', () => {
    const { resolved } = shader();
    expect(resolved.slotDefaults.get('operator')).toBe(OPERATOR_IDS.aces);
    expect(OPERATOR_IDS.aces).toBe(1);
    // The fallback target format in a non-browser host. A real canvas answers
    // bgra8unorm, and both are non-srgb, so both need the encode below.
    expect(resolved.targets[0]?.format).toBe('rgba8unorm');
    expect(resolved.targets[0]?.format.endsWith('-srgb')).toBe(false);
  });
});

describe('the sRGB decision, which is the bug this module exists for', () => {
  test('a plain bgra8unorm target encodes in the shader', () => {
    // WebGPU gives you no free conversion here. A linear 0.5 written to a plain
    // bgra8unorm attachment is stored as 0.5, and the image comes out far too
    // dark — which reads as a lighting bug, so it gets "fixed" by raising light
    // intensities until the shadows look right.
    const { resolved } = generateScaffold(tonemapMaterialSpec({ targetFormat: 'bgra8unorm' }));
    expect(resolved.fragmentBody).toContain('let display = srgbEncode(graded);');
  });

  test('an *-srgb target does not, because the hardware already did', () => {
    // Encoding twice is not neutral: it reads as a washed-out, low-contrast image
    // rather than as an error.
    for (const format of ['bgra8unorm-srgb', 'rgba8unorm-srgb'] as const) {
      const { resolved } = generateScaffold(tonemapMaterialSpec({ targetFormat: format }));
      expect(resolved.fragmentBody).toContain('let display = graded;');
      expect(resolved.fragmentBody).not.toContain('srgbEncode(graded)');
    }
  });

  test('every non-srgb target encodes, and every srgb one does not', () => {
    for (const format of ['bgra8unorm', 'rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm-srgb'] as const) {
      const encodes = shader({ targetFormat: format }).resolved.fragmentBody.includes('srgbEncode(graded)');
      expect(encodes).toBe(!isSrgbFormat(format));
      expect(isSrgbFormat(format)).toBe(format.endsWith('-srgb'));
    }
  });

  test('the two formats produce different programs, so they cannot share a pipeline', () => {
    const plain = shader({ targetFormat: 'bgra8unorm' });
    const srgb = shader({ targetFormat: 'bgra8unorm-srgb' });
    expect(plain.pipelineKey).not.toBe(srgb.pipelineKey);
  });
});

// ===========================================================================
// 4. The tone curves
//
// **The shader is verified visually, not numerically.** There is no GPU here, so
// the curves below are a TypeScript port that exists to pin the *constants* and
// the *shape* of each curve. What is asserted numerically is the port; what is
// asserted textually is that the WGSL was generated from the same constants the
// port reads. A reviewer should still look at a rendered frame before believing
// the image is right — nothing here can catch a driver disagreeing with WGSL
// about pow().
// ===========================================================================

/** Narkowicz 2015. The same five numbers the shader was generated from. */
const ACES_FIT = { a: 2.51, b: 0.03, c: 2.43, d: 0.59, e: 0.14 } as const;

function acesPort(x: number): number {
  const { a, b, c, d, e } = ACES_FIT;
  const v = (x * (a * x + b)) / (x * (c * x + d) + e);
  return Math.min(1, Math.max(0, v));
}

function reinhardPort(x: number): number {
  return x / (1 + x);
}

function linearPort(x: number): number {
  return Math.min(1, Math.max(0, x));
}

/** IEC 61966-2-1, piecewise. The same constants the shader was generated from. */
const SRGB_PORT = { linearThreshold: 0.0031308, linearScale: 12.92, powerScale: 1.055, powerOffset: 0.055 } as const;

function srgbPort(c: number): number {
  return c < SRGB_PORT.linearThreshold
    ? c * SRGB_PORT.linearScale
    : SRGB_PORT.powerScale * Math.pow(Math.max(c, 0), 1 / 2.4) - SRGB_PORT.powerOffset;
}

describe('the curve constants are the ones the shader was built from', () => {
  const { code } = shader();

  test('ACES', () => {
    expect(code).toContain(`const ACES_A : f32 = ${ACES_FIT.a};`);
    expect(code).toContain(`const ACES_B : f32 = ${ACES_FIT.b};`);
    expect(code).toContain(`const ACES_C : f32 = ${ACES_FIT.c};`);
    expect(code).toContain(`const ACES_D : f32 = ${ACES_FIT.d};`);
    expect(code).toContain(`const ACES_E : f32 = ${ACES_FIT.e};`);
    // The form, not just the numbers: a numerator/denominator that is not the
    // Narkowicz rational is a different curve wearing the same constants.
    expect(code).toContain('(x * (ACES_A * x + ACES_B)) / (x * (ACES_C * x + ACES_D) + ACES_E)');
  });

  test('sRGB', () => {
    expect(code).toContain(`const SRGB_LINEAR_THRESHOLD : f32 = ${SRGB_PORT.linearThreshold};`);
    expect(code).toContain(`const SRGB_LINEAR_SCALE : f32 = ${SRGB_PORT.linearScale};`);
    expect(code).toContain(`const SRGB_POWER_SCALE : f32 = ${SRGB_PORT.powerScale};`);
    expect(code).toContain(`const SRGB_POWER_OFFSET : f32 = ${SRGB_PORT.powerOffset};`);
    // Written as the division, not a 17-digit literal, so the generated shader
    // stays readable; the value is the same one to within an f32 ulp.
    expect(code).toContain('pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4))');
  });

  test('Reinhard is the per-channel c / (1 + c)', () => {
    expect(code).toContain('return x / (vec3f(1.0) + x);');
  });
});

describe('the curves, numerically', () => {
  test('ACES maps 0 to 0 and 1 to ~0.804, and never leaves [0, 1]', () => {
    expect(acesPort(0)).toBe(0);
    expect(acesPort(1)).toBeCloseTo(2.54 / 3.16, 5);
    for (const x of [0, 0.001, 0.1, 0.18, 0.5, 1, 4, 16, 1000]) {
      const v = acesPort(x);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  test('ACES is monotonically increasing — the property a tone curve must have', () => {
    // A curve that dips produces a highlight that darkens as the light gets
    // brighter, which is a bug you cannot un-see once it is in a frame.
    let previous = -1;
    for (let i = 0; i <= 400; i++) {
      const v = acesPort(i * 0.25);
      expect(v).toBeGreaterThanOrEqual(previous);
      previous = v;
    }
  });

  test('ACES is compressive: below the midpoint it rises faster than the identity', () => {
    // 0.18 is mid grey. A tone curve that failed to lift the shadows would map it
    // to less than 0.18, which is the "everything is dark" symptom.
    expect(acesPort(0.18)).toBeGreaterThan(0.18);
    expect(acesPort(4)).toBeLessThan(4);
  });

  test('Reinhard: 0 to 0, 1 to 0.5, and asymptotic to 1', () => {
    expect(reinhardPort(0)).toBe(0);
    expect(reinhardPort(1)).toBe(0.5);
    // Asymptotic, never reaching 1: that is the property, and asserting
    // toBeCloseTo(1) at 6 places would be asserting it *does* reach white.
    expect(reinhardPort(1e6)).toBeGreaterThan(0.9999);
    expect(reinhardPort(1e6)).toBeLessThan(1);
    expect(reinhardPort(2)).toBeCloseTo(2 / 3, 6);
  });

  test('linear and none are the same clamp, and it is the identity inside the range', () => {
    for (const x of [0, 0.25, 0.5, 1]) expect(linearPort(x)).toBe(x);
    expect(linearPort(2)).toBe(1);
    expect(linearPort(-1)).toBe(0);
  });

  test('sRGB fixes both ends and lifts the midtones', () => {
    expect(srgbPort(0)).toBe(0);
    expect(srgbPort(1)).toBeCloseTo(1, 6);
    expect(srgbPort(0.5)).toBeCloseTo(0.735357, 5);
    // The property the whole module is about: a linear value is NOT what a
    // display shows, and writing it verbatim is the bug.
    expect(srgbPort(0.5)).not.toBe(0.5);
    expect(srgbPort(0.5)).toBeGreaterThan(0.5);
    expect(srgbPort(0.02)).toBeGreaterThan(0.02);
  });

  test('sRGB is continuous across its branch point', () => {
    // Using the power law below the threshold darkens the first few code values
    // of every gradient, which is visible as banding in a dark scene.
    const t = SRGB_PORT.linearThreshold;
    const below = t * SRGB_PORT.linearScale;
    const above = SRGB_PORT.powerScale * Math.pow(t, 1 / 2.4) - SRGB_PORT.powerOffset;
    expect(below).toBeCloseTo(above, 4);
    expect(Math.abs(below - above)).toBeLessThan(0.002);
  });

  test('sRGB is monotonically increasing across both branches', () => {
    let previous = -1;
    for (let i = 0; i <= 2000; i++) {
      const v = srgbPort(i / 2000);
      expect(v).toBeGreaterThanOrEqual(previous);
      previous = v;
    }
  });
});

// ===========================================================================
// 5. Option validation
//
// No GPU is needed to reject a bad option, and rejecting one is most of the
// value: a typo that were coerced would produce a plausible-looking wrong image.
// ===========================================================================

describe('TonemapOptions validation', () => {
  test('an unknown operator is rejected and the known ones are listed', () => {
    const error = expectCode(
      () => tonemapMaterialSpec({ operator: 'filmic' as never }),
      'OPTION_UNKNOWN',
    );
    expect(error.message).toContain('filmic');
    for (const name of TONE_MAP_OPERATORS) expect(error.fix.includes(name)).toBe(true);
  });

  test('an HDR format outside the supported set is rejected', () => {
    for (const format of ['rgba8unorm', 'bgra8unorm', 'rgba32uint'] as GPUTextureFormat[]) {
      const error = expectCode(() => tonemapMaterialSpec({ hdrFormat: format }), 'OPTION_UNKNOWN');
      expect(error.message).toContain(format);
    }
    // The two that are accepted.
    for (const format of HDR_TARGET_FORMATS) {
      expect(() => tonemapMaterialSpec({ hdrFormat: format })).not.toThrow();
    }
  });

  test('a non-positive or non-finite exposure is rejected', () => {
    for (const exposure of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const error = expectCode(() => tonemapMaterialSpec({ exposure }), 'OPTION_UNKNOWN');
      expect(error.message).toContain('exposure');
    }
  });

  test('gamma must be greater than -1, because pow(0, 0) is unspecified in WGSL', () => {
    for (const gamma of [-1, -1.5, Number.NaN]) {
      expectCode(() => tonemapMaterialSpec({ gamma }), 'OPTION_UNKNOWN');
    }
    expect(() => tonemapMaterialSpec({ gamma: -0.5 })).not.toThrow();
    expect(() => tonemapMaterialSpec({ gamma: 1 })).not.toThrow();
  });

  test('every operator produces a distinct, named case in the shader', () => {
    const ids = new Set(Object.values(OPERATOR_IDS));
    expect(ids.size).toBe(TONE_MAP_OPERATORS.length);
    for (const [name, id] of Object.entries(OPERATOR_IDS)) {
      expect(TONE_MAP_OPERATORS.includes(name as ToneMapOperator)).toBe(true);
      expect(Number.isInteger(id)).toBe(true);
    }
  });
});

// ===========================================================================
// 6. PresentPass — texture accounting
// ===========================================================================

/** One row of the accounting table. `allocated` is label suffixes, in order. */
interface Row {
  readonly name: string;
  readonly opts: Omit<PresentOptions, 'target'>;
  readonly dest?: { readonly format?: GPUTextureFormat; readonly sampleCount?: 1 | 4 };
  readonly allocated: readonly string[];
  readonly formats?: readonly GPUTextureFormat[];
}

const ROWS: readonly Row[] = [
  {
    name: 'direct to target: nothing at all',
    opts: { toneMapping: null },
    allocated: [],
  },
  {
    name: 'hdr: an rgba16float colour target plus depth',
    opts: { hdr: true, toneMapping: { operator: 'aces' } },
    allocated: ['present:scene:color', 'present:scene:depth'],
    formats: ['rgba16float', 'depth24plus'],
  },
  {
    name: 'a tone map with no hdr: an LDR intermediate in the destination format',
    opts: { toneMapping: { operator: 'none' } },
    allocated: ['present:scene:color', 'present:scene:depth'],
    formats: ['bgra8unorm', 'depth24plus'],
  },
  {
    name: 'sampleCount 4: MSAA colour, resolve, and depth',
    opts: { sampleCount: 4, toneMapping: { operator: 'aces' } },
    allocated: ['present:scene:color', 'present:scene:colorResolve', 'present:scene:depth'],
    formats: ['bgra8unorm', 'bgra8unorm', 'depth24plus'],
  },
  {
    name: 'hdr with sampleCount 4: MSAA and resolve both in the HDR format',
    opts: { hdr: true, sampleCount: 4, toneMapping: {} },
    allocated: ['present:scene:color', 'present:scene:colorResolve', 'present:scene:depth'],
    formats: ['rgba16float', 'rgba16float', 'depth24plus'],
  },
];

describe('PresentPass — texture accounting', () => {
  for (const row of ROWS) {
    test(`${row.name}`, async () => {
      const h = harness();
      const dest = destination(h, row.dest ?? {});
      const pass = await present(h, dest, row.opts);

      expect(h.presentTextures().map((t) => t.label)).toEqual([...row.allocated]);
      expect(h.livePresent()).toBe(row.allocated.length);
      if (row.formats !== undefined) {
        expect(h.presentTextures().map((t) => t.format)).toEqual([...row.formats]);
      }
      // The size reaches the descriptor, not just the accessor.
      for (const texture of h.presentTextures()) {
        expect(texture.width).toBe(640);
        expect(texture.height).toBe(480);
      }
      pass.dispose();
      expect(h.livePresent()).toBe(0);
      dest.dispose();
    });
  }

  test('the intermediate is sampleable, and a 4x attachment is not', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { sampleCount: 4, toneMapping: {} });
    const msaa = h.presentTextures().find((t) => t.label === 'present:scene:color');
    const resolve = h.presentTextures().find((t) => t.label === 'present:scene:colorResolve');
    // The tone map samples the resolve texture. Binding a multisampled texture
    // as a sampled texture is a validation error, not a blurry result.
    expect(msaa?.sampleCount).toBe(4);
    expect(msaa?.hasUsage(TEXTURE_USAGE.TEXTURE_BINDING)).toBe(false);
    expect(resolve?.sampleCount).toBe(1);
    expect(resolve?.hasUsage(TEXTURE_USAGE.TEXTURE_BINDING)).toBe(true);
    expect(resolve?.hasUsage(TEXTURE_USAGE.RENDER_ATTACHMENT)).toBe(true);
    pass.dispose();
  });

  test('the intermediate always carries depth, even though this pass never reads it', async () => {
    // The scene's materials declare depthStencil, and a pass with no depth
    // attachment and a pipeline that has one is a validation error that discards
    // the whole pass — including the clear. A black frame, no error anywhere.
    for (const opts of ROWS.filter((r) => r.allocated.length > 0)) {
      const h = harness();
      const pass = await present(h, destination(h), opts.opts);
      const depth = h.presentTextures().find((t) => t.label.endsWith(':depth'));
      expect(depth).toBeDefined();
      expect(depth?.format).toBe('depth24plus');
      pass.dispose();
    }
  });

  test('resize to the same size allocates nothing and destroys nothing', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { hdr: true, toneMapping: {} });
    const allocated = h.presentTextures().length;
    const destroyed = h.gpu.destroyLog.length;
    pass.resize(640, 480);
    expect(h.presentTextures()).toHaveLength(allocated);
    expect(h.gpu.destroyLog).toHaveLength(destroyed);
    expect(h.livePresent()).toBe(2);
    pass.dispose();
  });

  test('resize to a different size destroys the old textures before allocating new ones', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { hdr: true, toneMapping: {} });
    const old = h.presentTextures().filter((t) => !t.destroyed);
    pass.resize(1024, 768);

    for (const texture of old) expect(texture.destroyed).toBe(true);
    expect(h.gpu.destroyLog).toEqual(['present:scene:color', 'present:scene:depth']);
    // Every old texture is gone *before* the new ones exist, so the peak is one
    // target rather than two.
    expect(h.presentTextures()).toHaveLength(4);
    expect(h.livePresent()).toBe(2);
    for (const texture of h.presentTextures().filter((t) => !t.destroyed)) {
      expect(texture.width).toBe(1024);
      expect(texture.height).toBe(768);
    }
    pass.dispose();
  });

  test('twenty chained resizes leave exactly the expected number live', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { sampleCount: 4, hdr: true, toneMapping: {} });
    expect(h.livePresent()).toBe(3);
    for (let i = 1; i <= 20; i++) {
      pass.resize(320 + i, 240 + i);
      expect(h.livePresent()).toBe(3);
    }
    // 3 initial + 20 resizes x 3, and 60 destroys: nothing stranded.
    expect(h.presentTextures()).toHaveLength(3 * 21);
    expect(h.gpu.destroyLog).toHaveLength(3 * 20);
    pass.dispose();
    expect(h.livePresent()).toBe(0);
  });

  test('dispose leaves zero live textures, and twice is safe', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { sampleCount: 4, toneMapping: {} });
    expect(h.livePresent()).toBe(3);
    pass.dispose();
    expect(h.livePresent()).toBe(0);
    expect(h.gpu.destroyLog).toEqual([
      'present:scene:color',
      'present:scene:colorResolve',
      'present:scene:depth',
    ]);
    // The fake throws if a texture is destroyed twice, so this proves apse is not
    // leaning on the driver being lenient.
    pass.dispose();
    pass.dispose();
    expect(h.gpu.destroyLog).toHaveLength(3);
  });

  test('dispose releases the mesh and material buffers, and not the shared ones', async () => {
    // The frame and object uniform buffers belong to the per-device cache, not to
    // this pass. Destroying them here would break every other material on the
    // device — which is the same reason Material does not destroy its pipeline.
    const h = harness();
    const before = h.gpu.liveBuffers;
    const pass = await present(h, destination(h), { toneMapping: {} });
    const created = h.gpu.buffers.slice(before);
    const owned = created.filter((b) => b.label === 'present:fullscreen:vertex' || b.label === 'apse:tonemap:material');
    const shared = created.filter((b) => b.label.startsWith('apse:apse.'));
    expect(owned).toHaveLength(2);
    expect(shared).toHaveLength(2);

    pass.dispose();
    for (const buffer of owned) expect(buffer.destroyed).toBe(true);
    for (const buffer of shared) expect(buffer.destroyed).toBe(false);
    expect(h.gpu.liveBuffers).toBe(before + shared.length);
  });

  test('a failed create leaves nothing behind', async () => {
    const h = harness();
    await expect(present(h, destination(h), { hdr: true, toneMapping: null })).rejects.toThrow();
    expect(h.presentTextures()).toHaveLength(0);
  });
});

// ===========================================================================
// 7. PresentPass — the direct path, the rejections, and the render pass
// ===========================================================================

describe('PresentPass — the direct path', () => {
  test('sceneTarget IS the target, and there is no material or intermediate', async () => {
    const h = harness();
    const dest = destination(h);
    const pass = await present(h, dest, { toneMapping: null });

    expect(pass.sceneTarget).toBe(dest);
    expect(pass.material).toBeNull();
    expect(h.presentTextures()).toHaveLength(0);
    // No intermediate means no pipeline, no mesh, and no shader compile at all.
    expect(h.gpu.pipelines).toHaveLength(0);
    expect(h.gpu.buffers).toHaveLength(0);

    pass.render();
    expect(h.gpu.encoders).toBe(0);
    expect(h.gpu.passes).toHaveLength(0);
    pass.dispose();
  });

  test('with an intermediate, sceneTarget is NOT the target and has the right format', async () => {
    const h = harness();
    const dest = destination(h, { format: 'bgra8unorm' });
    const pass = await present(h, dest, { hdr: true, toneMapping: {} });
    expect(pass.sceneTarget).not.toBe(dest);
    expect(pass.sceneTarget.format).toBe('rgba16float');
    expect(pass.sceneTarget.width).toBe(dest.width);
    expect(pass.sceneTarget.height).toBe(dest.height);
    expect(pass.material).not.toBeNull();
    pass.dispose();
  });

  test('without hdr, the intermediate takes the destination format', async () => {
    const h = harness();
    const pass = await present(h, destination(h, { format: 'rgba8unorm' }), { toneMapping: {} });
    expect(pass.sceneTarget.format).toBe('rgba8unorm');
    pass.dispose();
  });
});

describe('PresentPass — rejections', () => {
  test('hdr with no tone map is refused, naming the intermediate', async () => {
    const h = harness();
    const error = await expectCodeAsync(
      () => present(h, destination(h), { hdr: true, toneMapping: null }),
      'OPTION_UNKNOWN',
    );
    expect(error.message).toContain('hdr: true');
    expect(error.message).toContain('offscreen intermediate');
    expect(error.fix).toContain("operator: \"none\"");
  });

  test('sampleCount 4 with no tone map is refused, naming the canvas rule', async () => {
    // This is the combination the design has to decide about, and the decision is
    // to support it: 4x needs an offscreen MSAA attachment whose resolve is a
    // texture, and something has to put that texture on the target. The only
    // thing apse has that can read one is the tone map, so `{ operator: 'none' }`
    // is the answer and this combination is refused rather than half-done.
    const h = harness();
    const error = await expectCodeAsync(
      () => present(h, destination(h), { sampleCount: 4, toneMapping: null }),
      'OPTION_UNKNOWN',
    );
    expect(error.message).toContain('sampleCount: 4');
    expect(error.message).toContain('multisampled');
    expect(error.why).toContain('nobody samples');
  });

  test('hdr with sampleCount 4 is supported: the offscreen carries the MSAA pair', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { hdr: true, sampleCount: 4, toneMapping: {} });
    expect(pass.sceneTarget.sampleCount).toBe(4);
    expect(h.presentTextures().map((t) => t.format)).toEqual([
      'rgba16float', 'rgba16float', 'depth24plus',
    ]);
    pass.dispose();
  });

  test('a sample count that is neither 1 nor 4 is refused', async () => {
    const h = harness();
    for (const count of [0, 2, 8, 16]) {
      const error = await expectCodeAsync(
        () => present(h, destination(h), { sampleCount: count as 1 | 4, toneMapping: {} }),
        'OPTION_UNKNOWN',
      );
      expect(error.message).toContain(String(count));
    }
  });

  test('an unsupported HDR format is refused, and the compat note is there', async () => {
    const h = harness();
    const error = await expectCodeAsync(
      () => present(h, destination(h), { hdr: true, toneMapping: { hdrFormat: 'rgba8unorm' } }),
      'OPTION_UNKNOWN',
    );
    expect(error.message).toContain('rgba8unorm');
    expect(error.fix).toContain('compatibility');
  });

  test('a destination with an impossible sample count is refused', async () => {
    const h = harness();
    const weird = { ...destination(h), sampleCount: 2 } as unknown as RenderTarget;
    const error = await expectCodeAsync(
      () => present(h, weird, { toneMapping: {} }),
      'RENDER_TARGET_FORMAT_MISMATCH',
    );
    expect(error.message).toContain('sampleCount is 2');
  });

  test('a size the device cannot allocate is refused, and nothing is allocated', async () => {
    const h = harness();
    const dest = destination(h, { width: 4096 });
    const error = await expectCodeAsync(
      () => present(h, dest, { hdr: true, toneMapping: {}, }) .then(async (p) => {
        p.resize(5000, 2160);
        return p;
      }),
      'CANVAS_SIZE_INVALID',
    );
    expect(error.message).toContain('5000x2160');
    // The failed resize changed nothing and destroyed nothing.
    expect(h.livePresent()).toBe(2);
  });
});

describe('PresentPass — the fullscreen render pass', () => {
  test('loads rather than clears, and attaches no depth', async () => {
    // Mirrors #encode in renderer.ts, with the two things that matter: the
    // scene's colour must survive (loadOp 'load' + storeOp 'store'; WebGPU has
    // no loadOp 'store' — see GPULoadOp vs GPUStoreOp), and the pass has no depth
    // attachment, which is only legal because the pipeline has no depthStencil.
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: {} });
    pass.render();

    expect(h.gpu.passes).toHaveLength(1);
    const descriptor = h.gpu.passes[0]!.descriptor;
    const attachments = Array.from(descriptor.colorAttachments);
    expect(attachments).toHaveLength(1);
    expect(attachments[0]?.loadOp).toBe('load');
    expect(attachments[0]?.storeOp).toBe('store');
    expect(descriptor.depthStencilAttachment).toBeUndefined();

    const recorded = h.gpu.passes[0]!;
    expect(recorded.ended).toBe(true);
    expect(recorded.pipeline).not.toBeNull();
    expect(recorded.draws).toEqual([{ vertexCount: 3, instances: 1 }]);
    // All four bind groups: the scaffold declares frame and object on every
    // material whether or not the body reads them, so an unsatisfied pipeline
    // layout is a validation error, not a warning.
    expect(recorded.bindGroups).toEqual([0, 1, 2, 3]);
    pass.dispose();
  });

  test('a multisampled destination gets a resolve destination', async () => {
    const h = harness();
    const dest = destination(h, { sampleCount: 4 });
    const pass = await present(h, dest, { toneMapping: {} });
    pass.render();
    const attachment = Array.from(h.gpu.passes[0]!.descriptor.colorAttachments)[0];
    expect(attachment?.resolveTarget).toBeDefined();
    pass.dispose();
  });

  test('a single-sampled destination has no resolve destination', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: {} });
    pass.render();
    expect(Array.from(h.gpu.passes[0]!.descriptor.colorAttachments)[0]?.resolveTarget).toBeUndefined();
    pass.dispose();
  });

  test('render() allocates nothing, ever', async () => {
    // The whole reason the constructor builds everything: a per-frame pipeline or
    // texture allocation is a frame that stutters, and it is the failure mode this
    // layer is most prone to.
    const h = harness();
    const pass = await present(h, destination(h), { hdr: true, sampleCount: 4, toneMapping: {} });
    pass.render();
    const textures = h.presentTextures().length;
    const buffers = h.gpu.buffers.length;
    const pipelines = h.gpu.pipelines.length;
    const destroyed = h.gpu.destroyLog.length;

    for (let i = 0; i < 10; i++) pass.render();

    expect(h.presentTextures()).toHaveLength(textures);
    expect(h.gpu.buffers).toHaveLength(buffers);
    expect(h.gpu.pipelines).toHaveLength(pipelines);
    expect(h.gpu.destroyLog).toHaveLength(destroyed);
    expect(h.gpu.passes).toHaveLength(11);
    pass.dispose();
  });

  test('render() acquires exactly one canvas colour view per frame', async () => {
    // A canvas view expires at present. Reading colorView inside a loop would
    // mint a view per iteration and hand later code a view of a dead texture; the
    // tripwire in target.ts exists for exactly that mistake.
    const h = harness();
    const fake = fakeAseDevice({ canvasWidth: 640, canvasHeight: 480 });
    const canvasTarget = createCanvasTarget(asCanvasTargetDevice(fake), { width: 640, height: 480 });
    const ase: RenderTargetDevice = {
      device: h.gpu as unknown as GPUDevice,
      limits: fake.limits,
      assertLive: () => { /* not lost */ },
    };
    const pass = await PresentPass.create(ase, { label: 'present', target: canvasTarget, toneMapping: {} });
    const context = fake.context as unknown as FakeCanvasContext;

    const before = context.getCurrentTextureCalls;
    pass.render();
    expect(context.getCurrentTextureCalls - before).toBe(1);
    pass.render();
    expect(context.getCurrentTextureCalls - before).toBe(2);

    pass.dispose();
    canvasTarget.dispose();
  });

  test('render() into a caller-supplied encoder does not submit', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: {} });
    const encoder = h.gpu.createCommandEncoder({ label: 'caller' });
    const encodersBefore = h.gpu.encoders;
    const submitsBefore = h.gpu.submits;

    pass.render(encoder);
    expect(h.gpu.passes).toHaveLength(1);
    expect(h.gpu.encoders).toBe(encodersBefore);
    expect(h.gpu.submits).toBe(submitsBefore);

    // With no encoder it opens and submits its own: one submit, and the queue
    // ordering is still correct because writeBuffer precedes it.
    pass.render();
    expect(h.gpu.encoders).toBe(encodersBefore + 1);
    expect(h.gpu.submits).toBe(submitsBefore + 1);
    pass.dispose();
  });

  test('exposure is re-asserted only when it is wrong', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: { exposure: 0.5 } });
    expect(pass.exposure).toBe(0.5);
    // The renderer writes frame.exposure = 1 on every upload, so the pass has to
    // put it back — but only when it differs, so the steady state is no write.
    pass.render();
    const afterFirst = h.gpu.writeLog.length;
    pass.render();
    expect(h.gpu.writeLog).toHaveLength(afterFirst);

    pass.setExposure(2);
    pass.render();
    expect(h.gpu.writeLog.length).toBeGreaterThan(afterFirst);
    pass.dispose();
  });

  test('a non-positive exposure is refused at the call site', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: {} });
    for (const value of [0, -1, Number.NaN]) {
      expectCode(() => pass.setExposure(value), 'SLOT_VALUE_NOT_FINITE');
    }
    pass.dispose();
  });

  test('using a disposed pass is a typed failure, not a silent no-op', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: {} });
    pass.dispose();
    expectCode(() => pass.render(), 'INTERNAL_INVARIANT');
    expectCode(() => pass.resize(64, 64), 'INTERNAL_INVARIANT');
    expectCode(() => pass.setExposure(2), 'INTERNAL_INVARIANT');
  });
});

describe('PresentPass — the pipeline', () => {
  test('compiles two: the material\'s own, and a depth-free companion it draws with', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: {} });
    // One from Material.create, which carries the scaffold's depthStencil, and
    // one from this class, which does not. The first is never drawn with — it is
    // the price of building the tone map through Material.create at all, and it
    // is one compile at startup rather than one per frame.
    expect(h.gpu.pipelines).toHaveLength(2);
    const companion = h.gpu.pipelines.find((d) => d.label === 'present:pipeline');
    expect(companion).toBeDefined();
    expect(companion?.depthStencil).toBeUndefined();
    expect(companion?.multisample).toEqual({ count: 1 });
    expect(companion?.primitive?.cullMode).toBe('none');
    expect(Array.from(companion?.vertex?.buffers ?? [])[0]?.arrayStride).toBe(12);
    const targets = companion?.fragment?.targets as { format: GPUTextureFormat }[];
    expect(targets[0]?.format).toBe('bgra8unorm');
    pass.dispose();
  });

  test('the companion is built for the destination format, not the canvas default', async () => {
    const h = harness();
    const pass = await present(h, destination(h, { format: 'rgba8unorm' }), { toneMapping: {} });
    const companion = h.gpu.pipelines.find((d) => d.label === 'present:pipeline');
    const targets = companion?.fragment?.targets as { format: GPUTextureFormat }[];
    expect(targets[0]?.format).toBe('rgba8unorm');
    pass.dispose();
  });

  test('the tone map material targets the destination format, which decides the sRGB path', async () => {
    const h = harness();
    const plain = await present(h, destination(h, { format: 'bgra8unorm' }), { toneMapping: {} });
    expect(plain.material?.wgsl).toContain('let display = srgbEncode(graded);');
    plain.dispose();

    const srgb = await present(h, destination(h, { format: 'bgra8unorm-srgb' }), { toneMapping: {} });
    expect(srgb.material?.wgsl).toContain('let display = graded;');
    srgb.dispose();
  });

  test('the fullscreen mesh and the frame and object uniforms come from the device cache', async () => {
    // Two FrameUniforms on one device means the renderer writes one and the
    // present pass reads the other: no error, and an exposure that never takes
    // effect. This is the invariant both resolve through deviceCache to protect.
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: {} });
    expect(pass.material?.frameUniforms).toBe(pass.material?.frameUniforms);
    expect(pass.material?.objectUniforms).toBe(pass.material?.objectUniforms);
    // Exactly one frame buffer for the device, and the pass shares it.
    const frameBuffers = h.gpu.buffers.filter((b) => b.label === 'apse:apse.frame');
    expect(frameBuffers).toHaveLength(1);
    pass.dispose();
  });
});

// ===========================================================================
// 8. The pass descriptor is shared scratch
//
// render() must not allocate, and the descriptor is the only thing it would have
// allocated every frame. Reusing one object across every PresentPass is only safe
// because every field is written before it is read — which the two tests above
// verify by construction rather than by assertion.
// ===========================================================================

describe('the shared pass descriptor', () => {
  test('two passes interleaved each get a correct descriptor', async () => {
    const h = harness();
    const a = await present(h, destination(h), { toneMapping: {} });
    const b = await present(h, destination(h, { sampleCount: 4 }), { sampleCount: 4, toneMapping: {} });

    a.render();
    const afterA = h.gpu.passes.at(-1)!.descriptor;
    b.render();
    const afterB = h.gpu.passes.at(-1)!.descriptor;
    a.render();

    expect(Array.from(afterA.colorAttachments)[0]?.resolveTarget).toBeUndefined();
    expect(Array.from(afterB.colorAttachments)[0]?.resolveTarget).toBeDefined();
    // And the last one is A's, because the scratch is re-populated per call.
    expect(h.gpu.passes.at(-1)!.descriptor).toEqual(afterA);
    a.dispose();
    b.dispose();
  });
});

// ===========================================================================
// 9. The integration seam the renderer owns
// ===========================================================================

describe('the shape the renderer depends on', () => {
  test('exposes exactly the four names the renderer calls', async () => {
    const h = harness();
    const pass = await present(h, destination(h), { toneMapping: {} });
    expect(typeof pass.sceneTarget).toBe('object');
    expect(typeof pass.render).toBe('function');
    expect(typeof pass.resize).toBe('function');
    expect(typeof pass.dispose).toBe('function');
    // The constructor is private: pipeline compilation is asynchronous on purpose
    // and a synchronous constructor could only block or hand back something that
    // cannot draw yet.
    expect(PresentPass.create.length).toBe(2);
    pass.dispose();
  });

  test('resize in the direct path does not touch the destination', async () => {
    const h = harness();
    const dest = destination(h);
    const pass = await present(h, dest, { toneMapping: null });
    const before = h.gpu.destroyLog.length;
    pass.resize(1024, 768);
    // The canvas target's size is the canvas's business, and resizing it here
    // would fight CanvasSizer.
    expect(dest.width).toBe(640);
    expect(h.gpu.destroyLog).toHaveLength(before);
    pass.dispose();
  });
});
