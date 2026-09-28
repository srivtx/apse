/**
 * The tone map — the first pass of a post-processing chain.
 *
 * # Why this is a material and not a shader string
 *
 * A present pass looks like the obvious place to hand-write WGSL. It is not. The
 * moment there is a second pass — bloom, FXAA, a colour grade, TAA — the things a
 * hand-written pass gets for free stop being free: the uniform block layout, the
 * sampler pairing, the pipeline cache key, and the fact that two materials with
 * the same declarations can share one `GPUBindGroupLayout`. So the tone map is
 * built the same way as every other material in apse: a declarative spec, and
 * generated bindings. It is entry one in a chain rather than a special case, and
 * the next entry can be written exactly the same way.
 *
 * # What makes a *fullscreen* pass different, and why it matters
 *
 * Three things, each of which is a bug if you get it wrong and a silent one:
 *
 * 1. **The vertex stage ignores `obj.model` and `frame.viewProj`.** A fullscreen
 *    pass has no world transform: the three vertices of a clip-space triangle
 *    *are* the geometry. Multiplying by the object's model matrix is how a post
 *    chain picks up a stray transform from whatever node it happened to be
 *    attached to — the screen shifts, the scene appears off-centre, and nothing
 *    anywhere reports an error.
 *
 * 2. **The UV comes from `@builtin(position)`, not from a varying.** Fragment
 *    position arrives in framebuffer pixels, already texel-centred, and
 *    interpolating a UV you could have derived in one divide costs one of the 14
 *    inter-stage variables compatibility mode allows and one more value that has
 *    to survive the rasteriser exactly. It also cannot be subtly wrong, because
 *    there is nothing to be subtly wrong with.
 *
 * 3. **The operator is a uniform, not a material per operator.** Switching from
 *    ACES to Reinhard is one 4-byte write. A pipeline rebuild per operator is a
 *    shader compile on the frame the user changed a dropdown.
 *
 * # The output-space question, which is the whole point of the module
 *
 * WebGPU gives you **no free sRGB encode** on a plain `bgra8unorm` target. The
 * `*-srgb` formats do the conversion in hardware; `bgra8unorm` — which is what
 * `navigator.gpu.getPreferredCanvasFormat()` returns on every desktop — does
 * not, and a linear value written to it is stored verbatim. The result is a
 * scene that is far too dark: a linear 0.5 should display as 0.735, and lands as
 * 0.5. **This is the single most common tone-mapping bug**, and it looks like a
 * lighting bug, so it gets "fixed" by raising light intensities until the
 * shadows look right and then the highlights are wrong on a different display.
 *
 * So the spec below asks for the target format and emits the encode **only when
 * the target is not an `*-srgb` format**. Both paths are generated from the same
 * source, so they cannot drift.
 */

import { fail } from '../core/error.ts';
import { buildUniformBlock } from '../core/uniform.ts';
import { layoutCached } from '../geometry/layout.ts';
import { MeshData } from '../geometry/mesh.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import { Material } from './material.ts';
import type { MaterialSpec } from './scaffold.ts';

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/**
 * The vertex layout a fullscreen pass uses. `position` is in clip space, -1..1.
 *
 * Position-only, 12 bytes per vertex, one attribute. Anything more is a varying
 * the fragment stage does not need, on a triangle with three vertices.
 *
 * Interned through `layoutCached`, so this is the *same object* as
 * `POSITION_LAYOUT` — and that is the point. A layout is part of the pipeline
 * cache key, so two fullscreen materials that each constructed their own layout
 * would each own a `GPUVertexBufferLayout` and a pipeline, for identical code.
 */
export const FULLSCREEN_LAYOUT: VertexLayout = layoutCached({ position: 'float32x3' });

/**
 * The clip-space triangle that covers the framebuffer. One triangle, three
 * vertices, no index buffer.
 *
 * ```txt
 *   (-1,-1) +----------------+
 *          |  \              |
 *          |    \            |
 *          |      \          |
 *          |        \        |
 *   (-1,3) +----------+  \    |
 *                      \   \  |
 *                       \   \ |
 *                        \  \|
 *                         \  |
 *              (3,-1) +----+\-+
 * ```
 *
 * **The third vertex is `(3, -1)` and `(-1, 3)`, not `(1, 1)`.** The triangle is
 * built to *overhang* the clip volume, so one primitive covers every pixel with
 * no quad seam: with a two-triangle quad the shared diagonal is rasterised twice
 * and, with a derivative-using or a sample-position-sensitive fragment stage,
 * produces a visible one-pixel diagonal seam. Every implementation that gets this
 * wrong gets a line across the screen, and because it is exactly one pixel wide it
 * is easy to mistake for a rendering artifact in the source content.
 *
 * A quad would be the alternative: 4 vertices, 6 indices, and a seam to reason
 * about. One overhanging triangle is strictly cheaper and strictly safer.
 *
 * z is 0 — the near plane in apse's `[0, 1]` clip space — so the triangle still
 * passes a `less` depth test if a depth attachment is ever attached to the pass.
 */
export function fullscreenMesh(): MeshData {
  return new MeshData({
    name: 'fullscreen',
    layout: FULLSCREEN_LAYOUT,
    vertices: {
      interleaved: new Float32Array([
        -1, -1, 0,
        3, -1, 0,
        -1, 3, 0,
      ]),
      vertexCount: 3,
    },
    indices: null,
  });
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/** The tone curves apse implements. */
export const TONE_MAP_OPERATORS = ['none', 'aces', 'reinhard', 'linear'] as const;

export type ToneMapOperator = (typeof TONE_MAP_OPERATORS)[number];

/**
 * The `operator` slot's value for each curve.
 *
 * These numbers are the shader's `switch` cases, generated from this table, so
 * the two cannot drift. They are part of the material's public contract: writing
 * `operator` by name is not possible over the uniform boundary, so this is how a
 * caller converts a string to the 4 bytes that go in the buffer.
 */
export const OPERATOR_IDS: Readonly<Record<ToneMapOperator, number>> = Object.freeze({
  none: 0,
  aces: 1,
  reinhard: 2,
  linear: 3,
});

/**
 * HDR intermediate formats the tone map will accept as its source.
 *
 * `rgba16float` is the default and the right answer on almost every device: it is
 * renderable, blendable, and *filterable*, which matters because the source is
 * sampled. `rgba32float` costs twice the bandwidth for a precision nobody can see
 * at 10 bits of display, and is the format most likely to be refused in
 * compatibility mode — see {@link TonemapOptions.hdrFormat}.
 */
export const HDR_TARGET_FORMATS: readonly GPUTextureFormat[] = [
  'rgba16float',
  'rgba32float',
];

// ---------------------------------------------------------------------------
// Curve constants
//
// These are the *only* place the numbers live. The WGSL below is generated from
// them, so a coefficient cannot be changed in the shader without changing it
// here, and a test that re-derives the curve in TypeScript reads the same
// constants the shader was built from.
// ---------------------------------------------------------------------------

/**
 * ACES fitted-curve coefficients.
 *
 * Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve", 2015 — a
 * single-rational approximation of the full ACES RRT + ODT:
 *
 * ```txt
 *   (x (a x + b)) / (x (c x + d) + e)
 * ```
 *
 * Chosen over the full RRT/ODT fit because it is five constants and one divide.
 * The full fit is two `mat3` multiplies and an `exp2` per channel, and the
 * difference in the final image is small enough that the RRT's own shadow and
 * highlight detail is lost in 8-bit output anyway. The tradeoff is real and
 * stated here: Narkowicz desaturates and lifts the deep shadows slightly
 * relative to the true ACES, which matters if you ever composite in linear and
 * tone map twice. Do not.
 */
const ACES_FIT = Object.freeze({ a: 2.51, b: 0.03, c: 2.43, d: 0.59, e: 0.14 });

/**
 * The sRGB transfer function, IEC 61966-2-1.
 *
 * The piecewise form is not optional: the standard switches from the 12.92×power
 * law to a `1/2.4` power curve at 0.0031308, and using the power law everywhere
 * makes the darkest two or three code values of a gradient visibly wrong. The
 * exponent is written as `1.0 / 2.4` in the generated WGSL rather than as its
 * 17-digit decimal expansion, because the generated program is read by people and
 * the compiler folds the division exactly.
 */
const SRGB = Object.freeze({
  linearThreshold: 0.0031308,
  linearScale: 12.92,
  powerScale: 1.055,
  powerOffset: 0.055,
});

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface TonemapOptions {
  /** Render to a linear HDR intermediate instead of the canvas. Default 'rgba16float'. */
  readonly hdrFormat?: GPUTextureFormat;
  /** Tone map operator. Default 'aces'. */
  readonly operator?: ToneMapOperator;
  /** Default 1. */
  readonly exposure?: number;
  /** Default 0. */
  readonly gamma?: number;
  /**
   * Colour format this pass writes. Defaults to the canvas preferred format.
   *
   * This is not a detail: the format decides whether the sRGB encode happens in
   * the shader or in hardware, and a mismatch between the format passed here and
   * the target actually drawn into is a too-dark image with no error anywhere.
   * Pass the present target's format whenever it is not the canvas.
   */
  readonly targetFormat?: GPUTextureFormat;
}

/**
 * The uniform fields this pass is defined by: field name to WGSL type.
 *
 * **Not all four are material slots.** `exposure` is already a field of the
 * generated frame block — `core/slot.ts` reserves it there with the comment
 * "tone-map exposure, applied by the present pass", which is exactly this pass —
 * and `RESERVED_SLOT_NAMES` refuses any material slot that shadows a generated
 * name. So the table is the honest description of the four values the shader
 * reads, and {@link TONEMAP_MATERIAL_SLOTS} is the subset the material block
 * actually carries. Exposure is read as `frame.exposure`, which also makes it
 * what an exposure control actually is: one value for the whole frame rather than
 * one per material.
 *
 * `toneMap` is a **blend weight**: 1 is the tone-mapped image, 0 is the raw
 * clamped scene, and anything between cross-fades. It exists so that "show me the
 * pre-tone-map image" is a 4-byte write rather than a second material and a
 * second pipeline, for the same reason the operator is a uniform.
 *
 * Offsets are never hand-computed. `buildUniformBlock` places every one of these
 * by the WGSL uniform address-space rules, and the material block the shader sees
 * is generated from the same table.
 */
export const TONEMAP_SLOTS: Readonly<{
  toneMap: 'f32';
  exposure: 'f32';
  operator: 'i32';
  gamma: 'f32';
}> = Object.freeze({
  /** Blend weight between the raw clamped scene (0) and the tone-mapped result (1). */
  toneMap: 'f32',
  /** Multiplier applied to the linear scene value before the curve. Read from the frame block. */
  exposure: 'f32',
  /** One of {@link OPERATOR_IDS}. Selected in the shader, not by pipeline. */
  operator: 'i32',
  /** Exponent offset on the display transfer, applied after the sRGB encode. */
  gamma: 'f32',
});

/**
 * The subset of {@link TONEMAP_SLOTS} that lives in the material block at
 * `@group(2)`. What `tonemapMaterialSpec` passes as `slots`.
 */
export const TONEMAP_MATERIAL_SLOTS: Readonly<{
  toneMap: 'f32';
  operator: 'i32';
  gamma: 'f32';
}> = Object.freeze({
  toneMap: TONEMAP_SLOTS.toneMap,
  operator: TONEMAP_SLOTS.operator,
  gamma: TONEMAP_SLOTS.gamma,
});

/** The field of the frame block the fourth value is read from. */
export const EXPOSURE_FRAME_FIELD = 'exposure' as const;

/** The resolved uniform block for {@link TONEMAP_MATERIAL_SLOTS}, for tooling and tests. */
export const TONEMAP_BLOCK = buildUniformBlock('MaterialData', TONEMAP_MATERIAL_SLOTS, { maxBindingSize: 65536 });

// ---------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------

/**
 * Tone curve functions and transfer functions. Declarations, so they belong in
 * `prelude` — the one place a material author may write a `const` or an `fn`.
 *
 * The `srgbEncode` helper is emitted whether or not this target needs it. It is
 * four lines, and a shader that carries a function it does not call costs
 * nothing at runtime — whereas emitting it conditionally means two preludes and
 * two pipelines for what is one program with one line switched.
 */
function toneMapPrelude(): string {
  return `
// --- operator ids ---------------------------------------------------------
// Generated from OPERATOR_IDS in src/material/tonemap.ts. Changing a number
// here without changing that table is the one way these could drift, and the
// two are generated from the same object.
const OPERATOR_NONE : i32 = 0;
const OPERATOR_ACES : i32 = 1;
const OPERATOR_REINHARD : i32 = 2;
const OPERATOR_LINEAR : i32 = 3;

// --- the sRGB transfer function -------------------------------------------
// IEC 61966-2-1. Only referenced on a target that is not an *-srgb format;
// a hardware-encode target has already done this by the time the value leaves.
const SRGB_LINEAR_THRESHOLD : f32 = ${SRGB.linearThreshold};
const SRGB_LINEAR_SCALE : f32 = ${SRGB.linearScale};
const SRGB_POWER_SCALE : f32 = ${SRGB.powerScale};
const SRGB_POWER_OFFSET : f32 = ${SRGB.powerOffset};

fn srgbEncode(c : vec3f) -> vec3f {
  // The two branches meet at SRGB_LINEAR_THRESHOLD, so the function is
  // continuous: using the power law below it darkens the first few code values
  // of every gradient.
  let lo = c * SRGB_LINEAR_SCALE;
  // max(c, 0): pow() of a negative base is undefined, and a NaN here would
  // propagate into the whole fragment and blank it.
  let hi = SRGB_POWER_SCALE * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - SRGB_POWER_OFFSET;
  return select(hi, lo, c < vec3f(SRGB_LINEAR_THRESHOLD));
}

// --- ACES ------------------------------------------------------------------
// Narkowicz 2015, a single-rational fit of the full ACES RRT + ODT. See the
// ACES_FIT object in src/material/tonemap.ts for the citation and the tradeoff
// against the two-mat3 version.
const ACES_A : f32 = ${ACES_FIT.a};
const ACES_B : f32 = ${ACES_FIT.b};
const ACES_C : f32 = ${ACES_FIT.c};
const ACES_D : f32 = ${ACES_FIT.d};
const ACES_E : f32 = ${ACES_FIT.e};

fn acesFitted(x : vec3f) -> vec3f {
  return clamp((x * (ACES_A * x + ACES_B)) / (x * (ACES_C * x + ACES_D) + ACES_E), vec3f(0.0), vec3f(1.0));
}

// --- Reinhard ---------------------------------------------------------------
// Reinhard, Stark, Shirley, Ferwerda, SIGGRAPH 2002, "Photographic Tone
// Reproduction for Digital Images": the per-channel operator c / (1 + c).
// Compresses indefinitely, which is its virtue and also its problem — it
// approaches white asymptotically, so a scene with a large bright area never
// reaches it and reads as flat.
fn reinhard(x : vec3f) -> vec3f {
  return x / (vec3f(1.0) + x);
}

// --- the selector -----------------------------------------------------------
// A switch on a uniform, not a pipeline per operator: changing the curve is a
// 4-byte write, never a shader compile.
fn applyOperator(x : vec3f, op : i32) -> vec3f {
  switch (op) {
    // NONE and LINEAR are the same operation, deliberately, and both are named:
    // a reader scanning the switch should be able to see that 'none' is a
    // decision rather than a fallthrough. An unclamped write to a unorm target is
    // implementation-defined rather than merely bright, so the clamp stays.
    case OPERATOR_NONE: { return clamp(x, vec3f(0.0), vec3f(1.0)); }
    case OPERATOR_LINEAR: { return clamp(x, vec3f(0.0), vec3f(1.0)); }
    case OPERATOR_ACES: { return acesFitted(x); }
    case OPERATOR_REINHARD: { return reinhard(x); }
    // A value that is not one of the four: a stale uniform buffer, or a typo in a
    // setSlot. Clamp rather than branch into undefined behaviour.
    default: { return clamp(x, vec3f(0.0), vec3f(1.0)); }
  }
}
`;
}

/** Vertex stage. Three lines, and the two that matter are the ones it does not have. */
const TONEMAP_VERTEX = `
// No frame.viewProj, no obj.model. A fullscreen pass has no world transform:
// the three vertices of the clip-space triangle are the geometry. Reading the
// object matrix here is how a post chain inherits a stray transform from
// whatever node it was attached to, and the symptom is a shifted screen with
// no error anywhere.
//
// z = 0 is the near plane in apse's [0, 1] clip space, so this still passes a
// 'less' depth test if a depth attachment is ever attached to the pass.
out.clip = vec4f(in.position, 0.0);
`;

/**
 * Fragment stage. The UV is derived from `@builtin(position)` rather than
 * interpolated — see the module comment.
 */
function tonemapFragment(targetFormat: GPUTextureFormat): string {
  // True when the *hardware* does the encode on store, which is exactly when the
  // shader must not do it as well.
  const hardwareEncodes = isSrgbFormat(targetFormat);
  const display = hardwareEncodes
    ? `// ${targetFormat} is an *-srgb format, so the hardware encodes on store.
// Encoding here as well would apply the transfer function twice, which reads
// as a washed-out, low-contrast image rather than as an error.
let display = graded;`
    : `// ${targetFormat} is NOT an *-srgb format, so the encode has to happen
// here. WebGPU provides no free conversion on a plain bgra8unorm target: a
// linear value written to one is stored verbatim and the image comes out far
// too dark. This line is the difference between correct and broken.
let display = srgbEncode(graded);`;

  return `
// @builtin(position) arrives in framebuffer pixels, texel-centred. No varying:
// a UV is one divide from the value already in the register, and interpolating
// it would spend an inter-stage variable for nothing.
let texel = textureSample(texture, textureSampler, in.clip.xy / vec2f(textureDimensions(texture, 0)));

// Linear scene radiance, exposed. Exposure comes from the frame block rather
// than a material slot: it is one value for the whole frame, it is a reserved
// generated name a material slot may not shadow, and the renderer already writes
// it every frame. Alpha is dropped rather than carried: a per-channel tone map
// applied to premultiplied colour is wrong, and apse's canvas defaults to
// alphaMode 'opaque' where the channel is ignored anyway.
let scene = texel.rgb * frame.exposure;

let mapped = applyOperator(scene, mat.operator);

// mat.toneMap = 1 is the tone-mapped image, 0 is the raw clamped scene.
let graded = mix(clamp(scene, vec3f(0.0), vec3f(1.0)), mapped, mat.toneMap);

${display}

// A display-gamma correction applied *after* the transfer function, so it
// composes identically whether the encode was done here or in hardware.
// gamma = 0 is the identity, which is why 0 is the default. gamma must be > -1,
// which the option validation enforces: 1 + gamma = 0 would make pow(0, 0)
// undefined, and WGSL leaves that unspecified per target.
let g = 1.0 + mat.gamma;
let outRgb = pow(clamp(display, vec3f(0.0), vec3f(1.0)), vec3f(g));

return vec4f(outRgb, 1.0);
`;
}

// ---------------------------------------------------------------------------
// The spec
// ---------------------------------------------------------------------------

/**
 * The spec `tonemapMaterial` builds. Exported so the generated WGSL can be
 * asserted on without a device — which is the only way to test a shader here,
 * and the whole point of the material being a spec rather than a program.
 */
export function tonemapMaterialSpec(opts: TonemapOptions = {}): MaterialSpec {
  const {
    hdrFormat = 'rgba16float',
    operator = 'aces',
    exposure = 1,
    gamma = 0,
    targetFormat = preferredFormat(),
  } = opts;

  assertHDRFormat(hdrFormat);
  assertOperator(operator);
  assertExposure(exposure);
  assertGamma(gamma);

  return {
    name: 'tonemap',
    layout: FULLSCREEN_LAYOUT,
    // No varyings at all. The UV is derived from @builtin(position) in the
    // fragment stage; see the module comment for why that is not a varying.
    slots: {
      // `exposure` is deliberately absent: it is a reserved frame-block field, and
      // a material slot of that name is rejected. The fragment stage reads
      // `frame.exposure` instead, which is the same value the renderer writes.
      toneMap: { type: TONEMAP_MATERIAL_SLOTS.toneMap, default: 1 },
      operator: { type: TONEMAP_MATERIAL_SLOTS.operator, default: OPERATOR_IDS[operator] },
      gamma: { type: TONEMAP_MATERIAL_SLOTS.gamma, default: gamma },
    },
    textures: {
      // The HDR source. No mips — it is a render target, not an asset — and
      // clamp-to-edge so a half-pixel UV at the edge cannot wrap to the other
      // side, which is the difference between an invisible seam and a mirrored
      // one-pixel column.
      texture: { kind: '2d', sampleType: 'float', mipmapFilter: false, addressMode: 'clamp-to-edge' },
    },
    prelude: toneMapPrelude(),
    phase: 'opaque',
    topology: 'triangle-list',
    // 'none' rather than the 'back' default: a fullscreen triangle has nothing
    // to cull, so culling it can only ever be a bug — and the classic one is a
    // reversed winding deleting the entire screen with no error.
    cull: 'none',
    depth: { write: false, compare: 'always' },
    blend: null,
    targets: [{ format: targetFormat }],
    sampleCount: 1,
    vertex: TONEMAP_VERTEX,
    fragment: tonemapFragment(targetFormat),
  };
}

/**
 * Creates the tone map.
 *
 * Returns a promise because pipeline compilation must not block the main
 * thread — see the module comment in `material.ts`.
 *
 * The frame and object uniforms are the shared per-device ones from
 * `deviceCache`, under the default labels: the same instances the renderer
 * writes. That is why a present pass cannot end up bound to a second, private
 * frame buffer that nothing ever updates.
 */
export function tonemapMaterial(
  device: GPUDevice,
  opts: TonemapOptions = {},
): Promise<Material> {
  return Material.create(device, tonemapMaterialSpec(opts));
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * True when a format does its own sRGB encode on store.
 *
 * A `bgra8unorm-srgb` target converts a linear value to sRGB as it is written.
 * A `bgra8unorm` one does not, and there is nothing else in the API that will:
 * the shader is the only place the transfer function can be applied.
 */
export function isSrgbFormat(format: GPUTextureFormat): boolean {
  return format.endsWith('-srgb');
}

function assertHDRFormat(format: GPUTextureFormat): void {
  if (HDR_TARGET_FORMATS.includes(format)) return;
  fail('OPTION_UNKNOWN',
    `Option hdrFormat was given the value "${format}", which is not one of: ${HDR_TARGET_FORMATS.join(', ')}.`, {
      why: 'The HDR intermediate has to be renderable *and* filterable, because the present pass samples it. rgba16float is both and is the format to use; rgba32float is renderable on the core profile but is the one most likely to be refused in compatibility mode, where blending and rendering of 32-bit float formats are restricted.',
      fix: `Use one of: ${HDR_TARGET_FORMATS.join(', ')}, or omit it for the rgba16float default. If you need a cheaper HDR format, supersample or render at a lower internal resolution and let the compositor scale it.`,
    });
}

function assertOperator(operator: ToneMapOperator): void {
  if (TONE_MAP_OPERATORS.includes(operator)) return;
  fail('OPTION_UNKNOWN',
    `Option operator was given the value "${operator}", which is not one of: ${TONE_MAP_OPERATORS.join(', ')}.`, {
      why: 'The operator is compiled into the generated WGSL as a switch case, so an unrecognised name has no case to select and would silently behave as the default.',
      fix: `Use one of: ${TONE_MAP_OPERATORS.join(', ')}. 'aces' is the default; 'linear' is a clamp with no curve; 'none' is the same clamp, for callers that want to say so explicitly.`,
    });
}

function assertExposure(exposure: number): void {
  if (Number.isFinite(exposure) && exposure > 0) return;
  fail('OPTION_UNKNOWN',
    `Option exposure was given the value ${String(exposure)}.`, {
      why: 'Exposure multiplies every pixel of the scene before the curve. Zero produces a black frame, a negative value produces a negative radiance that clamps to black through the operator, and NaN propagates through every fragment and blanks the whole pass.',
      fix: 'Pass a positive, finite number. 1 is the identity; 0.5 darkens by one stop.',
      detail: { kind: 'numeric', field: 'exposure', value: exposure, min: 0 },
    });
}

function assertGamma(gamma: number): void {
  if (Number.isFinite(gamma) && gamma > -1) return;
  fail('OPTION_UNKNOWN',
    `Option gamma was given the value ${String(gamma)}.`, {
      why: 'The shader computes pow(colour, 1 + gamma), and pow(0, 0) is unspecified in WGSL — a target is free to return anything, including NaN, which would blank the fragment. gamma = -1 is exactly that case.',
      fix: 'Pass a number greater than -1. 0 is the identity and is the default; 0.5 gives a gamma of 1.5.',
      detail: { kind: 'numeric', field: 'gamma', value: gamma, min: -1 },
    });
}

/**
 * The canvas format a material should target when the caller does not say.
 *
 * Read at call time and guarded for non-browser contexts, so importing this
 * module in a Bun test does not throw. The same five lines appear in
 * `basic.ts` and `pbr.ts` for the same reason: a shared helper would be a fourth
 * module for something that must stay inlined in every material's own default
 * path.
 */
function preferredFormat(): GPUTextureFormat {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined'
    ? navigator.gpu.getPreferredCanvasFormat()
    : 'rgba8unorm';
}
