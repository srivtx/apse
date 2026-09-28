/**
 * The emissive material — a physically-shaped unlit surface.
 *
 * # Why this is not `basicMaterial`
 *
 * `basicMaterial` takes a colour in 0..1 and draws it. That is the right tool
 * for UI, debug geometry, and a vertex-colour tint, and it is what you want when
 * the value is *not* a physical quantity. An emitter is a physical quantity: a
 * filament at 2700 K emits a radiance, that radiance is tens or hundreds of
 * times a display white, and the only reason it looks white is that the tone map
 * compresses it. So this material:
 *
 *   - **carries radiance, not colour** — `intensity` is a real multiplier, and a
 *     value of 12 is a legitimate and useful number that a `basicMaterial` cannot
 *     express (it clips at the write);
 *   - **has a view-dependent term**, because a thin emissive coating really does
 *     get brighter at grazing angles — that is Fresnel reflectance, not a glow
 *     fudge, and it is why energy shields and holograms have a bright silhouette;
 *   - **can be driven by the frame clock**, so a beacon or a screen can pulse
 *     without the application writing a per-frame `setSlot`.
 *
 * # The working space, stated once
 *
 * Every shipped material writes **linear radiance** and nothing else. No
 * transfer function, no exposure, no clamp, no `pow(1/2.2)`. The chain that
 * finishes the job is: material → optional `rgba16float` intermediate → the
 * present pass's tone map and sRGB encode → the display. A material that
 * "corrected" its own output would be processed twice, and the second pass is
 * the one that cannot be undone by changing a number.
 *
 * **The practical consequence for this material specifically:** `intensity` above
 * 1 is only meaningful if the frame is rendered into an HDR intermediate, which
 * is what the present pass's `hdr: true` does. Draw it straight to the canvas
 * and the attachment is `unorm`, so everything above 1.0 is clipped at the write
 * and a carefully chosen 8.0 comes out identical to 1.0.
 */

import { fail } from '../core/error.ts';
import { STANDARD_LAYOUT } from '../geometry/layout.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import { VECTOR_CORE } from './lighting.ts';
import { Material } from './material.ts';
import type { MaterialOptions } from './material.ts';
import { preferredTargetFormat, samplerNameFor } from './scaffold.ts';
import type { MaterialSpec, VaryingDefs } from './scaffold.ts';
import type { TextureSlotSpec } from './texture-slot.ts';

export interface EmissiveMaterialOptions extends MaterialOptions {
  /** Defaults to `'emissive'`. */
  name?: string;
  /**
   * Emitted colour, linear. Not a display colour: 0.18 is mid grey in *linear*
   * light, and a value of 1.0 is a display white after the present pass's curve,
   * not before it. Defaults to white.
   */
  color?: readonly [number, number, number];
  /**
   * Radiance multiplier. Defaults to 1.
   *
   * Values above 1 are the point of this material — a lamp at 6 and a lamp at 60
   * differ by exactly the tone curve — and they require an `rgba16float`
   * intermediate to survive to the present pass. Against a `unorm` target every
   * value above 1 is clipped at the write, with no warning.
   */
  intensity?: number;
  /** Alpha. 1 by default; only meaningful with `transparent`. */
  opacity?: number;
  /**
   * Declare an `emissive` texture slot and multiply the colour by it.
   *
   * The map is **emission**, not albedo: it is what the surface gives off, so
   * black is black and nothing in it is ever lit. For a texture that should also
   * catch light, compose this with `basicMaterial` or `pbrMaterial`.
   */
  textured?: boolean;
  /**
   * Multiply the radiance by a Fresnel-Schlick term, so the silhouette glows.
   *
   * The physics: a thin emissive layer over a dark substrate is a dielectric
   * coating, and a dielectric's reflectance rises to 1 at grazing incidence. The
   * result is that a flat quad seen edge-on emits *more* than one seen face-on,
   * which is what a real energy shield does. `fresnelStrength` is the peak gain;
   * 1.0 means "twice as bright at the silhouette as face-on".
   */
  fresnel?: boolean;
  /** Peak gain of the view-dependent term. 0 by default. */
  fresnelStrength?: number;
  /**
   * Oscillation frequency in Hz, driven by `frame.time`. 0 (the default) is
   * steady. Frame-time driven rather than a per-frame `setSlot` so a beacon
   * costs nothing on the CPU and stays in step across a pause.
   */
  pulseHz?: number;
  /**
   * How deep the pulse goes, 0..1: 0 is no pulse, 1 fades to black twice a
   * second. Defaults to 0.
   */
  pulseDepth?: number;
  /** Alpha-blend and skip depth writes. Defaults to false. */
  transparent?: boolean;
  /** Disable back-face culling. Defaults to false. */
  doubleSided?: boolean;
  /** Defaults to {@link STANDARD_LAYOUT}. */
  layout?: VertexLayout;
  /** Defaults to the canvas preferred format. See `pbrMaterial`. */
  targetFormat?: GPUTextureFormat;
}

/**
 * The spec `emissiveMaterial` builds. Exported so the generated WGSL can be read
 * and asserted on with no device — the material *is* its spec.
 */
export function emissiveMaterialSpec(opts: EmissiveMaterialOptions = {}): MaterialSpec {
  const {
    name = 'emissive',
    color = [1, 1, 1],
    intensity = 1,
    opacity = 1,
    textured = false,
    fresnel = false,
    fresnelStrength = 0,
    pulseHz = 0,
    pulseDepth = 0,
    transparent = false,
    doubleSided = false,
    layout = STANDARD_LAYOUT,
    targetFormat = preferredTargetFormat(),
  } = opts;

  if (fresnel && fresnelStrength <= 0) {
    fail('OPTION_UNKNOWN',
      `Material "${name}" was given \`fresnel: true\` with a fresnelStrength of ${String(fresnelStrength)}.`, {
      why: 'The two options together say "the surface gets brighter at grazing angles" and "by nothing". One of them is always a mistake, and which one is not knowable from here — so it is asked rather than resolved.',
      fix: 'Drop `fresnel: true` (the term is already off at strength 0), or give a positive `fresnelStrength` — 1.0 is a doubling at the silhouette.',
      detail: { kind: 'numeric', field: 'fresnelStrength', value: fresnelStrength, min: 0 },
    });
  }
  if (pulseHz !== 0 && pulseDepth <= 0) {
    fail('OPTION_UNKNOWN',
      `Material "${name}" was given a pulseHz of ${String(pulseHz)} with a pulseDepth of ${String(pulseDepth)}.`, {
      why: 'A frequency with no depth is a uniform write every frame for no visible change, which is the shape of a bug rather than an intention.',
      fix: 'Set `pulseDepth` between 0 and 1, or set `pulseHz: 0` to leave the surface steady.',
      detail: { kind: 'numeric', field: 'pulseDepth', value: pulseDepth, min: 0 },
    });
  }
  assertNonNegative(name, 'intensity', intensity);
  assertNonNegative(name, 'fresnelStrength', fresnelStrength);
  assertNonNegative(name, 'pulseHz', pulseHz);

  const textures: Record<string, TextureSlotSpec> = {};
  if (textured) textures.emissive = { kind: '2d' };
  const sampler = samplerNameFor(textures, 'emissive');

  const pulse = pulseHz !== 0 && pulseDepth > 0;
  const sheen = fresnel || fresnelStrength > 0;

  // Only the terms that are switched on get a varying. An emitter with no
  // view-dependent term needs no normal and no world position at all, and two
  // unused inter-stage variables are two locations, two interpolators, and two
  // more things that have to survive the rasteriser for nothing.
  const varyings: VaryingDefs = {
    ...(sheen ? { worldPos: 'vec3f' as const } : {}),
    ...(sheen ? { normal: 'vec3f' as const } : {}),
    ...(textured ? { uv: 'vec2f' as const } : {}),
  };

  const fragment = [
    sheen
      ? `
// --- geometry ---------------------------------------------------------------
// Read for one reason only: the view-dependent term below. An emitter that is
// flat in all directions does not need either, and should use basicMaterial
// instead — which is also why the Fresnel option is the reason this material is
// not a renamed basicMaterial.
let n = normalize(in.normal);
let v = safeNormalize(frame.camPos - in.worldPos);
let nDotV = max(dot(n, v), 0.0);`
      : '',
    textured
      ? `
// The map is emission, so it multiplies the radiance and is never added to a lit
// term. Black in the map is genuinely black.
let radiance = textureSample(emissive, ${sampler}, in.uv).rgb * mat.color * mat.intensity;`
      : `
let radiance = mat.color * mat.intensity;`,
    sheen
      ? `
// --- view-dependent term ---------------------------------------------------
// A Fresnel-Schlick reflectance with F0 = 1: a bare emitter has nothing behind
// it to show through, so the only thing that varies with angle is the
// reflectance of the layer above it. The gain is 1 face-on and 1 + strength at
// the silhouette, and it is bounded, so one grazing pixel cannot take the frame
// with it the way an unbounded pow() would.
radiance = radiance * (1.0 + fresnelSchlick(nDotV, vec3f(1.0)) * mat.fresnelStrength);`
      : '',
    pulse
      ? `
// --- pulse -----------------------------------------------------------------
// frame.time is seconds since start, so the pulse is frame-rate independent and
// survives a tab being backgrounded. A raised cosine rather than a sine: sin has
// a non-zero derivative at its trough, and a surface sitting exactly at
// pulseDepth 1 visibly bounces there twice a cycle.
let phase = 2.0 * PI * mat.pulseHz * frame.time;
radiance = radiance * (1.0 - 0.5 * (1.0 + cos(phase)) * clamp(mat.pulseDepth, 0.0, 1.0));`
      : '',
    `
// Linear radiance out. The present pass owns exposure, the tone curve, and the
// sRGB transfer function; doing any of them here would process the frame twice.
return vec4f(radiance, mat.opacity * frame.alpha);`,
  ].map((part) => part.trim()).filter((part) => part.length > 0).join('\n\n');

  return {
    name,
    layout,
    varyings,
    slots: {
      color: { type: 'vec3f', default: color },
      intensity: { type: 'f32', default: intensity },
      opacity: { type: 'f32', default: opacity },
      ...(sheen ? { fresnelStrength: { type: 'f32' as const, default: fresnelStrength } } : {}),
      ...(pulse ? {
        pulseHz: { type: 'f32' as const, default: pulseHz },
        pulseDepth: { type: 'f32' as const, default: pulseDepth },
      } : {}),
    },
    ...(textured ? { textures } : {}),
    // Assembled from the two terms that need it. An emitter with neither the
    // sheen nor the pulse carries an empty prelude, which is honest: it needs no
    // helper function, and shipping an unused one would cost a reader more than
    // it costs the compiler nothing.
    prelude: [pulse ? 'const PI : f32 = 3.14159265359;' : '', sheen ? VECTOR_CORE : '']
      .filter((part) => part.length > 0)
      .join('\n'),
    phase: transparent ? 'transparent' : 'opaque',
    topology: 'triangle-list',
    cull: doubleSided ? 'none' : 'back',
    depth: transparent ? { write: false, compare: 'less' } : { write: true, compare: 'less' },
    blend: transparent
      ? {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      }
      : null,
    targets: [{ format: targetFormat }],
    sampleCount: 1,
    vertex: `
out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
${sheen ? 'out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;\nout.normal = normalize(obj.normalMatrix * in.normal);' : ''}
${textured ? 'out.uv = in.uv;' : ''}
`,
    fragment,
  };
}

function assertNonNegative(material: string, field: string, value: number): void {
  if (Number.isFinite(value) && value >= 0) return;
  fail('OPTION_UNKNOWN',
    `Material "${material}" was given a ${field} of ${String(value)}.`, {
    why: 'Every value this material carries scales a radiance, and radiance is not negative. A negative value clips to black through the tone map; a NaN propagates through every fragment that reads the uniform and blanks the draw rather than raising an error anywhere.',
    fix: 'Pass a finite, non-negative number. 0 is off, 1 is the identity.',
    detail: { kind: 'numeric', field, value, min: 0 },
  });
}

/**
 * Creates the emissive material.
 *
 * Returns a promise because pipeline compilation must not block the main
 * thread — see the comment at the top of `material.ts`.
 */
export function emissiveMaterial(
  device: GPUDevice,
  opts: EmissiveMaterialOptions = {},
): Promise<Material> {
  const { frame, object, maxObjects } = opts;
  return Material.create(device, emissiveMaterialSpec(opts), { frame, object, maxObjects });
}
