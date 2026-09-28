/**
 * The declarative lighting layer shared by the shipped lit materials.
 *
 * # Why this module exists
 *
 * "Two lights and a fresnel rim" is the single largest gap between apse and a
 * library like three.js, and the *wrong* way to close it is to write a second
 * WGSL string by hand. A hand-written two-light shader is a shader: it has a
 * fixed light count baked into its text, so a third light is another string,
 * and the uniform block, the packer, and the documentation all have to be
 * updated to match by hand, at which point they can disagree — which is the
 * exact failure this project exists to remove.
 *
 * So the light count is **data**. {@link resolveLights} turns a
 * `readonly DirectionalLight[]` into slot declarations, and
 * {@link directLightLoop} turns the same array into the matching statement
 * block. Both walk the same array, so the number of uniform fields and the
 * number of shading statements cannot drift apart: adding a light to the array
 * adds its `lightDir`/`lightColor` slots *and* its loop iteration, and touches
 * no WGSL text, no hand-computed byte offset, and no other file.
 *
 * # The naming rule for the first light, and why it is not uniform
 *
 * The first light's slots are `lightDir` and `lightColor`; the n-th is
 * `light{n}Dir` and `light{n}Color`. The asymmetry is deliberate: `lightDir` and
 * `lightColor` shipped in `pbrMaterial`, and `material.setSlot('lightDir', …)`
 * is in the wild. Renaming them to `light0Dir` would break that call with a
 * typed `OPTION_UNKNOWN` at runtime rather than a compile error, so the
 * un-suffixed pair stays and every additional light is suffixed with its index.
 * {@link lightDirSlot} and {@link lightColorSlot} are the only place that rule
 * exists, and a test pins it.
 *
 * # Radiance, not intensity
 *
 * A light carries `color` and `intensity`, and the spec folds them into one
 * `vec3f` radiance slot. Two slots per light instead of three, and — more to the
 * point — a light's contribution is `radiance * N·L`, so a light retuned at
 * runtime is a `setSlot('lightColor', …)` and nothing else. There is no separate
 * intensity that can be forgotten.
 *
 * # What is here and what is not
 *
 * The BRDFs are `prelude` fragments — the one place a material may write a
 * declaration — and they are *assembled* from flags rather than emitted whole,
 * so a Lambert material does not carry a GGX distribution function it never
 * calls. Everything a lit shader needs — the light slots, the loop, the shadow
 * term, the ambient, the image-based term, the rim — is generated from the same
 * option objects the material factory takes.
 */

import { fail } from '../core/error.ts';
import type { SlotDefs, SlotType } from '../core/slot.ts';

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------

/**
 * How many directional lights a shipped material will accept.
 *
 * Four, and the reason is a byte budget as much as an aesthetic one: each light
 * is two `vec3f` slots, 32 bytes of uniform block, and one unrolled iteration of
 * the direct-lighting loop. A fifth light in a forward renderer is a real
 * per-fragment cost on top of a loop that is already unrolled, and a material
 * block that is approaching the compatibility profile's 16 KiB binding is a
 * material that will not build on a phone.
 *
 * The constant is here rather than in each material so that raising it is one
 * edit which every shipped material follows.
 */
export const MAX_DIRECTIONAL_LIGHTS = 4;

/** One directional light, as data. */
export interface DirectionalLight {
  /**
   * Direction **towards** the light, in world space.
   *
   * Not the direction the light travels: a light at `(0, 10, 0)` is
   * `direction: [0, 1, 0]`. Need not be normalised, and is not — the shader
   * normalises it, so `normalize(lightPos - worldPos)` can be precomputed once
   * per light per frame instead of once per fragment.
   */
  readonly direction: readonly [number, number, number];
  /** Radiance, linear. Defaults to white. Multiplied by `intensity`. */
  readonly color?: readonly [number, number, number];
  /** Scalar multiplier on `color`. Defaults to 1. Negative is rejected. */
  readonly intensity?: number;
  /**
   * Multiply this light's direct term by the shadow map's visibility estimate.
   *
   * Only the **first** light may set it, and only when the material was built
   * with `shadows: true`. There is one shadow map and one light matrix — the
   * generated `lightViewProj` slot — so a second shadowed light would need a
   * second matrix and a second depth texture, which is a second pass rather than
   * a parameter. Both mistakes are rejected with the reason rather than silently
   * ignored.
   */
  readonly castShadow?: boolean;
}

/** A light after validation: slot names resolved, intensity folded into radiance. */
export interface ResolvedLight {
  readonly index: number;
  /** The `mat.` slot holding this light's direction. */
  readonly dirSlot: string;
  /** The `mat.` slot holding this light's radiance. */
  readonly colorSlot: string;
  readonly direction: readonly [number, number, number];
  readonly radiance: readonly [number, number, number];
  readonly castShadow: boolean;
}

/** The first light keeps the shipped un-suffixed slot names. See the module note. */
export function lightDirSlot(index: number): string {
  return index === 0 ? 'lightDir' : `light${index}Dir`;
}

/** The first light keeps the shipped un-suffixed slot names. See the module note. */
export function lightColorSlot(index: number): string {
  return index === 0 ? 'lightColor' : `light${index}Color`;
}

/**
 * Validates a light array and resolves it to slot names and folded radiance.
 *
 * `shadows` is the *material's* setting, not the light's: it says whether a
 * depth texture is declared at all, and a `castShadow` on a material with no
 * shadow map has nothing to multiply by.
 */
export function resolveLights(
  lights: readonly DirectionalLight[] | undefined,
  opts: { readonly shadows: boolean; readonly material: string },
): ResolvedLight[] {
  if (lights === undefined) return [];
  if (lights.length === 0) {
    fail('OPTION_UNKNOWN',
      `Material "${opts.material}" was given an empty light array.`, {
      why: 'A lit material with no light is a material that shades only its ambient term, which is a different material. An empty array is nearly always a lights variable that was built conditionally and came out empty this frame.',
      fix: `Omit the option to get the default single light, or pass at least one: \`lights: [{ direction: [0, 1, 0], color: [3, 3, 3] }]\`. apse accepts between 1 and ${MAX_DIRECTIONAL_LIGHTS}.`,
    });
  }
  if (lights.length > MAX_DIRECTIONAL_LIGHTS) {
    fail('OPTION_UNKNOWN',
      `Material "${opts.material}" declares ${lights.length} lights; apse's shipped materials accept at most ${MAX_DIRECTIONAL_LIGHTS}.`, {
      why: 'Each light is two vec3f uniform slots and one unrolled iteration of the direct-lighting loop, so the cost of a light is a fixed 32 bytes of uniform block plus a per-fragment BRDF evaluation. The cap is what keeps a shipped material inside the compatibility profile\'s 16 KiB uniform binding and inside a fragment cost a mobile GPU can carry.',
      fix: `Keep the brightest ${MAX_DIRECTIONAL_LIGHTS} and fold the rest into the ambient or environment term, which is what a single ambient probe is for. To raise the cap, edit MAX_DIRECTIONAL_LIGHTS in src/material/lighting.ts — one constant, and every shipped material follows it.`,
    });
  }

  const out: ResolvedLight[] = [];
  for (let i = 0; i < lights.length; i++) {
    const light: DirectionalLight = lights[i];
    const where = `Light ${i} of material "${opts.material}"`;

    const direction = assertVec3(light.direction, `${where} direction`);
    if (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2] === 0) {
      fail('OPTION_UNKNOWN',
        `${where} has a direction of [0, 0, 0].`, {
        why: 'The shader normalises the direction, and normalize(vec3f(0)) is not a direction — it is a NaN, which propagates through the whole direct-lighting term and blanks the draw. The check is here precisely because nothing downstream of it can report the mistake.',
        fix: 'Give the direction towards the light, e.g. `[0, 1, 0]` for a light directly overhead. It need not be normalised.',
        detail: { kind: 'numeric', field: `light${i}.direction`, value: 0, min: 1e-6 },
      });
    }

    const color = light.color ?? [1, 1, 1];
    const base = assertVec3(color, `${where} color`);
    const intensity: number = light.intensity ?? 1;
    if (!Number.isFinite(intensity) || intensity < 0) {
      fail('OPTION_UNKNOWN',
        `${where} has an intensity of ${String(intensity)}.`, {
        why: 'Intensity multiplies the light\'s radiance. A negative value subtracts energy from the surface, so an unlit face could be darker than the ambient term alone; a NaN blanks every fragment that reads the light.',
        fix: 'Pass a finite, non-negative number. 1 is the identity; 3 is a reasonably bright key light against an ambient of 0.2.',
        detail: { kind: 'numeric', field: `light${i}.intensity`, value: intensity, min: 0 },
      });
    }

    if (light.castShadow === true && !opts.shadows) {
      fail('OPTION_UNKNOWN',
        `${where} sets castShadow, but material "${opts.material}" was built with shadows: false.`, {
        why: 'The shadow term multiplies the direct lighting by a 3x3 PCF estimate sampled from a declared `shadowDepth` texture. With `shadows: false` no such texture is declared and no `lightViewProj` matrix exists, so the flag would have nothing to act on and the light would be lit as if unoccluded — silently, because a light *is* usually unoccluded.',
        fix: `Build the material with \`shadows: true\`, or drop \`castShadow\` from ${where}. A shadowed material cannot be drawn until a depth texture is assigned with setTexture('shadowDepth', view).`,
      });
    }
    if (light.castShadow === true && i !== 0) {
      fail('OPTION_UNKNOWN',
        `${where} sets castShadow, but only the first light can be shadowed.`, {
        why: 'apse declares one shadow map, one comparison sampler, and one `lightViewProj` matrix, and that matrix belongs to light 0 by construction. A second shadowed light would therefore be sampling the first light\'s depth from the first light\'s point of view, which puts a shadow on the wrong surface — worse than no shadow at all.',
        fix: 'Reorder the array so the shadowed light is first, or shadow the key light only, which is what a single cascade is for. Two shadowed lights need two cascades and two passes.',
      });
    }

    out.push(Object.freeze({
      index: i,
      dirSlot: lightDirSlot(i),
      colorSlot: lightColorSlot(i),
      direction,
      radiance: Object.freeze([base[0] * intensity, base[1] * intensity, base[2] * intensity]) as readonly [number, number, number],
      castShadow: light.castShadow === true && i === 0,
    }));
  }
  return out;
}

function assertVec3(value: readonly number[], what: string): readonly [number, number, number] {
  if (value.length !== 3) {
    fail('OPTION_UNKNOWN', `${what} must be three numbers, but it has ${value.length}.`, {
      why: 'A vec3f uniform slot is written from exactly three floats. A two-element value would leave the third reading whatever the previous frame wrote, so the light would change brightness whenever the slot was reused.',
      fix: 'Pass a three-element array, e.g. `[0, 1, 0]`.',
    });
  }
  for (const c of value) {
    if (typeof c !== 'number' || !Number.isFinite(c)) {
      fail('OPTION_UNKNOWN', `${what} contains ${String(c)}, which is not a finite number.`, {
        why: 'NaN and Infinity propagate through every fragment that reads the uniform and blank the draw rather than raising an error anywhere.',
        fix: 'Pass finite numbers. There is no sentinel for "unset" — omit the option instead.',
      });
    }
  }
  return value as readonly [number, number, number];
}

/**
 * The `lightDir`/`lightColor` slot pairs, in declaration order.
 *
 * The keys are the material block's field order, and the order of a uniform
 * block is its layout: apse places each field by the WGSL uniform rules, so this
 * object is simultaneously the struct, the byte offsets, and the JS packer.
 */
export function lightSlots(lights: readonly ResolvedLight[]): SlotDefs {
  const slots: Record<string, SlotType | { type: SlotType; default: readonly number[] }> = {};
  for (const light of lights) {
    slots[light.dirSlot] = { type: 'vec3f', default: light.direction };
    slots[light.colorSlot] = { type: 'vec3f', default: light.radiance };
  }
  return slots;
}

// ---------------------------------------------------------------------------
// The direct-lighting loop
// ---------------------------------------------------------------------------

/**
 * The two per-light expressions a BRDF contributes to the direct term.
 *
 * Each is WGSL *statement* text that must declare exactly one `vec3f` — the
 * names are fixed so the loop body that wraps them is identical for every
 * material. `specular` may declare `nDotH` / `vDotH` / tangent-frame quantities
 * itself; the loop only guarantees `l`, `nDotL`, and `h`.
 */
export interface DirectBrdf {
  /** Statements declaring `spec : vec3f` for this light. */
  readonly specular: string;
  /** Statements declaring `diff : vec3f` for this light. */
  readonly diffuse: string;
  /** Statements emitted once before the loop, after `n`, `v`, `f0` are bound. */
  readonly setup?: string;
  /**
   * Whether the loop needs the halfway vector.
   *
   * Cook-Torrance does, because D and V are both functions of `H`. Oren-Nayar
   * does not — it wants `L·V` instead, which is the whole reason it is a
   * different model — and emitting `h`, `nDotH`, and `vDotH` for it puts three
   * dead `let`s in a file people read to learn the library, and three live
   * normalizations per light in a fragment shader. Defaults to true.
   */
  readonly halfway?: boolean;
}

/**
 * The direct-lighting loop, one unrolled block per light.
 *
 * **Unrolled rather than a `for` loop over a uniform count** for two reasons,
 * and both are correctness rather than speed. A `for` loop indexed by a uniform
 * cannot be unrolled by the compiler, so every light pays for the maximum count
 * and the shader carries a branch; and a uniform-count loop makes the *shader's*
 * arithmetic depend on a *value* rather than on the spec, which is precisely the
 * coupling the scaffold exists to remove. Here the count is a compile-time
 * consequence of the `lights` array, so the WGSL, the uniform block, and the
 * documentation are all generated from one thing.
 *
 * `n`, `v`, `f0`, `diffuseColor`, `roughness`, and the shadow factor `shadow`
 * (declared to 1 when shadows are off) must already be in scope.
 */
export function directLightLoop(
  lights: readonly ResolvedLight[],
  brdf: DirectBrdf,
): string {
  const blocks: string[] = [];
  if (brdf.setup !== undefined) blocks.push(brdf.setup);
  const anyShadowed = lights.some((l) => l.castShadow);

  for (const light of lights) {
    // Light 0 is the only one that can carry a shadow, so the occlusion comes
    // from the resolved flag rather than from a separate material option.
    const occlusion = light.castShadow ? ' * shadow' : '';
    const note = light.castShadow
      ? ''
      : `\n    // ${anyShadowed
        ? 'This light is not the shadow caster, so the term is unoccluded.'
        : 'No shadow map is declared for this material, so the term is unoccluded.'}`;
    const half = brdf.halfway === false
      ? ''
      : `
    let h = safeNormalize(v + l);
    let nDotH = max(dot(n, h), 0.0);
    let vDotH = max(dot(v, h), 0.0);`;
    blocks.push(`{
    let l = safeNormalize(mat.${light.dirSlot});
    let nDotL = max(dot(n, l), 0.0);${half}
${brdf.specular}
${brdf.diffuse}
    direct = direct + (diff + spec) * mat.${light.colorSlot} * nDotL${occlusion};${note}
  }`);
  }
  return blocks.join('\n');
}

// ---------------------------------------------------------------------------
// Shadow term
// ---------------------------------------------------------------------------

/**
 * The 3x3 PCF visibility estimate, as statements.
 *
 * `textureSampleCompare` is only legal in uniform control flow, and a fragment
 * stage that branched per pixel is exactly what that rule forbids. So the
 * bounds test is folded into the result through `select` instead of being used
 * as an early-out: outside the map, the clamped edge is sampled and the result
 * discarded. A test asserts the `select` survives, because "simplifying" it
 * back into an `if` compiles on some drivers and fails on others.
 */
export function shadowBlock(): string {
  return `
// --- shadow ----------------------------------------------------------------
// The light is directional and its projection is orthographic, so the transform
// is affine and w is 1. Transform into the light's clip space, then into texture
// space: xy in [0,1] with y flipped because NDC y is up and texture v is down,
// and z already in [0,1] because the comparison sampler wants [0,1] and a
// WebGPU depth format is 0-to-1 rather than OpenGL's -1-to-1.
let lightClip = mat.lightViewProj * vec4f(in.worldPos, 1.0);
let ndc = lightClip.xyz;
let shadowUV = ndc.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);

let inside = all(shadowUV >= vec2f(0.0)) && all(shadowUV <= vec2f(1.0)) && ndc.z <= 1.0;
let clampedUV = clamp(shadowUV, vec2f(0.0), vec2f(1.0));
let reference = ndc.z - mat.shadowBias;

// 3x3 PCF. A constant-bounded loop is still uniform control flow, so the
// comparison sample is legal here.
let texel = vec2f(1.0) / vec2f(textureDimensions(shadowDepth));
var sum = 0.0;
for (var y = 0; y < 3; y = y + 1) {
  for (var x = 0; x < 3; x = x + 1) {
    let offset = vec2f(f32(x - 1), f32(y - 1)) * texel;
    sum = sum + textureSampleCompare(shadowDepth, shadowDepthSampler, clampedUV + offset, reference);
  }
}
let shadow = select(1.0, sum / 9.0, inside);`;
}

// ---------------------------------------------------------------------------
// Ambient, rim, image-based lighting
// ---------------------------------------------------------------------------

/**
 * The hemisphere ambient, the roughness-driven AO approximation, and the rim.
 *
 * All three are one function because all three are one idea: how much of the
 * sky a point on the surface can see. `upness` is the only geometric input, and
 * it is the cheapest possible ambient-occlusion proxy — a surface facing up sees
 * more sky than one facing down, which is true for almost every real scene and
 * costs two ALU.
 *
 * **The rim is a Fresnel-Schlick term, not `pow(1 - N·V, k)`.** That is the
 * whole difference. `pow(1 - N·V, k)` is a curve with an arbitrary exponent that
 * happens to be zero head-on and one at grazing; Fresnel-Schlick is an
 * *approximation to a reflectance*, with a physically meaningful F0 (`rimColor`),
 * and it composes with the rest of the energy budget because the value it returns
 * is the fraction of light the surface would reflect at that angle. That is what
 * lets {@link rimBlock} subtract the rim from the diffuse instead of adding to
 * it and hoping.
 *
 * Emitted only when at least one of the three is non-default, so a material that
 * wants neither the sky gradient nor the sheen does not carry the arithmetic.
 */
export function ambientBlock(opts: {
  readonly hemisphere: boolean;
  readonly ao: boolean;
  readonly rim: boolean;
  /**
   * What the rim is masked by. A metal already has a specular lobe, so its rim
   * is a sheen on top of one and is scaled down; a material with no metalness
   * passes `'1.0'`. Written as an expression rather than a flag because the
   * emitted line is the same either way and two lines of WGSL that differ only in
   * this token are two lines to keep in step.
   */
  readonly rimMask?: string;
}): string {
  if (!opts.hemisphere && !opts.ao && !opts.rim) return '';
  const lines: string[] = [];
  if (opts.hemisphere || opts.ao) {
    lines.push(`// upness is the only geometric input: 1 facing +Y, 0 facing -Y.
let upness = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);`);
  }
  if (opts.hemisphere) {
    lines.push(`// A hemisphere rather than a constant: sky above, a darker ground colour
// below, blended by the normal. Cheaper than an irradiance probe and, at this
// scale, indistinguishable from one.
let hemi = mix(mat.groundColor, mat.ambientColor, upness) * mat.ambientIntensity;`);
  }
  if (opts.ao) {
    lines.push(`// One-tap ambient occlusion: a surface facing up sees more sky, so it is
// brighter. aoStrength = 0 disables it and leaves a flat ambient.
let ao = mix(1.0, upness, clamp(mat.aoStrength, 0.0, 1.0));`);
  }
  if (opts.rim) {
    lines.push(`// --- rim -------------------------------------------------------------------
// fresnelSchlick(N·V, rimColor) is a real reflectance: rimColor at normal
// incidence, 1 at grazing, nothing in between that the power law invents. It is
// a sheen rather than a light, so it is emitted rather than reflected and is
// unaffected by the light rig — set rimStrength to 0 for a pure BRDF.
let rim = fresnelSchlick(nDotV, mat.rimColor) * max(mat.rimStrength, 0.0) * ${opts.rimMask ?? '(1.0 - metallic)'};`);
  }
  return `\n// --- ambient and rim --------------------------------------------------------\n${lines.join('\n')}`;
}

/**
 * The image-based lighting term, as statements.
 *
 * **`environment` is one equirectangular (lat-long) 2D texture**, and it is
 * documented as *y-up, +X at u = 0.5 increasing with atan2(z, x)*, because a
 * caller has to author to that convention and apse has no way to discover it.
 * The alternative — a cube map — would be the better choice in every way, and it
 * is not offered because `texture_cube` cannot be created from an image file
 * without a compute or copy pass that is squarely outside what a material can
 * own. See the note in `pbr.ts` on migrating once apse has an asset path.
 *
 * The same texture serves both lookups, and the two mip knobs are the honest way
 * to say what that costs:
 *
 *   - **Diffuse irradiance** is read at `irradianceMip`. A 32x16 map with no mips
 *     wants 0; a full-resolution sky wants its top mip, which is where the
 *     low-frequency radiance the diffuse term needs actually lives.
 *   - **The specular reflection** is read at `environmentMip + sqrt(roughness) *
 *     ENV_MIP_RANGE`, along `reflect(-V, N)`. WGSL clamps a mip level to the
 *     texture's own maximum, so `ENV_MIP_RANGE` is a taste parameter and not a
 *     correctness requirement — a map with no mips just stays sharp.
 *
 * **No multi-scatter energy compensation.** The split-sum specular term is the
 * single-scattering one, so a rough metal loses the energy a real one bounces
 * around; the standard fix is a DFG lookup table and a second specular bounce.
 * That is deliberately not here, and a material with strong image-based
 * specular is better served by a prefiltered map plus an ambient occlusion
 * value than by this approximation.
 */
export function environmentBlock(specular: boolean): string {
  const lines = [
    `// Irradiance is the cosine-weighted average radiance over the hemisphere, so
// it already carries the 1/pi and the diffuse colour multiplies it directly.
let envDiffuse = irradianceAt(n);`,
  ];
  if (specular) {
    lines.push(`let envSpec = environmentSpecular(v, n, roughness);
let envBRDF = envBRDFApprox(f0, roughness, nDotV);`);
  }
  lines.push(`let env = envDiffuse * diffuseColor * ao${specular ? ' + envSpec * envBRDF' : ''};`);
  lines.push(`color = color + env * mat.environmentIntensity;`);
  return `\n// --- image-based lighting ---------------------------------------------------\n${lines.join('\n')}`;
}

/** How many mips of prefiltered environment a fully rough surface reaches for. */
export const ENV_MIP_RANGE = 4;

/** The `environment` texture slot. Mipped and repeating, because it is a sky. */
export const ENVIRONMENT_TEXTURE_SLOT = 'environment' as const;

/** The uniform knobs the image-based term reads. See {@link environmentBlock}. */
export const ENVIRONMENT_SLOTS = Object.freeze({
  /** Scales the whole IBL contribution. 0 leaves the material lit only by lights. */
  environmentIntensity: 'f32',
  /** Radians about +Y, applied before the equirect lookup. */
  environmentRotation: 'f32',
  /** Mip the diffuse irradiance is read at. */
  irradianceMip: 'f32',
  /** Base mip for the specular reflection; roughness adds up to ENV_MIP_RANGE. */
  environmentMip: 'f32',
} as const satisfies Record<string, SlotType>);

/**
 * The denominator guard both GGX forms use.
 *
 * The only requirement is that it not bind for any roughness the shader can
 * receive, and {@link BRDF_GGX} explains at length what happens when it does.
 * Both forms share it, because the anisotropic one has the same structure: at
 * the roughness floor its unguarded `v2` is about 1.7e-11.
 */
const GGX_GUARD = 1e-12;

// ---------------------------------------------------------------------------
// BRDF preludes
//
// Assembled from parts rather than emitted whole, so a Lambert material does not
// carry a GGX distribution function it never calls. The first two parts are
// always present: PI, a NaN-safe normalize, and Fresnel — which the rim needs
// even when there is no specular lobe at all.
// ---------------------------------------------------------------------------

/** `safeNormalize`, `pow5`, and Schlick Fresnel. Every lit material gets these. */
export const VECTOR_CORE = `
/** normalize() with a floor under the length. v + l is exactly zero at a
 *  silhouette, and normalize(vec3f(0)) is a NaN that then propagates through
 *  every term the fragment computes — a black speck, or a blanked draw. */
fn safeNormalize(v : vec3f) -> vec3f {
  return v * inverseSqrt(max(dot(v, v), 1e-12));
}

/** (1 - c)^5, shared by Schlick's Fresnel and the rim. Written as two squares
 *  because a pow() call in the middle of a BRDF is a transcendental for
 *  something that is a product. */
fn pow5(c : f32) -> f32 {
  let c2 = c * c;
  return c2 * c2 * c;
}

/** Schlick's approximation to Fresnel: f0 at normal incidence, 1 at grazing.
 *  Pass the actual N·V or V·H — passing 1 - N·V inverts the whole function. */
fn fresnelSchlick(cosTheta : f32, f0 : vec3f) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow5(1.0 - clamp(cosTheta, 0.0, 1.0));
}
`;

/** `PI` plus {@link VECTOR_CORE}. Every BRDF material gets these. */
export const BRDF_CORE = `
const PI : f32 = 3.14159265359;
${VECTOR_CORE}`;

/**
 * Isotropic Cook-Torrance: GGX distribution, height-correlated Smith visibility.
 *
 * Visibility is already divided by `4 * N·L * N·V`, so `D * V * F` *is* the
 * BRDF and the direct term is `(D * V * F + diffuse) * radiance * N·L`. Keeping
 * that division inside the helper is why the two halves cannot later be scaled
 * wrongly relative to each other.
 */
export const BRDF_GGX = `
/** A GGX lobe of zero roughness is a delta function: no rasteriser can integrate
 *  it, and visibilitySmith divides by a quantity that vanishes with it. 0.045 is
 *  the floor "Real-Time Rendering" uses, and it is a *roughness* floor rather
 *  than an epsilon — a mirror is 0.045 here, not 0. */
const MIN_ROUGHNESS : f32 = 0.045;

/** GGX / Trowbridge-Reitz: how tightly the microfacets around the halfway
 *  vector are aligned. Peaks at N·H = 1, and its integral over the hemisphere is
 *  1 at every roughness, which is what makes the metallic workflow conserve
 *  energy.
 *
 *  The ${GGX_GUARD} denominator guard is load-bearing in its *smallness*, and
 *  this is the one number in the module that was wrong before it was right. It
 *  exists only so a zero roughness cannot divide by zero, and MIN_ROUGHNESS
 *  already excludes that. At r = 0.045 the true peak is 1 / (PI r^4) ≈ 7.8e4
 *  and the unguarded denominator is 5.3e-11, so a guard of 1e-7 — a
 *  perfectly plausible-looking epsilon — clips the answer to 41. That is a
 *  polished metal with no highlight at all, and nothing anywhere reports it. The
 *  guard only has to be small enough not to bind. */
fn distributionGGX(nDotH : f32, roughness : f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, ${GGX_GUARD});
}

/** Smith's height-correlated visibility term. The 1e-6 here is a different
 *  guard from the one above and it *is* allowed to bind: it only reaches that
 *  size when N·L and N·V are both essentially zero, and the direct term
 *  multiplies by N·L afterwards, so the product still goes to zero. */
fn visibilitySmith(nDotV : f32, nDotL : f32, roughness : f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let gv = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
  let gl = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-6);
}
`;

/**
 * Anisotropic Cook-Torrance: Burley's aniso NDF and aniso Smith visibility.
 *
 * Two alphas — one along the tangent, one along the bitangent — are the whole
 * difference from the isotropic case. Everything else (the halfway vector, the
 * energy-conservation factor, the ambient, the rim, the light rig) is shared,
 * which is why this is a variant of the same generated loop rather than a second
 * material's worth of code.
 *
 * `at` and `ab` are **alpha**, not perceptual roughness: the caller squares
 * them, because a formulation that takes perceptual roughness and forms `at*ab`
 * internally produces a lobe one squaring wider than the isotropic path, and the
 * same `roughness` would then mean two different materials.
 */
export const BRDF_GGX_ANISO = `
/** GGX with an elliptical lobe, so a brushed surface stretches its highlight
 *  across the grain and a polished one stays round. */
fn distributionGGXAniso(nDotH : f32, tDotH : f32, bDotH : f32, at : f32, ab : f32) -> f32 {
  let a2 = at * ab;
  let v = vec3f(ab * tDotH, at * bDotH, a2 * nDotH);
  let v2 = dot(v, v);
  // Same guard, same reason: see distributionGGX. At the roughness floor the
  // unguarded v2 is ~1.7e-11, so a 1e-9 guard would clip the peak by seven
  // orders of magnitude.
  let w2 = a2 / max(v2, ${GGX_GUARD});
  return a2 * w2 * w2 * (1.0 / PI);
}

/** Smith's height-correlated visibility, anisotropic. */
fn visibilitySmithAniso(at : f32, ab : f32, tDotV : f32, bDotV : f32,
                        tDotL : f32, bDotL : f32, nDotV : f32, nDotL : f32) -> f32 {
  let lv = nDotL * length(vec3f(at * tDotV, ab * bDotV, nDotV));
  let ll = nDotV * length(vec3f(at * tDotL, ab * bDotL, nDotL));
  return 0.5 / max(lv + ll, 1e-6);
}
`;

/**
 * Oren-Nayar 1994, the s/t formulation.
 *
 * Lambert assumes a perfectly smooth surface, which is why plaster and chalk
 * look like plastic under a raking light: the real reflectance *increases* at
 * grazing angles, by up to a factor of four, and Lambert's flat N·L cannot
 * express that. The correction is a view-dependent term, which means it needs
 * L·V — the one dot product the isotropic loop does not otherwise compute, and
 * the reason this BRDF cannot be dropped into the GGX path unchanged.
 */
export const BRDF_OREN_NAYAR = `
/** Oren-Nayar diffuse reflectance. sigma2 is the squared surface roughness:
 *  0 reduces to Lambert exactly, 1 is a chalky retroreflective surface. */
fn orenNayar(nDotL : f32, nDotV : f32, lDotV : f32, sigma2 : f32) -> f32 {
  let s = lDotV - nDotL * nDotV;
  // t is the denominator of the retro-reflection lobe; flooring it keeps the
  // division finite at a perfectly grazing view.
  let t = select(1.0, max(nDotL, nDotV) + 1e-4, s > 0.0);
  let a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
  let b = 0.45 * sigma2 / (sigma2 + 0.09);
  // The unclamped form exceeds 1 at grazing incidence, which is the model's
  // known failure and not energy conservation.
  return min(nDotL / PI * (a + b * s / t), 1.0);
}
`;

/**
 * The image-based helpers, parameterised by the sampler the body must use.
 *
 * Emitted only when the material declares the `environment` texture, because
 * they reference it by name and a helper naming an undeclared binding is a WGSL
 * error. The sampler name is a *parameter* rather than a literal because a
 * material with two float 2D slots on the same sampler configuration has them
 * share one binding under the first slot's name — see `samplerNameFor`.
 */
export function environmentPrelude(sampler: string): string {
  return `
/** How many mips a fully rough surface reaches for. WGSL clamps the level to
 *  the texture's own maximum, so this is a taste parameter: too large and a
 *  smooth metal samples the top of the chain, too small and a matte one keeps a
 *  mirror reflection. 4 is the top of a 256-wide equirect chain. */
const ENV_MIP_RANGE : f32 = ${ENV_MIP_RANGE};

/** Karis 2013: an analytic fit to the split-sum DFG term, which is the piece of
 *  an environment BRDF lookup table that actually matters. A mat3 multiply and
 *  an exp2 per channel in a LUT become three ALU here, at the cost of an error
 *  that peaks exactly where a rim light lives. */
fn envBRDFApprox(f0 : vec3f, roughness : f32, nDotV : f32) -> vec3f {
  let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
  let r = vec4f(roughness) * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * nDotV)) * r.x + r.y;
  let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + vec3f(ab.y);
}

/** Equirectangular (lat-long) lookup. y is up, u runs anticlockwise from +X:
 *  atan2(z, x) over 2pi. A caller authoring the map has to match this
 *  convention — apse has no way to discover it, and getting it wrong mirrors
 *  the environment rather than breaking it visibly. */
fn equirectUv(d : vec3f) -> vec2f {
  let rot = mat.environmentRotation;
  let c = cos(rot);
  let s = sin(rot);
  let rotated = vec3f(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
  return vec2f(
    atan2(rotated.z, rotated.x) / (2.0 * PI) + 0.5,
    acos(clamp(rotated.y, -1.0, 1.0)) / PI
  );
}

fn irradianceAt(n : vec3f) -> vec3f {
  return textureSampleLevel(environment, ${sampler}, equirectUv(n), mat.irradianceMip).rgb;
}

fn environmentSpecular(v : vec3f, n : vec3f, roughness : f32) -> vec3f {
  let r = reflect(-v, n);
  let mip = mat.environmentMip + sqrt(clamp(roughness, 0.0, 1.0)) * ENV_MIP_RANGE;
  return textureSampleLevel(environment, ${sampler}, equirectUv(r), mip).rgb;
}
`;
}

/**
 * The `shadowDepth` texture declaration, or nothing when shadows are off.
 *
 * Kept here rather than in `pbr.ts` so a second material that wants shadows
 * declares the same slot, the same sampler configuration, and therefore shares
 * the same bind group layout.
 */
export const SHADOW_TEXTURE: Readonly<{ shadowDepth: { readonly kind: 'depth-2d'; readonly sampleType: 'depth'; readonly compare: true; readonly mipmapFilter: false } }> =
  Object.freeze({
    shadowDepth: Object.freeze({
      kind: 'depth-2d' as const,
      sampleType: 'depth' as const,
      compare: true as const,
      // A depth buffer has one level; asking for mipmapped sampling is a
      // validation error rather than a slower path.
      mipmapFilter: false as const,
    }),
  });

/** The slots the shadow term reads, on top of the per-light pair. */
export function shadowSlots(bias: number): SlotDefs {
  return {
    // The light's orthographic matrix, written per frame from the light's view.
    // Declared as a slot so it can be retuned with setSlot and no pipeline rebuild.
    lightViewProj: { type: 'mat4x4f', default: new Float32Array(16) },
    shadowBias: { type: 'f32', default: bias },
  };
}
