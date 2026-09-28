/**
 * The anisotropic GGX material — a brushed or stretched specular lobe.
 *
 * # What anisotropy actually is
 *
 * An isotropic GGX lobe is a circle of microfacets, so its highlight is a point
 * that stays a point however the surface is scaled. Real surfaces are not
 * circular: brushed metal, vinyl, hair, satin fabric, and a polished floor all
 * have microfacets arranged along a direction, and their highlight is an *ellipse*
 * stretched across the grain. The physically correct way to say that is to give
 * the distribution **two** roughnesses, one along the surface tangent and one
 * along the bitangent, and to replace the isotropic normal distribution and
 * visibility terms with their anisotropic forms.
 *
 * That is the whole difference. The halfway vector, the metallic workflow, the
 * energy-conservation factor, the ambient, the rim, and the whole light rig are
 * shared with `pbrMaterial` — which is why this is a variant of the same
 * generated loop rather than a second material's worth of code.
 *
 * # It needs a tangent, and apse does not have one for you
 *
 * **No apse primitive produces a `tangent` attribute.** This material defaults
 * to {@link TANGENT_LAYOUT} (`position | normal | uv | tangent`, with the
 * tangent's `w` carrying the bitangent's sign) and refuses a layout without one,
 * naming the attribute rather than letting a WGSL error point into generated
 * code. Meshes for it have to come with tangents.
 *
 * The frame is rebuilt with a Gram-Schmidt projection rather than trusting the
 * interpolated tangent, and that is not defensive programming: an interpolated
 * tangent is *not* guaranteed orthogonal to an interpolated normal, and a
 * non-orthonormal frame makes the highlight visibly change shape with the
 * triangle's tessellation — the same mesh shaded at a different density looks
 * like a different material.
 *
 * # What is deliberately absent
 *
 * - **No alpha-driven anisotropy** (hair, where the shift follows the tangent
 *   map). It needs a tangent *map* and a per-strand mask, which is a different
 *   material.
 * - **No multi-scatter compensation.** The single-scattering GGX loses energy at
 *   high roughness; a proper aniso fix is a second lobe and a fitted DFG term.
 *   At `anisotropy` values below about 0.8 the loss is below what 8-bit output
 *   can show.
 */

import { fail } from '../core/error.ts';
import { TANGENT_LAYOUT } from '../geometry/layout.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import {
  BRDF_CORE,
  BRDF_GGX,
  BRDF_GGX_ANISO,
  ENVIRONMENT_SLOTS,
  ENVIRONMENT_TEXTURE_SLOT,
  SHADOW_TEXTURE,
  ambientBlock,
  directLightLoop,
  environmentBlock,
  environmentPrelude,
  lightSlots,
  resolveLights,
  shadowBlock,
  shadowSlots,
} from './lighting.ts';
import type { DirectionalLight, DirectBrdf } from './lighting.ts';
import { Material } from './material.ts';
import type { MaterialOptions } from './material.ts';
import { preferredTargetFormat, samplerNameFor } from './scaffold.ts';
import type { MaterialSpec, VaryingDefs } from './scaffold.ts';
import type { TextureSlotSpec } from './texture-slot.ts';

export interface AnisotropicMaterialOptions extends MaterialOptions {
  /** Defaults to `'anisotropic'`. */
  name?: string;
  /** Albedo, linear. The specular reflectance at `metallic: 1`. */
  baseColor?: readonly [number, number, number];
  /** 0 = dielectric, 1 = metal. Defaults to 1, because that is what is brushed. */
  metallic?: number;
  /**
   * The roughness **across** the grain, 0..1. The tangent direction gets
   * `roughness * (1 + anisotropy)`, so the highlight always stretches along the
   * tangent and never across it.
   */
  roughness?: number;
  /**
   * -1..1. 0 is isotropic, ±1 is the limit.
   *
   * Positive stretches the highlight along the tangent; negative stretches it
   * along the bitangent, which is the same surface brushed the other way. The
   * two are not symmetric in appearance even though they cost the same, which is
   * why it is a signed value rather than a magnitude.
   */
  anisotropy?: number;
  /** The light rig, as data. One to {@link MAX_DIRECTIONAL_LIGHTS} lights. */
  lights?: readonly DirectionalLight[];
  /** Shorthand for the first light's direction. Rejected alongside `lights`. */
  lightDir?: readonly [number, number, number];
  /** Shorthand for the first light's radiance. Rejected alongside `lights`. */
  lightColor?: readonly [number, number, number];
  /** Upper hemisphere ambient colour, linear. */
  ambientColor?: readonly [number, number, number];
  /** Lower hemisphere ambient. Defaults to `ambientColor * 0.35`. */
  groundColor?: readonly [number, number, number];
  /** Scales the whole hemisphere ambient. Defaults to 1. */
  ambientIntensity?: number;
  /** How strongly the normal drives the ambient occlusion approximation. 0..1. */
  aoStrength?: number;
  /** The rim's reflectance at normal incidence, and therefore its colour. */
  rimColor?: readonly [number, number, number];
  /** Rim strength. 0 by default; see `pbrMaterial`. */
  rimStrength?: number;
  /** Declare the `environment` texture slot and add image-based lighting. */
  environment?: boolean;
  /** Scales the image-based contribution. Defaults to 1. */
  environmentIntensity?: number;
  /** Radians of rotation about +Y. Defaults to 0. */
  environmentRotation?: number;
  /** Also reflect the environment, roughness-blurred. See `pbrMaterial`. */
  environmentSpecular?: boolean;
  /** Mip the diffuse irradiance is read at. Defaults to 0. */
  irradianceMip?: number;
  /** Base mip for the specular reflection. Defaults to 0. */
  environmentMip?: number;
  /** Declare a `shadowDepth` depth texture and shadow the first light. */
  shadows?: boolean;
  /** Depth bias for the shadow's penumbra. Defaults to 0.02. */
  shadowBias?: number;
  /** Disable back-face culling. Defaults to false. */
  doubleSided?: boolean;
  /**
   * Must declare a `tangent` attribute. Defaults to {@link TANGENT_LAYOUT}.
   */
  layout?: VertexLayout;
  /** Defaults to the canvas preferred format. See `pbrMaterial`. */
  targetFormat?: GPUTextureFormat;
}

/**
 * The per-light BRDF. The `setup` block is the tangent frame, emitted once
 * because it does not depend on the light.
 */
const ANISO_BRDF: DirectBrdf = {
  setup: `
// --- tangent frame ----------------------------------------------------------
// Gram-Schmidt, and not optional. An interpolated tangent is not guaranteed
// orthogonal to an interpolated normal, and a frame that is not orthonormal makes
// the highlight's shape depend on the triangle's tessellation — the same mesh at
// a different density reads as a different material.
let tRaw = in.tangent;
let t = safeNormalize(tRaw.xyz - n * dot(n, tRaw.xyz));
// The w component is the bitangent's sign: a tangent frame can be left- or
// right-handed and the shader has no other way to know which the mesh meant.
let b = cross(n, t) * tRaw.w;

// Two roughnesses from one value, then squared into alpha. The squaring is not
// cosmetic: distributionGGXAniso forms at * ab internally, so handing it
// perceptual roughness produces a lobe one squaring wider than the isotropic
// path, and the same \`roughness\` would then mean two different materials in
// two files of the same library. Both are floored at MIN_ROUGHNESS first,
// because a zero roughness is a delta function.
let roughT = clamp(mat.roughness * (1.0 + mat.anisotropy), MIN_ROUGHNESS, 1.0);
let roughB = clamp(mat.roughness * (1.0 - mat.anisotropy), MIN_ROUGHNESS, 1.0);
let at = roughT * roughT;
let ab = roughB * roughB;
let tDotV = dot(t, v);
let bDotV = dot(b, v);`,
  specular: `    let spec = distributionGGXAniso(nDotH, dot(t, h), dot(b, h), at, ab)
      * visibilitySmithAniso(at, ab, tDotV, bDotV, dot(t, l), dot(b, l), nDotV, nDotL)
      * fresnelSchlick(vDotH, f0);`,
  // Identical to the isotropic case, and deliberately so: energy conservation is
  // a property of the material, not of the lobe's shape.
  diffuse: `    let diff = (vec3f(1.0) - fresnelSchlick(nDotV, f0)) * diffuseColor / PI;`,
};

/**
 * The spec `anisotropicMaterial` builds. Exported so the generated WGSL can be
 * read and asserted on with no device.
 */
export function anisotropicMaterialSpec(opts: AnisotropicMaterialOptions = {}): MaterialSpec {
  const {
    name = 'anisotropic',
    baseColor = [0.9, 0.9, 0.92],
    metallic = 1,
    roughness = 0.35,
    anisotropy = 0.7,
    lights,
    lightDir,
    lightColor,
    ambientColor = [0.15, 0.17, 0.22],
    groundColor,
    ambientIntensity = 1,
    aoStrength = 1,
    rimColor = [1, 1, 1],
    rimStrength = 0,
    environment = false,
    environmentIntensity = 1,
    environmentRotation = 0,
    environmentSpecular = false,
    irradianceMip = 0,
    environmentMip = 0,
    shadows = false,
    shadowBias = 0.02,
    doubleSided = false,
    layout = TANGENT_LAYOUT,
    targetFormat = preferredTargetFormat(),
  } = opts;

  if (lights !== undefined && (lightDir !== undefined || lightColor !== undefined)) {
    fail('OPTION_UNKNOWN',
      `Material "${name}" was given both \`lights\` and the \`lightDir\`/\`lightColor\` shorthand.`, {
      why: 'The shorthand is exactly `lights: [{ direction: lightDir, color: lightColor }]`, so passing both names the first light twice and one of them is ignored.',
      fix: 'Pass `lights` alone.',
    });
  }

  assertRange(name, 'anisotropy', anisotropy, -1, 1);
  assertRange(name, 'metallic', metallic, 0, 1);
  assertRange(name, 'roughness', roughness, 0, 1);

  // Named here rather than letting Tint report it: the generated vertex body
  // reads `in.tangent`, and a layout without it produces a WGSL error whose line
  // number points into generated code the author never wrote.
  if (layout.attribute('tangent') === undefined) {
    fail('ATTRIBUTE_MISSING',
      `Anisotropic material "${name}" was given a layout with no "tangent" attribute.`, {
      why: 'The anisotropy lives in a tangent frame, so the generated vertex stage writes the tangent to a varying and the fragment stage rebuilds the frame from it. Without the attribute there is no frame, and a direction picked arbitrarily in the shader would give a highlight that rotates as the mesh turns — which looks like a bug in the lighting rather than a missing attribute.',
      fix: 'Build the layout with TANGENT_LAYOUT (position, normal, uv, tangent as float32x4) or declare `tangent: "float32x4"` yourself. Note that no apse primitive generates tangents; the mesh has to supply them.',
    });
  }

  const resolved = resolveLights(lights ?? [{
    direction: lightDir ?? [0.5, 1, 0.3],
    color: lightColor ?? [3, 3, 3],
    castShadow: shadows,
  }], { shadows, material: name });

  const textures: Record<string, TextureSlotSpec> = {};
  if (shadows) Object.assign(textures, SHADOW_TEXTURE);
  if (environment) {
    textures[ENVIRONMENT_TEXTURE_SLOT] = { kind: '2d', mipmapFilter: true, addressMode: 'repeat' };
  }
  const envSampler = samplerNameFor(textures, ENVIRONMENT_TEXTURE_SLOT);

  const ground = groundColor ?? ([
    ambientColor[0] * 0.35, ambientColor[1] * 0.35, ambientColor[2] * 0.35,
  ] as const);

  const varyings: VaryingDefs = { worldPos: 'vec3f', normal: 'vec3f', tangent: 'vec4f' };

  const fragment = [
    `
// --- geometry ---------------------------------------------------------------
let n = normalize(in.normal);
let v = safeNormalize(frame.camPos - in.worldPos);
let nDotV = max(dot(n, v), 1e-4);

let metallic = clamp(mat.metallic, 0.0, 1.0);
let roughness = clamp(mat.roughness, MIN_ROUGHNESS, 1.0);
let f0 = mix(vec3f(0.04), mat.baseColor, metallic);
let diffuseColor = mat.baseColor * (1.0 - metallic);`,
    ambientBlock({ hemisphere: true, ao: aoStrength > 0, rim: rimStrength > 0 }),
    shadows ? shadowBlock() : '',
    `var direct = vec3f(0.0);
${directLightLoop(resolved, ANISO_BRDF)}`,
    `var color = direct + diffuseColor * hemi * ao;`,
    environment ? environmentBlock(environmentSpecular) : '',
    rimStrength > 0 ? 'color = color + mat.rimColor * rim;' : '',
    `// Linear radiance out; the present pass owns exposure and the transfer
// function. See the note at the top of pbr.ts.
return vec4f(color, 1.0);`,
  ].map((part) => part.trim()).filter((part) => part.length > 0).join('\n\n');

  return {
    name,
    layout,
    varyings,
    slots: {
      baseColor: { type: 'vec3f', default: baseColor },
      metallic: { type: 'f32', default: metallic },
      roughness: { type: 'f32', default: roughness },
      anisotropy: { type: 'f32', default: anisotropy },
      ...lightSlots(resolved),
      ambientColor: { type: 'vec3f', default: ambientColor },
      groundColor: { type: 'vec3f', default: ground },
      ambientIntensity: { type: 'f32', default: ambientIntensity },
      aoStrength: { type: 'f32', default: aoStrength },
      rimColor: { type: 'vec3f', default: rimColor },
      rimStrength: { type: 'f32', default: rimStrength },
      ...(environment ? {
        environmentIntensity: { type: ENVIRONMENT_SLOTS.environmentIntensity, default: environmentIntensity },
        environmentRotation: { type: ENVIRONMENT_SLOTS.environmentRotation, default: environmentRotation },
        irradianceMip: { type: ENVIRONMENT_SLOTS.irradianceMip, default: irradianceMip },
        environmentMip: { type: ENVIRONMENT_SLOTS.environmentMip, default: environmentMip },
      } : {}),
      ...(shadows ? shadowSlots(shadowBias) : {}),
    },
    ...(Object.keys(textures).length > 0 ? { textures } : {}),
    prelude: [
      BRDF_CORE,
      BRDF_GGX,
      BRDF_GGX_ANISO,
      environment ? environmentPrelude(envSampler) : '',
    ].join('\n'),
    phase: 'opaque',
    topology: 'triangle-list',
    cull: doubleSided ? 'none' : 'back',
    depth: { write: true, compare: 'less' },
    blend: null,
    targets: [{ format: targetFormat }],
    sampleCount: 1,
    vertex: `
out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;
out.normal = normalize(obj.normalMatrix * in.normal);
// The tangent is transformed by the model matrix, not by the normal matrix: it
// is a direction in the surface, and the inverse-transpose is the right
// transform for a *normal*, not for a tangent lying in the tangent plane. A
// uniform scale makes the two identical, which is exactly why using the wrong
// one is invisible until the object is scaled unevenly.
out.tangent = vec4f((obj.model * vec4f(in.tangent.xyz, 0.0)).xyz, in.tangent.w);
`,
    fragment,
  };
}

function assertRange(material: string, field: string, value: number, lo: number, hi: number): void {
  if (Number.isFinite(value) && value >= lo && value <= hi) return;
  fail('OPTION_UNKNOWN',
    `Material "${material}" was given a ${field} of ${String(value)}, outside ${lo}..${hi}.`, {
    why: 'This value is consumed by a clamp in the shader as well, so an out-of-range number is not a crash — it is a silently different material from the one that was asked for. A NaN is worse than that: it propagates through every fragment and blanks the draw.',
    fix: `Pass a number between ${lo} and ${hi}. The endpoints are the limits of the model, not approximations of them.`,
    detail: { kind: 'numeric', field, value, min: lo, max: hi },
  });
}

/**
 * Creates the anisotropic GGX material.
 *
 * Returns a promise because pipeline compilation must not block the main
 * thread — see the comment at the top of `material.ts`.
 */
export function anisotropicMaterial(
  device: GPUDevice,
  opts: AnisotropicMaterialOptions = {},
): Promise<Material> {
  const { frame, object, maxObjects } = opts;
  return Material.create(device, anisotropicMaterialSpec(opts), { frame, object, maxObjects });
}
