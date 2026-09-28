/**
 * The Oren-Nayar diffuse material — a non-metallic surface with a real rim.
 *
 * # Why this exists next to `pbrMaterial`
 *
 * A metallic-roughness BRDF answers "what does a physically measured surface do
 * when light hits it", and it is the wrong tool for a large part of what people
 * actually render: plaster, paper, chalk, unglazed ceramic, foliage, skin, the
 * entire non-metal half of a stylised scene. Those surfaces are *not* Lambert.
 * Lambert assumes a perfectly smooth surface, which is why a raking light across
 * plaster in a PBR renderer looks like plastic: the real reflectance of a rough
 * dielectric **rises** at grazing angles, by as much as a factor of four, and a
 * flat `N·L` cannot express it.
 *
 * Oren & Nayar (1994) is the cheapest model that does express it, and it is a
 * genuinely different BRDF rather than a rename: it needs `L·V`, the one dot
 * product the isotropic loop does not otherwise compute, and it has no specular
 * lobe at all — which is exactly why its rim has to do the energy conservation
 * the GGX lobe would otherwise do.
 *
 * # The rim is the missing specular lobe
 *
 * A dielectric reflects about 4% head-on and 100% at grazing. `pbrMaterial`
 * can ignore that because its GGX lobe already carries the Fresnel. This
 * material has no lobe, so the rim *is* that reflection, and the code says so:
 *
 * ```wgsl
 * let diff = diffuseColor * orenNayar(...) * (vec3f(1.0) - rim);
 * ```
 *
 * Energy conservation is not a nicety here. Adding a grazing-angle sheen on top
 * of a full-energy diffuse lobe makes a white surface brighter than 100% of the
 * light falling on it, and the error is worst exactly where the eye is most
 * sensitive to it. Because the rim is a Fresnel-Schlick *reflectance* rather
 * than a `pow(1 - N·V, k)` curve, subtracting it is meaningful: the two terms
 * are parts of one budget.
 *
 * # Not built here, on purpose
 *
 * - **No metalness.** That is `pbrMaterial`, and a two-material system where one
 *   is a strict subset of the other is how you get a scene with three
 *   near-identical shaders and no idea which is which.
 * - **No subsurface scattering.** A wrapped-diffuse term (`(N·L + w) / (1 + w)`)
 *   is two lines and looks like translucency; it is *not* translucency, and it
 *   reads as plastic the moment the light goes behind the surface. Real SSS
 *   needs a thickness map and a translucency colour, which is a material of its
 *   own rather than a flag.
 * - **No texture beyond the albedo.** A normal map, a height-derived normal, and
 *   an AO map are three more texture slots and three more varyings; they belong
 *   in the material that wants all three, not in a second copy of the light rig.
 */

import { fail } from '../core/error.ts';
import { STANDARD_LAYOUT } from '../geometry/layout.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import {
  BRDF_CORE,
  BRDF_OREN_NAYAR,
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

export interface DiffuseMaterialOptions extends MaterialOptions {
  /** Defaults to `'diffuse'`. */
  name?: string;
  /** Albedo, linear. Defaults to a light warm grey. */
  baseColor?: readonly [number, number, number];
  /**
   * Surface roughness, 0..1, as the Oren-Nayar sigma.
   *
   * 0 is exactly Lambert, so this is a continuous knob rather than a model
   * switch. 0.5 is chalk; 1 is a velvety retroreflective surface. It is the
   * *only* meaning `roughness` has on this material — there is no specular
   * lobe for it to blur.
   */
  roughness?: number;
  /** The light rig, as data. One to {@link MAX_DIRECTIONAL_LIGHTS} lights. */
  lights?: readonly DirectionalLight[];
  /** Shorthand for the first light's direction. Rejected alongside `lights`. */
  lightDir?: readonly [number, number, number];
  /** Shorthand for the first light's radiance. Rejected alongside `lights`. */
  lightColor?: readonly [number, number, number];
  /** Upper hemisphere ambient colour, linear. Defaults to a dim blue. */
  ambientColor?: readonly [number, number, number];
  /** Lower hemisphere ambient. Defaults to `ambientColor * 0.35`. */
  groundColor?: readonly [number, number, number];
  /** Scales the whole hemisphere ambient. Defaults to 1. */
  ambientIntensity?: number;
  /** How strongly the normal drives the ambient occlusion approximation. 0..1. */
  aoStrength?: number;
  /**
   * The rim's reflectance at normal incidence, and therefore its colour.
   * Defaults to white.
   */
  rimColor?: readonly [number, number, number];
  /**
   * Rim strength. 0 by default.
   *
   * The rim here is the dielectric specular this material does not lobe, so
   * `rimStrength: 1` with a white `rimColor` is *physically* about right for a
   * surface with F0 near 1 — which no real dielectric has. 0.04 to 0.1 is the
   * honest range for a coated surface; anything above that is art, and it is
   * cheap because the term it replaces is the one it subtracts from.
   */
  rimStrength?: number;
  /** Declare the `environment` texture slot and add image-based irradiance. */
  environment?: boolean;
  /** Scales the image-based contribution. Defaults to 1. */
  environmentIntensity?: number;
  /** Radians of rotation about +Y. Defaults to 0. */
  environmentRotation?: number;
  /** Mip the diffuse irradiance is read at. Defaults to 0. */
  irradianceMip?: number;
  /** Declare an `albedo` texture slot and multiply the base colour by it. */
  textured?: boolean;
  /** Declare a `shadowDepth` depth texture and shadow the first light. */
  shadows?: boolean;
  /** Depth bias for the shadow's penumbra. Defaults to 0.02. */
  shadowBias?: number;
  /** Alpha. 1 by default; only meaningful with `transparent`. */
  opacity?: number;
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
 * The per-light BRDF. `spec` is the zero vector and that is the point: this
 * material has no specular lobe, so the rim does that job instead, and the loop
 * still wants a `spec` to add.
 */
const DIFFUSE_BRDF: DirectBrdf = {
  // No halfway vector: the model is a function of N·L, N·V and L·V, so emitting
  // h / N·H / V·H would be three dead lets in every generated light block.
  halfway: false,
  specular: `    // No specular lobe. The rim is the same reflection folded into one
    // view-dependent term, so this is exactly zero rather than a small constant.
    let spec = vec3f(0.0);`,
  diffuse: `    // Oren-Nayar needs L·V, which the isotropic loop does not otherwise
    // compute — that extra dot product is the entire price of the model.
    let lDotV = max(dot(l, v), 0.0);
    let diff = diffuseColor * orenNayar(nDotL, nDotV, lDotV, sigma2) * (vec3f(1.0) - rim);`,
};

/**
 * The spec `diffuseMaterial` builds. Exported so the generated WGSL can be read
 * and asserted on with no device — the material *is* its spec.
 */
export function diffuseMaterialSpec(opts: DiffuseMaterialOptions = {}): MaterialSpec {
  const {
    name = 'diffuse',
    baseColor = [0.72, 0.7, 0.66],
    roughness = 0.4,
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
    irradianceMip = 0,
    textured = false,
    shadows = false,
    shadowBias = 0.02,
    opacity = 1,
    transparent = false,
    doubleSided = false,
    layout = STANDARD_LAYOUT,
    targetFormat = preferredTargetFormat(),
  } = opts;

  if (lights !== undefined && (lightDir !== undefined || lightColor !== undefined)) {
    fail('OPTION_UNKNOWN',
      `Material "${name}" was given both \`lights\` and the \`lightDir\`/\`lightColor\` shorthand.`, {
      why: 'The shorthand is exactly `lights: [{ direction: lightDir, color: lightColor }]`, so passing both names the first light twice and one of them is ignored — a light rig that keeps the light you replaced.',
      fix: 'Pass `lights` alone.',
    });
  }

  const resolved = resolveLights(lights ?? [{
    direction: lightDir ?? [0.5, 1, 0.3],
    color: lightColor ?? [3, 3, 3],
    castShadow: shadows,
  }], { shadows, material: name });

  const textures: Record<string, TextureSlotSpec> = {};
  if (textured) textures.albedo = { kind: '2d' };
  if (shadows) Object.assign(textures, SHADOW_TEXTURE);
  if (environment) {
    textures[ENVIRONMENT_TEXTURE_SLOT] = { kind: '2d', mipmapFilter: true, addressMode: 'repeat' };
  }
  const albedoSampler = samplerNameFor(textures, 'albedo');
  const envSampler = samplerNameFor(textures, ENVIRONMENT_TEXTURE_SLOT);

  const ground = groundColor ?? ([
    ambientColor[0] * 0.35, ambientColor[1] * 0.35, ambientColor[2] * 0.35,
  ] as const);

  const varyings: VaryingDefs = textured
    ? { worldPos: 'vec3f', normal: 'vec3f', uv: 'vec2f' }
    : { worldPos: 'vec3f', normal: 'vec3f' };

  const fragment = [
    diffuseFragmentHead(textured ? albedoSampler : null),
    // No rimMask: this material has no metalness, so there is nothing to mask
    // the sheen by — every surface is a dielectric and the rim is all of it.
    ambientBlock({ hemisphere: true, ao: aoStrength > 0, rim: rimStrength > 0 }),
    shadows ? shadowBlock() : '',
    `var direct = vec3f(0.0);
${directLightLoop(resolved, DIFFUSE_BRDF)}`,
    `var color = direct + diffuseColor * hemi * ao;`,
    // Diffuse only. There is no lobe to reflect the environment with, and
    // reflecting a low-resolution irradiance map through the rim would smear a
    // constant rather than light anything.
    environment ? environmentBlock(false) : '',
    rimStrength > 0 ? 'color = color + mat.rimColor * rim;' : '',
    `return vec4f(color, mat.opacity * frame.alpha);`,
  ].map((part) => part.trim()).filter((part) => part.length > 0).join('\n\n');

  return {
    name,
    layout,
    varyings,
    slots: {
      baseColor: { type: 'vec3f', default: baseColor },
      roughness: { type: 'f32', default: roughness },
      ...lightSlots(resolved),
      ambientColor: { type: 'vec3f', default: ambientColor },
      groundColor: { type: 'vec3f', default: ground },
      ambientIntensity: { type: 'f32', default: ambientIntensity },
      aoStrength: { type: 'f32', default: aoStrength },
      rimColor: { type: 'vec3f', default: rimColor },
      rimStrength: { type: 'f32', default: rimStrength },
      opacity: { type: 'f32', default: opacity },
      ...(environment ? {
        environmentIntensity: { type: ENVIRONMENT_SLOTS.environmentIntensity, default: environmentIntensity },
        environmentRotation: { type: ENVIRONMENT_SLOTS.environmentRotation, default: environmentRotation },
        irradianceMip: { type: ENVIRONMENT_SLOTS.irradianceMip, default: irradianceMip },
      } : {}),
      ...(shadows ? shadowSlots(shadowBias) : {}),
    },
    ...(Object.keys(textures).length > 0 ? { textures } : {}),
    prelude: [
      BRDF_CORE,
      BRDF_OREN_NAYAR,
      environment ? environmentPrelude(envSampler) : '',
    ].join('\n'),
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
out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;
out.normal = normalize(obj.normalMatrix * in.normal);
${textured ? 'out.uv = in.uv;' : ''}
`,
    fragment,
  };
}

function diffuseFragmentHead(albedoSampler: string | null): string {
  return `
// --- geometry ---------------------------------------------------------------
let n = normalize(in.normal);
let v = safeNormalize(frame.camPos - in.worldPos);
let nDotV = max(dot(n, v), 1e-4);

let diffuseColor = ${albedoSampler === null
    ? 'mat.baseColor'
    : `textureSample(albedo, ${albedoSampler}, in.uv).rgb * mat.baseColor`};

// sigma2 is what the Oren-Nayar constants are written in terms of. Clamped on
// the CPU's behalf as well as here: a negative sigma2 would make the A term
// exceed 1 and the model reflect more than it receives.
let sigma2 = clamp(mat.roughness, 0.0, 1.0) * clamp(mat.roughness, 0.0, 1.0);
`;
}

/**
 * Creates the Oren-Nayar material.
 *
 * Returns a promise because pipeline compilation must not block the main
 * thread — see the comment at the top of `material.ts`.
 */
export function diffuseMaterial(
  device: GPUDevice,
  opts: DiffuseMaterialOptions = {},
): Promise<Material> {
  const { frame, object, maxObjects } = opts;
  return Material.create(device, diffuseMaterialSpec(opts), { frame, object, maxObjects });
}
