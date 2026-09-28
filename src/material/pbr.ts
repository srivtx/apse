/**
 * The Cook-Torrance GGX material.
 *
 * This is the module's proof that the scaffold scales. It is a real
 * physically-based shader — GGX distribution, Smith height-correlated
 * visibility, Schlick Fresnel, a metallic workflow with energy conservation, a
 * hemisphere ambient, a Fresnel-Schlick rim, optional image-based lighting, and a
 * 3x3 PCF shadow — and every single binding, uniform field, and varyings struct
 * it needs is *declared as data* and generated. There is not one `@group`,
 * `@binding`, or `struct` in this file.
 *
 * # The light count is data
 *
 * Read the `fragment` body below and notice that it contains no number of
 * lights. It does not contain one because there is no place to put it: the loop
 * is emitted by {@link directLightLoop} from the same `lights` array that
 * {@link lightSlots} turns into uniform fields, and the WGSL, the struct, the
 * byte offsets, and the JS packer are all generated from that one array. Passing
 * two lights instead of one changes three generated things at once and no
 * hand-written text — which is the difference between a material and a shader.
 *
 * # What the BRDF owes the rest of the pipeline
 *
 * Every value below is **linear radiance**, not a colour in 0..1. Lighting
 * constants (`lightColor: [3, 3, 3]`, `baseColor: [0.6, 0.6, 0.6]`) only make
 * sense in linear light, and the chain that consumes them is: this material
 * writes linear radiance → an optional HDR intermediate → the present pass's
 * tone map and sRGB encode → the display. Nothing here applies a transfer
 * function, applies `frame.exposure`, or clamps, because a material that did any
 * of the three would be double-processed the moment it was drawn into a
 * post-processing chain. The absence is deliberate and it is the reason the
 * unlit `basicMaterial` looks too dark and this one does not, once the present
 * pass is on.
 *
 * Read the `fragment` string as the tutorial it is meant to be: it is the
 * most-read shader in the library, because it is the shortest complete example of
 * what a material author can and cannot write.
 */

import { fail } from '../core/error.ts';
import { STANDARD_LAYOUT } from '../geometry/layout.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import {
  BRDF_CORE,
  BRDF_GGX,
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

export interface PbrMaterialOptions extends MaterialOptions {
  /** Defaults to `'pbr'`. */
  name?: string;
  /** Albedo, linear. Ignored by `metallic: 1`, where albedo *is* the specular
   *  reflectance. Defaults to mid grey. */
  baseColor?: readonly [number, number, number];
  /** 0 = dielectric, 1 = metal. Defaults to 0. */
  metallic?: number;
  /** Perceptual roughness. Clamped to a floor of 0.045 in the shader. Defaults to 0.5. */
  roughness?: number;
  /**
   * The light rig, as data. One to {@link MAX_DIRECTIONAL_LIGHTS} lights; the
   * default is the single key light `lightDir`/`lightColor` describe.
   *
   * The uniform fields, the shading statements, and the documentation all come
   * from this array, so adding a light is a change to it and nothing else.
   */
  lights?: readonly DirectionalLight[];
  /**
   * Direction towards the first light, world space. A shorthand for
   * `lights: [{ direction: lightDir, color: lightColor }]`.
   *
   * Rejected alongside `lights` rather than ignored: two ways to say the same
   * thing, one of which quietly loses, is how a scene ends up lit by a light
   * the author replaced.
   */
  lightDir?: readonly [number, number, number];
  /** Radiance of the first light. The shorthand's other half. */
  lightColor?: readonly [number, number, number];
  /** Upper hemisphere ambient colour, linear. Defaults to a dim blue. */
  ambientColor?: readonly [number, number, number];
  /**
   * Lower hemisphere ambient colour. Defaults to `ambientColor * 0.35`, which
   * is what the material used before the ground colour was a parameter.
   */
  groundColor?: readonly [number, number, number];
  /** Scales the whole hemisphere ambient. Defaults to 1; 0 removes it. */
  ambientIntensity?: number;
  /** How strongly the normal drives the ambient occlusion approximation. 0..1. */
  aoStrength?: number;
  /**
   * The rim's reflectance at normal incidence, and therefore its colour.
   * Defaults to white, which gives a neutral sheen.
   */
  rimColor?: readonly [number, number, number];
  /**
   * Rim strength. **0 by default**, so a PBR material is a PBR material until
   * somebody asks for a sheen — the rim is the one term here that is an
   * artistic addition rather than a physical consequence of the surface.
   */
  rimStrength?: number;
  /**
   * Declare the `environment` texture slot and add image-based lighting.
   *
   * The texture is **one equirectangular (lat-long) 2D image**, y-up, with u
   * running anticlockwise from +X — `u = atan2(z, x) / 2pi + 0.5`,
   * `v = acos(y) / pi` — because that is the convention {@link environmentBlock}
   * looks up with. A 32x16 blurred sky is enough for the diffuse term; a
   * prefiltered, mipped environment is needed for the specular one.
   *
   * It is a 2D image and not a cube because a `texture_cube` cannot be created
   * from an image file without a copy or compute pass, which is outside what a
   * material can own. Once apse has an asset path, the equirect lookup here is
   * the only function that has to change.
   *
   * A material with this on cannot be drawn until a view is assigned with
   * `setTexture('environment', view)`.
   */
  environment?: boolean;
  /** Scales the whole image-based contribution. Defaults to 1. */
  environmentIntensity?: number;
  /** Radians of rotation about +Y applied before the lookup. Defaults to 0. */
  environmentRotation?: number;
  /**
   * Also reflect the environment, roughness-blurred.
   *
   * **Off by default**, because the supplied map is normally an irradiance map:
   * reflecting it would give a mirror surface a blurry grey constant rather than
   * the room. Turn it on only with a prefiltered, mipped environment.
   *
   * The term is the single-scattering split-sum one, with no multi-scatter
   * energy compensation, so a rough metal under strong IBL loses some of the
   * energy a real one bounces around. See the note in `lighting.ts`.
   */
  environmentSpecular?: boolean;
  /** Mip the diffuse irradiance is read at. Defaults to 0. */
  irradianceMip?: number;
  /** Base mip for the specular reflection. Defaults to 0. */
  environmentMip?: number;
  /**
   * Declare an `albedo` texture slot and multiply the base colour by it — linear
   * by convention, so an sRGB-tagged source must be decoded by the sampler (an
   * `*-srgb` view format) rather than by this shader.
   */
  textured?: boolean;
  /**
   * Declare a `shadowDepth` depth texture and multiply the shadowed light's
   * direct term by a 3x3 PCF visibility estimate.
   *
   * Off by default, because a shadow map is a whole other pass and a material
   * that declares the slot but has no map bound cannot be drawn. When this is on
   * you must also write `lightViewProj` — an orthographic light matrix matching
   * the shadow map you rendered — and assign the depth texture. A shadowed
   * material is `lights: [{ …, castShadow: true }, …]`.
   */
  shadows?: boolean;
  /** Tint for the shadow's penumbra. 0 = fully hard. Defaults to 0.02. */
  shadowBias?: number;
  /** Disable back-face culling. Defaults to false. */
  doubleSided?: boolean;
  /** Defaults to {@link STANDARD_LAYOUT}. */
  layout?: VertexLayout;
  /**
   * Colour target format. Defaults to `navigator.gpu.getPreferredCanvasFormat()`.
   *
   * WebGPU bakes the attachment format into a pipeline, so a material is
   * permanently bound to the format it was compiled for. The preferred canvas
   * format is `bgra8unorm` on desktop and `rgba8unorm` on Android, so a
   * hardcoded default is wrong on one of them, and the failure is a Dawn
   * validation error at `setPipeline` that names a format instead of the
   * mistake.
   *
   * Pass this explicitly when drawing into an offscreen target — an HDR
   * `rgba16float` post-processing chain, for instance — since the canvas format
   * is the wrong answer there. **A lit material belongs in an `rgba16float`
   * intermediate**: the canvas format is a `unorm` target, so a lit value above
   * 1.0 is clipped at the write, long before any tone curve could compress it.
   */
  targetFormat?: GPUTextureFormat;
}

/**
 * Joins the generated fragment sections.
 *
 * Trimming and dropping the empties is what keeps a program with one term
 * switched off from carrying a run of three blank lines through the middle —
 * generated code is read by people, and its shape is part of what it teaches.
 */
const FRAGMENT_JOIN = '\n\n';
/** The per-light BRDF, in the shape {@link directLightLoop} splices it into. */
const PBR_BRDF: DirectBrdf = {
  specular: `    let spec = distributionGGX(nDotH, roughness) * visibilitySmith(nDotV, nDotL, roughness) * fresnelSchlick(vDotH, f0);`,
  // Energy conservation. Whatever the specular lobe reflects is not also
  // available as diffuse; without this a rough metal glows white, because its
  // albedo is its specular reflectance and there is no diffuse left to compete.
  diffuse: `    let diff = (vec3f(1.0) - fresnelSchlick(nDotV, f0)) * diffuseColor / PI;`,
};

/** The vertex stage. World-space shading, so nothing is transformed twice. */
const PBR_VERTEX = `
out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;

// The normal matrix is a mat3x3f, and in WGSL a matCxR multiplies a vecC — so
// this is a vec3f in and a vec3f out. (GLSL's mat3 * vec4, which ignores w, is
// not valid WGSL and is a common porting slip.) The matrix is the
// inverse-transpose of the model's upper 3x3, packed as three 16-byte-aligned
// columns, which is why a non-uniform scale still produces perpendicular normals.
out.normal = normalize(obj.normalMatrix * in.normal);
`;

/**
 * The spec `pbrMaterial` builds. Exported so it can be read and tested without a
 * device — the material *is* its spec.
 */
export function pbrMaterialSpec(opts: PbrMaterialOptions = {}): MaterialSpec {
  const {
    name = 'pbr',
    baseColor = [0.6, 0.6, 0.6],
    metallic = 0,
    roughness = 0.5,
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
    textured = false,
    shadows = false,
    shadowBias = 0.02,
    doubleSided = false,
    layout = STANDARD_LAYOUT,
    targetFormat = preferredTargetFormat(),
  } = opts;

  if (lights !== undefined && (lightDir !== undefined || lightColor !== undefined)) {
    fail('OPTION_UNKNOWN',
      `Material "${name}" was given both \`lights\` and the \`lightDir\`/\`lightColor\` shorthand.`, {
      why: 'The shorthand is exactly `lights: [{ direction: lightDir, color: lightColor }]`, so passing both names the first light twice. One of them would be ignored, and a light rig that quietly keeps the light you replaced is a bug you only see in the render.',
      fix: `Pass \`lights\` alone: \`lights: [{ direction: ${fmt(lightDir ?? [0.5, 1, 0.3])}, color: ${fmt(lightColor ?? [3, 3, 3])} }, { … }]\`.`,
    });
  }

  const resolved = resolveLights(lights ?? [{
    direction: lightDir ?? [0.5, 1, 0.3],
    color: lightColor ?? [3, 3, 3],
    castShadow: shadows,
  }], { shadows, material: name });

  // The texture set has to exist before the body is generated, because the body
  // names a sampler and a shared sampler has one name for the whole group.
  const textures: Record<string, TextureSlotSpec> = {};
  if (textured) textures.albedo = { kind: '2d' };
  if (shadows) Object.assign(textures, SHADOW_TEXTURE);
  if (environment) {
    textures[ENVIRONMENT_TEXTURE_SLOT] = {
      kind: '2d',
      // Mipped: the specular term reads a roughness-chosen level. A map with no
      // mips simply clamps to level 0.
      mipmapFilter: true,
      // Repeat, not clamp-to-edge: u wraps a full turn around an equirect map,
      // and clamping would put a hard seam down the back of everything.
      addressMode: 'repeat',
    };
  }
  const albedoSampler = samplerNameFor(textures, 'albedo');
  const envSampler = samplerNameFor(textures, ENVIRONMENT_TEXTURE_SLOT);

  const ground = groundColor ?? ([
    ambientColor[0] * 0.35, ambientColor[1] * 0.35, ambientColor[2] * 0.35,
  ] as const);

  const slots = {
    baseColor: { type: 'vec3f' as const, default: baseColor },
    metallic: { type: 'f32' as const, default: metallic },
    roughness: { type: 'f32' as const, default: roughness },
    ...lightSlots(resolved),
    ambientColor: { type: 'vec3f' as const, default: ambientColor },
    groundColor: { type: 'vec3f' as const, default: ground },
    ambientIntensity: { type: 'f32' as const, default: ambientIntensity },
    aoStrength: { type: 'f32' as const, default: aoStrength },
    rimColor: { type: 'vec3f' as const, default: rimColor },
    rimStrength: { type: 'f32' as const, default: rimStrength },
    ...(environment ? {
      environmentIntensity: { type: ENVIRONMENT_SLOTS.environmentIntensity, default: environmentIntensity },
      environmentRotation: { type: ENVIRONMENT_SLOTS.environmentRotation, default: environmentRotation },
      irradianceMip: { type: ENVIRONMENT_SLOTS.irradianceMip, default: irradianceMip },
      environmentMip: { type: ENVIRONMENT_SLOTS.environmentMip, default: environmentMip },
    } : {}),
    ...(shadows ? shadowSlots(shadowBias) : {}),
  };

  const varyings: VaryingDefs = textured
    ? { worldPos: 'vec3f', normal: 'vec3f', uv: 'vec2f' }
    : { worldPos: 'vec3f', normal: 'vec3f' };

  const fragment = [
    pbrFragmentHead(textured ? albedoSampler : null),
    ambientBlock({ hemisphere: true, ao: aoStrength > 0, rim: rimStrength > 0 }),
    shadows ? shadowBlock() : '',
    `var direct = vec3f(0.0);
${directLightLoop(resolved, PBR_BRDF)}`,
    `var color = direct + diffuseColor * hemi * ao;`,
    environment ? environmentBlock(environmentSpecular) : '',
    rimStrength > 0 ? PBR_RIM_APPLY : '',
    PBR_FRAGMENT_TAIL,
  ].map((part) => part.trim()).filter((part) => part.length > 0).join(FRAGMENT_JOIN);

  return {
    name,
    layout,
    // Varyings in declaration order, so worldPos is @location(0) and normal is
    // @location(1). `clip` is the builtin and consumes none.
    varyings,
    slots,
    ...(Object.keys(textures).length > 0 ? { textures } : {}),
    prelude: [
      BRDF_CORE,
      BRDF_GGX,
      environment ? environmentPrelude(envSampler) : '',
    ].join('\n'),
    phase: 'opaque',
    topology: 'triangle-list',
    cull: doubleSided ? 'none' : 'back',
    depth: { write: true, compare: 'less' },
    blend: null,
    targets: [{ format: targetFormat }],
    sampleCount: 1,
    vertex: textured
      ? `${PBR_VERTEX}\nout.uv = in.uv;`
      : PBR_VERTEX,
    fragment,
  };
}

/**
 * The head of the fragment stage: everything that does not depend on which lights
 * were declared. Split out so the light loop reads as what it is.
 */
function pbrFragmentHead(albedoSampler: string | null): string {
  return `
// --- geometry ---------------------------------------------------------------
// Shading in world space, so the view and light vectors are both plain
// differences of world-space positions and no normal has to be transformed a
// second time.
let n = normalize(in.normal);
let v = safeNormalize(frame.camPos - in.worldPos);

// Clamped rather than trusted: a zero-length v at a silhouette would otherwise
// produce a NaN that propagates through the whole term.
let nDotV = max(dot(n, v), 1e-4);

let metallic = clamp(mat.metallic, 0.0, 1.0);
let roughness = clamp(mat.roughness, MIN_ROUGHNESS, 1.0);

// --- material ---------------------------------------------------------------
// Metallic workflow: a metal *is* its specular reflectance, a dielectric sits
// on a 0.04 base — the standard F0 for an organic surface at normal incidence.
// 0.04 is a reflectance, not a colour, and it is the one number in this shader
// that came from a measurement rather than from taste.
let base = ${albedoSampler === null
    ? 'mat.baseColor'
    : `textureSample(albedo, ${albedoSampler}, in.uv).rgb * mat.baseColor`};
let f0 = mix(vec3f(0.04), base, metallic);
let diffuseColor = base * (1.0 - metallic);
`;
}

/**
 * The tail: nothing here is a physical consequence of the surface, so it is the
 * part worth reading for "what have I forgotten to make tone-referred".
 */
const PBR_FRAGMENT_TAIL = `
// No tone map and no sRGB encode here, and no frame.exposure either. This stage
// writes **linear radiance** into whatever target it was given; the present
// pass owns exposure, the curve, and the transfer function, so a material in an
// offscreen pass is not processed twice. Encoding here as well is the single
// most common way to make a correct render look wrong on one display and right
// on another.
return vec4f(color, 1.0);`;

/**
 * The rim, added.
 *
 * **Additive here, and that is a consequence of the GGX lobe rather than an
 * inconsistency.** The physical Fresnel the rim approximates is already inside
 * the specular term, so the rim is a *sheen on top of* a correct BRDF and must
 * not also subtract from the diffuse. `diffuseMaterial`, which has no specular
 * lobe at all, does the opposite — there the rim *is* the missing specular, and
 * the diffuse is attenuated by it.
 */
const PBR_RIM_APPLY = 'color = color + mat.rimColor * rim;';

/**
 * Creates the GGX material.
 *
 * Returns a promise because pipeline compilation must not block the main
 * thread — see the comment at the top of `material.ts`.
 */
export function pbrMaterial(
  device: GPUDevice,
  opts: PbrMaterialOptions = {},
): Promise<Material> {
  const { frame, object, maxObjects } = opts;
  return Material.create(device, pbrMaterialSpec(opts), { frame, object, maxObjects });
}

/** Compact array literal for an error message. */
function fmt(v: readonly number[]): string {
  return `[${v.map((n) => (Number.isInteger(n) ? String(n) : n.toFixed(3))).join(', ')}]`;
}
