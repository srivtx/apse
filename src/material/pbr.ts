/**
 * A Cook-Torrance GGX material.
 *
 * This is the module's proof that the scaffold scales. It is a real
 * physically-based shader — GGX distribution, Smith height-correlated
 * visibility, Schlick Fresnel, a hemisphere ambient term, and a
 * percentage-closer-filtered directional shadow — and every single binding,
 * uniform field, and varyings struct it needs is *declared as data* and
 * generated. There is not one `@group`, `@binding`, or `struct` in this file.
 *
 * The BRDF itself lives in `prelude`, which is the documented escape hatch for
 * user declarations. That is where helper `fn`s belong, and it is the reason
 * `prelude` is not a loophole: it sits after the generated declarations and
 * before the entry points, so it can add code but cannot redefine a binding.
 *
 * Read the `fragment` string below as the tutorial it is meant to be. It is the
 * most-read shader in the library, because it is the shortest complete example
 * of what a material author can and cannot write.
 */

import { STANDARD_LAYOUT } from '../geometry/layout.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import { Material } from './material.ts';
import type { MaterialOptions } from './material.ts';
import type { MaterialSpec } from './scaffold.ts';

export interface PbrMaterialOptions extends MaterialOptions {
  /** Defaults to `'pbr'`. */
  name?: string;
  /** Albedo, linear. Defaults to mid grey. */
  baseColor?: readonly [number, number, number];
  /** 0 = dielectric, 1 = metal. Defaults to 0. */
  metallic?: number;
  /** Perceptual roughness. Defaults to 0.5. */
  roughness?: number;
  /** Direction *towards* the light, in world space. Need not be normalised. */
  lightDir?: readonly [number, number, number];
  /** Radiance of the directional light. */
  lightColor?: readonly [number, number, number];
  /** Upper hemisphere ambient colour. */
  ambientColor?: readonly [number, number, number];
  /** How strongly the normal drives the ambient occlusion approximation. */
  aoStrength?: number;
  /**
   * Declare a `shadowDepth` depth texture and multiply the direct term by a 3×3
   * PCF visibility estimate.
   *
   * Off by default, because a shadow map is a whole other pass and a material
   * that declares the slot but has no map bound cannot be drawn. When you turn
   * this on you must also set `lightViewProj` — an orthographic light matrix
   * matching the shadow map you rendered.
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
   * validation error at `setPipeline` that names a format rather than the
   * mistake.
   *
   * Pass this explicitly when drawing into an offscreen target — an HDR
   * `rgba16float` post-processing chain, for instance — since the canvas format
   * is the wrong answer there.
   */
  targetFormat?: GPUTextureFormat;
}

/**
 * The BRDF. Three functions, each one line of real mathematics, placed in
 * `prelude` because they are declarations and declarations are not allowed in a
 * body.
 */
const BRDF_PRELUDE = `
// --- Cook-Torrance ----------------------------------------------------------
// D * V * F. Each term is separated here so it can be reasoned about — and
// plotted — independently, which is why real implementations keep them apart
// even though the product is what is needed.

const PI : f32 = 3.14159265359;

/** GGX / Trowbridge-Reitz normal distribution: how tightly the microfacets
 *  around the halfway vector are aligned. Peaks at NdotH = 1. */
fn distributionGGX(nDotH: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;      // perceptual -> linear roughness
  let a2 = a * a;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-7);
}

/** Smith's height-correlated visibility, already divided by
 *  4 * NdotL * NdotV so that the product D * V * F is the BRDF directly. */
fn visibilitySmith(nDotV: f32, nDotL: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let gv = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
  let gl = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-6);
}

/** Schlick's approximation to Fresnel, from F0 to 1 at grazing angles. */
fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (vec3f(1.0) - f0) * (m2 * m2 * m);
}
`;

/** The vertex stage. World-space shading, so nothing is transformed twice. */
const PBR_VERTEX = `
out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;

// The normal matrix is a mat3x3f, and in WGSL a matCxR multiplies a vecC — so
// this is a vec3f in and a vec3f out. (GLSL's mat3 * vec4, which ignores w,
// is not valid WGSL and is a common porting slip.)
out.normal = normalize(obj.normalMatrix * in.normal);
`;

/** The direct lighting term, shared by both shadow variants. */
const PBR_FRAGMENT_DIRECT = `
  // --- geometry -------------------------------------------------------------
  // Shading in world space, so the view and light vectors are both plain
  // differences of world-space positions.
  let n = normalize(in.normal);
  let v = normalize(frame.camPos - in.worldPos);
  let l = normalize(mat.lightDir);
  let h = normalize(v + l);

  // Clamped rather than trusted: a zero-length v + l at a silhouette would
  // otherwise produce a NaN that propagates through the whole term.
  let nDotV = max(dot(n, v), 1e-4);
  let nDotL = max(dot(n, l), 0.0);
  let nDotH = max(dot(n, h), 0.0);
  let vDotH = max(dot(v, h), 0.0);

  let roughness = clamp(mat.roughness, 0.045, 1.0);
  let metallic = clamp(mat.metallic, 0.0, 1.0);

  // --- material -------------------------------------------------------------
  // Metallic workflow: a metal *is* its specular reflectance, a dielectric
  // sits on a 0.04 base.
  let base = mat.baseColor;
  let f0 = mix(vec3f(0.04), base, metallic);
  let diffuseColor = base * (1.0 - metallic);
`;

/** Shadowed PCF. Uses `mat.shadowBias` and `mat.lightViewProj`. */
const PBR_FRAGMENT_SHADOW = `
  // --- shadow ---------------------------------------------------------------
  // The light is directional and its projection is orthographic, so the
  // transform is affine and w is 1. Transform into the light's clip space,
  // then into texture space: xy in [0,1] with y flipped because NDC y is up
  // and texture v is down, and z already in [0,1] because the comparison
  // sampler wants [0,1] and a WebGPU depth format is 0-to-1 rather than
  // OpenGL's -1-to-1.
  let lightClip = mat.lightViewProj * vec4f(in.worldPos, 1.0);
  let ndc = lightClip.xyz;
  let shadowUV = ndc.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);

  // The bounds test is folded into the result rather than used as an early-out.
  // textureSampleCompare requires uniform control flow, and a fragment that
  // took a different branch per pixel is exactly what that rule forbids —
  // outside the map, sample the clamped edge and discard the result.
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
  let shadow = select(1.0, sum / 9.0, inside);
`;

/**
 * Assembling the terms, the ambient, and the return. Shared by both shadow
 * variants; `SHADOW_APPLY` is the whole of what the shadow changes, which is
 * one line, because a shadow is a multiplier on the direct term and nothing
 * else.
 */
const PBR_FRAGMENT_COMBINE = `
  // --- direct ---------------------------------------------------------------
  let specular = distributionGGX(nDotH, roughness) * visibilitySmith(nDotV, nDotL, roughness) * fresnelSchlick(vDotH, f0);
  // Energy conservation: whatever the specular lobe reflects is not also
  // available as diffuse. Without this, rough metal glows white.
  let diffuse = (vec3f(1.0) - fresnelSchlick(nDotV, f0)) * diffuseColor / PI;

  var direct = (diffuse + specular) * mat.lightColor * nDotL;
  SHADOW_APPLY

  // --- ambient --------------------------------------------------------------
  // A hemisphere: the ambient colour above, a darker version below, blended by
  // how far the normal points up. Cheaper than an irradiance probe and, at
  // this scale, indistinguishable.
  let upness = n.y * 0.5 + 0.5;
  let hemi = mix(mat.ambientColor * 0.35, mat.ambientColor, upness);

  // A one-tap ambient occlusion approximation: surfaces facing up see more sky.
  let ao = mix(1.0, clamp(upness, 0.0, 1.0), mat.aoStrength);

  // A Fresnel rim keeps grazing angles from going flat and reads as a sheen.
  let rim = pow(1.0 - nDotV, 5.0) * (1.0 - metallic) * mat.aoStrength;

  var color = direct + diffuseColor * hemi * ao + mat.ambientColor * rim;

  // No tone map here: frame.exposure is applied by the present pass, so a
  // material in an offscreen pass is not double-exposed.
  return vec4f(color, 1.0);
`;

/**
 * The spec `pbrMaterial` builds. Exported so it can be read and tested without
 * a device — the material *is* its spec.
 */
export function pbrMaterialSpec(opts: PbrMaterialOptions = {}): MaterialSpec {
  const {
    name = 'pbr',
    baseColor = [0.6, 0.6, 0.6],
    metallic = 0,
    roughness = 0.5,
    lightDir = [0.5, 1, 0.3],
    lightColor = [3, 3, 3],
    ambientColor = [0.15, 0.17, 0.22],
    aoStrength = 1,
    shadows = false,
    shadowBias = 0.02,
    doubleSided = false,
    layout = STANDARD_LAYOUT,
    targetFormat = preferredFormat(),
  } = opts;

  const fragment = [
    PBR_FRAGMENT_DIRECT,
    shadows ? PBR_FRAGMENT_SHADOW : '',
    PBR_FRAGMENT_COMBINE.replace(
      'SHADOW_APPLY',
      // A comment rather than nothing, so the generated shader explains the
      // absence of a shadow term instead of leaving a silent gap.
      shadows
        ? 'direct = direct * shadow;'
        : '// No shadow map was declared, so the direct term is unoccluded.',
    ),
  ]
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .join('\n\n');

  return {
    name,
    layout,
    // Two varyings, both interpolated. Location 0 and 1, assigned in
    // declaration order; `clip` is the builtin and consumes none.
    varyings: { worldPos: 'vec3f', normal: 'vec3f' },
    slots: {
      // The seven the material is defined by.
      baseColor: { type: 'vec3f', default: baseColor },
      metallic: { type: 'f32', default: metallic },
      roughness: { type: 'f32', default: roughness },
      lightDir: { type: 'vec3f', default: lightDir },
      lightColor: { type: 'vec3f', default: lightColor },
      ambientColor: { type: 'vec3f', default: ambientColor },
      aoStrength: { type: 'f32', default: aoStrength },
      // Two more that the shadow term needs: the light's orthographic matrix
      // and a depth bias. Declaring them as slots means `setSlot` can retune
      // them at runtime with no pipeline rebuild.
      lightViewProj: { type: 'mat4x4f', default: new Float32Array(16) },
      shadowBias: { type: 'f32', default: shadowBias },
    },
    ...(shadows
      ? {
        textures: {
          // A depth texture with a comparison sampler. apse generates both the
          // `texture_depth_2d` binding and the `sampler_comparison` binding,
          // and shares the sampler with any other slot that asks for the same
          // configuration.
          shadowDepth: { kind: 'depth-2d' as const, sampleType: 'depth' as const, compare: true, mipmapFilter: false },
        },
      }
      : {}),
    prelude: BRDF_PRELUDE,
    phase: 'opaque',
    topology: 'triangle-list',
    cull: doubleSided ? 'none' : 'back',
    depth: { write: true, compare: 'less' },
    blend: null,
    targets: [{ format: targetFormat }],
    sampleCount: 1,
    vertex: PBR_VERTEX,
    fragment,
  };
}

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

/**
 * The canvas format a material should target when the caller does not say.
 *
 * Read at call time rather than captured at module load, and guarded for
 * non-browser contexts so importing a material factory in a Node test does not
 * throw. `rgba8unorm` is the documented fallback for anywhere there is no
 * canvas — an offscreen target, where the caller must pass the format anyway.
 */
function preferredFormat(): GPUTextureFormat {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined'
    ? navigator.gpu.getPreferredCanvasFormat()
    : 'rgba8unorm';
}
