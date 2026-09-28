/**
 * `apse/material` — materials, without writing a shader.
 *
 *   `scaffold`     the generator: a spec in, byte-identical WGSL out, plus the
 *                  validation that turns "undeclared identifier" into a typed
 *                  apse error naming the field. No device, importable anywhere.
 *   `material`     the compiled, drawable shader, and the shared frame/object
 *                  uniform blocks that 500 materials should not each own.
 *   `lighting`     the declarative light rig shared by the lit materials: the
 *                  light array becomes both the uniform fields and the shading
 *                  statements, so the count is data rather than shader text.
 *   `basic`        unlit, a colour in 0..1. The hello cube.
 *   `pbr`          Cook-Torrance GGX, metals, up to four lights, a
 *                  Fresnel-Schlick rim, image-based lighting, and a PCF shadow.
 *                  The proof that the abstraction is not a toy.
 *   `diffuse`      Oren-Nayar for non-metals, with the energy-conserving rim.
 *   `emissive`     a physically-shaped unlit emitter: radiance, a Fresnel
 *                  silhouette, and a frame-clock pulse.
 *   `anisotropic`  an elliptical GGX lobe in a tangent frame. Brushed metal.
 *   `tonemap`      the present pass's first entry: exposure, an operator chosen
 *                  by a uniform, and the sRGB encode the target format decides.
 *
 * Every one of the five lit/unlit factories is a function returning a
 * `MaterialSpec`, and every one of them is asserted on in `test/material.test.ts`
 * with no GPU in sight — which is only possible because a material is its spec.
 *
 * The contract the renderer depends on is `Drawable` in `src/render/types.ts`,
 * not anything else exported here.
 */

export {
  generateScaffold,
  describeMaterial,
  resolveSpec,
  validateBodyIdentifiers,
  validateGeneratedWGSL,
  stripComments,
  removeComments,
  tokenKeyOf,
  checkVaryingBudget,
  pipelineStateKeyOf,
  pipelineKeyOf,
  fnv1a,
  preferredTargetFormat,
  samplerNameFor,
  VARYING_TYPES,
  VARYING_TYPE_NAMES,
  CORE_INTER_STAGE,
  COMPAT_INTER_STAGE,
  type GeneratedShader,
  type MaterialDescription,
  type MaterialSpec,
  type ResolvedMaterialSpec,
  type ResolvedSampler,
  type ResolvedVarying,
  type ResolvedTarget,
  type VaryingDefs,
  type VaryingType,
  type DescribedAttribute,
  type DescribedBindGroup,
  type DescribedSlot,
  type DescribedTexture,
  type DescribedVarying,
  type InterStageLimits,
} from './scaffold.ts';

export {
  Material,
  FrameUniforms,
  ObjectUniforms,
  DeviceCache,
  deviceCache,
  stageAtLine,
  VERTEX_ENTRY,
  FRAGMENT_ENTRY,
  DEFAULT_FRAME_LABEL,
  DEFAULT_OBJECT_LABEL,
  type MaterialOptions,
  type ShaderStage,
} from './material.ts';

export { basicMaterial, basicMaterialSpec, type BasicMaterialOptions } from './basic.ts';
export { pbrMaterial, pbrMaterialSpec, type PbrMaterialOptions } from './pbr.ts';
export { diffuseMaterial, diffuseMaterialSpec, type DiffuseMaterialOptions } from './diffuse.ts';
export { emissiveMaterial, emissiveMaterialSpec, type EmissiveMaterialOptions } from './emissive.ts';
export { anisotropicMaterial, anisotropicMaterialSpec, type AnisotropicMaterialOptions } from './anisotropic.ts';

export {
  MAX_DIRECTIONAL_LIGHTS,
  ENV_MIP_RANGE,
  ENVIRONMENT_SLOTS,
  ENVIRONMENT_TEXTURE_SLOT,
  lightColorSlot,
  lightDirSlot,
  lightSlots,
  resolveLights,
  type DirectionalLight,
  type ResolvedLight,
} from './lighting.ts';

export {
  TEXTURE_KINDS,
  samplerTypeName,
  textureViewDimension,
  wgslTextureDecl,
  type ResolvedTextureSlot,
  type TextureKind,
  type TextureSampleType,
  type TextureSlotSpec,
} from './texture-slot.ts';

export {
  instancedMaterial,
  instancedMaterialSpec,
  type InstancedMaterialOptions,
} from './instanced.ts';

export {
  tonemapMaterial,
  tonemapMaterialSpec,
  fullscreenMesh,
  FULLSCREEN_LAYOUT,
  TONEMAP_SLOTS,
  TONEMAP_MATERIAL_SLOTS,
  TONEMAP_BLOCK,
  EXPOSURE_FRAME_FIELD,
  TONE_MAP_OPERATORS,
  OPERATOR_IDS,
  DEFAULT_TONE_MAP_OPERATOR,
  DEFAULT_HDR_FORMAT,
  SRGB_TRANSFER,
  ACES_COEFFICIENTS,
  isSrgbFormat,
  linearToSrgb,
  srgbToLinear,
  resolveTonemapOptions,
  writeToneMapSlots,
  type TonemapOptions,
  type ResolvedTonemapOptions,
  type ToneMapOperator,
  type ToneMapSlots,
} from './tonemap.ts';
