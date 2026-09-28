/**
 * `apse/material` — materials, without writing a shader.
 *
 *   `scaffold`     the generator: a spec in, byte-identical WGSL out, plus the
 *                  validation that turns "undeclared identifier" into a typed
 *                  apse error naming the field. No device, importable anywhere.
 *   `material`     the compiled, drawable shader, and the shared frame/object
 *                  uniform blocks that 500 materials should not each own.
 *   `basic`        unlit. The hello cube.
 *   `pbr`          Cook-Torrance GGX with a PCF shadow. The proof that the
 *                  abstraction is not a toy.
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
  VERTEX_ENTRY,
  FRAGMENT_ENTRY,
  type MaterialOptions,
} from './material.ts';

export { basicMaterial, basicMaterialSpec, type BasicMaterialOptions } from './basic.ts';
export { pbrMaterial, pbrMaterialSpec, type PbrMaterialOptions } from './pbr.ts';

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
  TONE_MAP_OPERATORS,
  isSrgbFormat,
  type TonemapOptions,
} from './tonemap.ts';
