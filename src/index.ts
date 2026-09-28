/**
 * apse — a WebGPU renderer.
 *
 * The import surface is the file tree, and every subpath is independently
 * tree-shakeable. `sideEffects: false` is a promise, not a decoration: no module
 * in `src` has top-level work, registers a listener, or patches a prototype.
 *
 * Start at `createRenderer` and the `geometry` primitives. Everything else is
 * reachable from there when you need it, and not before.
 */

// --- the one call that starts everything ---
export { Renderer, type RendererOptions, type FrameStats, type RenderBudget } from './render/renderer.ts';

// --- devices, targets, sizing ---
export {
  createDevice,
  withErrorScope,
  copyLimits,
  COMPAT_LIMITS,
  type AseDevice,
  type DeviceOptions,
  type FeatureLevel,
} from './render/device.ts';
export { RenderTargetImpl, createCanvasTarget, createColorTarget, createDepthTarget } from './render/target.ts';
export { CanvasSizer } from './render/context.ts';
export { PresentPass, DEFAULT_TONE_MAPPING, type PresentOptions } from './render/present.ts';
export {
  GpuTimer,
  type GpuTimerOptions,
  type GpuTimerFailure,
  type TimingDevice,
} from './render/timing.ts';
export {
  CaptureReadback,
  alignedBytesPerRow,
  assertBytesPerRowAligned,
  unpadRows,
  BYTES_PER_PIXEL_RGBA8,
  COPY_BYTES_PER_ROW_ALIGNMENT,
} from './render/readback.ts';
export {
  requireFeature,
  hasFeature,
  availableFeatures,
  readCapabilities,
  ASE_FEATURES,
  BUFFER_USAGE,
  MAP_MODE,
  TIMESTAMP_BYTES,
  type AseFeatureName,
  type GpuFeatureFailure,
  type DeviceCapabilities,
  type TextureCompressionFamily,
} from './render/device.ts';
export { sortDrawItems, compareDrawItems } from './render/sort.ts';
export * from './render/pipeline-state.ts';
export type {
  DrawItem,
  Drawable,
  DrawableGeometry,
  RenderTarget,
  FrameState,
  FrameTimingStats,
} from './render/types.ts';
export type { TargetCompatibility } from './render/target.ts';

// --- geometry: layouts, meshes, primitives ---
export {
  layout,
  layoutCached,
  vertexFormat,
  VertexLayout,
  VERTEX_FORMATS,
  VERTEX_FORMAT_NAMES,
  MAX_VERTEX_LOCATIONS,
  STANDARD_ATTRIBUTES,
  STANDARD_LAYOUT,
  TANGENT_ATTRIBUTES,
  TANGENT_LAYOUT,
  POSITION_UV_LAYOUT,
  POSITION_LAYOUT,
  type VertexFormat,
  type AttributeDefs,
} from './geometry/layout.ts';
export {
  MeshData,
  GpuMesh,
  upload,
  VertexWriter,
  allocateIndices,
  type DrawableGeometryRange,
  type InstanceBufferSource,
  type GpuMeshOptions,
  type MeshDataOptions,
  type MeshSource,
} from './geometry/mesh.ts';
export {
  InstanceData,
  GpuInstances,
  uploadInstances,
  instancedLayout,
  INSTANCE_ATTRIBUTES,
  INSTANCE_ATTRIBUTES_COLORED,
  TRANSFORM_ATTRIBUTES,
  TRANSFORM_STRIDE,
  COLORED_INSTANCE_STRIDE,
  INSTANCE_COLOR,
  INSTANCE_COLOR_COMPONENTS,
  type InstanceDataOptions,
} from './geometry/instanced.ts';
export {
  mergeMeshes,
  uploadBatch,
  BatchedMesh,
  GpuBatchedMesh,
  type BatchSource,
  type BatchedRange,
} from './geometry/batch.ts';
export {
  computeTangents,
  withTangents,
  TANGENT_FLOATS,
  type TangentBasis,
  type TangentInput,
  type TangentMeshResult,
} from './geometry/tangents.ts';
export {
  box,
  sphere,
  plane,
  torus,
  cylinder,
  grid,
  capsule,
  cone,
  roundedBox,
  BOX_FACES,
} from './geometry/primitives/index.ts';

// --- scene ---
export { Scene, OBJECT_UNIFORM_STRIDE, MIN_OBJECT_UNIFORM_STRIDE } from './scene/graph.ts';
export { Node, MeshNode, updateWorldMatrices, getTransformWriteCount, resetTransformWriteCount } from './scene/node.ts';
export { Camera, PerspectiveCamera, OrthographicCamera } from './scene/camera.ts';

// --- materials: the shader scaffold ---
export { Material, FrameUniforms, ObjectUniforms, deviceCache, type MaterialOptions } from './material/material.ts';
export { generateScaffold, describeMaterial, type MaterialSpec, type GeneratedShader } from './material/scaffold.ts';
export { basicMaterial, basicMaterialSpec } from './material/basic.ts';
export { pbrMaterial, pbrMaterialSpec } from './material/pbr.ts';
export { instancedMaterial, instancedMaterialSpec, type InstancedMaterialOptions } from './material/instanced.ts';
export { diffuseMaterial, diffuseMaterialSpec, type DiffuseMaterialOptions } from './material/diffuse.ts';
export { emissiveMaterial, emissiveMaterialSpec, type EmissiveMaterialOptions } from './material/emissive.ts';
export {
  anisotropicMaterial,
  anisotropicMaterialSpec,
  type AnisotropicMaterialOptions,
} from './material/anisotropic.ts';
export {
  tonemapMaterial, tonemapMaterialSpec, fullscreenMesh, FULLSCREEN_LAYOUT,
  TONEMAP_SLOTS, TONE_MAP_OPERATORS, type TonemapOptions,
} from './material/tonemap.ts';
export type { TextureSlotSpec, TextureKind } from './material/texture-slot.ts';
export type { SlotDefs, SlotSpec, SlotType } from './core/slot.ts';

// --- math, for callers who need it directly ---
export * as vec3 from './math/vec3.ts';
export * as mat4 from './math/mat4.ts';
export * as quat from './math/quat.ts';
export * as frustum from './math/frustum.ts';

// --- errors, because they are part of the API ---
export { AseError, isAseError, isErrorCode, fail, type AseErrorOptions } from './core/error.ts';
export {
  ERROR_CATALOG,
  ERROR_CODES,
  ERROR_BLAME,
  type AseErrorCode,
  type ErrorDetail,
  type ErrorBlame,
} from './core/error-catalog.ts';
export { ok, err, isOk, isErr, unwrap, unwrapOr, attempt, type Result, type Err } from './core/result.ts';
export { Resource, ResourceScope, type Disposable, type DisposedCode } from './core/resource.ts';
export { buildUniformBlock, UniformBlock, UNIFORM_TYPES, type UniformTypeInfo, type UniformBlockSpec } from './core/uniform.ts';

/** Library version. Kept in step with package.json by scripts/build.ts. */
export const VERSION = '0.0.1';
