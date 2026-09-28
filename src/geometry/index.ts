/**
 * `apse/geometry` — vertex layouts, mesh data, instancing, batching, and
 * primitives.
 *
 * Six layers, in dependency order:
 *
 *   `layout`       what a vertex is: names, formats, offsets, stride. One
 *                  description that the WGSL struct, the pipeline, and the CPU
 *                  buffer are all generated from, so they cannot disagree. A
 *                  layout carries the *instance* half too, which is what makes
 *                  instancing one more attribute map rather than a second
 *                  pipeline path.
 *   `mesh`         what a mesh is: interleaved data, a conservative bounding
 *                  sphere, and the GPU buffers that mirror it. `MeshData` has no
 *                  device in it, which is why every primitive here is testable
 *                  headless, and `GpuMesh` is where the draw contract is stated.
 *   `instanced`    the same idea one level out: per-instance data, the second
 *                  vertex buffer that carries it, and a dirty-range upload for
 *                  the frame where one of five thousand moved.
 *   `tangents`     the tangent basis a normal map needs, derived from
 *                  positions and uvs, with every degenerate case counted rather
 *                  than divided by.
 *   `batch`        many meshes in one buffer, with index rebasing, combined
 *                  bounds, and per-source sub-ranges so one of them can be
 *                  drawn without re-merging.
 *   `primitives`   the nine shapes, as functions.
 *
 * The whole module is import-safe in a plain Node or Bun process. Nothing here
 * touches `navigator.gpu` at import time or at construction time, so a build
 * script can generate geometry, a test can assert on it, and a renderer can
 * upload it — one code path, three hosts.
 */

export {
  layout,
  layoutCached,
  VertexLayout,
  STANDARD_LAYOUT,
  STANDARD_ATTRIBUTES,
  TANGENT_LAYOUT,
  TANGENT_ATTRIBUTES,
  TANGENT,
  POSITION_UV_LAYOUT,
  POSITION_LAYOUT,
  VERTEX_FORMATS,
  VERTEX_FORMAT_NAMES,
  vertexFormat,
  MAX_VERTEX_LOCATIONS,
  type AttributeDefs,
  type ResolvedAttribute,
  type VertexFormat,
  type VertexFormatInfo,
} from './layout.ts';

export {
  MeshData,
  GpuMesh,
  VertexWriter,
  upload,
  allocateIndices,
  type MeshSource,
  type MeshDataOptions,
  type GpuMeshOptions,
  type InstanceBufferSource,
  type DrawableGeometryRange,
} from './mesh.ts';

export {
  InstanceData,
  GpuInstances,
  uploadInstances,
  instancedLayout,
  TRANSFORM_ATTRIBUTES,
  TRANSFORM_STRIDE,
  COLORED_INSTANCE_STRIDE,
  INSTANCE_ATTRIBUTES,
  INSTANCE_ATTRIBUTES_COLORED,
  INSTANCE_COLOR,
  INSTANCE_COLOR_COMPONENTS,
  type InstancedLayoutOptions,
  type InstanceDataOptions,
  type InstanceDataMeta,
  type GpuInstancesOptions,
} from './instanced.ts';

export {
  computeTangents,
  withTangents,
  TANGENT_FLOATS,
  type TangentBasis,
  type TangentInput,
  type TangentMeshResult,
  type WithTangentsOptions,
} from './tangents.ts';

export {
  mergeMeshes,
  uploadBatch,
  BatchedMesh,
  GpuBatchedMesh,
  type BatchSource,
  type BatchOptions,
  type BatchedRange,
} from './batch.ts';

export * from './primitives/index.ts';
