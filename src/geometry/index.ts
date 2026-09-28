/**
 * `apse/geometry` — vertex layouts, mesh data, and primitives.
 *
 * Three layers, in dependency order:
 *
 *   `layout`       what a vertex is: names, formats, offsets, stride. One
 *                  description that the WGSL struct, the pipeline, and the CPU
 *                  buffer are all generated from, so they cannot disagree.
 *   `mesh`         what a mesh is: interleaved data, a bounding sphere, and the
 *                  GPU buffers that mirror it. `MeshData` has no device in it,
 *                  which is why every primitive here is testable headless.
 *   `instanced`    the same idea one level out: per-instance data, and the
 *                  second vertex buffer that carries it. A layout can declare
 *                  per-instance attributes, so instancing is one more
 *                  attribute map rather than a second pipeline path.
 *   `primitives`   the six shapes, as functions.
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
  upload,
  type MeshSource,
  type MeshDataOptions,
  type GpuMeshOptions,
} from './mesh.ts';

export {
  InstanceData,
  GpuInstances,
  uploadInstances,
  instancedLayout,
  TRANSFORM_ATTRIBUTES,
  TRANSFORM_STRIDE,
  INSTANCE_ATTRIBUTES,
  INSTANCE_ATTRIBUTES_COLORED,
  INSTANCE_COLOR,
  INSTANCE_COLOR_COMPONENTS,
  type InstanceDataOptions,
  type GpuInstancesOptions,
} from './instanced.ts';

export * from './primitives/index.ts';
