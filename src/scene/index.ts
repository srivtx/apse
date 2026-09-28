/**
 * The scene layer: a transform graph that only recomputes what moved, and a
 * draw-list builder that allocates nothing.
 *
 *     import { Scene, MeshNode, PerspectiveCamera } from 'apse/scene';
 *
 *     const scene = new Scene();
 *     scene.add(new MeshNode({ name: 'cube', mesh, material, position: [0, 0, -5] }));
 *
 *     const camera = new PerspectiveCamera({ fov: 60, near: 0.1, far: 1000 });
 *     camera.lookAt([0, 0, 0], [0, 0, -1]);
 *     camera.update(width / height);
 *
 *     const items = scene.collectDrawItems([], camera);
 *
 * Four files, one concern each:
 *
 *   - `node.ts`     transforms, hierarchy, and the pruned matrix pass
 *   - `graph.ts`    the {@link Scene}, culling, and the pooled draw list
 *   - `camera.ts`   view and projection, in WebGPU's clip space
 *   - `index.ts`    this file
 *
 * The transform contract — setters bump a version, a clean node prunes its
 * whole subtree, a static scene costs two integer comparisons per frame — is
 * documented in full at the top of `node.ts`, and asserted in
 * `test/scene.test.ts`.
 */

export {
  // Class
  Node,
  MeshNode,
  // Traversal
  updateWorldMatrices,
  // Diagnostics
  getTransformWriteCount,
  resetTransformWriteCount,
  getNodeVisitCount,
  resetNodeVisitCount,
  getGraphRevision,
  // Constants
  DEFAULT_LAYER,
  // Types
  type NodeOptions,
  type MeshNodeOptions,
} from './node.ts';

export {
  // Classes
  Scene,
  // Constants
  OBJECT_UNIFORM_STRIDE,
  OBJECT_UNIFORM_STRIDE_F32,
  OBJECT_UNIFORM_SIZE,
  // Types
  type SceneOptions,
} from './graph.ts';

export {
  // Classes
  Camera,
  PerspectiveCamera,
  OrthographicCamera,
  // Constants
  FRUSTUM_PLANE_COUNT,
  FRUSTUM_PLANE_LENGTH,
  // Types
  type CameraOptions,
  type PerspectiveCameraOptions,
  type OrthographicCameraOptions,
  type Frustum,
  type Ray,
} from './camera.ts';
