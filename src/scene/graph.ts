/**
 * The scene: a root, a version, and the draw-list builder.
 *
 * ## The two passes, and why they are separate
 *
 * {@link Scene.collectDrawItems} does two things, and conflating them is how
 * you end up with a renderer that is fast in the profiler and slow on a phone:
 *
 *   1. **Transform maintenance**, in `src/scene/node.ts`. A tree walk that
 *      touches only the nodes whose world matrix is actually stale. For a
 *      static scene this is two integer comparisons at the root.
 *   2. **Culling**, here. A tree walk that touches every drawable node, because
 *      the camera may have moved since the last frame and every bounding sphere
 *      has to be re-tested against six planes.
 *
 * Pass 1 is O(1) on a static scene. Pass 2 is O(nodes) on *any* scene, and no
 * amount of version bookkeeping can make it otherwise: if the camera moved, the
 * answer to "which objects are on screen" changed, and answering it means
 * asking about each object. Caching that answer needs a static camera, which
 * is not a thing worth pretending to.
 *
 * What pass 2 does *not* do is recompute a single matrix. It reads world data
 * that pass 1 has already brought up to date, so a static scene with a moving
 * camera costs 1000 sphere tests and zero matrix multiplies — which is the
 * actual difference from the design `node.ts` exists to replace, where the same
 * frame would cost 1000 matrix multiplies *and* 1000 sphere tests.
 *
 * ## Draw items are pooled, and that is not an optimisation detail
 *
 * {@link Scene.collectDrawItems} reuses the `DrawItem` objects it handed out
 * last frame. At 60 Hz with 5,000 visible objects, allocating them fresh is
 * 300,000 short-lived objects per second, and the GC's response is to stop the
 * world for a few milliseconds at the worst possible moment. A young-generation
 * collector makes short-lived objects cheap on average and unboundedly
 * expensive at the worst moment, which is the worst possible trade for a
 * renderer.
 *
 * The pool is indexed, not a stack: item *k* of this frame is the same object
 * as item *k* of last frame whenever the count and order are stable. That
 * keeps identity comparison meaningful for a renderer caching anything per
 * item, and it makes the behaviour testable — `test/scene.test.ts` asserts
 * `items[0] === items2[0]`.
 */

import { fail } from '../core/error.ts';
import { transformPoint } from '../math/mat4.ts';
import { containsSphere, create as createFrustum } from '../math/frustum.ts';
import type { Sphere } from '../math/sphere.ts';
import type { DrawItem, DrawPhase } from '../render/types.ts';
import type { Camera } from './camera.ts';
import { MeshNode, Node, updateWorldMatrices } from './node.ts';

// ---------------------------------------------------------------------------
// The object uniform block
// ---------------------------------------------------------------------------

/**
 * Bytes per object in the object uniform buffer.
 *
 * ```txt
 *   mat4x4f  model          64
 *   mat3x3f  normal         48   (padded to a vec4's alignment: 16 + 32)
 *   u32      objectId        4
 *   u32      flags           4
 *   f32      materialIndex   4   (+ 4 bytes of tail padding to 16)
 *                        ----
 *                          124 used, 128 to the next 16-byte boundary
 * ```
 *
 * 128 would satisfy the uniform block's own alignment. It is not what this
 * constant is, because the block is bound with a **dynamic offset**: WebGPU
 * requires `minUniformBufferOffsetAlignment` bytes between consecutive
 * dynamic offsets, and that limit is 256 on the core profile and the
 * compatibility profile. So consecutive objects sit 256 bytes apart and 132 of
 * them are padding.
 *
 * The alternative — std140-compatible 256-float blocks with every field padded
 * out — wastes the same memory and buys nothing, because the alignment
 * requirement is imposed by the binding, not by the layout.
 *
 * Exported in both units, because getting this wrong by a factor of four is
 * easy and produces a validation error at `setBindGroup` time rather than a
 * wrong picture: {@link DrawItem.objectOffset} is in **bytes**.
 */
export const OBJECT_UNIFORM_STRIDE = 256;

/** {@link OBJECT_UNIFORM_STRIDE} in `f32` elements: 64 floats, 256 bytes. */
export const OBJECT_UNIFORM_STRIDE_F32 = OBJECT_UNIFORM_STRIDE / 4;

/** Bytes of the object block actually carrying data. See the table above. */
export const OBJECT_UNIFORM_SIZE = 124;

/**
 * The `DrawItem` fields apse fills in. `material` and `geometry` are
 * `readonly` on the interface, so the pooled implementation declares them
 * mutable and is assignable to it — the pool is the point.
 */
class PooledDrawItem implements DrawItem {
  objectId = 0;
  phase: DrawPhase = 'opaque';
  order = 0;
  depth = 0;
  objectOffset = 0;
  /** A live reference to the node's world matrix — see `DrawItem.model`. */
  model!: Float32Array;
  worldVersion = 0;
  material!: DrawItem['material'];
  geometry!: DrawItem['geometry'];
  instanceCount = 1;
  firstInstance = 0;
  visible = true;
}

// ---------------------------------------------------------------------------
// Scratch — module-level. `collectDrawItems` allocates nothing.
// ---------------------------------------------------------------------------

const _frustum: Float32Array = createFrustum();
const _viewPoint = new Float32Array(3);

/**
 * A bounding sphere for the cull test, borrowing each node's `worldPosition`
 * array for the duration of the test. `center` is reassigned per node, which is
 * one store, and never copied.
 */
const _sphere: Sphere = { center: new Float32Array(3), radius: 0 };

const _collectStack: Node[] = [];

const _traverseStack: Node[] = [];

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

export interface SceneOptions {
  /** Shown in tooling. Defaults to `'scene'`. */
  name?: string;
}

/**
 * A scene graph, and the draw list it produces.
 *
 * The `root` is a plain {@link Node} with no transform of its own, so a scene
 * graph is the tree you asked for and not a tree plus a mandatory invisible
 * extra level. It is the one node that is never culled and never drawn, and
 * because nothing dirties it implicitly it is the single comparison that makes
 * a static frame free.
 */
export class Scene {
  readonly name: string;

  /** The tree root. Add to this, or to any of its descendants. */
  readonly root: Node;

  /**
   * Structural revision. Increments on every add / remove / clear.
   *
   * This counts *hierarchy* changes only, because a node does not know which
   * scene owns it and a global counter smuggled into every node would be a
   * coupling worth more than the feature. For the "did anything move?" signal
   * use {@link getGraphRevision}, which counts transform changes too — compare
   * it against a value you saved after the last frame, and a match means the
   * world matrices are all still correct.
   */
  #version = 0;

  /** The `DrawItem` pool, indexed by draw order. Never shrinks. */
  #pool: PooledDrawItem[] = [];

  /** Highest number of items ever collected. */
  #peakDrawCount = 0;

  constructor(options: SceneOptions | string = {}) {
    const opts: SceneOptions = typeof options === 'string' ? { name: options } : options;
    this.name = opts.name ?? 'scene';
    this.root = new Node({ name: `${this.name}:root` });
  }

  /** Structural revision. See {@link Scene}. */
  get version(): number {
    return this.#version;
  }

  /** The largest number of draw items collected in any single frame so far. */
  get peakDrawCount(): number {
    return this.#peakDrawCount;
  }

  /** Attaches `node` to the scene root. Chainable. */
  add(node: Node): this {
    this.root.add(node);
    this.#version++;
    return this;
  }

  /**
   * Detaches `node` from the scene.
   *
   * Throws `NODE_NOT_ATTACHED` when it was never in a scene, because "removed
   * something that was not there" is nearly always a variable that was never
   * added and a bug worth seeing.
   */
  remove(node: Node): this {
    if (node.parent === null) {
      fail('NODE_NOT_ATTACHED',
        `Cannot remove "${node.name}": it is not in scene "${this.name}".`, {
        why: 'Removing a node that was never added usually means the add was conditional, or the node was already removed once and the caller kept going. Either way the graph is not what the caller believes it is.',
        fix: `Add it with \`scene.add(${node.name})\` first, or check \`${node.name}.parent\` before removing.`,
      });
    }
    node.removeFromParent();
    this.#version++;
    return this;
  }

  /** Detaches everything. */
  clear(): this {
    this.root.clear();
    this.#version++;
    return this;
  }

  /**
   * Pre-order walk of the whole tree, ignoring the dirty state.
   *
   * This is a *structure* walk, not the transform pass: it visits every node
   * every time, because a scene walk means "tell me about the nodes", not
   * "recompute the transforms". Return `false` to skip a subtree. Iterative and
   * allocation-free.
   */
  traverse(fn: (node: Node) => boolean | void): void {
    const stack = _traverseStack;
    stack.length = 0;
    stack.push(this.root);
    while (stack.length > 0) {
      const node = stack.pop()!;
      if (fn(node) === false) continue;
      const children = node.children;
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
    }
  }

  /** Every mesh node in the tree, in draw order, without culling. */
  forEachMesh(fn: (node: MeshNode) => boolean | void): void {
    this.traverse((node) => {
      if (node instanceof MeshNode) {
        if (fn(node) === false) return false;
      }
      return true;
    });
  }

  /**
   * Resolves world matrices, culls, and fills `out` with one pooled
   * {@link DrawItem} per visible mesh node.
   *
   * `out` is cleared first, so the array length is the draw count and a culled
   * object is *absent* rather than present-and-invisible. Absent is the
   * stronger contract: a `visible: false` item still has to be walked, sorted,
   * and skipped, and that is work this module is trying not to do.
   *
   * `objectUniformStride` is in **bytes** and defaults to
   * {@link OBJECT_UNIFORM_STRIDE}. `item.objectOffset` is `index * stride` and
   * `item.objectId` is `index`, both indexed into `out` — the uniform packer
   * writes object *k*'s transform at byte offset *k* × 256, and a shader reads
   * the same number back out of the flat `objectId`.
   *
   * The camera's frustum is extracted once, into module scratch, and every
   * node is tested against it. Nothing here allocates, on any frame, ever:
   * a scene that grew to 100,000 objects would still allocate zero.
   */
  /**
   * MeshNodes the last {@link collectDrawItems} call tested and rejected.
   *
   * This has to be recorded here, at the walk, because the caller cannot derive
   * it: `out` only ever receives the survivors, so `candidates - out.length` is
   * a subtraction of a number by itself and always zero. A statistic that can
   * only ever read one value is worse than no statistic, because it looks like
   * culling is working.
   */
  culledCount = 0;
  /** MeshNodes the last walk visited, before and after culling. */
  meshNodeCount = 0;

  collectDrawItems(
    out: DrawItem[],
    camera: Camera,
    objectUniformStride: number = OBJECT_UNIFORM_STRIDE,
  ): DrawItem[] {
    updateWorldMatrices(this.root);
    camera.getFrustum(_frustum);
    out.length = 0;

    const stack = _collectStack;
    stack.length = 0;
    stack.push(this.root);
    let meshNodes = 0;
    this.culledCount = 0;
    const layers = camera.layers;

    while (stack.length > 0) {
      const node = stack.pop()!;
      // A hidden node hides its subtree. A layer mismatch does too, which is
      // what makes `node.layers = 0` a valid way to switch a whole rig off.
      if (!node.visible || (node.layer & layers) === 0) continue;

      if (node instanceof MeshNode) {
        meshNodes++;
        _sphere.center = node.worldPosition;
        _sphere.radius = node.worldBoundingRadius;
        if (containsSphere(_frustum, _sphere)) {
          out.push(this.#take(node, camera, objectUniformStride, out.length));
        } else {
          this.culledCount++;
        }
      }

      const children = node.children;
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
    }

    this.meshNodeCount = meshNodes;
    this.#peakDrawCount = out.length > this.#peakDrawCount ? out.length : this.#peakDrawCount;
    return out;
  }

  /**
   * Reuses the item at `index` of the pool, growing it if this is the first
   * time the scene has needed that many.
   */
  #take(node: MeshNode, camera: Camera, stride: number, index: number): DrawItem {
    const item = this.#pool[index] ?? (this.#pool[index] = new PooledDrawItem());
    item.objectId = index;
    item.objectOffset = index * stride;
    // A reference, not a copy. The transform pass writes into this array in
    // place and its identity is stable, so the renderer can pack it whenever it
    // likes and always see the current value.
    item.model = node.world;
    item.worldVersion = node.worldVersion;
    item.phase = node.material.phase;
    item.order = node.order;
    item.material = node.material;
    item.geometry = node.mesh;
    item.instanceCount = node.instanceCount;
    item.firstInstance = node.firstInstance;
    item.visible = true;
    // View-space depth, positive in front of the camera. The sort key for
    // back-to-front transparency; opaque items are front-to-back in every
    // engine that cares, and both orders are one comparator apart.
    item.depth = -transformPoint(_viewPoint, node.worldPosition, camera.view)[2];
    return item;
  }
}
