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
import { sceneObjectOffset } from '../core/slot.ts';
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
 * The smallest object-uniform stride {@link Scene.collectDrawItems} will accept,
 * in bytes.
 *
 * `minUniformBufferOffsetAlignment` is 256 in the WebGPU specification, and a
 * device may only ever report a *smaller* one — the compatibility profile takes
 * the floor, and no profile raises it. So "a multiple of 256" is the one stride
 * rule that is correct on every device without asking the device anything, and
 * that is what the parameter is checked against rather than against a value
 * passed in from the renderer. It also happens to be the default, so the common
 * case needs no arithmetic at all.
 */
export const MIN_OBJECT_UNIFORM_STRIDE = OBJECT_UNIFORM_STRIDE;

/**
 * The `DrawItem` fields apse fills in. `material` and `geometry` are
 * `readonly` on the interface, so the pooled implementation declares them
 * mutable and is assignable to it — the pool is the point.
 *
 * Every field is assigned on every frame by {@link Scene.collectDrawItems}, and
 * that is not a matter of tidiness. A pooled item is an object that last frame's
 * scene left holding last frame's node, and anything not overwritten is
 * *someone else's values* read as if they were this frame's — a stale
 * `instanceCount` draws the wrong number of copies, a stale `objectId` puts two
 * objects' transforms in one uniform slot. The initialiser is one function for
 * exactly that reason, and {@link Scene.collectDrawItems} validates the result
 * of it rather than the inputs, so a field added later fails loudly instead of
 * reading a default.
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
// Draw-item validation
// ---------------------------------------------------------------------------

/**
 * `Resource` state, read structurally.
 *
 * `DrawableGeometry` and `Drawable` are interfaces in `src/render`, and they
 * describe what a draw *needs* — not how the thing behind them counts its
 * references. `GpuMesh` and `Material` both answer these questions; a hand-rolled
 * geometry does not, and `undefined` here means "not a ref-counted resource",
 * which is not an error. `undefined` is a valid answer because a *missing*
 * `disposed` cannot mean a disposed object.
 */
interface ResourceState {
  readonly disposed?: boolean;
  readonly name?: string;
}

/**
 * True when a draw with these two counts would issue anything at all.
 *
 * Zero is legal input and must not become a draw item: `GpuInstances` accepts an
 * empty transform list — it allocates a 4-byte floor so `createBuffer` stays well
 * defined — and a mesh with no indices is equally legal. `drawIndexed` with
 * `instanceCount: 0` is a silent no-op, so a node that will draw nothing is
 * counted instead. It is a *not-submitted* node, which is a different fact from a
 * culled one, and the two are counted separately for exactly that reason.
 *
 * Takes the counts rather than the node because the walk needs them again a few
 * lines later, and `node.mesh` and `node.instanceCount` are getters. Reading
 * them once here and passing the results in costs the walk two calls per node
 * instead of four, and `instanceCount` is a *live* read either way — hoisting it
 * does not turn a per-frame read into a snapshot, it just stops reading the same
 * geometry three times in the same ten lines.
 */
function canDraw(indexCount: number, instanceCount: number): boolean {
  return indexCount > 0 && instanceCount > 0;
}

/**
 * Rejects a draw item the renderer could not encode correctly.
 *
 * Runs on the *item*, after it has been filled in, and on every item. The
 * alternative — checking the node on the way in — is what makes a validation
 * conditional on which path emitted the item, and a validation that some paths
 * skip is not a validation.
 *
 * The three failures are the three ways a draw item can be structurally
 * plausible and still produce a corrupt frame: a dynamic offset the API will
 * reject, a geometry whose buffers are gone, and a `model` that is not a
 * `mat4x4f`. The middle one is what refcounted ownership in `node.ts` makes
 * unreachable by normal use — a node holds a reference to what it draws — so it
 * is the backstop for the paths ownership does not cover: a mesh disposed by hand
 * while a node still points at it.
 */
function assertDrawable(item: DrawItem): void {
  if (item.objectOffset % MIN_OBJECT_UNIFORM_STRIDE !== 0) {
    fail('INTERNAL_INVARIANT',
      `Draw item ${item.objectId} would bind its object block at byte offset ${item.objectOffset}.`, {
      why: `A dynamic offset must be a multiple of ${MIN_OBJECT_UNIFORM_STRIDE}; one that is not invalidates the whole command buffer, with no exception and an empty frame. The stride is validated on entry to collectDrawItems, so this offset came from somewhere other than \`objectId * stride\`.`,
      fix: 'Report this: `objectOffset` is assigned in exactly one place.',
      detail: { kind: 'numeric', field: 'objectOffset', value: item.objectOffset, min: MIN_OBJECT_UNIFORM_STRIDE },
    });
  }

  const geometry = item.geometry as ResourceState;
  if (geometry.disposed === true) {
    fail('MESH_DISPOSED',
      `Draw item ${item.objectId} draws mesh "${geometry.name ?? 'unnamed'}", whose buffers have been released.`, {
      why: 'The buffers behind this item are destroyed, so the draw encodes against freed memory: every triangle of this object silently vanishes while the draw call succeeds.',
      fix: 'Hold a reference with `mesh.ref()`. A MeshNode in a graph already holds one, so this means the mesh was disposed out from under the node.',
    });
  }
  if (item.model.length !== 16) {
    fail('INTERNAL_INVARIANT',
      `Draw item ${item.objectId} carries a ${item.model.length}-element model matrix.`, {
      why: 'The model is a mat4x4f in the object block. Any other length leaves the rest of the record unwritten, so every object after this one inherits the previous one\'s transform.',
      fix: 'Report this: `DrawItem.model` is a live reference to a node\'s `Float32Array(16)`.',
      detail: { kind: 'numeric', field: 'model.length', value: item.model.length, min: 16, max: 16 },
    });
  }
}

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
 *
 * ## A scene owns what is in it
 *
 * {@link add} and {@link remove} are also the retain and release pair. Adding a
 * `MeshNode` takes a reference to its mesh and its material; removing one gives
 * it back. So the question "is this mesh still alive?" is answered by the graph
 * rather than by whoever happened to keep a handle:
 *
 * ```ts
 * const mesh = upload(device, boxData());           // refCount 1, yours
 * const node = new MeshNode({ mesh, material });    // not in a graph: still 1
 * scene.add(node);                                  // refCount 2, the scene's
 * mesh.unref();                                     // you are done with it
 * // ...render for as long as you like: the node holds the last reference
 * scene.remove(node);                               // refCount 0, released
 * ```
 *
 * Without the `add` line, that `unref` destroys the buffers out from under a
 * scene that is still drawing them, and the symptom is a screen that goes empty
 * with no error and no failing draw call.
 *
 * The same rule applies one level down, which is the part that has to be here
 * rather than in {@link add}: a node built into a group *before* the group is
 * added is covered by exactly the same code path, because the reference is taken
 * by `Node.add` and given back by the one detach path every removal funnels
 * through. Ownership is a property of being in a graph, not of which graph.
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

  /**
   * Attaches `node` to the scene root. Chainable.
   *
   * Takes a reference to everything `node` draws, and to everything its existing
   * subtree draws, so a group assembled before it reaches the scene arrives with
   * its resources already retained. See the class comment.
   */
  add(node: Node): this {
    this.root.add(node);
    this.#version++;
    return this;
  }

  /**
   * Detaches `node` from the scene, wherever in the tree it is.
   *
   * Releases the references taken by {@link add}, so a mesh or material whose
   * only remaining reference was the scene's is disposed here and not before.
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

  /**
   * Detaches everything, releasing every reference the scene held.
   *
   * A shared mesh is released once per node that held it, so a thousand nodes on
   * one geometry survive this call with the geometry intact and dispose it on the
   * thousandth detach — not on the first.
   */
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
   * MeshNodes the last {@link collectDrawItems} call tested and rejected.
   *
   * Recorded at the walk because the caller cannot derive it: `out` only ever
   * receives the survivors, so `candidates - out.length` is a subtraction of a
   * number by itself and always zero. A statistic that can only ever read one
   * value is worse than no statistic, because it looks like culling is working.
   *
   * This counts **frustum rejections and nothing else**. A node that was never a
   * candidate — hidden, on a layer the camera does not see, or holding a
   * geometry with no instances to draw — is in {@link emptyCount} or in neither,
   * and folding those in here is how a "culled" statistic starts reporting how
   * much of the scene the author happened to switch off.
   */
  culledCount = 0;

  /**
   * MeshNodes the last walk saw, before culling.
   *
   * The three counters partition it exactly, and the partition is the point:
   *
   * ```txt
   *   meshNodeCount === out.length + culledCount + emptyCount
   * ```
   *
   * so any one of them can be checked against the other two, and a caller that
   * wants "how much of my scene did I not draw, and why" has both answers
   * rather than one ambiguous one. A node under a hidden ancestor is in none of
   * them, and {@link meshNodeCount} counts it: the walk skipped that subtree on
   * the ancestor's behalf, and pretending otherwise would make the identity
   * above false.
   */
  meshNodeCount = 0;

  /**
   * MeshNodes that were candidates and produced no draw item for a reason other
   * than culling — today, a geometry with no instances or no indices.
   *
   * Separate from {@link culledCount} because the two mean opposite things: a
   * culled node is off screen and a closer camera will find it, while an empty
   * one is on screen, in frustum, and still has nothing to draw. A statistic
   * that adds them together reports a culling win for a scene that has nothing
   * to draw.
   */
  emptyCount = 0;

  /**
   * Resolves world matrices, culls, and fills `out` with one pooled
   * {@link DrawItem} per drawable, unculled mesh node.
   *
   * `out` is cleared first, so the array length is the draw count and a culled
   * object is *absent* rather than present-and-invisible. Absent is the
   * stronger contract: a `visible: false` item still has to be walked, sorted,
   * and skipped, and that is work this module is trying not to do.
   *
   * `item.objectOffset` is {@link sceneObjectOffset} of `index` and
   * `item.objectId` is `index`. The offset is **not** `index * 256`: byte 0 of
   * the shared scene buffer is the frame, so object *k* begins at
   * `SCENE_FRAME_BYTES + k * SCENE_UNIFORM_STRIDE`. A draw binding offset 0
   * would read the camera as a world matrix — an in-range read, a successful
   * draw, and every object at the wrong place.
   *
   * Instanced meshes are one item, not `instanceCount` of them. The count rides
   * on the item, so a thousand copies of a cube cost a thousand sphere tests, a
   * thousand uniform slots, and **one** `drawIndexed`. The per-instance transforms
   * themselves are vertex attributes read from `node.instanceBuffer` — see
   * `MeshNode.instanceCount` for why that count is read per frame rather than
   * captured.
   *
   * The camera's frustum is extracted once, into module scratch, and every
   * node is tested against it. Nothing here allocates, on any frame, ever:
   * a scene that grew to 100,000 objects would still allocate zero.
   *
   * ## One pass, and there is nothing left to fuse
   *
   * Visibility, layer, emptiness and the frustum test are all decided on the
   * node as the walk reaches it, and a node that fails any of them is never
   * pushed. There is no candidate list, no filter pass, and no second array:
   * an absent node costs the walk exactly one `continue`, where an
   * `if (it.visible) out.push(it)` filter would cost a second traversal of
   * every item and a second array to build. The only pass `out` sees is the
   * sort the renderer runs over it afterwards.
   */
  collectDrawItems(out: DrawItem[], camera: Camera): DrawItem[] {
    updateWorldMatrices(this.root);
    camera.getFrustum(_frustum);
    out.length = 0;

    // Hoisted out of the per-item path. `camera.view` is a stable array; see
    // the depth note below for what the per-item path was re-deriving from it.
    const view = camera.view;
    // `transformPoint` divides by the homogeneous `w`, and for an affine view
    // matrix `w` is the constant `view[15]` — so the reciprocal is the same
    // number every time, and computing it once takes 100,000 divisions out of a
    // 100,000-object frame. The three zeros are the affine test, also once per
    // frame: if any of them is set the fast path below is skipped entirely.
    const affineView = view[3] === 0 && view[7] === 0 && view[11] === 0;
    const viewW = 1 / view[15];

    const stack = _collectStack;
    stack.length = 0;
    stack.push(this.root);
    let meshNodes = 0;
    let culled = 0;
    let empty = 0;
    // `out` is sized to the pool's high-water mark before the walk rather than
    // truncated and appended to. The pool never shrinks, so its length is an
    // upper bound on this frame's draw count, and the fill below is therefore
    // always in range.
    //
    // This is worth a comment because the two forms look identical and are not:
    // `out.length = 0` followed by n `push`es measures 3.2 ns per element against
    // 0.72 ns for the same n stores into an array that is already long enough,
    // on this machine, at 10k and at 100k alike. The truncation is free; the
    // *appends* are what cost, because every one of them re-enters the array's
    // growth path. At 100,000 objects that is 0.25 ms of the frame that has
    // nothing to do with any of the work being done.
    const capacity = this.#pool.length;
    if (out.length < capacity) out.length = capacity;
    let count = 0;
    const layers = camera.layers;

    while (stack.length > 0) {
      const node = stack.pop()!;
      // A hidden node hides its subtree. A layer mismatch does too, which is
      // what makes `node.layers = 0` a valid way to switch a whole rig off.
      if (!node.visible || (node.layer & layers) === 0) continue;

      if (node instanceof MeshNode) {
        meshNodes++;
        const geometry = node.mesh;
        // Live, every frame — the geometry is the authority on this and a
        // `GpuInstances` transform list is re-uploaded from a moving source.
        const instanceCount = node.instanceCount;
        // Not a cull decision and not a draw: a node with nothing to draw is
        // never submitted, and reporting it as culled would credit the
        // frustum with rejecting something that was never in front of it.
        if (!canDraw((geometry as { indexCount?: number }).indexCount ?? 0, instanceCount)) {
          empty++;
        } else {
          const position = node.worldPosition;
          _sphere.center = position;
          _sphere.radius = node.worldBoundingRadius;
          if (containsSphere(_frustum, _sphere)) {
            // View-space depth, positive in front of the camera. The sort key for
            // back-to-front transparency; opaque items are front-to-back in every
            // engine that cares, and both orders are one comparator apart.
            //
            // Only the `z` row of the view matrix is needed, and only that row is
            // computed. `transformPoint` builds all three output components, stores
            // all three, and this call site then reads one back and throws the
            // other two away — about forty operations and a divide where a dot
            // product will do, once per drawn object, on the frame's hottest loop.
            // The `w` reciprocal is hoisted; the affine test falls back to the
            // general path rather than assuming a view matrix's last row is
            // `(0, 0, 0, 1)`, because `Camera.view` is a writable array and
            // nothing stops a caller putting a projective matrix in it.
            //
            // `Math.fround` is the single-precision rounding `transformPoint`
            // applied when it stored its result, and it is kept on purpose. The
            // depth key is a function of this number all the way down to which
            // of 1024 buckets a draw lands in, so evaluating the dot product at
            // double precision is *more* accurate and produces a *different*
            // value: about one part in 10^8, invisible in a picture and exactly
            // enough to move a depth sitting near a bucket boundary. A sort key
            // is one of the few places where "more accurate" is not the same as
            // "the same". `Math.fround` is bit-identical to a `Float32Array`
            // store and load — `test/perf-graph.test.ts` checks that over the
            // signed zeros, the subnormals, the overflow threshold and `NaN`.
            const depth = Math.fround(affineView
              ? -(view[2] * position[0] + view[6] * position[1] + view[10] * position[2] + view[14]) * viewW
              : -transformPoint(_viewPoint, position, view)[2]);
            out[count] = this.#emit(node, count, geometry, instanceCount, depth);
            count++;
          } else {
            culled++;
          }
        }
      }

      const children = node.children;
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
    }

    // Back to the draw count. Everything above the last write is a leftover
    // reference to a pooled item, invisible to every reader — which walks
    // `out` to its `length`, as all of them do — and the pool holds those items
    // anyway, so truncating retains nothing extra.
    out.length = count;
    this.meshNodeCount = meshNodes;
    this.culledCount = culled;
    this.emptyCount = empty;
    this.#peakDrawCount = count > this.#peakDrawCount ? count : this.#peakDrawCount;
    return out;
  }

  /**
   * The single place a draw item is created, filled, and checked.
   *
   * One function for three reasons, each of which is a bug the moment it is
   * spread out:
   *
   *   1. **The pool.** A reused item is last frame's object. Every field is
   *      assigned unconditionally below, so nothing survives from a previous
   *      frame — a stale `instanceCount` draws the wrong number of copies and a
   *      stale `objectId` puts two objects in one uniform slot, both silently.
   *   2. **The validation.** It runs on the item, once, here. Any second place
   *      that pushed into `out` would be a path that skips it, and a validation
   *      some paths skip is not a validation.
   *   3. **The slot.** `objectId` is the item's own index, so every item has a
   *      unique dynamic-offset slot and the uniform packer can key on it
   *      without a second lookup. One slot per *item*, not per instance: an
   *      instanced draw binds one object block and reads each instance's
   *      transform from the vertex buffer, so a per-instance slot would be a
   *      second copy of the same 124 bytes `instanceCount` times over.
   *
   * `geometry`, `instanceCount` and `depth` are passed in rather than read off
   * the node here. Each of them is already in hand — or has to be computed —
   * somewhere the walk reaches once, and `node.mesh` and `node.instanceCount`
   * are getters, so re-reading them would be two calls per drawn object for
   * values that cannot have changed in between.
   */
  #emit(
    node: MeshNode,
    index: number,
    geometry: DrawItem['geometry'],
    instanceCount: number,
    depth: number,
  ): DrawItem {
    const item = this.#pool[index] ?? (this.#pool[index] = new PooledDrawItem());
    item.objectId = index;
    item.objectOffset = sceneObjectOffset(index);
    // A reference, not a copy. The transform pass writes into this array in
    // place and its identity is stable, so the renderer can pack it whenever it
    // likes and always see the current value.
    item.model = node.world;
    item.worldVersion = node.worldVersion;
    const material = node.material;
    item.phase = material.phase;
    item.order = node.order;
    item.material = material;
    item.geometry = geometry;
    // The count the walk read through the node, so the draw list has one place
    // that knows a `MeshNode` can carry instances and the pooled item never
    // holds a count from the frame before the geometry changed.
    item.instanceCount = instanceCount;
    item.firstInstance = node.firstInstance;
    item.visible = true;
    item.depth = depth;
    assertDrawable(item);
    return item;
  }
}
