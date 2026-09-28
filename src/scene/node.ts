/**
 * Scene graph nodes — and the transform propagation they exist to avoid.
 *
 * ## The pathology this module is written against
 *
 * The obvious design, and the one every mature 3D library shipped for a
 * decade, is a `Node` base class with a `matrixAutoUpdate` flag. Its
 * `updateMatrixWorld` looks like this:
 *
 * ```ts
 * updateMatrixWorld(force) {
 *   if (this.matrixAutoUpdate) this.updateMatrix();   // every frame
 *   this.matrixWorldNeedsUpdate = true;              // ...and always dirty
 *   if (this.matrixWorldNeedsUpdate || force) {
 *     force = true;                                  // <-- unconditional
 *     for (const c of this.children) c.updateMatrixWorld(force);
 *   }
 * }
 * ```
 *
 * The `force = true` line is the whole problem. It says "if I recomputed, then
 * everything below me must recompute too", on the assumption that a parent can
 * only become dirty by moving. The assumption is wrong in the one case that
 * matters: the scene root extends the same class with `matrixAutoUpdate` on, so
 * the root is *always* dirty, so the entire graph is traversed and every matrix
 * is rewritten on every frame — including the nine hundred static nodes the
 * player will never look at twice. A scene that has not changed costs exactly
 * as much as a scene in which everything moves.
 *
 * ## The rule here instead
 *
 * A node's `world` is a pure function of its `local` and its parent's `world`.
 * So it only needs rewriting when one of those two inputs has changed:
 *
 * ```txt
 * ownChanged    = localVersion !== localVersionWritten
 * parentChanged = parentWorldVersion !== parent.worldVersion
 * changed       = ownChanged || parentChanged
 * ```
 *
 * When it has not changed, the entire subtree below it is provably unchanged
 * too — every descendant is a function of this node's `world`, and nothing
 * below it has been invalidated — so the traversal returns without descending.
 * The cost of a static frame is two integer comparisons, not `n` matrix
 * multiplies.
 *
 * ## Two signals, not one, and the second one is the whole trick
 *
 * "Is this node stale?" and "does this node need visiting?" are different
 * questions, and answering them with one flag is the second-order version of
 * the same bug as the first.
 *
 * - `localVersion` — *did this node's own transform change?* Bumped by this
 *   node's setters, and by nothing else. This is the question that decides
 *   whether a world matrix is rewritten.
 * - `subtreeDirty` — *has anything at or below this node changed?* Set on the
 *   node and on every ancestor, by every change anywhere in its subtree. This
 *   is the question that decides whether the traversal descends.
 *
 * The traversal **visits** a node when `subtreeDirty` is set, and **writes** it
 * when `ownChanged || parentChanged`. A moving leaf therefore marks five nodes
 * on its way to the root, and the traversal visits those five. It writes one.
 *
 * Collapsing the two flags into one — the obvious first cut, and what the
 * `localVersion`-bumped-on-every-ancestor version does — is what forces a
 * parent to be rewritten when a child moves, and a rewritten parent
 * invalidates *its* children, and the cost of moving one leaf in a thousand-node
 * scene silently becomes a thousand matrix multiplies. The distinction between
 * "this matrix is stale" and "somewhere below me is stale" is the entire
 * difference between O(depth) and O(n).
 *
 * The result: **a static scene costs O(1) per frame; a scene with one moving
 * leaf costs O(depth) node visits and one matrix write.** Both are asserted in
 * `test/scene.test.ts` by counting real visits and real writes into real
 * `Float32Array`s, not by timing.
 *
 * ## Float equality is never used
 *
 * Dirty tracking is version-based, never value-based. Comparing sixteen floats
 * per node per frame to discover that nothing moved costs as much as just
 * writing the matrix, and it cannot distinguish "moved and came back" from
 * "never moved" — a transform animated out and back would be recorded as static
 * and the two would then disagree about the next frame. The setters
 * ({@link Node.setPosition} and friends) are the dirty-tracking mechanism: they
 * mutate the exposed typed arrays **in place** — the reference never changes,
 * because the uniform packer holds it — and bump the version. Nothing here
 * diffs a float.
 *
 * ## Direct writes
 *
 * `local`, `position`, `rotation` and `scale` are live views, not snapshots,
 * and writing into them directly is supported. It is your responsibility to
 * call {@link Node.markDirty}, exactly as you would after mutating a
 * `Float32Array` you handed to a GPU yourself. A write without a `markDirty()`
 * is silently not picked up — that is the one rule this module asks you to
 * remember, and it is the same rule WebGPU itself imposes on `writeBuffer`
 * inputs.
 */

import { fail } from '../core/error.ts';
import { fromQuat, fromScale, mul } from '../math/mat4.ts';
import type { Drawable, DrawableGeometry } from '../render/types.ts';

/**
 * The default layer mask. Four bits, so `0b1111` covers the "layer 0..3"
 * setups almost every scene starts with. See {@link Node.layer}.
 */
export const DEFAULT_LAYER = 0b1111;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

/**
 * Monotonic traversal clock.
 *
 * Incremented once per {@link updateWorldMatrices} pass and stamped into every
 * node whose `world` the pass rewrites. A node carrying the current `clock` in
 * `worldVersion` was written by the pass in flight, which makes a stale write
 * from an earlier frame detectable by identity rather than by comparison.
 */
let clock = 0;

/**
 * Monotonic count of invalidations anywhere in any graph. Bumped by every
 * transform or structural change. Distinct from `clock`, which ticks per frame:
 * this one ticks per *change*, so "did anything move since I last looked?" is a
 * single integer compare.
 */
let graphRevision = 0;

/** Count of `world` array writes since the last reset. Diagnostic. */
let transformWrites = 0;

/** Count of nodes examined by the last {@link updateWorldMatrices} pass. */
let nodeVisits = 0;

/** World matrices written since the last {@link resetTransformWriteCount}. */
export function getTransformWriteCount(): number {
  return transformWrites;
}

/** Zeroes the write counter. Call before a frame you want to measure. */
export function resetTransformWriteCount(): void {
  transformWrites = 0;
}

/**
 * Nodes examined by the last transform pass, including the single node whose
 * flags caused a whole subtree to be pruned.
 *
 * The companion to {@link getTransformWriteCount}, and the one that measures
 * the *traversal* rather than the arithmetic. A moving leaf in a deep chain
 * shows up here as `depth` — the five nodes on the path to the root — and
 * nowhere in the write count, because only the leaf's matrix actually changed.
 */
export function getNodeVisitCount(): number {
  return nodeVisits;
}

/** Zeroes the visit counter. Call before a frame you want to measure. */
export function resetNodeVisitCount(): void {
  nodeVisits = 0;
}

/** Invalidations since the process started. See {@link graphRevision}. */
export function getGraphRevision(): number {
  return graphRevision;
}

// ---------------------------------------------------------------------------
// Scratch — module-level, reused forever. Nothing here allocates per frame.
// ---------------------------------------------------------------------------

const _rs = new Float32Array(16);
const _ss = new Float32Array(16);
const _tmp = new Float32Array(16);

/**
 * Explicit stack for {@link updateWorldMatrices}. Grows, never shrinks.
 *
 * Dedicated, and not borrowed from {@link borrowStack}, because the matrix
 * pass invokes no user code: it cannot be re-entered, so it cannot clobber a
 * walk that a callback is in the middle of.
 */
const _updateStack: Node[] = [];

/**
 * A pool of walk stacks, so a hierarchy walk is allocation-free *and*
 * re-entrant. A callback that mutates the graph, or that starts its own walk,
 * borrows the next one instead of overwriting the outer walk's.
 */
const _stackPool: Node[][] = [];
let _stackDepth = 0;

function borrowStack(): Node[] {
  const stack = _stackPool[_stackDepth] ?? (_stackPool[_stackDepth] = []);
  stack.length = 0;
  _stackDepth++;
  return stack;
}

function releaseStack(): void {
  _stackDepth--;
}

/**
 * The largest axis scale of a column-major matrix: the length of its longest
 * basis vector. This is the factor a sphere must be scaled by to still contain
 * its world-space image under a non-uniform scale.
 */
function maxAxisScale(m: Float32Array): number {
  const x = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
  const y = m[4] * m[4] + m[5] * m[5] + m[6] * m[6];
  const z = m[8] * m[8] + m[9] * m[9] + m[10] * m[10];
  const longest = x > y ? (x > z ? x : z) : (y > z ? y : z);
  return Math.sqrt(longest);
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface NodeOptions {
  /** Shown in errors and tooling. Defaults to `'node'`. */
  name?: string;
  /** Local translation. Defaults to the origin. */
  position?: readonly number[];
  /** Local rotation as a quaternion `[x, y, z, w]`. Defaults to identity. */
  rotation?: readonly number[];
  /** Local scale. Defaults to `[1, 1, 1]`. */
  scale?: readonly number[];
  /** Layer bitmask. Defaults to {@link DEFAULT_LAYER}. */
  layer?: number;
  /** Default visibility. Defaults to `true`. */
  visible?: boolean;
  /**
   * Half-extent of the bounding sphere around the local origin, in local
   * units. Defaults to 0, which means "no bounds of its own" — correct for a
   * group node, wrong for a mesh. `MeshNode` defaults to 1.
   */
  boundingRadius?: number;
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

export class Node {
  name: string;

  /**
   * The transform relative to the parent, column-major, `T * R * S`.
   *
   * A stable reference: setters write into this array and never replace it, so
   * a uniform packer that captured `node.local` last frame is still looking at
   * live data. Write into it directly and call {@link markDirty}.
   */
  readonly local: Float32Array;

  /** The transform in world space. Written by {@link updateWorldMatrices}. */
  readonly world: Float32Array;

  /** Bumped on every write to {@link world}. See the {@link worldVersion} getter. */
  #worldVersion = 0;

  /** Live view of the local translation, kept in sync with `local[12..14]`. */
  readonly position: Float32Array;

  /** Live view of the local rotation quaternion `[x, y, z, w]`. */
  readonly rotation: Float32Array;

  /** Live view of the local scale. */
  readonly scale: Float32Array;

  /** The world-space origin of this node, refreshed on every world write. */
  readonly worldPosition: Float32Array;

  /**
   * This node's own transform revision, bumped by this node's setters and by
   * nothing else. Compared against {@link localVersionWritten} to decide
   * whether `world` must be rewritten — which is the question this field is
   * for, and the reason it is never bumped on a node's behalf by a descendant.
   */
  localVersion = 0;

  /** `localVersion` as of the last write to {@link world}. */
  localVersionWritten = -1;

  /**
   * Set when this node or anything below it has changed, and cleared by the
   * transform pass once the node has been examined.
   *
   * This is the traversal's "keep going" signal, as opposed to `localVersion`,
   * which is the "this matrix is stale" signal. Merging them is what makes a
   * static subtree cost O(n) again.
   */
  subtreeDirty = false;

  /**
   * Increments every time {@link world} is written.
   *
   * The renderer uses this to avoid re-uploading an object whose transform has
   * not changed. Combined with the identity of the `world` array itself — which
   * is unique per node and never replaced — this is a free, correct invalidation
   * key. The version alone would not be: it comes from a global clock, so two
   * different nodes written in the same pass carry the same token with entirely
   * different matrices.
   */
  get worldVersion(): number { return this.#worldVersion; }

  /** Advances the token. Used by the transform pass, which numbers its own. */
  bumpWorldVersion(): void { this.#worldVersion++; }

  /** The transform pass assigns the clock directly, to keep tokens monotonic. */
  setWorldVersion(v: number): void { if (v > this.#worldVersion) this.#worldVersion = v; }

  /** The parent's {@link worldVersion} as of this node's last world write. */
  parentWorldVersion = 0;

  /**
   * Value of the global clock when {@link world} was last written. Within the
   * transform pass a write only happens when the content actually changed, so
   * this doubles as *when this node's world transform last changed value* — and
   * that is what makes `parentWorldVersion === parent.worldVersion` a sound
   * test of "my parent has not moved". A matrix rewritten to the same bytes
   * never gets a new token, so it never invalidates a child.
   */

  /** Half-extent of the local bounding sphere, in local units. */
  boundingRadius: number;

  /**
   * Half-extent of the world bounding sphere: {@link boundingRadius} times the
   * largest axis scale of the world matrix. Refreshed on every world write.
   */
  worldBoundingRadius = 0;

  /** Skipped by the cull pass, along with its whole subtree. */
  visible: boolean;

  /** Layer bitmask. A node is drawn only if `layer & camera.layers` is set. */
  layer: number;

  /** True once {@link freeze} has been called. */
  frozen = false;

  #parent: Node | null = null;
  #children: Node[] = [];

  constructor(options: NodeOptions | string = {}) {
    const opts: NodeOptions = typeof options === 'string' ? { name: options } : options;
    this.name = opts.name ?? 'node';
    this.local = new Float32Array(16);
    this.world = new Float32Array(16);
    this.position = new Float32Array(3);
    this.rotation = new Float32Array(4);
    this.scale = new Float32Array(3);
    this.worldPosition = new Float32Array(3);
    this.boundingRadius = opts.boundingRadius ?? 0;
    this.visible = opts.visible ?? true;
    this.layer = opts.layer ?? DEFAULT_LAYER;

    this.scale[0] = 1;
    this.scale[1] = 1;
    this.scale[2] = 1;
    this.rotation[3] = 1;
    this.local[0] = 1;
    this.local[5] = 1;
    this.local[10] = 1;
    this.local[15] = 1;

    if (opts.position !== undefined) this.setPosition(opts.position[0], opts.position[1], opts.position[2]);
    if (opts.rotation !== undefined) this.setQuaternion(opts.rotation[0], opts.rotation[1], opts.rotation[2], opts.rotation[3]);
    if (opts.scale !== undefined) this.setScale(opts.scale[0], opts.scale[1], opts.scale[2]);

    // `world` is seeded so that reading a freshly built node before any pass is
    // harmless, but `localVersionWritten` is deliberately left at −1: a new node
    // is stale until the traversal has actually resolved it. The pass costs one
    // write to settle that, once, and in exchange nothing in the graph is ever
    // clean on the strength of a matrix nobody ran the maths for.
    this.world.set(this.local);
    this.#worldVersion++;
    this.worldPosition.set(this.position);
  }

  // -------------------------------------------------------------------------
  // Hierarchy
  // -------------------------------------------------------------------------

  /** The parent, or `null` for a root or a detached node. */
  get parent(): Node | null {
    return this.#parent;
  }

  /**
   * The live child list, in insertion order. Do not mutate it — `add`,
   * `remove` and `clear` exist for that, and writing to this array directly
   * bypasses the invalidation the traversal depends on.
   */
  get children(): Node[] {
    return this.#children;
  }

  /** This node plus every descendant. */
  get descendantCount(): number {
    const stack = borrowStack();
    let n = 0;
    try {
      stack.push(this);
      while (stack.length > 0) {
        const node = stack.pop()!;
        n++;
        const children = node.children;
        for (let i = 0; i < children.length; i++) stack.push(children[i]);
      }
    } finally {
      releaseStack();
    }
    return n;
  }

  /**
   * Attaches `child`, making it a child of this node.
   *
   * ## `add` is explicit about re-parenting
   *
   * If `child` already has a parent this throws rather than silently moving it.
   * Implicit re-parenting is how a scene acquires a node that two owners both
   * believe they have: after a move, the old parent keeps drawing,
   * transforming, and culling a child it can no longer see, and nothing
   * reports it. The fix is one call — `child.removeFromParent()` — and it is
   * worth making the author of the moving code say so.
   */
  add(child: Node): this {
    // The cycle check comes first, so re-adding an ancestor of this node
    // reports the cycle rather than the re-parenting it also violates.
    // `child.isAncestorOf(this)` walks up from the child: the question is
    // whether *this* node sits somewhere above the child, not the other way
    // round. Getting that backwards is silent, and the graph it produces has no
    // root to stop an ancestor walk at.
    if (child === this || child.isAncestorOf(this)) {
      fail('NODE_CYCLE',
        `Cannot add "${child.name}" to "${this.name}": it is already an ancestor of it.`, {
        why: 'A scene graph is a tree, not a graph. Attaching a node to one of its own descendants would make it its own parent, and transform resolution would recurse forever.',
        fix: 'Detach first: `child.removeFromParent()`, then `parent.add(child)`. To give a second node the same transform, build a second node — sharing one node between two parents is impossible by design.',
      });
    }
    if (child.#parent !== null) {
      const from = child.#parent.name;
      fail('NODE_REPARENTED',
        `Cannot add "${child.name}" to "${this.name}": it is already a child of "${from}".`, {
        why: 'Attaching a node that already has a parent would leave the old parent holding a child it still draws, transforms, and culls. Neither owner could tell which one it is, and the node would be rendered twice or under a stale transform.',
        fix: 'Detach from the current parent first: `child.removeFromParent()`, then add.',
      });
    }

    child.#parent = this;
    this.#children.push(child);
    // Structural, therefore unconditional: the child, its whole subtree, and
    // this node's ancestor chain. Without the last one a clean ancestor would
    // prune straight past the node that was just attached.
    child.#invalidate(true, true);
    return this;
  }

  /** Detaches `child`. Returns `true` if it was a child of this node. */
  remove(child: Node): boolean {
    const index = this.#children.indexOf(child);
    if (index < 0) return false;
    this.#children.splice(index, 1);
    child.#detach();
    return true;
  }

  /** Detaches every child. */
  clear(): void {
    for (const child of this.#children) child.#detach();
    this.#children.length = 0;
  }

  /** Detaches from the current parent. Returns the old parent, or `null`. */
  removeFromParent(): Node | null {
    const parent = this.#parent;
    if (parent === null) return null;
    const index = parent.#children.indexOf(this);
    if (index >= 0) parent.#children.splice(index, 1);
    this.#detach();
    return parent;
  }

  /**
   * Throws unless this node is attached to a scene.
   *
   * For operations whose whole subject is the scene — its world transform, its
   * place in the draw list — a detached node is silently wrong, and
   * `NODE_NOT_ATTACHED` names the fix.
   */
  assertAttached(): this {
    if (this.#parent === null) {
      fail('NODE_NOT_ATTACHED',
        `"${this.name}" is not attached to a scene, so it has no world transform.`, {
        why: 'World matrices are resolved by walking down from the scene root. A detached node is never visited, so its `world` holds whatever the last attachment left behind.',
        fix: 'Add it first: `scene.add(node)`.',
      });
    }
    return this;
  }

  /** True when `other` is this node or one of its ancestors. */
  isAncestorOf(other: Node): boolean {
    for (let n: Node | null = other; n !== null; n = n.#parent) {
      if (n === this) return true;
    }
    return false;
  }

  /**
   * Walks from this node up to the root, `this` first. Return `false` from `fn`
   * to stop. Iterative — hierarchy depth is not bounded by the JS call stack.
   */
  traverseUp(fn: (node: Node) => boolean | void): void {
    const stack = borrowStack();
    try {
      stack.push(this);
      while (stack.length > 0) {
        const node = stack.pop()!;
        if (fn(node) === false) return;
        const parent = node.parent;
        if (parent !== null) stack.push(parent);
      }
    } finally {
      releaseStack();
    }
  }

  /**
   * Pre-order walk of this node and its descendants. Return `false` from `fn`
   * to skip that node's subtree. Iterative — a 10,000-deep chain will not
   * overflow the stack, and nothing is allocated per call.
   *
   * The callback may itself walk or mutate the graph: {@link borrowStack}
   * hands re-entrant walks their own storage.
   */
  traverseDown(fn: (node: Node) => boolean | void): void {
    const stack = borrowStack();
    try {
      stack.push(this);
      while (stack.length > 0) {
        const node = stack.pop()!;
        if (fn(node) === false) continue;
        const children = node.children;
        for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
      }
    } finally {
      releaseStack();
    }
  }

  // -------------------------------------------------------------------------
  // Transform
  // -------------------------------------------------------------------------

  /** Local translation, in the parent's units. Bumps {@link localVersion}. */
  setPosition(x: number, y: number, z: number): this {
    if (this.frozen) return this;
    const l = this.local;
    l[12] = x;
    l[13] = y;
    l[14] = z;
    this.position[0] = x;
    this.position[1] = y;
    this.position[2] = z;
    this.#invalidate(false, false);
    return this;
  }

  /** Adds `dx, dy, dz` to the local translation. */
  translate(dx: number, dy: number, dz: number): this {
    if (this.frozen) return this;
    const l = this.local;
    return this.setPosition(l[12] + dx, l[13] + dy, l[14] + dz);
  }

  /**
   * Local rotation as a quaternion `[x, y, z, w]`. Not normalised — the caller
   * owns that, and a non-unit quaternion is a caller bug that should be visible
   * rather than silently papered over.
   */
  setQuaternion(x: number, y: number, z: number, w: number): this {
    if (this.frozen) return this;
    const q = this.rotation;
    q[0] = x;
    q[1] = y;
    q[2] = z;
    q[3] = w;
    this.#composeBasis();
    this.#invalidate(false, false);
    return this;
  }

  /**
   * Local scale. `setScale(2)` is uniform; `setScale(x, y, z)` is per axis.
   *
   * Recomposes the basis, so the quaternion and the scale stay the single
   * source of truth for the 3×3.
   */
  setScale(x: number, y?: number, z?: number): this {
    if (this.frozen) return this;
    const s = this.scale;
    s[0] = x;
    s[1] = y ?? x;
    s[2] = z ?? x;
    this.#composeBasis();
    this.#invalidate(false, false);
    return this;
  }

  /**
   * Replaces the local transform wholesale. Copies the sixteen floats in place
   * and decomposes them back into {@link position}, {@link rotation} and
   * {@link scale} so the accessors keep telling the truth.
   *
   * The decomposition assumes `T * R * S` with a positive determinant, i.e. a
   * basis that is a rotation composed with a scale. A matrix carrying a
   * negative scale — a mirror — decomposes to a positive scale plus a rotation
   * that contains the flip; the matrix itself is always preserved exactly, so
   * this is a statement about the accessors, not about `local`.
   */
  setLocalMatrix(matrix: ArrayLike<number>): this {
    if (this.frozen) return this;
    const l = this.local;
    for (let i = 0; i < 16; i++) l[i] = matrix[i] ?? l[i];
    this.#decompose();
    this.#invalidate(false, false);
    return this;
  }

  /**
   * Declares this transform changed: bumps this node's {@link localVersion} and
   * sets {@link subtreeDirty} here and on every ancestor, which is what lets the
   * traversal find it and prune everything else.
   *
   * Also re-derives {@link position}, {@link rotation} and {@link scale} from
   * `local`, so a direct write to the array followed by this call is
   * indistinguishable from having used a setter.
   */
  markDirty(): void {
    if (this.frozen) return;
    this.#decompose();
    this.#invalidate(false, false);
  }

  /**
   * Declares this node's whole subtree changed — for re-attaching a subtree
   * that already had a parent somewhere else.
   *
   * Structural in strength, so it ignores {@link freeze} and forces a rewrite
   * of every node below: a frozen *transform* is not a frozen *position in the
   * tree*, and pruning a re-parented subtree would leave every one of them
   * showing a world matrix computed against the old parent.
   */
  markSubtreeDirty(): void {
    this.#invalidate(true, true);
  }

  /**
   * Permanently stops this node from being invalidated by transform changes.
   *
   * A static scene is already O(1) per frame without this, so `freeze` is an
   * optimisation and never a requirement. What it buys on a large static graph
   * is that a mutation elsewhere cannot propagate down into it at all: a
   * frozen region is provably inert rather than merely unmoved, and the
   * ancestor-chain walk shortens the moment a change happens next to it.
   *
   * After `freeze`, {@link setPosition}, {@link setQuaternion},
   * {@link setScale}, {@link setLocalMatrix} and {@link markDirty} return
   * without touching anything.
   *
   * Two things still update a frozen node, because leaving them stale would be
   * a rendering bug rather than a performance win:
   *   - **its world matrix, whenever an ancestor's moves.** The traversal
   *     reaches a frozen node whenever its parent's `worldVersion` has changed,
   *     independently of any version bump, so it follows its parent correctly
   *     with no bookkeeping at all.
   *   - **its whole subtree, on a structural change** (`add` / `remove`).
   */
  freeze(): this {
    this.frozen = true;
    return this;
  }

  /** Undoes {@link freeze}. */
  unfreeze(): this {
    this.frozen = false;
    return this;
  }

  /**
   * Resolves this node's `world` now, against whatever parent it currently
   * has, and marks the subtree so the next pass agrees.
   *
   * {@link updateWorldMatrices} does this in bulk; reach for it directly for a
   * detached subtree, or after writing a parent's `world` array by hand. It
   * always writes, even when the result is identical, so prefer the batch pass
   * unless you specifically have a single node in hand.
   */
  updateWorldMatrix(): this {
    const parent = this.#parent;
    if (parent === null) {
      this.world.set(this.local);
    } else {
      mul(this.world, parent.world, this.local);
    }
    this.worldPosition[0] = this.world[12];
    this.worldPosition[1] = this.world[13];
    this.worldPosition[2] = this.world[14];
    this.worldBoundingRadius = this.boundingRadius * maxAxisScale(this.world);
    this.localVersionWritten = this.localVersion;
    this.parentWorldVersion = parent === null ? 0 : parent.worldVersion;
    // A fresh world token. Because this is the only path that assigns one, a
    // token change means the content changed, which is what stops a needlessly
    // rewritten parent from invalidating its children.
    this.bumpWorldVersion();
    transformWrites++;
    // A new world token is a new parent token, which is how the children below
    // learn to recompute without anybody touching their own flags.
    this.subtreeDirty = true;
    return this;
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /**
   * The single invalidation primitive.
   *
   * `subtree` also invalidates every descendant; `unconditional` ignores
   * {@link freeze}, which is what a structural change needs — a frozen
   * *transform* is not a frozen *position in the tree*, and pruning a
   * re-parented subtree would leave it showing its previous parent's world
   * matrix.
   *
   * Note what is and is not frozen here. The `subtreeDirty` walk up the
   * ancestors always runs, even through frozen nodes, because a frozen node
   * must not be able to *hide* a change happening below it from the traversal.
   * What a frozen node suppresses is its own `localVersion` bump — its matrix
   * is not rewritten — and the traversal still reaches it whenever its parent
   * moves, which is handled by `parentWorldVersion` and needs no flag at all.
   */
  #invalidate(subtree: boolean, unconditional: boolean): void {
    if (!unconditional && this.frozen) return;
    this.localVersion++;
    this.subtreeDirty = true;
    if (subtree) {
      const stack = borrowStack();
      try {
        stack.push(this);
        while (stack.length > 0) {
          const node = stack.pop()!;
          if (unconditional || !node.frozen) node.localVersion++;
          node.subtreeDirty = true;
          const children = node.children;
          for (let i = 0; i < children.length; i++) stack.push(children[i]);
        }
      } finally {
        releaseStack();
      }
    }
    graphRevision++;
    // Cheap and usually O(1): an already-dirty ancestor implies every ancestor
    // above it is dirty too, so the walk can stop there.
    for (let p = this.#parent; p !== null; p = p.#parent) {
      if (p.subtreeDirty) break;
      p.subtreeDirty = true;
    }
  }

  /** Rewrites the 3×3 of `local` from `rotation` and `scale`. */
  #composeBasis(): void {
    fromQuat(_rs, this.rotation);
    fromScale(_ss, this.scale[0], this.scale[1], this.scale[2]);
    mul(_tmp, _rs, _ss);
    const l = this.local;
    for (let i = 0; i < 12; i++) l[i] = _tmp[i];
  }

  /** Recovers position, rotation and scale from the 3×3 and the translation. */
  #decompose(): void {
    const l = this.local;
    this.position[0] = l[12];
    this.position[1] = l[13];
    this.position[2] = l[14];

    const sx = Math.sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
    const sy = Math.sqrt(l[4] * l[4] + l[5] * l[5] + l[6] * l[6]);
    const sz = Math.sqrt(l[8] * l[8] + l[9] * l[9] + l[10] * l[10]);
    const s = this.scale;
    s[0] = sx;
    s[1] = sy;
    s[2] = sz;

    // A zero axis has no direction to recover. Leave the rotation alone rather
    // than dividing by zero and writing NaN into a uniform.
    if (sx === 0 || sy === 0 || sz === 0) return;
    const r = this.rotation;
    const m00 = l[0] / sx, m10 = l[1] / sx, m20 = l[2] / sx;
    const m01 = l[4] / sy, m11 = l[5] / sy, m21 = l[6] / sy;
    const m02 = l[8] / sz, m12 = l[9] / sz, m22 = l[10] / sz;
    const trace = m00 + m11 + m22;
    let x: number, y: number, z: number, w: number;
    if (trace > 0) {
      const k = Math.sqrt(trace + 1) * 2;
      w = 0.25 * k;
      x = (m21 - m12) / k;
      y = (m02 - m20) / k;
      z = (m10 - m01) / k;
    } else if (m00 > m11 && m00 > m22) {
      const k = Math.sqrt(1 + m00 - m11 - m22) * 2;
      w = (m21 - m12) / k;
      x = 0.25 * k;
      y = (m01 + m10) / k;
      z = (m02 + m20) / k;
    } else if (m11 > m22) {
      const k = Math.sqrt(1 + m11 - m00 - m22) * 2;
      w = (m02 - m20) / k;
      x = (m01 + m10) / k;
      y = 0.25 * k;
      z = (m12 + m21) / k;
    } else {
      const k = Math.sqrt(1 + m22 - m00 - m11) * 2;
      w = (m10 - m01) / k;
      x = (m02 + m20) / k;
      y = (m12 + m21) / k;
      z = 0.25 * k;
    }
    const inv = 1 / Math.sqrt(x * x + y * y + z * z + w * w);
    r[0] = x * inv;
    r[1] = y * inv;
    r[2] = z * inv;
    r[3] = w * inv;
  }

  /** Unlinks from the parent and leaves `world` self-consistent again. */
  #detach(): void {
    this.#parent = null;
    // With no parent, the world transform *is* the local one. Writing it here
    // rather than leaving a stale parent-relative matrix means a detached node
    // is still correct on its own terms — and it is the one place this module
    // writes a world matrix outside the traversal.
    this.world.set(this.local);
    this.#worldVersion++;
    this.worldPosition[0] = this.local[12];
    this.worldPosition[1] = this.local[13];
    this.worldPosition[2] = this.local[14];
    this.worldBoundingRadius = this.boundingRadius * maxAxisScale(this.world);
    this.localVersionWritten = this.localVersion;
    this.parentWorldVersion = 0;
    this.bumpWorldVersion();
    this.subtreeDirty = true;
    graphRevision++;
  }
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

/**
 * Resolves every `world` matrix under `root`, writing only the stale ones.
 *
 * This is the whole performance claim of the module, in one function, and it
 * turns on two different questions:
 *
 * ```txt
 *   visit this node?  subtreeDirty || parentChanged
 *   write its world?  ownChanged || parentChanged
 *   where
 *       ownChanged    = localVersion !== localVersionWritten
 *       parentChanged = parentWorldVersion !== parent.worldVersion
 * ```
 *
 * A node that does not need visiting prunes its whole subtree: everything below
 * it is a function of a `world` that did not change, and nothing below it was
 * invalidated. A node that needs visiting but whose own matrix is unchanged is
 * not written — it is on the path between a change and its owner, and its
 * children are not affected by it. That second line is what makes the cost of a
 * moving leaf `depth` node visits and **one** matrix write, instead of `depth`
 * writes each of which then invalidates a subtree and puts you back at O(n).
 *
 * Cost, therefore:
 *   - nothing moved → two comparisons at the root, and the pass returns;
 *   - one leaf moved → one comparison per node on its path to the root, one
 *     multiply at the leaf;
 *   - a parent moved → one multiply per node in its subtree, because every one
 *     of those matrices genuinely changed.
 *
 * `visible` and `layer` are deliberately ignored here. A hidden subtree still
 * needs correct world matrices, because the moment it is shown again they are
 * read. Visibility is a cull-pass decision and belongs to
 * `src/scene/graph.ts`.
 *
 * The stack is explicit and module-level: hierarchy depth is bounded by the
 * application's memory rather than by the JS call stack, so a 10,000-deep chain
 * resolves without recursing and without allocating.
 *
 * May be called on any node, not only a scene root — a detached subtree
 * resolves against its own.
 */
export function updateWorldMatrices(root: Node): void {
  const pass = ++clock;
  const stack = _updateStack;
  stack.length = 0;
  stack.push(root);
  nodeVisits = 0;
  while (stack.length > 0) {
    const node = stack.pop()!;
    nodeVisits++;
    const parent = node.parent;
    const parentChanged = parent !== null && parent.worldVersion !== node.parentWorldVersion;
    if (!node.subtreeDirty && !parentChanged) continue;
    node.subtreeDirty = false;
    if (node.localVersion !== node.localVersionWritten || parentChanged) {
      if (parent === null) {
        node.world.set(node.local);
        node.bumpWorldVersion();
      } else {
        mul(node.world, parent.world, node.local);
      }
      node.worldPosition[0] = node.world[12];
      node.worldPosition[1] = node.world[13];
      node.worldPosition[2] = node.world[14];
      node.worldBoundingRadius = node.boundingRadius * maxAxisScale(node.world);
      node.localVersionWritten = node.localVersion;
      node.parentWorldVersion = parent === null ? 0 : parent.worldVersion;
      // A fresh token, and only here: because this is the only path that
      // assigns one, a token change means the content changed, which is what
      // keeps a needlessly rewritten parent from invalidating its children.
      node.setWorldVersion(pass);
      transformWrites++;
    }
    const children = node.children;
    for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
  }
}

// ---------------------------------------------------------------------------
// MeshNode
// ---------------------------------------------------------------------------

export interface MeshNodeOptions extends NodeOptions {
  /** The GPU geometry this node draws. */
  mesh: DrawableGeometry;
  /** The material it is drawn with. */
  material: Drawable;
  /** Rendered into the shadow pass. Defaults to `true`. */
  castShadow?: boolean;
  /** Sort key within a phase. Higher draws later. Defaults to 0. */
  order?: number;
}

/**
 * A node that draws something.
 *
 * Everything a {@link Node} does, plus a mesh and a material. The only node
 * type the draw-list builder emits, and the only one ever culled: a group
 * node's own bounding sphere says nothing about what is inside it, so bounding
 * a group by its origin would cull its children.
 */
export class MeshNode extends Node {
  readonly mesh: DrawableGeometry;
  readonly material: Drawable;
  readonly castShadow: boolean;
  readonly order: number;

  /** `1` for a plain mesh, `n` for an instanced draw. */
  readonly instanceCount: number;

  /** First instance index, for instanced draws. */
  readonly firstInstance: number;

  constructor(options: MeshNodeOptions) {
    super({ ...options, boundingRadius: options.boundingRadius ?? 1 });
    this.mesh = options.mesh;
    this.material = options.material;
    this.castShadow = options.castShadow ?? true;
    this.order = options.order ?? 0;
    this.instanceCount = options.mesh.instanceCount;
    this.firstInstance = options.mesh.firstInstance;
  }
}
