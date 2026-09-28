/**
 * Scene graph tests.
 *
 * The first two are the ones that matter. Everything else in this file checks
 * behaviour; those two check the *performance claim*, by counting real writes
 * into real `Float32Array`s rather than by timing. A timing assertion passes on
 * a fast machine with a broken implementation and fails on a slow one with a
 * correct one; a write count is exact.
 *
 * The claim being defended, in the module's own words:
 *
 *   - a static scene costs O(1) per frame, not O(n);
 *   - a scene with one moving leaf costs O(depth of that leaf), not O(n).
 *
 * A regression to the design `node.ts` exists to replace — recompute my own
 * matrix every frame, then force every descendant — fails both.
 */

import { describe, expect, test } from 'bun:test';
import { FRAME_BLOCK, SCENE_UNIFORM_STRIDE, sceneObjectOffset } from '../src/core/slot.ts';
import { Resource } from '../src/core/resource.ts';
import { STANDARD_LAYOUT } from '../src/geometry/layout.ts';
import { mul, transformVec4 } from '../src/math/mat4.ts';
import { compareDrawItems, sortDrawItems } from '../src/render/sort.ts';
import type { Drawable, DrawableGeometry, DrawItem } from '../src/render/types.ts';
import {
  MeshNode,
  Node,
  OBJECT_UNIFORM_STRIDE,
  OrthographicCamera,
  PerspectiveCamera,
  Scene,
  getNodeVisitCount,
  getGraphRevision,
  getTransformWriteCount,
  resetNodeVisitCount,
  resetTransformWriteCount,
} from '../src/scene/index.ts';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/**
 * The scene layer only ever reads `phase` and `name` off a material, and a
 * layout, a couple of buffers and two counts off a geometry. Everything else on
 * those interfaces is GPU state that cannot exist in a headless test, so the
 * fakes are `as unknown as` casts — a deliberate boundary, not a shortcut.
 */
const material = {
  name: 'test',
  phase: 'opaque',
} as unknown as Drawable;

const mesh = {
  name: 'mesh',
  layout: STANDARD_LAYOUT,
  vertexBuffer: null as unknown as GPUBuffer,
  indexBuffer: null,
  indexCount: 36,
  instanceCount: 1,
  firstInstance: 0,
  instanceBuffer: null,
} as unknown as DrawableGeometry;

/**
 * A mesh that carries `count` instances from `first`, optionally with a
 * per-instance vertex buffer. A fresh object each call, so a test can change a
 * count and see whether the draw list notices.
 */
function instancedGeometry(
  count: number,
  first = 0,
  buffer: GPUBuffer | null = null,
): DrawableGeometry {
  return {
    name: `instanced:${count}`,
    layout: STANDARD_LAYOUT,
    vertexBuffer: null as unknown as GPUBuffer,
    indexBuffer: null,
    indexCount: 36,
    instanceCount: count,
    firstInstance: first,
    instanceBuffer: buffer,
  } as unknown as DrawableGeometry;
}

function meshNode(name: string, x = 0, y = 0, z = 0, radius = 0.5): MeshNode {
  return new MeshNode({
    name,
    mesh,
    material,
    position: [x, y, z],
    boundingRadius: radius,
  });
}

function sceneWith(count: number): Scene {
  const scene = new Scene();
  for (let i = 0; i < count; i++) scene.add(meshNode(`m${i}`));
  return scene;
}

/** A camera at the origin looking down −Z, which is the whole scene's front. */
function lookingAtScene(fov = 60, near = 0.1, far = 1000): PerspectiveCamera {
  const camera = new PerspectiveCamera({ fov, near, far, aspect: 1 });
  camera.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]);
  camera.update(1);
  return camera;
}

/** Asserts that `fn` throws an apse error with exactly `code`. */
function expectCode(fn: () => void, code: string): void {
  try {
    fn();
  } catch (error) {
    expect((error as { code?: string }).code).toBe(code);
    expect((error as { name?: string }).name).toBe('AseError');
    return;
  }
  throw new Error(`Expected a throw with code ${code}, but nothing was thrown.`);
}

/** The view-space depths the sort keys on, for asserting on order. */
function depthsOf(items: DrawItem[]): number[] {
  return items.map((item) => item.depth);
}

const _point = new Float32Array(4);
const _clip = new Float32Array(4);

/**
 * The NDC coordinates of the world point `(x, y, z)` under `m`, after the
 * perspective divide. Scratch, so a test can call it in a loop.
 */
function ndc(m: Float32Array, x: number, y: number, z: number): [number, number, number] {
  _point[0] = x;
  _point[1] = y;
  _point[2] = z;
  _point[3] = 1;
  transformVec4(_clip, _point, m);
  const w = _clip[3];
  return [_clip[0] / w, _clip[1] / w, _clip[2] / w];
}

// ---------------------------------------------------------------------------
// The performance claim
// ---------------------------------------------------------------------------

describe('transform write count', () => {
  test('a static scene of 1000 nodes writes 1000 matrices on frame 1 and 0 on frame 2', () => {
    const scene = sceneWith(999);
    // Root plus 999 children: exactly 1000 nodes, every one of which needs its
    // world matrix resolved once.
    expect(scene.root.descendantCount).toBe(1000);

    const camera = lookingAtScene();
    const items: DrawItem[] = [];

    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(1000);
    expect(getNodeVisitCount()).toBe(1000);

    // Nothing moved. This is the assertion the module exists for: the traversal
    // compares two integers at the root, prunes the whole graph, and returns.
    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(0);
    expect(getNodeVisitCount()).toBe(1);

    // And it stays zero, rather than being a one-off.
    resetTransformWriteCount();
    scene.collectDrawItems(items, camera);
    scene.collectDrawItems(items, camera);
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(0);
  });

  test('one moving leaf writes 1 matrix in a 1000-node scene, and leaves its ancestors alone', () => {
    // A chain of exactly five: root → a → b → c → leaf, the leaf being a mesh.
    // Its four ancestors are a, b, c and the root.
    const scene = new Scene();
    const a = new Node({ name: 'a' });
    const b = new Node({ name: 'b' });
    const c = new Node({ name: 'c' });
    const leaf = meshNode('leaf', 0, 0, -5);
    a.add(b);
    b.add(c);
    c.add(leaf);
    scene.add(a);

    // ...plus 995 static siblings, so the write count below is a statement about
    // the scene and not about the size of the moving branch. Root, a, b, c, leaf
    // and 995 fillers is exactly 1000 nodes.
    for (let i = 0; i < 995; i++) scene.add(meshNode(`static${i}`));
    expect(scene.root.descendantCount).toBe(1000);

    const camera = lookingAtScene();
    const items: DrawItem[] = [];

    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(1000);
    expect(getNodeVisitCount()).toBe(1000);

    const ancestorTokens = [scene.root.worldVersion, a.worldVersion, b.worldVersion, c.worldVersion];
    leaf.setPosition(0, 0, -6);

    resetTransformWriteCount();
    scene.collectDrawItems(items, camera);
    // One matrix written: the leaf's. The four ancestors are on the path between
    // the change and the root, so the traversal visits them, but their world
    // matrices are byte-identical to what is already there. Writing a parent
    // whose content did not change would hand it a new world token and so
    // invalidate all 995 of its other children — which is precisely how one
    // moving leaf becomes a 1000-write frame.
    expect(getTransformWriteCount()).toBe(1);
    // Proof that the ancestors were not rewritten: their tokens are unchanged.
    expect([scene.root.worldVersion, a.worldVersion, b.worldVersion, c.worldVersion])
      .toEqual(ancestorTokens);

    // The world matrices are right, not just the counts.
    expect(leaf.worldPosition[2]).toBeCloseTo(-6, 5);
  });

  test('moving a branch rewrites only that branch', () => {
    const scene = new Scene();
    const branch = new Node({ name: 'branch' });
    const leaf = meshNode('leaf', 0, 0, -5);
    branch.add(leaf);
    scene.add(branch);
    for (let i = 0; i < 100; i++) scene.add(meshNode(`filler${i}`));

    const camera = lookingAtScene();
    const items: DrawItem[] = [];
    scene.collectDrawItems(items, camera);

    branch.setPosition(1, 0, 0);
    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems(items, camera);
    // Two writes: the branch and the leaf it carries.
    expect(getTransformWriteCount()).toBe(2);
    // 103 examinations: the root, its 101 children, the branch and its child.
    // Each of the 100 fillers is looked at once and pruned on a single flag
    // read — that is the difference from the design being replaced, where
    // looking at it would have cost a 4×4 multiply.
    expect(getNodeVisitCount()).toBe(103);
    expect(leaf.worldPosition[0]).toBeCloseTo(1, 5);
  });

  test('a frozen node does not follow a transform setter, but does follow its parent', () => {
    const scene = new Scene();
    const parent = new Node({ name: 'parent' });
    const child = meshNode('child', 0, 0, -5);
    parent.add(child);
    scene.add(parent);

    const camera = lookingAtScene();
    const items: DrawItem[] = [];
    scene.collectDrawItems(items, camera);

    child.freeze();
    child.setPosition(0, 0, -50);
    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(0);
    expect(getNodeVisitCount()).toBe(1);
    expect(child.worldPosition[2]).toBeCloseTo(-5, 5);

    // A frozen *transform* is not a frozen *world position*: the child still
    // has to follow a moving parent, and it does so with no bookkeeping of its
    // own at all, because the traversal reaches it whenever the parent's
    // worldVersion has moved on.
    parent.setPosition(0, 10, 0);
    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(2);
    expect(child.worldPosition[1]).toBeCloseTo(10, 5);
    expect(child.worldPosition[2]).toBeCloseTo(-5, 5);
  });

  test('setLocalMatrix decomposes back into the accessors, and 180° survives', () => {
    const node = new Node({ name: 'n' });
    // A scale and a 90° turn about Y: sin(45°) = √2/2.
    const h = Math.SQRT1_2;
    node.setQuaternion(0, h, 0, h);
    node.setScale(2, 3, 4);
    expect(node.scale[1]).toBeCloseTo(3, 5);

    // Round-trip the matrix that the setters just built.
    const matrix = Float32Array.from(node.local);
    node.setLocalMatrix(matrix);
    expect(node.rotation[0]).toBeCloseTo(0, 5);
    expect(node.rotation[1]).toBeCloseTo(h, 5);
    expect(node.rotation[2]).toBeCloseTo(0, 5);
    expect(node.rotation[3]).toBeCloseTo(h, 5);
    expect(node.scale[0]).toBeCloseTo(2, 5);
    expect(node.scale[1]).toBeCloseTo(3, 5);
    expect(node.scale[2]).toBeCloseTo(4, 5);

    // The trace-based branches, where the naive formula divides by zero: half
    // turns about each axis.
    for (const q of [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]) {
      const n = new Node({ name: 'half' });
      n.setQuaternion(q[0], q[1], q[2], q[3]);
      const m = Float32Array.from(n.local);
      const back = new Node({ name: 'back' });
      back.setLocalMatrix(m);
      expect(back.rotation[3]).toBeCloseTo(0, 4);
      const expected = Math.abs(q[0]) + Math.abs(q[1]) + Math.abs(q[2]);
      expect(
        Math.abs(back.rotation[0]) + Math.abs(back.rotation[1]) + Math.abs(back.rotation[2]),
      ).toBeCloseTo(expected, 4);
    }

    // A zero scale axis has no direction to recover, and must not produce NaN.
    const flat = new Node({ name: 'flat' });
    flat.setLocalMatrix([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]);
    expect(flat.scale[0]).toBe(0);
    expect(Number.isNaN(flat.rotation[0])).toBe(false);
    expect(flat.position[0]).toBe(5);
  });

  test('markDirty on a node written through directly is what invalidates it', () => {
    const scene = new Scene();
    const node = meshNode('n', 0, 0, -5);
    scene.add(node);
    const camera = lookingAtScene();
    const items: DrawItem[] = [];
    scene.collectDrawItems(items, camera);

    // Write straight into the exposed array, as a physics solver or a loader
    // would. Nothing happens, which is the documented contract.
    node.local[12] = 25;
    resetTransformWriteCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(0);
    expect(node.worldPosition[0]).toBeCloseTo(0, 5);

    node.markDirty();
    resetTransformWriteCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(1);
    expect(node.worldPosition[0]).toBeCloseTo(25, 5);
    // markDirty also re-derives the accessors, so they do not disagree with it.
    expect(node.position[0]).toBeCloseTo(25, 5);
  });
});

// ---------------------------------------------------------------------------
// Hierarchy
// ---------------------------------------------------------------------------

describe('hierarchy', () => {
  test('attaching an ancestor is a cycle', () => {
    const root = new Node({ name: 'root' });
    const mid = new Node({ name: 'mid' });
    const leaf = new Node({ name: 'leaf' });
    root.add(mid);
    mid.add(leaf);

    // `root` is an ancestor of `mid`, and it also already has a parent — the
    // cycle has to win, or the caller is told to detach something that may not
    // be detached.
    expectCode(() => mid.add(root), 'NODE_CYCLE');
    expectCode(() => mid.add(mid), 'NODE_CYCLE');
    expectCode(() => leaf.add(root), 'NODE_CYCLE');
    // The graph is untouched by a rejected attach.
    expect(root.parent).toBeNull();
    expect(mid.parent).toBe(root);
  });

  test('attaching a node that already has a parent is refused, not silently moved', () => {
    const a = new Node({ name: 'a' });
    const b = new Node({ name: 'b' });
    const child = new Node({ name: 'child' });
    a.add(child);
    expectCode(() => b.add(child), 'NODE_REPARENTED');
    expect(child.parent).toBe(a);

    // The documented way through.
    child.removeFromParent();
    b.add(child);
    expect(child.parent).toBe(b);
    expect(a.children).toHaveLength(0);
  });

  test('removing a node from a scene it was never added to names the fix', () => {
    const scene = new Scene({ name: 'main' });
    expectCode(() => scene.remove(new Node({ name: 'orphan' })), 'NODE_NOT_ATTACHED');
    expectCode(() => new Node({ name: 'detached' }).assertAttached(), 'NODE_NOT_ATTACHED');
  });

  test('a detached node resolves to its own local transform', () => {
    const scene = new Scene();
    const parent = new Node({ name: 'parent', position: [10, 0, 0] });
    const child = meshNode('child', 1, 2, 3);
    parent.add(child);
    scene.add(parent);

    const camera = lookingAtScene();
    const items: DrawItem[] = [];
    scene.collectDrawItems(items, camera);
    expect(child.worldPosition[0]).toBeCloseTo(11, 5);

    scene.remove(child);
    expect(child.parent).toBeNull();
    // Written at detach time, not left holding a parent-relative matrix.
    expect(child.worldPosition[0]).toBeCloseTo(1, 5);
    expect(child.world[12]).toBeCloseTo(1, 5);
  });

  test('walks are iterative, ordered, and pruneable', () => {
    const root = new Node({ name: 'root' });
    const a = new Node({ name: 'a' });
    const b = new Node({ name: 'b' });
    const c = new Node({ name: 'c' });
    root.add(a);
    root.add(b);
    a.add(c);

    const down: string[] = [];
    root.traverseDown((n) => {
      down.push(n.name);
    });
    expect(down).toEqual(['root', 'a', 'c', 'b']);

    const pruned: string[] = [];
    root.traverseDown((n) => {
      pruned.push(n.name);
      return n.name !== 'a';
    });
    expect(pruned).toEqual(['root', 'a', 'b']);

    const up: string[] = [];
    c.traverseUp((n) => {
      up.push(n.name);
    });
    expect(up).toEqual(['c', 'a', 'root']);

    // 10,000 deep must not blow the JS stack. A recursive walk would.
    const deep = new Node({ name: '0' });
    let tip = deep;
    for (let i = 1; i < 10000; i++) {
      const next = new Node({ name: String(i) });
      tip.add(next);
      tip = next;
    }
    let count = 0;
    deep.traverseDown(() => {
      count++;
    });
    expect(count).toBe(10000);
  });
});

// ---------------------------------------------------------------------------
// Culling
// ---------------------------------------------------------------------------

describe('collectDrawItems', () => {
  test('returns nothing when the camera faces away, everything when it faces the scene', () => {
    const scene = new Scene();
    scene.add(meshNode('ahead', 0, 0, -5));
    scene.add(meshNode('farToTheSide', 20, 0, -5));

    const camera = new PerspectiveCamera({ fov: 60, near: 0.1, far: 100, aspect: 1 });
    const items: DrawItem[] = [];

    // Turned around: the whole scene is behind the camera.
    camera.lookAt([0, 0, 0], [0, 0, 1], [0, 1, 0]);
    camera.update(1);
    expect(scene.collectDrawItems(items, camera)).toHaveLength(0);

    camera.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]);
    camera.update(1);
    // 'ahead' is dead centre; 'farToTheSide' is 20 units off-axis at a distance
    // of 5, well outside a 60° frustum, and its sphere is not big enough to
    // reach in.
    expect(scene.collectDrawItems(items, camera)).toHaveLength(1);
    expect(items[0].objectId).toBe(0);
  });

  test('an out-of-frustum mesh is absent rather than present-and-invisible', () => {
    const scene = new Scene();
    scene.add(meshNode('inView', 0, 0, -5));
    scene.add(meshNode('offToTheSide', 500, 0, -5, 0.1));

    const items = scene.collectDrawItems([], lookingAtScene());
    expect(items).toHaveLength(1);
    expect(items[0].visible).toBe(true);
  });

  test('hidden and layer-masked nodes take their subtrees with them', () => {
    const scene = new Scene();
    const rig = new Node({ name: 'rig' });
    rig.add(meshNode('inRig', 0, 0, -5));
    scene.add(rig);
    scene.add(meshNode('loose', 0, 0, -5));
    const camera = lookingAtScene();
    expect(scene.collectDrawItems([], camera)).toHaveLength(2);

    rig.visible = false;
    expect(scene.collectDrawItems([], camera)).toHaveLength(1);

    rig.visible = true;
    // The camera can see layer 0 only, and the rig is on layer 1 — so the rig
    // and the mesh inside it both go, and the loose node stays.
    camera.layers = 0b0001;
    rig.layer = 0b0010;
    expect(scene.collectDrawItems([], camera)).toHaveLength(1);

    rig.layer = 0b0001;
    expect(scene.collectDrawItems([], camera)).toHaveLength(2);
  });

  test('draw items are reused across frames, by reference', () => {
    const scene = new Scene();
    for (let i = 0; i < 32; i++) scene.add(meshNode(`m${i}`, 0, 0, -5 - i));
    const camera = lookingAtScene();

    const first = scene.collectDrawItems([], camera);
    const second = scene.collectDrawItems([], camera);

    expect(first).toHaveLength(32);
    expect(second).toHaveLength(32);
    // Same objects, so anything a renderer caches per item survives the frame.
    for (let i = 0; i < 32; i++) expect(second[i]).toBe(first[i]);
    // ...and the pool grew to the high-water mark exactly once.
    expect(scene.peakDrawCount).toBe(32);
  });

  test('objectOffset is the dynamic-offset-aligned stride, and objectId is the index', () => {
    const scene = new Scene();
    for (let i = 0; i < 4; i++) scene.add(meshNode(`m${i}`, 0, 0, -5 - i));

    const items = scene.collectDrawItems([], lookingAtScene());
    expect(items).toHaveLength(4);
    expect(OBJECT_UNIFORM_STRIDE).toBe(256);
    items.forEach((item, i) => {
      expect(item.objectId).toBe(i);
      // Offset from the frame region, not from byte 0.
      expect(item.objectOffset).toBe(sceneObjectOffset(i));
      expect(item.objectOffset % 256).toBe(0);
      expect(item.objectOffset).toBeGreaterThanOrEqual(FRAME_BLOCK.size);
      expect(item.instanceCount).toBe(1);
      expect(item.firstInstance).toBe(0);
      expect(item.material).toBe(material);
      expect(item.geometry).toBe(mesh);
      expect(item.phase).toBe('opaque');
    });

    // Consecutive objects are exactly one stride apart, and the first is clear
    // of the frame region.
    expect(items[1].objectOffset - items[0].objectOffset).toBe(256);
  });

  test('culling needs the world bounding sphere, which scales with the transform', () => {
    const scene = new Scene();
    const big = meshNode('big', 5, 0, -5, 1);
    scene.add(big);

    const camera = new PerspectiveCamera({ fov: 60, near: 0.1, far: 100, aspect: 1 });
    camera.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]);
    camera.update(1);
    // At z = −5 a 60° frustum is 5·tan(30°) ≈ 2.9 units half-width, so a unit
    // sphere centred 5 units off-axis cannot reach into it.
    expect(scene.collectDrawItems([], camera)).toHaveLength(0);

    // Scaled to 3, the same sphere spans 2..8 and now overlaps the frustum.
    // The bound that gets tested is the *world* one, not the local value — and
    // it is only refreshed by the pass, so the collect comes before the read.
    big.setScale(3, 3, 3);
    expect(scene.collectDrawItems([], camera)).toHaveLength(1);
    expect(big.worldBoundingRadius).toBeCloseTo(3, 4);

    // And a non-uniform scale takes the largest axis, which is the conservative
    // choice: a sphere maps to an ellipsoid, and only the longest semi-axis
    // still contains it.
    big.setScale(3, 1, 1);
    expect(scene.collectDrawItems([], camera)).toHaveLength(1);
    expect(big.worldBoundingRadius).toBeCloseTo(3, 4);
  });

  test('the scene version tracks structural change, and the graph revision tracks all change', () => {
    const scene = new Scene();
    const v0 = scene.version;
    const node = meshNode('m', 0, 0, -5);
    scene.add(node);
    expect(scene.version).toBe(v0 + 1);

    const r0 = getGraphRevision();
    node.setPosition(1, 0, 0);
    expect(getGraphRevision()).toBeGreaterThan(r0);
    // A transform is not a structural change, and the scene version says so —
    // it is the graph revision that answers "did anything move since I last
    // looked?", because a node does not know which scene owns it.
    expect(scene.version).toBe(v0 + 1);

    scene.remove(node);
    expect(scene.version).toBe(v0 + 2);
  });
});

// ---------------------------------------------------------------------------
// Draw-item validation
// ---------------------------------------------------------------------------

describe('draw item validation', () => {
  test('object offsets are stride-aligned and clear of the frame region', () => {
    const scene = new Scene();
    for (let i = 0; i < 8; i++) scene.add(meshNode(`m${i}`, 0, 0, -5 - i));
    const camera = lookingAtScene();

    // The stride is no longer a parameter, so there is no misaligned-stride
    // refusal to test here -- the whole point of the change was that a caller
    // cannot choose one. What must hold is that every offset is a multiple of
    // minUniformBufferOffsetAlignment and that object 0 starts past the frame,
    // because a draw binding offset 0 would read the camera as a world matrix.
    const items = scene.collectDrawItems([], camera);
    for (let i = 0; i < items.length; i++) {
      expect(items[i]!.objectOffset % SCENE_UNIFORM_STRIDE).toBe(0);
      expect(items[i]!.objectOffset).toBeGreaterThanOrEqual(FRAME_BLOCK.size);
      expect(items[i]!.objectOffset).toBe(sceneObjectOffset(i));
    }
    // Consecutive, whole strides apart.
    expect(items[1]!.objectOffset - items[0]!.objectOffset).toBe(SCENE_UNIFORM_STRIDE);
    expect(items[7]!.objectOffset - items[6]!.objectOffset).toBe(SCENE_UNIFORM_STRIDE);
  });

  test('a draw item whose geometry was released fails loudly instead of encoding freed buffers', () => {
    const scene = new Scene();
    const released = new TrackedMesh('released');
    const node = new MeshNode({
      name: 'm',
      mesh: released as unknown as DrawableGeometry,
      material,
      position: [0, 0, -5],
    });
    scene.add(node);
    const camera = lookingAtScene();
    expect(scene.collectDrawItems([], camera)).toHaveLength(1);

    // Forced, from outside, while the node still points at it. Ownership makes
    // this unreachable through scene.add / scene.remove; this is the backstop
    // for the paths ownership does not cover.
    released.dispose();
    expectCode(() => scene.collectDrawItems([], camera), 'MESH_DISPOSED');
  });

  test('every emitted item is checked, not just the ones on some path', () => {
    // Five items, the fourth of them backed by a disposed mesh. If the check
    // were conditional on anything about the item — its index, its geometry
    // identity, whether the pool had to grow — one of these five would slip
    // through and produce a draw against destroyed buffers.
    const scene = new Scene();
    const released = new TrackedMesh('late');
    const nodes: MeshNode[] = [];
    for (let i = 0; i < 5; i++) {
      const g = i === 3 ? (released as unknown as DrawableGeometry) : mesh;
      const n = new MeshNode({ name: `m${i}`, mesh: g, material, position: [0, 0, -5 - i] });
      nodes.push(n);
      scene.add(n);
    }
    const camera = lookingAtScene();
    expect(scene.collectDrawItems([], camera)).toHaveLength(5);

    released.dispose();
    // The pool is warm, this is the third frame, and the offending item is at
    // index 3 rather than 0 — none of which is a reason to skip the check.
    expectCode(() => scene.collectDrawItems([], camera), 'MESH_DISPOSED');
  });
});

// ---------------------------------------------------------------------------
// Culled, submitted, and not-submitted
// ---------------------------------------------------------------------------

describe('culled, submitted, and not-submitted', () => {
  test('culledCount is exactly the number of candidates the frustum rejected', () => {
    const scene = new Scene();
    // Seven dead ahead, at 5..11 units down −Z, all inside a 60-degree frustum.
    for (let i = 0; i < 7; i++) scene.add(meshNode(`in${i}`, 0, 0, -5 - i));
    // Three far off axis at a distance of 5, where the frustum is about
    // 5 * tan(30 degrees) ~ 2.9 units half-wide. A 0.1-radius sphere at x = 500
    // cannot reach into it, so these are frustum rejections and nothing else.
    for (let i = 0; i < 3; i++) scene.add(meshNode(`out${i}`, 500, 0, -5, 0.1));

    const items = scene.collectDrawItems([], lookingAtScene());
    expect(items).toHaveLength(7);
    expect(scene.culledCount).toBe(3);
    expect(scene.emptyCount).toBe(0);
    expect(scene.meshNodeCount).toBe(10);
    // The partition the three counters promise. Checked as an identity rather
    // than as three literals, because this is the property a caller who derives
    // one counter from the other two is relying on.
    expect(scene.meshNodeCount).toBe(items.length + scene.culledCount + scene.emptyCount);
  });

  test('culledCount is zero when nothing is culled, and the count tracks the camera', () => {
    const scene = new Scene();
    // Spread down the view axis. A node sitting *on* the camera is a degenerate
    // case for any frustum test, and a test built on one proves nothing.
    for (let i = 0; i < 12; i++) scene.add(meshNode(`m${i}`, 0, 0, -5 - i * 0.25));
    const camera = lookingAtScene();
    scene.collectDrawItems([], camera);
    expect(scene.culledCount).toBe(0);
    expect(scene.emptyCount).toBe(0);

    // Turn around and every one of them is a frustum rejection.
    camera.lookAt([0, 0, 0], [0, 0, 1], [0, 1, 0]);
    camera.update(1);
    const items = scene.collectDrawItems([], camera);
    expect(items).toHaveLength(0);
    expect(scene.culledCount).toBe(12);

    // And back, exactly: the counters are per-collection, not cumulative.
    camera.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]);
    camera.update(1);
    expect(scene.collectDrawItems([], camera)).toHaveLength(12);
    expect(scene.culledCount).toBe(0);
  });

  test('a node with nothing to draw is not-submitted, not culled', () => {
    const scene = new Scene();
    scene.add(meshNode('draws', 0, 0, -5));
    // Legal inputs, both of them: GpuInstances accepts an empty transform list,
    // and a mesh with no indices has nothing to step over.
    const zeroInstances = new MeshNode({
      name: 'zero',
      mesh: instancedGeometry(0),
      material,
      position: [0, 0, -6],
    });
    const noIndices = new MeshNode({
      name: 'noIndices',
      mesh: { ...instancedGeometry(1), indexCount: 0 } as unknown as DrawableGeometry,
      material,
      position: [0, 0, -7],
    });
    scene.add(zeroInstances);
    scene.add(noIndices);

    // Both are dead centre, so a frustum has no opinion about them at all.
    const items = scene.collectDrawItems([], lookingAtScene());
    expect(items).toHaveLength(1);
    expect(items[0].geometry).toBe(mesh);
    expect(scene.culledCount).toBe(0);
    expect(scene.emptyCount).toBe(2);
    expect(scene.meshNodeCount).toBe(3);
    // The identity that distinguishes the two facts. This is the case where
    // `meshNodeCount - items.length` is 2 and `culledCount` is 0, and a
    // statistic derived by that subtraction credits the frustum with rejecting
    // two objects that were never off screen.
    expect(scene.meshNodeCount - items.length).toBe(2);
  });

  test('hidden and layer-masked subtrees are in none of the three counters', () => {
    const scene = new Scene();
    const rig = new Node({ name: 'rig' });
    rig.add(meshNode('inRig', 0, 0, -5));
    rig.add(meshNode('inRig2', 0, 0, -6));
    scene.add(rig);
    scene.add(meshNode('loose', 0, 0, -5));
    const camera = lookingAtScene();
    expect(scene.collectDrawItems([], camera)).toHaveLength(3);
    expect(scene.meshNodeCount).toBe(3);

    rig.visible = false;
    expect(scene.collectDrawItems([], camera)).toHaveLength(1);
    // The walk skipped the subtree on the ancestor's behalf, so its mesh nodes
    // are neither candidates nor frustum rejections. Counting them would make
    // the partition identity false; counting them as culled would credit the
    // frustum with hiding a rig that a boolean hid.
    expect(scene.meshNodeCount).toBe(1);
    expect(scene.culledCount).toBe(0);
    expect(scene.emptyCount).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Instancing in the draw list
// ---------------------------------------------------------------------------

describe('instancing in the draw list', () => {
  test('a thousand instances are one draw item, carrying the count and the buffer', () => {
    const scene = new Scene();
    const buffer = {} as GPUBuffer;
    const node = new MeshNode({
      name: 'forest',
      mesh: instancedGeometry(1000, 0, buffer),
      material,
      position: [0, 0, -5],
    });
    scene.add(node);

    const items = scene.collectDrawItems([], lookingAtScene());
    expect(items).toHaveLength(1);
    expect(items[0].instanceCount).toBe(1000);
    expect(items[0].firstInstance).toBe(0);
    // The renderer binds vertex slot 1 from the node, so the draw list never has
    // to reach into the geometry for it and cannot get it wrong by omission.
    expect(node.instanceBuffer).toBe(buffer);
    expect(scene.culledCount).toBe(0);
  });

  test('firstInstance survives, so a window of one buffer draws the right copies', () => {
    const scene = new Scene();
    scene.add(new MeshNode({
      name: 'tail',
      mesh: instancedGeometry(8, 24),
      material,
      position: [0, 0, -5],
    }));
    const items = scene.collectDrawItems([], lookingAtScene());
    expect(items[0].instanceCount).toBe(8);
    expect(items[0].firstInstance).toBe(24);
  });

  test('instanced items each get their own object slot, and cull and sort like any other', () => {
    const scene = new Scene();
    const near = new MeshNode({
      name: 'near', mesh: instancedGeometry(4), material, position: [0, 0, -5],
    });
    const far = new MeshNode({
      name: 'far', mesh: instancedGeometry(7), material, position: [0, 0, -20],
    });
    scene.add(near);
    scene.add(far);
    const camera = lookingAtScene();

    const items = scene.collectDrawItems([], camera);
    expect(items).toHaveLength(2);
    // One slot per draw item, and a *unique* one per item. The uniform packer
    // writes at `objectId * stride` and the draw binds at `objectOffset`, so two
    // items sharing an id would mean one object's matrix drawn for both. That
    // is the instancing bug the brief names, and it is invisible in the picture:
    // both objects simply appear in the same place.
    expect(items[0].objectId).toBe(0);
    expect(items[1].objectId).toBe(1);
    // Object 0 starts after the frame region, not at byte 0.
    expect(items[0].objectOffset).toBe(sceneObjectOffset(0));
    expect(items[1].objectOffset).toBe(sceneObjectOffset(1));
    expect(new Set(items.map((i) => i.objectOffset)).size).toBe(2);
    // The count is per item, not per instance: a slot per instance would be the
    // same 124 bytes `instanceCount` times over, for a transform the vertex
    // stage already reads from the instance buffer.
    expect(items.map((i) => i.instanceCount)).toEqual([4, 7]);

    // Culling is per draw item and bounds the whole instance set, so an
    // instanced node is one sphere test, not one per copy — 11 instances, and
    // the culled count is still one.
    far.setPosition(900, 0, -20);
    expect(scene.collectDrawItems([], camera)).toHaveLength(1);
    expect(scene.culledCount).toBe(1);
    far.setPosition(0, 0, -20);

    // Sorting is by depth and the instance count does not perturb it: an
    // instanced item is an ordinary item with a bigger extent.
    const ordered = sortDrawItems(scene.collectDrawItems([], camera));
    expect(ordered).toHaveLength(2);
    const depths = depthsOf(ordered);
    expect(depths).toEqual([...depths].sort((a, b) => a - b));
    expect(compareDrawItems(ordered[0], ordered[1])).toBeLessThan(0);
    expect(ordered[0].instanceCount).toBe(4);
    expect(ordered[1].instanceCount).toBe(7);
  });

  test('the instance count is read per frame, not captured when the node was built', () => {
    const scene = new Scene();
    const geometry = instancedGeometry(2);
    const node = new MeshNode({ name: 'n', mesh: geometry, material, position: [0, 0, -5] });
    scene.add(node);

    expect(scene.collectDrawItems([], lookingAtScene())[0].instanceCount).toBe(2);

    // A GpuInstances buffer is re-uploaded from a moving transform list, so the
    // number of instances is not a property of the node — it is a property of
    // the buffer, read on the frame that draws it. A count captured in the
    // constructor is right on frame 1 and wrong on every frame after, in the
    // direction that draws the wrong number of objects.
    (geometry as { instanceCount: number }).instanceCount = 64;
    (geometry as { firstInstance: number }).firstInstance = 8;
    const grown = scene.collectDrawItems([], lookingAtScene())[0];
    expect(grown.instanceCount).toBe(64);
    expect(grown.firstInstance).toBe(8);
  });

  test('a pooled item is fully reinitialised, so no instance state crosses a frame', () => {
    const scene = new Scene();
    const instanced = new MeshNode({
      name: 'a', mesh: instancedGeometry(32, 5), material, position: [0, 0, -5],
    });
    const plain = meshNode('b', 0, 0, -6);
    scene.add(instanced);
    scene.add(plain);

    const first = scene.collectDrawItems([], lookingAtScene());
    expect(first[0].instanceCount).toBe(32);
    expect(first[0].firstInstance).toBe(5);
    // The same pooled object is handed out again — identity is what makes a
    // per-item renderer cache safe, so this is worth asserting before claiming
    // anything about its contents.
    expect(scene.collectDrawItems([], lookingAtScene())[0]).toBe(first[0]);

    // Reorder the scene so slot 0 is now the plain mesh. Every field of the
    // reused item has to be overwritten: a surviving `instanceCount` of 32 would
    // draw 32 copies of a cube that has one, and a surviving `firstInstance` of
    // 5 would start the vertex fetch five records into a one-record buffer.
    scene.remove(instanced);
    scene.remove(plain);
    scene.add(plain);
    scene.add(instanced);
    const swapped = scene.collectDrawItems([], lookingAtScene());
    expect(swapped[0]).toBe(first[0]);
    expect(swapped[0].instanceCount).toBe(1);
    expect(swapped[0].firstInstance).toBe(0);
    expect(swapped[0].objectId).toBe(0);
    expect(swapped[0].objectOffset).toBe(sceneObjectOffset(0));
    expect(swapped[0].model).toBe(plain.world);
    // ...and the instanced node, now at index 1, still carries its own state.
    expect(swapped[1].instanceCount).toBe(32);
    expect(swapped[1].firstInstance).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// Resource ownership
// ---------------------------------------------------------------------------

/**
 * A ref-counted stand-in for a `GpuMesh` or a `Material`.
 *
 * Real `Resource`, so the count and the disposal are the production ones and the
 * test is about the graph's pairing rather than about a mock. `disposeCount`
 * distinguishes "released" from "released twice", which a boolean cannot.
 */
class TrackedMesh extends Resource {
  disposeCount = 0;

  // Enough of a `DrawableGeometry` to be walked, culled and emitted headlessly.
  // Without these the draw list would classify it as a node with nothing to
  // draw, and every assertion below would be about the wrong thing.
  readonly layout = STANDARD_LAYOUT;
  readonly vertexBuffer = null as unknown as GPUBuffer;
  readonly indexBuffer = null;
  readonly indexCount = 36;
  readonly instanceCount = 1;
  readonly firstInstance = 0;
  readonly instanceBuffer = null;

  constructor(readonly name: string = 'tracked') {
    super();
  }
  protected onDispose(): void {
    this.disposeCount++;
  }
}

describe('resource ownership', () => {
  test('attaching retains, detaching releases, and a detached node costs nothing', () => {
    const scene = new Scene();
    const gpuMesh = new TrackedMesh('mesh');
    const gpuMaterial = new TrackedMesh('material');
    expect(gpuMesh.refCount).toBe(1);

    const node = new MeshNode({
      name: 'm',
      mesh: gpuMesh as unknown as DrawableGeometry,
      material: gpuMaterial as unknown as Drawable,
      position: [0, 0, -5],
    });
    // Not in a graph, not drawing anything, so nothing is retained: the
    // reference the creator holds is still the only one.
    expect(gpuMesh.refCount).toBe(1);
    expect(gpuMaterial.refCount).toBe(1);

    scene.add(node);
    expect(gpuMesh.refCount).toBe(2);
    expect(gpuMaterial.refCount).toBe(2);

    scene.remove(node);
    expect(gpuMesh.refCount).toBe(1);
    expect(gpuMesh.disposed).toBe(false);

    // Now the creator lets go, and this is the moment the mesh is released. A
    // scene that is still drawing it is what makes that the wrong moment, which
    // is why `add` has to have retained.
    gpuMesh.unref();
    gpuMaterial.unref();
    expect(gpuMesh.disposed).toBe(true);
    expect(gpuMaterial.disposed).toBe(true);
    expect(gpuMesh.disposeCount).toBe(1);
  });

  test('a mesh whose every external reference is dropped survives a collection while attached', () => {
    const scene = new Scene();
    const camera = lookingAtScene();

    // Everything the test itself holds goes out of scope. The only thing left
    // pointing at the mesh is the node, through the reference the graph took.
    const weak = ((): WeakRef<TrackedMesh> => {
      const gpuMesh = new TrackedMesh('orphan');
      const gpuMaterial = new TrackedMesh('orphanMaterial');
      scene.add(new MeshNode({
        name: 'm',
        mesh: gpuMesh as unknown as DrawableGeometry,
        material: gpuMaterial as unknown as Drawable,
        position: [0, 0, -5],
      }));
      // The caller's own handles go, exactly as `mesh.unref()` in real code
      // would; the graph's reference is the only one left.
      gpuMesh.unref();
      gpuMaterial.unref();
      return new WeakRef(gpuMesh);
    })();

    // A real, forced collection, not a "the garbage collector will get round to
    // it" hand-wave. If the node did not hold the mesh, this collects it and the
    // deref below is undefined — which is the failure this whole mechanism
    // exists to prevent, and the one that produces a scene that renders
    // nothing with every draw call succeeding.
    Bun.gc(true);

    const live = weak.deref();
    expect(live).toBeDefined();
    expect(live!.disposed).toBe(false);
    expect(live!.refCount).toBe(1);
    expect(scene.collectDrawItems([], camera)).toHaveLength(1);

    // Detaching is the release point.
    scene.clear();
    expect(live!.disposed).toBe(true);
    expect(live!.disposeCount).toBe(1);
  });

  test('a mesh shared by a thousand nodes is released a thousand times and freed once', () => {
    const shared = new TrackedMesh('shared');
    const sharedMaterial = new TrackedMesh('sharedMaterial');
    const scene = new Scene();
    const nodes: MeshNode[] = [];
    for (let i = 0; i < 1000; i++) {
      const n = new MeshNode({
        name: `m${i}`,
        mesh: shared as unknown as DrawableGeometry,
        material: sharedMaterial as unknown as Drawable,
        position: [0, 0, -5 - i * 0.01],
      });
      nodes.push(n);
      scene.add(n);
    }
    // One creator plus a thousand nodes.
    expect(shared.refCount).toBe(1001);

    // Remove them one at a time. The whole point of a count rather than a flag
    // is that the nine hundred and ninety-ninth removal frees nothing.
    for (let i = 0; i < 999; i++) scene.remove(nodes[i]!);
    expect(shared.refCount).toBe(2);
    expect(shared.disposed).toBe(false);

    scene.remove(nodes[999]!);
    expect(shared.refCount).toBe(1);
    expect(shared.disposed).toBe(false);

    // The creator's own reference is the last one, and it frees exactly once.
    shared.unref();
    sharedMaterial.unref();
    expect(shared.disposed).toBe(true);
    expect(shared.disposeCount).toBe(1);
    expect(sharedMaterial.disposeCount).toBe(1);
  });

  test('ownership follows the graph, not the scene, and moving a node is balanced', () => {
    const shared = new TrackedMesh('moved');
    const a = new Node({ name: 'a' });
    const b = new Node({ name: 'b' });
    const node = new MeshNode({
      name: 'm',
      mesh: shared as unknown as DrawableGeometry,
      material,
      position: [0, 0, -5],
    });

    // Built into a group before anything is in a scene. Same rule, same code.
    a.add(node);
    expect(shared.refCount).toBe(2);
    const scene = new Scene();
    scene.add(a);
    // Already retained by `a.add`; attaching the group does not double it.
    expect(shared.refCount).toBe(2);

    // Ten moves between two live parents. Each is one retain and one release,
    // so the count returns to where it started every time — this is what makes
    // it safe for `add` to retain unconditionally.
    for (let i = 0; i < 10; i++) {
      node.removeFromParent();
      expect(shared.refCount).toBe(1);
      (i % 2 === 0 ? b : a).add(node);
      expect(shared.refCount).toBe(2);
    }

    // The other routes out of a graph release identically.
    node.removeFromParent();
    b.add(node);
    expect(shared.refCount).toBe(2);
    expect(b.remove(node)).toBe(true);
    expect(shared.refCount).toBe(1);
    expect(shared.disposed).toBe(false);

    // `scene.clear()` detaches the scene's own children, and a node still
    // attached to one of them is still in a graph — a detached one, which a
    // later `scene.add(group)` can revive. So its reference is not given back,
    // and the count falls by exactly the scene's share.
    a.add(node);
    expect(shared.refCount).toBe(2);
    scene.clear();
    expect(shared.refCount).toBe(2);
    expect(shared.disposed).toBe(false);
    expect(node.parent).toBe(a);

    a.remove(node);
    expect(shared.refCount).toBe(1);
    shared.unref();
    expect(shared.disposed).toBe(true);
    expect(shared.disposeCount).toBe(1);
  });

  test('setMesh moves the reference, and the old mesh is released exactly once', () => {
    const scene = new Scene();
    const first = new TrackedMesh('first');
    const second = new TrackedMesh('second');
    const node = new MeshNode({
      name: 'm',
      mesh: first as unknown as DrawableGeometry,
      material,
      position: [0, 0, -5],
    });
    scene.add(node);
    expect(first.refCount).toBe(2);

    node.setMesh(second as unknown as DrawableGeometry);
    expect(first.refCount).toBe(1);
    expect(second.refCount).toBe(2);
    expect(node.mesh).toBe(second);
    expect(scene.collectDrawItems([], lookingAtScene())[0].geometry).toBe(second);

    // Setting the mesh it already has is a no-op, not a release and a re-take:
    // doing it the other way would drop the count to zero and dispose a mesh the
    // node is about to draw.
    node.setMesh(second as unknown as DrawableGeometry);
    expect(second.refCount).toBe(2);
    expect(second.disposed).toBe(false);

    // A detached node owes nothing, so a swap there releases without retaining
    // — and releasing the caller's only reference to the old mesh frees it,
    // which is the correct outcome and not an accident: nobody is drawing it
    // any more and this was the last handle on it.
    scene.remove(node);
    expect(second.refCount).toBe(1);
    node.setMesh(first as unknown as DrawableGeometry);
    expect(first.refCount).toBe(1);
    expect(second.refCount).toBe(0);
    expect(second.disposed).toBe(true);
    expect(second.disposeCount).toBe(1);

    // Re-attaching then retains the new one, which is the whole contract.
    scene.add(node);
    expect(first.refCount).toBe(2);
    scene.remove(node);
    expect(first.refCount).toBe(1);
  });

  test('a hand-rolled geometry with no reference count is left alone', () => {
    // `DrawableGeometry` is an interface, so this is a supported thing to draw.
    // There is no count to take and none is owed; `add` must not assume one.
    const scene = new Scene();
    const node = meshNode('m', 0, 0, -5);
    expect(() => scene.add(node)).not.toThrow();
    expect(() => scene.remove(node)).not.toThrow();
    expect(scene.collectDrawItems([], lookingAtScene())).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Removal and reparenting
// ---------------------------------------------------------------------------

describe('removal and reparenting', () => {
  test('a node that has been moved keeps its children in step with it', () => {
    // The regression this exists for. `worldVersion` used to be stamped from a
    // global clock that the detach path could not see, so a node whose version
    // had run ahead of the clock was written by the pass without its token
    // advancing. Every child below it then compared its recorded parent version
    // against an unchanged one, decided its parent had not moved, and kept a
    // world matrix from before the move — permanently, with no error and no
    // extra matrix write to notice.
    const scene = new Scene();
    const parent = new Node({ name: 'parent' });
    const child = meshNode('child', 0, 0, -5);
    parent.add(child);
    scene.add(parent);
    const camera = lookingAtScene();

    // Settle: the child records the parent's token.
    scene.collectDrawItems([], camera);
    const settled = [parent.worldVersion, child.parentWorldVersion];
    expect(settled[0]).toBe(settled[1]);

    // Churn: fifty attach/detach cycles, each of which advances the parent's
    // own token twice.
    for (let i = 0; i < 50; i++) {
      parent.removeFromParent();
      scene.add(parent);
    }
    scene.collectDrawItems([], camera);
    expect(child.parentWorldVersion).toBe(parent.worldVersion);

    parent.setPosition(10, 0, 0);
    resetTransformWriteCount();
    scene.collectDrawItems([], camera);

    // Two writes: the parent that moved and the child it carries.
    expect(getTransformWriteCount()).toBe(2);
    expect(parent.worldPosition[0]).toBeCloseTo(10, 5);
    expect(child.worldPosition[0]).toBeCloseTo(10, 5);
    // The token is what the second write depends on, so assert it directly.
    expect(child.parentWorldVersion).toBe(parent.worldVersion);
  });

  test('every write advances the token, whoever performed it', () => {
    // The property the fix rests on, asserted without reference to how many
    // frames have run: a `worldVersion` that does not move when `world` did is
    // a node whose children cannot learn that it moved. The old stamp came from
    // a global clock, so a node whose own counter had been advanced by detaching
    // was written by the pass and kept its token — and whether that happened
    // depended on how many passes the process had already run, which is exactly
    // why it survived every test written against the count of matrix writes.
    //
    // `add` bumps the node's `localVersion`, so the pass is guaranteed to write
    // it on the next collection. The token must therefore move.
    const scene = new Scene();
    const parent = new Node({ name: 'parent' });
    const mid = new Node({ name: 'mid' });
    const child = meshNode('child', 0, 0, -5);
    parent.add(mid);
    mid.add(child);
    scene.add(parent);
    const camera = lookingAtScene();
    scene.collectDrawItems([], camera);

    for (let i = 0; i < 200; i++) {
      parent.removeFromParent();
      const detached = parent.worldVersion;
      scene.add(parent);
      scene.collectDrawItems([], camera);
      expect(parent.worldVersion).toBeGreaterThan(detached);
      // The two halves of the parent-change test, both of which have to hold at
      // every step rather than only at the end.
      expect(mid.parentWorldVersion).toBe(parent.worldVersion);
      expect(child.parentWorldVersion).toBe(mid.worldVersion);
    }

    // And the symptom it produces: after all that churn, a move at the top of
    // the tree reaches the leaf.
    mid.setPosition(0, 7, 0);
    scene.collectDrawItems([], camera);
    expect(child.worldPosition[1]).toBeCloseTo(7, 5);
    expect(child.worldPosition[2]).toBeCloseTo(-5, 5);
  });

  test('a removed node stops costing transform work, and re-adding resumes it', () => {
    const scene = new Scene();
    for (let i = 0; i < 200; i++) scene.add(meshNode(`m${i}`, 0, 0, -5 - i * 0.01));
    const taken = meshNode('taken', 0, 0, -30);
    const group = new Node({ name: 'group' });
    group.add(taken);
    group.add(meshNode('stays', 0, 0, -30));
    scene.add(group);

    const camera = lookingAtScene();
    scene.collectDrawItems([], camera);
    expect(scene.collectDrawItems([], camera)).toHaveLength(202);

    // The mover is *inside* the group, so a change to it forces the pass to
    // descend into the group's children and the visit count becomes a statement
    // about the walk rather than about which ancestor happened to be dirty.
    const mover = group.children[1] as MeshNode;
    expect(mover.name).toBe('stays');

    // Attached: root, 200 loose meshes, the group and both its children — 204
    // — and one matrix written.
    mover.setPosition(0, 0, -31);
    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems([], camera);
    expect(getNodeVisitCount()).toBe(204);
    expect(getTransformWriteCount()).toBe(1);

    // Out of the graph: not visited, not written, and not in the draw list. The
    // world matrix it kept is its own, and nothing has to maintain it.
    scene.remove(taken);
    expect(scene.collectDrawItems([], camera)).toHaveLength(201);
    mover.setPosition(0, 0, -32);
    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems([], camera);
    // 203, not 204: the removed node contributes nothing to either number.
    expect(getNodeVisitCount()).toBe(203);
    expect(getTransformWriteCount()).toBe(1);

    // Back in, and it costs its own subtree again — including the subtree it
    // brings with it, which is the part a re-attached graph usually gets wrong.
    group.add(taken);
    expect(scene.collectDrawItems([], camera)).toHaveLength(202);
    mover.setPosition(0, 0, -33);
    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems([], camera);
    expect(getNodeVisitCount()).toBe(204);
    expect(getTransformWriteCount()).toBe(1);
    expect(taken.parentWorldVersion).toBe(group.worldVersion);
    // ...and its world matrix is resolved against the parent it is back under,
    // not left holding one computed against wherever it used to be. The group is
    // at the origin, so `taken`'s own -30 is the answer.
    expect(taken.world[14]).toBeCloseTo(-30, 5);
    expect(taken.worldPosition[2]).toBeCloseTo(-30, 5);
  });

  test('a re-attached subtree is resolved against its new parent, not its old one', () => {
    const scene = new Scene();
    const oldParent = new Node({ name: 'old', position: [100, 0, 0] });
    const newParent = new Node({ name: 'new', position: [0, 0, 0] });
    const branch = new Node({ name: 'branch' });
    branch.add(meshNode('leaf', 1, 2, 3));
    oldParent.add(branch);
    scene.add(oldParent);

    const camera = lookingAtScene();
    scene.collectDrawItems([], camera);
    const leaf = branch.children[0] as MeshNode;
    expect(leaf.worldPosition[0]).toBeCloseTo(101, 4);

    branch.removeFromParent();
    // The branch itself is its own world the moment it is detached — not the
    // old parent's, and not a stale mix of the two. Its descendants keep the
    // matrix they had until the next pass, which is the documented contract: the
    // detach path rewrites this node and nothing else.
    expect(branch.worldPosition[0]).toBeCloseTo(0, 4);

    newParent.add(branch);
    scene.add(newParent);
    scene.collectDrawItems([], camera);
    expect(leaf.worldPosition[0]).toBeCloseTo(1, 4);
    expect(leaf.worldPosition[1]).toBeCloseTo(2, 4);
  });

  test('a static scene is still free after all of that', () => {
    // The claim the whole module is written for, asserted once more at the end
    // so none of the ownership and validation work above has quietly bought its
    // correctness with a per-frame traversal.
    const scene = new Scene();
    for (let i = 0; i < 999; i++) scene.add(meshNode(`m${i}`, 0, 0, -5 - i * 0.05));
    expect(scene.root.descendantCount).toBe(1000);
    const camera = lookingAtScene();
    const items: DrawItem[] = [];

    scene.collectDrawItems(items, camera);
    scene.collectDrawItems(items, camera);
    resetTransformWriteCount();
    resetNodeVisitCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(0);
    expect(getNodeVisitCount()).toBe(1);

    // ...and one moving leaf is still one write, with the item's `model` still a
    // live reference to the node's array rather than a copy of last frame's. A
    // copy is the failure this catches: the array identity would differ, and the
    // value would be last frame's.
    expect(items).toHaveLength(999);
    const leaf = scene.root.children[10] as MeshNode;
    // Nudged sideways, not parked on the eye: a sphere centred where the camera
    // is straddles four frustum planes at once, and whether it survives is a
    // statement about the near plane rather than about this test.
    leaf.setPosition(0.5, 0, -5 - 10 * 0.05);
    resetTransformWriteCount();
    scene.collectDrawItems(items, camera);
    expect(getTransformWriteCount()).toBe(1);
    expect(items[10].model).toBe(leaf.world);
    expect(items[10].model[12]).toBeCloseTo(0.5, 5);
  });
});

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

describe('perspective camera', () => {
  test('near maps to NDC z = 0 and far to NDC z = 1 — WebGPU depth, not OpenGL', () => {
    const near = 0.25;
    const far = 250;
    const camera = lookingAtScene(60, near, far);

    // Under a [-1, 1] projection this would be −1, and a depth buffer cleared
    // to 0 would then be putting the near plane at the *far* end of the range.
    expect(ndc(camera.viewProj, 0, 0, -near)[2]).toBeCloseTo(0, 6);
    expect(ndc(camera.viewProj, 0, 0, -far)[2]).toBeCloseTo(1, 6);

    // Half the *range* in NDC is the harmonic mean of the two distances, not
    // their arithmetic mean — which is the perspective divide doing its job and
    // is the reason depth precision is not uniform.
    const harmonic = 2 * near * far / (near + far);
    expect(ndc(camera.viewProj, 0, 0, -harmonic)[2]).toBeCloseTo(0.5, 5);

    // The arithmetic mean, for contrast, is already at the far end.
    expect(ndc(camera.viewProj, 0, 0, -(near + far) / 2)[2]).toBeGreaterThan(0.99);
  });

  test('x and y stay in [-1, 1] and the frustum matches the projection', () => {
    const camera = lookingAtScene(90, 0.1, 100);
    // A 45°-off-axis point at unit distance is exactly on the edge of a 90° fov.
    expect(ndc(camera.viewProj, 1, 0, -1)[0]).toBeCloseTo(1, 5);
    expect(ndc(camera.viewProj, 0, 0, -1)[1]).toBeCloseTo(0, 5);

    const planes = new Float32Array(24);
    camera.getFrustum(planes);
    for (let i = 0; i < 6; i++) {
      const length = Math.hypot(planes[i * 4], planes[i * 4 + 1], planes[i * 4 + 2]);
      expect(length).toBeCloseTo(1, 4);
    }
  });

  test('view and inverse view are inverses, and the eye maps to the origin', () => {
    const camera = new PerspectiveCamera({ fov: 50, near: 0.5, far: 90, aspect: 16 / 9 });
    camera.lookAt([3, 4, 5], [0, 0, 0], [0, 1, 0]);
    camera.update();

    const [vx, vy, vz] = ndc(camera.view, 3, 4, 5);
    expect(vx).toBeCloseTo(0, 5);
    expect(vy).toBeCloseTo(0, 5);
    expect(vz).toBeCloseTo(0, 5);
    expect(camera.worldPosition[0]).toBeCloseTo(3, 5);
    expect(camera.worldPosition[2]).toBeCloseTo(5, 5);

    // invView * view is the identity.
    const product = new Float32Array(16);
    mul(product, camera.invView, camera.view);
    for (let i = 0; i < 16; i++) {
      expect(product[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 4);
    }
  });

  test('an orthographic camera projects a cube edge to the frustum edges', () => {
    const near = 0.1;
    const far = 10;
    const camera = new OrthographicCamera({
      left: -2, right: 2, bottom: -2, top: 2, near, far,
    });
    // The eye is 5 units back, so the near plane is the plane 0.1 in front of
    // it, at world z = 4.9.
    camera.lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]);
    camera.update(1);

    // The right edge of the view volume, at the far plane.
    expect(ndc(camera.viewProj, 2, 0, 5 - far)[0]).toBeCloseTo(1, 5);
    expect(ndc(camera.viewProj, 0, 2, 5 - far)[1]).toBeCloseTo(1, 5);
    // The near plane, and the far plane.
    expect(ndc(camera.viewProj, 0, 0, 5 - near)[2]).toBeCloseTo(0, 5);
    expect(ndc(camera.viewProj, 0, 0, 5 - far)[2]).toBeCloseTo(1, 5);
  });

  test('a degenerate lens is refused with a message that names the field', () => {
    expectCode(() => new PerspectiveCamera({ fov: 0 }), 'INTERNAL_INVARIANT');
    expectCode(() => new PerspectiveCamera({ fov: 180 }), 'INTERNAL_INVARIANT');
    expectCode(() => new PerspectiveCamera({ fov: Number.NaN }), 'INTERNAL_INVARIANT');
    expectCode(() => new PerspectiveCamera({ near: 0 }), 'INTERNAL_INVARIANT');
    expectCode(() => new PerspectiveCamera({ far: -1 }), 'INTERNAL_INVARIANT');
    expectCode(() => new PerspectiveCamera({ aspect: 0 }), 'INTERNAL_INVARIANT');

    // An infinite far plane is not degenerate — it is the skydome case, and it
    // is the answer whenever the horizon is coming out wrong.
    const sky = new PerspectiveCamera({ far: Number.POSITIVE_INFINITY });
    sky.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]);
    sky.update(1);
    expect(ndc(sky.viewProj, 0, 0, -1e7)[2]).toBeCloseTo(1, 3);
  });

  test('a frustum too small to hold six planes is refused before it is written', () => {
    const camera = lookingAtScene();
    expectCode(() => camera.getFrustum(new Float32Array(20)), 'INTERNAL_INVARIANT');
  });
});
