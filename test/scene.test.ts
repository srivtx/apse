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
import { STANDARD_LAYOUT } from '../src/geometry/layout.ts';
import { mul, transformVec4 } from '../src/math/mat4.ts';
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
  layout: STANDARD_LAYOUT,
  vertexBuffer: null as unknown as GPUBuffer,
  indexBuffer: null,
  indexCount: 36,
  instanceCount: 1,
  firstInstance: 0,
} as unknown as DrawableGeometry;

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
      expect(item.objectOffset).toBe(i * 256);
      expect(item.objectOffset % 256).toBe(0);
      expect(item.instanceCount).toBe(1);
      expect(item.material).toBe(material);
      expect(item.geometry).toBe(mesh);
      expect(item.phase).toBe('opaque');
    });

    // The stride is a parameter, not a constant baked into the loop.
    const tight = scene.collectDrawItems([], lookingAtScene(), 64);
    expect(tight[3].objectOffset).toBe(3 * 64);
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
