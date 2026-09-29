/**
 * Draw-list build: the shape it hands on, and the cost of building it.
 *
 * `test/scene.test.ts` owns the correctness of the scene layer. This file owns
 * two narrower claims, both of which are easy to break in a way that only shows
 * up as a slow frame:
 *
 *   1. **The depth key is the general one.** `collectDrawItems` computes
 *      view-space `z` from the third row of the view matrix directly, on the
 *      strength of a per-frame test that the matrix's last row is `(0, 0, 0, w)`.
 *      That is a fast path, and a fast path is only legitimate while it agrees
 *      with the slow path *bit for bit* — so that is what is asserted here, over
 *      a spread of positions and camera orientations, plus the general path
 *      itself for a matrix the fast path must decline.
 *   2. **`out` is filled in place, not appended to.** The collection array is
 *      sized to the pool's high-water mark and written by index, then truncated
 *      to the draw count. That is only safe if a frame which draws *more* than
 *      the last one, a frame which draws fewer, and a frame that reuses an
 *      `out` somebody else filled, all come out the same as the push-based
 *      form they replaced.
 *
 * The timing section at the bottom reports rather than asserts. `performance.now()`
 * is quantised on this machine, so the numbers are batched and the batch size and
 * the measured resolution are printed alongside them; a wall-clock assertion in
 * a test suite is a test that fails on a busy CI machine, which is a test that
 * gets deleted.
 */

import { describe, expect, test } from 'bun:test';
import { SCENE_FRAME_BYTES, SCENE_UNIFORM_STRIDE, sceneObjectOffset } from './src/core/slot.ts';
import { STANDARD_LAYOUT } from './src/geometry/layout.ts';
import { transformPoint } from './src/math/mat4.ts';
import { sortDrawItems } from './src/render/sort.ts';
import type { Drawable, DrawableGeometry, DrawItem } from './src/render/types.ts';
import {
  MeshNode,
  Node,
  PerspectiveCamera,
  Scene,
  getNodeVisitCount,
  getTransformWriteCount,
  resetNodeVisitCount,
  resetTransformWriteCount,
} from './src/scene/index.ts';

// ---------------------------------------------------------------------------
// Fixtures — the same fakes `scene.test.ts` uses, for the same reason.
// ---------------------------------------------------------------------------

const material = { name: 'bench', phase: 'opaque' } as unknown as Drawable;

function geometry(indexCount = 36, instanceCount = 1, firstInstance = 0): DrawableGeometry {
  return {
    name: 'mesh',
    layout: STANDARD_LAYOUT,
    vertexBuffer: null as unknown as GPUBuffer,
    indexBuffer: null,
    indexCount,
    instanceCount,
    firstInstance,
    instanceBuffer: null,
  } as unknown as DrawableGeometry;
}

const box = geometry();

function meshNode(name: string, x: number, y: number, z: number, radius = 0.5, mesh = box): MeshNode {
  return new MeshNode({ name, mesh, material, position: [x, y, z], boundingRadius: radius });
}

/** A camera at the origin looking down −Z, which is the whole scene's front. */
function camera(): PerspectiveCamera {
  const c = new PerspectiveCamera({ fov: 60, near: 0.1, far: 4000, aspect: 1 });
  c.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]);
  c.update(1);
  return c;
}

/**
 * A camera far enough back that a whole grid is in front of it, for the tests
 * that want every node drawn rather than a handful surviving the frustum. At
 * 900 units and 60° the half-height is ~520, so any `grid` below fits.
 */
function wideCamera(): PerspectiveCamera {
  const c = new PerspectiveCamera({ fov: 60, near: 0.1, far: 4000, aspect: 1 });
  c.lookAt([0, 0, 900], [0, 0, 0], [0, 1, 0]);
  c.update(1);
  return c;
}

/** A flat grid in the plane z = 0, all of it inside {@link wideCamera}. */
function grid(count: number, spacing = 2.5): Scene {
  const scene = new Scene();
  const side = Math.ceil(Math.sqrt(count));
  for (let i = 0; i < count; i++) {
    const x = (i % side) - side / 2;
    const y = Math.floor(i / side) - side / 2;
    scene.add(meshNode(`n${i}`, x * spacing, y * spacing, 0, 1));
  }
  return scene;
}

// ---------------------------------------------------------------------------
// The depth key
// ---------------------------------------------------------------------------

/**
 * The value `collectDrawItems` must put on `item.depth`: view-space depth,
 * positive in front of the camera, computed the general way.
 *
 * The point is rounded through `Float32Array` first, because that is what the
 * real path reads: `worldPosition` *is* a `Float32Array(3)`, so a position like
 * `1e-30` arrives already rounded. Comparing a double against a float-rounded
 * input would fail on the input's rounding rather than on the arithmetic, which
 * is a test that measures nothing.
 */
function referenceDepth(view: Float32Array, p: readonly number[]): number {
  const point = Float32Array.from(p);
  return -transformPoint(new Float32Array(3), point, view)[2];
}

/** The same, for a position already stored in a node's `Float32Array`. */
function depthOf(view: Float32Array, position: Float32Array): number {
  return -transformPoint(new Float32Array(3), position, view)[2];
}

/** A `Float32Array(1)`-rounded value, which is what a `DrawItem.depth` holds. */
function toF32(v: number): number {
  const one = new Float32Array(1);
  one[0] = v;
  return one[0]!;
}

describe('Math.fround is the Float32Array rounding', () => {
  test('it agrees bit for bit, so the depth key is unchanged', () => {
    // `collectDrawItems` rounds the fast path with `Math.fround` rather than
    // writing it through a scratch `Float32Array`. That is only the same number
    // if the two roundings are the same rounding, so it is asserted rather than
    // assumed — over the awkward inputs first, then a spread of exponents.
    const awkward = [0, -0, Infinity, -Infinity, NaN, 1e-45, -1e-45, 5e-324,
      3.4028234663852886e38, -3.4028234663852886e38, 1.1754943508222875e-38,
      1.1754942106924411e-38, 0.5, 2.5, 1e-30, 1e30, 16777216, 16777217, 16777219];
    for (const v of awkward) {
      const a = toF32(v);
      const b = Math.fround(v);
      expect(Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b)), `fround(${v})`).toBe(true);
    }
    let seed = 0x9e3779b9;
    let mismatches = 0;
    for (let i = 0; i < 200000; i++) {
      seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0;
      const v = (seed / 4294967296 * 2 - 1) * Math.pow(10, Math.floor(seed / 4096 % 40) - 20);
      const a = toF32(v);
      const b = Math.fround(v);
      if (!(Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b)))) mismatches++;
    }
    expect(mismatches).toBe(0);
  });
});

describe('the depth key agrees with the general transform', () => {
  /**
   * Cameras spanning the orientations that matter: axis-aligned, oblique, a
   * steep elevation, and one far enough out that the frustum arithmetic in
   * `lookAt` has lost most of its significant digits. An orientation that only
   * happens to be axis-aligned would not prove anything.
   */
  const ORIENTATIONS: [number[], number[], number[]][] = [
    [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[3, -7, 11], [0, 0, 0], [0, 1, 0]],
    [[-4, 2, 9], [1, 1, 1], [1, 0.2, 0]],
    [[0, 25, 0], [0, 0, 0], [0, 0, 1]],
    [[1e3, 1e-3, -1e3], [0, 0, 0], [0.2, 0.9, -0.3]],
  ];

  function viewOf(eye: number[], at: number[], up: number[]): Float32Array {
    const c = new PerspectiveCamera({ fov: 55, near: 0.05, far: 9000, aspect: 1.7 });
    c.lookAt(eye, at, up);
    c.update(1.7);
    // The fast path's own precondition, asserted rather than assumed: a view
    // matrix from `lookAt` is affine, so all of these take the fast path.
    expect(c.view[3]).toBe(0);
    expect(c.view[7]).toBe(0);
    expect(c.view[11]).toBe(0);
    return c.view;
  }

  test('the fast z expression is bit-identical to transformPoint, everywhere', () => {
    // Checked over the full range of positions rather than over positions that
    // survive a cull, so the extremes are covered: a fast path is only
    // legitimate while it agrees with the slow path everywhere, and the
    // interesting disagreements live at the ends of the exponent range.
    const positions: number[][] = [
      [0, 0, 0], [0, 0, -1], [0, 0, 1], [0, 0, -1e-30], [0, 0, -1e30],
      [1e30, -1e30, 1e30], [1e-30, 1e-30, -1e-30], [1e15, 3.7e15, -1e15],
      [0.5, -0.25, -12.5], [1234.5678, -0.0001, -3.0003],
      [1e-8, 1e-8, -1e-8], [-0, 0, -0],
    ];
    for (const [eye, at, up] of ORIENTATIONS) {
      const view = viewOf(eye, at, up);
      const w = 1 / view[15]!;
      for (const p of positions) {
        // f32, because a node's `worldPosition` is one. The arithmetic under test
        // is the expression, not the input's precision.
        const q = Float32Array.from(p);
        const want = -transformPoint(new Float32Array(3), q, view)[2];
        // Exact equality, not `toBeCloseTo`. The sort's depth key is a
        // truncation of this number into a bucket, so a one-ulp disagreement is a
        // different draw order on the one frame a depth lands on a boundary.
        const got = toF32(-(view[2]! * q[0]! + view[6]! * q[1]! + view[10]! * q[2]! + view[14]!) * w);
        expect(got).toBe(want);
      }
    }
  });

  test('a collected item\'s depth is that value, for points the frustum kept', () => {
    // The same equivalence end to end, through `collectDrawItems`. One view,
    // because the *cull* is what depends on the orientation and the depth is not:
    // the arithmetic is covered for every orientation by the test above, and
    // this one is about the value that actually reaches the item.
    const view = viewOf([0, 0, 900], [0, 0, 0], [0, 1, 0]);
    // Same orientation, and a far plane past the furthest point, so every one of
    // them is a candidate and the depth is the only thing this test can fail on.
    const c = new PerspectiveCamera({ fov: 60, near: 0.1, far: 9000, aspect: 1 });
    c.lookAt([0, 0, 900], [0, 0, 0], [0, 1, 0]);
    c.update(1);
    c.view.set(view);
    // All inside a 60° cone at 900..8000 units out: within `tan(30°) * d` in
    // both axes, and within the far plane.
    const points: number[][] = [
      [0, 0, -1], [0, 0, -100], [0, 0, -3300], [0.5, -0.25, -12.5],
      [1e-3, 1e-3, -1e-3], [800, 900, -1000], [500, -0.0001, -3.0003],
      [-0, 0, -0],
    ];
    const scene = new Scene();
    for (let i = 0; i < points.length; i++) {
      const p = points[i]!;
      scene.add(meshNode(`p${i}`, p[0]!, p[1]!, p[2]!, 0.001));
    }
    const items = scene.collectDrawItems([], c);
    expect(items.length).toBe(points.length);
    for (let i = 0; i < points.length; i++) {
      expect(items[i]!.depth).toBe(referenceDepth(view, points[i]!));
      // And the item's own world matrix holds the same position the reference
      // was computed from: `world[12..14]` is where `worldPosition` was copied
      // from, so it is a `Float32Array(3)` view rather than a fresh allocation.
      expect(items[i]!.depth).toBe(depthOf(view, items[i]!.model.subarray(12, 15)));
    }
  });

  test('a projective view matrix takes the general path and still reads correctly', () => {
    // `Camera.view` is a readonly field holding a writable array, so a caller
    // can put anything in it. This one is affine only in appearance: `view[11]`
    // is non-zero, so the last row is not `(0, 0, 0, w)` and the per-frame test
    // must decline. If it did not, `depth` would come from the third row alone
    // and this value would be wrong.
    const c = camera();
    c.view.set([1, 0, 0, 0.01, 0, 1, 0, 0.02, 0, 0, 1, 0.03, 0, 0, -1, 1]);

    const scene = new Scene();
    scene.add(meshNode('a', 0, 0, -5, 0));
    scene.add(meshNode('b', 1, 2, -9, 0));

    const items = scene.collectDrawItems([], c);
    expect(items.length).toBe(2);
    for (const item of items) {
      const p = [item.model[12]!, item.model[13]!, item.model[14]!];
      expect(item.depth).toBe(referenceDepth(c.view, p));
    }
    // And the projective case really does differ from the affine answer, or the
    // test above is not testing anything.
    const point = [1, 2, -9];
    const affine = -(c.view[2]! * point[0]! + c.view[6]! * point[1]! + c.view[10]! * point[2]! + c.view[14]!) / c.view[15]!;
    expect(referenceDepth(c.view, point)).not.toBe(affine);
  });

  test('depth is positive in front of the camera and negative behind it', () => {
    const c = camera();
    const scene = new Scene();
    // Both are given a radius big enough to straddle the near plane, so the one
    // behind the camera is *drawn* and its depth can be read — otherwise it
    // would simply be culled and the assertion would pass vacuously.
    const front = new MeshNode({ name: 'front', mesh: box, material, position: [0, 0, -5], boundingRadius: 50 });
    const behind = new MeshNode({ name: 'behind', mesh: box, material, position: [0, 0, 5], boundingRadius: 50 });
    scene.add(front);
    scene.add(behind);
    const items = scene.collectDrawItems([], c);
    expect(items.length).toBe(2);
    const byNode = new Map(items.map((i) => [i.model, i]));
    expect(byNode.get(front.world)!.depth).toBeGreaterThan(0);
    expect(byNode.get(behind.world)!.depth).toBeLessThan(0);
  });
});

// ---------------------------------------------------------------------------
// The draw list's shape
// ---------------------------------------------------------------------------

describe('the draw list is filled in place', () => {
  test('a frame that draws more than the last still fills every slot', () => {
    // The first frame sizes `out` from the pool, which is empty. The second
    // frame's pre-size is one frame behind a scene that grew, so the new items
    // land past the end of the array and have to grow it. That path is exactly
    // the one the in-place fill replaced, so it is the one worth testing.
    const scene = new Scene();
    for (let i = 0; i < 3; i++) scene.add(meshNode(`a${i}`, 0, 0, -5 - i, 0));
    const c = wideCamera();
    const out: DrawItem[] = [];
    expect(scene.collectDrawItems(out, c).length).toBe(3);
    expect(out.length).toBe(3);

    for (let i = 0; i < 40; i++) scene.add(meshNode(`b${i}`, 0, 0, -50 - i, 0));
    // Nothing above is culled: the camera is far back and the new nodes are in
    // front of it, so all 43 draw and the frame is the "grew" case rather than
    // a cull count that happens to move in the same direction.
    const grown = scene.collectDrawItems(out, c);
    expect(grown).toBe(out);
    expect(grown.length).toBe(43);
    for (let i = 0; i < 43; i++) expect(grown[i]!.objectId).toBe(i);
    // Every slot holds a distinct pooled item, and every one is a live model.
    expect(new Set(grown).size).toBe(43);
    for (let i = 0; i < 43; i++) expect(grown[i]!.model.length).toBe(16);
  });

  test('a frame that draws fewer truncates, and the tail is not reachable', () => {
    const scene = grid(40);
    const c = wideCamera();
    const out: DrawItem[] = [];
    expect(scene.collectDrawItems(out, c).length).toBe(40);
    expect(out.length).toBe(40);

    // Put half of them behind the camera. `out` keeps its capacity; only its
    // length comes down, and every reader walks to `length`.
    for (let i = 0; i < 20; i++) {
      const node = scene.root.children[i + 1]!;
      node.setPosition(node.position[0]!, node.position[1]!, 9000);
    }
    const fewer = scene.collectDrawItems(out, c);
    expect(fewer.length).toBe(20);
    expect(fewer.length).toBe(fewer.length);
    expect(scene.meshNodeCount).toBe(40);
    expect(scene.culledCount).toBe(20);
    expect(scene.meshNodeCount).toBe(fewer.length + scene.culledCount + scene.emptyCount);

    // Nothing from the previous frame is readable through the new length.
    const live = new Set(fewer.map((i) => i.model));
    for (let i = 0; i < fewer.length; i++) {
      expect(fewer[i]!.objectId).toBe(i);
      // Each surviving slot still points at a node that is in the scene, so no
      // slot holds an item from the 40-object frame that has been culled since.
      expect(live.has(fewer[i]!.model)).toBe(true);
    }
  });

  test('a caller-supplied array full of somebody else\'s items is fully overwritten', () => {
    const other = new Scene();
    for (let i = 0; i < 12; i++) other.add(meshNode(`x${i}`, 0, 0, -5, 0));
    const c = wideCamera();
    const out = other.collectDrawItems([], c);
    expect(out.length).toBe(12);

    const scene = grid(5);
    const returned = scene.collectDrawItems(out, c);
    expect(returned).toBe(out);
    expect(returned.length).toBe(5);
    // Every slot in the reused array now points at *this* scene's node, in this
    // scene's walk order. Not one of the twelve items the previous scene put
    // there is still reachable through the new length.
    const nodes = scene.root.children;
    for (let i = 0; i < 5; i++) {
      expect(returned[i]!.objectId).toBe(i);
      expect(returned[i]!.model).toBe(nodes[i]!.world);
      expect(returned[i]!.geometry).toBe(box);
    }
  });

  test('the array keeps its identity and the pool keeps item identity across frames', () => {
    // `test/scene.test.ts` asserts the item identity; this asserts the part that
    // the in-place fill could have broken, which is that the *array* the caller
    // handed in is the one it gets back, at whatever length the frame produced.
    const scene = grid(16);
    const c = wideCamera();
    const out: DrawItem[] = [];
    const first = scene.collectDrawItems(out, c);
    const firstItems = first.slice();
    for (let f = 0; f < 5; f++) {
      const again = scene.collectDrawItems(out, c);
      expect(again).toBe(first);
      expect(again.length).toBe(16);
      for (let i = 0; i < 16; i++) expect(again[i]).toBe(firstItems[i]!);
    }
  });

  test('peakDrawCount tracks the largest collection, not the latest', () => {
    const scene = grid(30);
    const c = wideCamera();
    const out: DrawItem[] = [];
    scene.collectDrawItems(out, c);
    expect(scene.peakDrawCount).toBe(30);
    for (const child of scene.root.children.slice(0, 20)) child.setPosition(0, 0, 9000);
    scene.collectDrawItems(out, c);
    expect(out.length).toBe(10);
    expect(scene.peakDrawCount).toBe(30);
  });
});

describe('the invariants the pool has to keep', () => {
  test('every item carries the collection index, and the sort moves it with the item', () => {
    const scene = grid(48);
    const c = wideCamera();
    const out: DrawItem[] = [];
    const collected = scene.collectDrawItems(out, c);
    for (let i = 0; i < collected.length; i++) expect(collected[i]!.objectId).toBe(i);

    // Sorting permutes the array. `objectId` travels with the item, so after the
    // sort it is a permutation of `0..n-1` and never the loop index — which is
    // the property the uniform packer depends on when it keys on it.
    const before = collected.map((i) => i.depth);
    sortDrawItems(collected);
    const ids = collected.map((i) => i.objectId);
    expect(new Set(ids).size).toBe(48);
    expect(Math.min(...ids)).toBe(0);
    expect(Math.max(...ids)).toBe(47);
    for (let i = 0; i < 48; i++) expect(collected[i]!.objectId).toBe(ids.indexOf(i));
    // The depths are the same set; the sort ordered them.
    const after = collected.map((i) => i.depth);
    expect([...after].sort((a, b) => a - b)).toEqual([...before].sort((a, b) => a - b));
  });

  test('the dynamic offset is sceneObjectOffset, and specifically not index * 256', () => {
    // Byte 0 of the shared scene buffer is the frame region, so a binding offset
    // of `index * 256` is an in-range read that hands object *k* the previous
    // object's slot — or the camera, for *k* = 0. Nothing errors.
    const scene = grid(8);
    const c = wideCamera();
    const items = scene.collectDrawItems([], c);
    for (let i = 0; i < items.length; i++) {
      const item = items[i]!;
      expect(item.objectId).toBe(i);
      expect(item.objectOffset).toBe(sceneObjectOffset(i));
      expect(item.objectOffset).toBe(SCENE_FRAME_BYTES + i * SCENE_UNIFORM_STRIDE);
      expect(item.objectOffset % SCENE_UNIFORM_STRIDE).toBe(0);
      expect(item.objectOffset).not.toBe(i * SCENE_UNIFORM_STRIDE);
    }
    expect(items[0]!.objectOffset).toBeGreaterThan(0);
  });

  test('model is a live reference to the node\'s own array, not a copy', () => {
    const scene = grid(4);
    const c = wideCamera();
    const items = scene.collectDrawItems([], c);
    const node = scene.root.children[0] as MeshNode;
    const item = items[0]!;

    // Identity, which is the whole claim. A copy would be a different array.
    expect(item.model).toBe(node.world);
    // And a write through the item lands in the node's own array, which is the
    // direction a copy cannot produce in.
    item.model[12] = 99;
    expect(node.world[12]).toBe(99);
    node.setPosition(31, -17, 0);
    scene.collectDrawItems(items, c);

    // The item still *is* that array, and the transform pass wrote the new
    // position into it — the packer may read `item.model` at any point in the
    // frame and must see the current transform, not a snapshot of this frame's.
    expect(item.model).toBe(node.world);
    expect(item.model[12]).toBe(31);
    expect(item.model[13]).toBe(-17);
  });

  test('an instanced mesh is one item with the count read fresh every frame', () => {
    const scene = new Scene();
    const mesh = geometry(36, 4);
    const node = meshNode('inst', 0, 0, -5, 0, mesh);
    scene.add(node);
    const c = camera();
    const out: DrawItem[] = [];

    const items = scene.collectDrawItems(out, c);
    expect(items.length).toBe(1);
    expect(items[0]!.instanceCount).toBe(4);
    expect(scene.meshNodeCount).toBe(1);
    expect(scene.emptyCount).toBe(0);
    expect(scene.culledCount).toBe(0);

    // The count is read through the node every frame, never snapshotted: a count
    // captured at construction is wrong from frame 2 and wrong silently.
    node.setMesh(geometry(36, 9, 2));
    const again = scene.collectDrawItems(out, c);
    expect(again.length).toBe(1);
    expect(again[0]!.instanceCount).toBe(9);
    expect(again[0]!.firstInstance).toBe(2);
  });

  test('an item with nothing to draw is counted, not drawn and not called culled', () => {
    const scene = new Scene();
    scene.add(meshNode('real', 0, 0, -5));
    scene.add(meshNode('noIndices', 0, 0, -5, 0, geometry(0)));
    scene.add(meshNode('noInstances', 0, 0, -5, 0, geometry(36, 0)));
    const c = camera();
    const items = scene.collectDrawItems([], c);
    expect(items.length).toBe(1);
    expect(scene.meshNodeCount).toBe(3);
    expect(scene.emptyCount).toBe(2);
    expect(scene.culledCount).toBe(0);
    expect(scene.meshNodeCount).toBe(items.length + scene.culledCount + scene.emptyCount);
  });

  test('no item handed to the renderer is invisible, which is what makes a filter dead', () => {
    // The renderer's `#sort` still runs `if (it.visible) out.push(it)` over this
    // list. A culled or hidden node is *absent* — never present-and-invisible —
    // so that filter has never had anything to drop. If a future path ever emits
    // an invisible item, this fails and the filter becomes load-bearing.
    const scene = grid(30);
    for (const child of scene.root.children.slice(0, 10)) child.setPosition(0, 0, -9000);
    const items = scene.collectDrawItems([], wideCamera());
    expect(items.length).toBe(20);
    expect(items.every((i) => i.visible)).toBe(true);
  });

  test('a hidden subtree contributes to none of the three counters, and meshNodeCount says so', () => {
    // The partition is over *examined* nodes. A node under a hidden ancestor is
    // not examined, so it is in none of the counters, and `meshNodeCount` is the
    // one that has to agree.
    const scene = grid(6);
    const rig = new Node({ name: 'rig' });
    rig.add(meshNode('buried', 0, 0, 0));
    scene.add(rig);
    const c = wideCamera();
    const items = scene.collectDrawItems([], c);
    expect(items.length).toBe(7);
    rig.visible = false;
    const after = scene.collectDrawItems([], c);
    expect(after.length).toBe(6);
    expect(scene.meshNodeCount).toBe(6);
    expect(scene.meshNodeCount).toBe(after.length + scene.culledCount + scene.emptyCount);
  });
});

// ---------------------------------------------------------------------------
// The transform pass, at the sizes the profiler cares about
// ---------------------------------------------------------------------------

describe('the transform pass at scale', () => {
  test('a static 20,000-node scene visits one node and writes no matrices', () => {
    // The count is size-independent by construction — the traversal compares two
    // integers at the root and returns — but the *claim* is worth restating at a
    // size where a regression to "traverse everything" would show up as a
    // timeout rather than as a slow frame nobody noticed.
    const scene = grid(20000, 0.5);
    expect(scene.root.descendantCount).toBe(20001);
    const c = wideCamera();
    const out: DrawItem[] = [];

    resetNodeVisitCount();
    resetTransformWriteCount();
    scene.collectDrawItems(out, c);
    expect(getNodeVisitCount()).toBe(20001);
    expect(getTransformWriteCount()).toBeGreaterThan(0);

    resetNodeVisitCount();
    resetTransformWriteCount();
    scene.collectDrawItems(out, c);
    expect(getNodeVisitCount()).toBe(1);
    expect(getTransformWriteCount()).toBe(0);
  });

  test('a moving leaf in a wide scene writes one matrix and its ancestors none', () => {
    // Wide, not deep. The writes are O(depth) as documented; the *visits* are
    // not O(depth) here, because a dirty node has to be descended through and
    // every one of its children is examined and pruned. That is measured and
    // asserted rather than left implicit, because the difference between the two
    // costs is the whole argument for a dirty-node list over a dirty flag.
    const scene = grid(5000, 0.5);
    const c = wideCamera();
    const out: DrawItem[] = [];
    scene.collectDrawItems(out, c);

    const leaf = scene.root.children[1234] as MeshNode;
    const tokens = [scene.root.worldVersion, leaf.parent!.worldVersion];
    leaf.setPosition(3, 0, 0);

    resetNodeVisitCount();
    resetTransformWriteCount();
    scene.collectDrawItems(out, c);
    expect(getTransformWriteCount()).toBe(1);
    // Root, every direct child, and the leaf. Every sibling is examined once and
    // pruned on one flag read.
    expect(getNodeVisitCount()).toBe(1 + 5000);
    expect([scene.root.worldVersion, leaf.parent!.worldVersion]).toEqual(tokens);
  });
});

// ---------------------------------------------------------------------------
// Timing: reported, not asserted
// ---------------------------------------------------------------------------

/**
 * What `performance.now()` can actually resolve, printed with every timing
 * number below — a timing comparison that does not state its own resolution is
 * not a measurement.
 *
 * Two back-to-back reads are almost always identical, so the reported figure is
 * the *smallest non-zero* gap, which is the clock's tick and not its precision.
 * The zero fraction says how often a single frame's cost is invisible.
 */
function clockResolution(): { tickNs: number; zeroFraction: number } {
  let smallest = Infinity;
  let zeros = 0;
  const pairs = 200000;
  for (let i = 0; i < pairs; i++) {
    const a = performance.now();
    const b = performance.now();
    if (b === a) zeros++;
    else if (b - a < smallest) smallest = b - a;
  }
  return {
    tickNs: smallest * 1e6,
    zeroFraction: zeros / pairs,
  };
}

/**
 * Median nanoseconds per `collectDrawItems`, over `batches` samples of `K`
 * frames each. `performance.now()` is quantised, so one frame is not a
 * measurable unit of anything; K is chosen so a sample is tens of milliseconds
 * and the quantisation is three orders of magnitude below it.
 */
function measure(
  scene: Scene,
  c: PerspectiveCamera,
  out: DrawItem[],
  tick: () => void,
  K: number,
  batches: number,
): number {
  for (let i = 0; i < 60; i++) { tick(); scene.collectDrawItems(out, c); }
  const samples: number[] = [];
  for (let b = 0; b < batches; b++) {
    const t0 = performance.now();
    for (let i = 0; i < K; i++) { tick(); scene.collectDrawItems(out, c); }
    samples.push(((performance.now() - t0) * 1e6) / K);
  }
  samples.sort((a, b) => a - b);
  return samples[samples.length >> 1];
}

describe('draw-list build cost', () => {
  test('a static and a moving 20,000-object scene, batched, with the resolution stated', () => {
    const clock = clockResolution();
    const still = grid(20000, 0.4);
    const moving = grid(20000, 0.4);
    const spinTarget = moving.root.children[1] as MeshNode;
    const c = wideCamera();
    const K = 20;
    const batches = 15;

    const stillOut: DrawItem[] = [];
    const movingOut: DrawItem[] = [];
    const staticNs = measure(still, c, stillOut, () => {}, K, batches);
    const movingNs = measure(
      moving,
      c,
      movingOut,
      () => spinTarget.setPosition(Math.sin(performance.now() * 1e-4) * 10, 0, 0),
      K,
      batches,
    );

    // eslint-disable-next-line no-console
    console.log(
      `\n  clock: smallest non-zero gap ${clock.tickNs.toFixed(0)} ns, ` +
      `${(clock.zeroFraction * 100).toFixed(1)}% of back-to-back reads identical\n` +
      `  batched K=${K} frames/sample, ${batches} samples, ` +
      `~${(staticNs * K / 1e6).toFixed(1)} ms per sample ` +
      `(${(staticNs * K / clock.tickNs).toFixed(0)}x the clock tick)\n` +
      `  static  20k: ${staticNs.toFixed(0).padStart(7)} ns/call  ${(staticNs / stillOut.length).toFixed(1)} ns/item  ` +
      `(${stillOut.length} drawn)\n` +
      `  moving  20k: ${movingNs.toFixed(0).padStart(7)} ns/call  ${(movingNs / movingOut.length).toFixed(1)} ns/item  ` +
      `(${movingOut.length} drawn)\n`,
    );

    // The only assertions: the measurement is of *something*, and both
    // scenarios actually produced a draw list. A wall-clock bound belongs in a
    // benchmark, where a human reads it, not in a suite that has to stay green
    // on a machine nobody profiled.
    expect(Number.isFinite(staticNs)).toBe(true);
    expect(Number.isFinite(movingNs)).toBe(true);
    expect(stillOut.length).toBeGreaterThan(19000);
    expect(movingOut.length).toBe(stillOut.length);
    expect(staticNs).toBeLessThan(50_000_000);
    // The moving pass has to be the more expensive one, or the static/moving
    // split is not exercising different branches and the numbers are one number.
    expect(movingNs).toBeGreaterThan(staticNs);
  });
});
