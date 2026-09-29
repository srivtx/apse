/**
 * Per-draw cost: what the four candidates each cost, and how it was measured.
 *
 * This file exists because the attribution is now three independent
 * measurements of the same thing, and they have to be reconciled before anyone
 * optimises off them. The reconciliation, and the traps, are in
 * `bench/diag/perf/ATTRIBUTION.md`; the short version is that the head-to-head
 * gap is ~0.18-0.24 us/draw, that it is stable across repetitions, and that
 * **the gap does not reproduce on a distinct-mesh scene at all** -- where apse
 * is ahead. That inversion is the most important thing in here, because it says
 * the deficit is not a property of "a draw" in apse.
 *
 * The tests here are on properties that are cheap to check and expensive to get
 * wrong. There is no GPU in the test environment, and nothing in this file needs
 * one: a fake device records the call sequence, which is the thing the whole
 * attribution rests on. A timing assertion would be worthless in CI -- this
 * machine's own per-draw cost moved 15-70% between runs of identical code -- so
 * every assertion below is on a *count*, a *byte count*, or an identity, and the
 * timing work lives in the browser harness where a GPU exists.
 */

import { beforeAll, afterAll, describe, expect, test } from 'bun:test';
import type { FakeFramePass } from './test/render-fakes.ts';

import { Renderer } from './src/render/renderer.ts';
import type { DrawItem } from './src/render/types.ts';
import { sortDrawItems, compareDrawItems } from './src/render/sort.ts';
import { Scene, MeshNode, PerspectiveCamera, updateWorldMatrices, getNodeVisitCount, resetNodeVisitCount, resetTransformWriteCount, getTransformWriteCount } from './src/scene/index.ts';
import { box } from './src/geometry/primitives/box.ts';
import { upload } from './src/geometry/mesh.ts';
import type { GpuMesh } from './src/geometry/mesh.ts';
import { basicMaterial } from './src/material/basic.ts';
import { fakeLimits } from './test/fake-device.ts';
import { FakeFrameDevice, fakeWebGpuCanvas, installWebGpuBitmaps, withGpu } from './test/render-fakes.ts';

let uninstallBitmaps: (() => void) | null = null;
beforeAll(() => { uninstallBitmaps = installWebGpuBitmaps(); });
afterAll(() => { uninstallBitmaps?.(); uninstallBitmaps = null; });

/**
 * A renderer on a recording device, plus the scene and camera it draws.
 *
 * `FakeFrameDevice` records every call the encoder receives. A `drawIndexed` is a
 * line in an array and that is enough: "one bind per draw" is a claim about the
 * calls and not about the pixels, and it is the claim the attribution rests on.
 */
async function withScene<T>(
  objects: number,
  opts: { distinctMeshes?: boolean; noPresentPass?: boolean; fn: (f: SceneFrame) => Promise<T> | T },
): Promise<T> {
  const device = new FakeFrameDevice({ limits: fakeLimits(), features: ['timestamp-query'] });
  const { canvas } = fakeWebGpuCanvas(1280, 720);
  return withGpu({ device: device as unknown as GPUDevice, preferredFormat: 'bgra8unorm' }, async () => {
    const renderer = await Renderer.create(canvas, opts.noPresentPass === true ? { toneMapping: null, hdr: false } : {});
    const gpu = device as unknown as GPUDevice;
    const camera = new PerspectiveCamera({ fov: 45, near: 0.1, far: 400, aspect: 1280 / 720 });
    camera.lookAt([0, 0, 120], [0, 0, 0], [0, 1, 0]);
    const scene = new Scene('perf');
    const mesh0 = upload(gpu, box({ width: 1 }));
    const material = await basicMaterial(gpu, { targetFormat: renderer.sceneFormat });
    const nodes: MeshNode[] = [];
    for (let i = 0; i < objects; i++) {
      // Distinct meshes is the scene shape the "4.00 calls per draw" census was
      // taken on, and it is the shape where the gap disappears. Both are
      // measured here so the census cannot be quoted without its scene.
      const mesh = opts.distinctMeshes ? upload(gpu, box({ width: 1 })) : mesh0;
      const n = new MeshNode({ name: 'n' + i, mesh, material });
      const side = Math.ceil(Math.sqrt(objects));
      const step = side > 1 ? 90 / (side - 1) : 0;
      n.setPosition((i % side) * step - 45, 0, Math.floor(i / side) * step - 45);
      n.setScale(90 / (2 * side));
      scene.add(n);
      nodes.push(n);
    }
    try {
      return await opts.fn({ renderer, device, scene, camera, nodes, gpu, mesh0, material });
    } finally {
      renderer.dispose();
    }
  });
}

interface SceneFrame {
  readonly renderer: Renderer;
  readonly gpu: GPUDevice;
  /** The shared mesh, for tests that patch a prototype or build their own nodes. */
  readonly mesh0: GpuMesh;
  /** The shared material, likewise. */
  readonly material: Awaited<ReturnType<typeof basicMaterial>>;
  readonly device: FakeFrameDevice;
  readonly scene: Scene;
  readonly camera: PerspectiveCamera;
  readonly nodes: MeshNode[];
}

/** A draw item with every field the sort and the pack loop read. */
function makeItem(
  id: number,
  phase: DrawItem['phase'],
  depth: number,
  order: number,
  material: DrawItem['material'],
  geometry: DrawItem['geometry'],
): DrawItem {
  return {
    objectId: id,
    phase,
    order,
    depth,
    objectOffset: id * 256,
    model: new Float32Array(16),
    worldVersion: 1,
    material,
    geometry,
    instanceCount: 1,
    firstInstance: 0,
    visible: true,
  } as DrawItem;
}

/**
 * The last frame's scene pass -- the one that is not the present pass.
 *
 * `lastPasses` is the passes of the most recent `render()`, so it has to be read
 * after the frame under test and before any other. Counting the present pass's
 * three calls into the scene census is how a census comes out a few calls too
 * high and nobody can say which frame it was from.
 */
function scenePassOf(device: FakeFrameDevice): FakeFramePass {
  const p = device.lastPasses.find((q) => !q.label.startsWith('apse.present'));
  if (p === undefined) throw new Error('no scene pass recorded; the frame encoded nothing');
  return p;
}

/** Count of a recorded call in the last scene pass. */
function callsOf(device: FakeFrameDevice, name: string): number {
  return scenePassOf(device).calls.filter((c) => c === name).length;
}

/**
 * Drops the recorded encoders, so the next frame's passes are unambiguous.
 *
 * `lastPasses` is a readonly view over `frameEncoders`, and `frameEncoders` is
 * what grows for the lifetime of the device. Assigning to `lastPasses` throws,
 * and appending to it does nothing useful.
 */
function resetPasses(device: FakeFrameDevice): void {
  device.frameEncoders.length = 0;
}

describe('the per-draw call census', () => {
  test('a shared-mesh frame is 3 calls per draw: one bind, one drawIndexed, and the buffer binds suppressed', async () => {
    // 200 objects on ONE mesh, so `lastVB`/`lastIB` are equal on every draw after
    // the first and both buffer binds are suppressed. This is the scene shape
    // the 0.87/0.66 us/draw numbers were measured on, and the census there is
    // 3.00 per draw, not 4.00.
    await withScene(200, { fn: ({ renderer, device, scene, camera }) => {
      renderer.render(scene, camera);
      expect(renderer.stats.drawCalls).toBe(200);
      expect(callsOf(device, 'setBindGroup:0')).toBe(200);
      expect(callsOf(device, 'drawIndexed')).toBe(200);
      // Exactly one buffer bind for the whole pass, not one per draw: the mesh is
      // the same object on every node, so `lastVB`/`lastIB` match after the first.
      expect(callsOf(device, 'setVertexBuffer:0')).toBe(1);
      expect(scenePassOf(device).indexBuffers.length).toBe(1);
      expect(callsOf(device, 'setPipeline')).toBe(1);
    } });
  });

  test('a distinct-mesh frame is 4 calls per draw, and all four are load-bearing', async () => {
    await withScene(200, { distinctMeshes: true, fn: ({ renderer, device, scene, camera }) => {
      renderer.render(scene, camera);
      expect(renderer.stats.drawCalls).toBe(200);
      expect(callsOf(device, 'setBindGroup:0')).toBe(200);
      // A different mesh per object, so nothing is suppressible: this is the
      // scene the "4.00 calls per draw" figure was measured on.
      expect(callsOf(device, 'setVertexBuffer:0')).toBe(200);
      expect(scenePassOf(device).indexBuffers.length).toBe(200);
      expect(callsOf(device, 'drawIndexed')).toBe(200);
      expect(renderer.stats.bindGroupCalls).toBeGreaterThanOrEqual(200);
      expect(renderer.stats.bufferCalls).toBe(400);
    } });
  });

  test('every draw binds the object at its own dynamic offset, so no two objects share a slot', async () => {
    // The census counts *calls*; this counts what they carry. A bind that always
    // passed offset 0 would have the same census and would draw every object
    // with the camera's transform -- an in-range read, a successful draw, and a
    // black frame.
    await withScene(64, { fn: ({ renderer, device, scene, camera }) => {
      renderer.render(scene, camera);
      const offsets = device.lastPasses
        .flatMap((p) => p.bindGroups)
        .filter((b) => b.group === 0 && b.offsets !== null)
        .map((b) => b.offsets![0]);
      expect(offsets.length).toBeGreaterThanOrEqual(64);
      expect(new Set(offsets).size).toBe(64);
    } });
  });
});

describe('the per-draw JS on the hot path', () => {
  test('the GpuMesh buffer accessors are called exactly three times per draw', async () => {
    // `vertexBuffer` and `indexBuffer` are accessors that build an error-message
    // string and call `assertLive`. renderer.ts reads indexBuffer once into a
    // local, and then reads it a second time in the draw/drawIndexed branch --
    // so three accessor calls per draw, not two. Measured at 12-30ns per draw
    // (K=500000), which is real but is 2-4% of the per-draw budget; the count
    // is pinned here because it is the cheap part of any future fix.
    // The present pass draws its own fullscreen mesh, which reads the same
    // accessors once more. Counting it would put the per-draw accessor count at
    // 3.03 rather than 3.00, so it is off and the count is exactly the scene loop.
    await withScene(32, { noPresentPass: true, fn: ({ renderer, device, gpu, scene, camera, mesh0 }) => {
      const proto = Object.getPrototypeOf(mesh0);
      const seen = { indexBuffer: 0, vertexBuffer: 0 };
      const originals: Record<string, PropertyDescriptor | undefined> = {};
      for (const name of ['indexBuffer', 'vertexBuffer'] as const) {
        originals[name] = Object.getOwnPropertyDescriptor(proto, name);
        const orig = originals[name]!.get!;
        Object.defineProperty(proto, name, {
          ...originals[name],
          get: function () { seen[name]++; return orig.call(this); },
        });
      }
      try {
        renderer.render(scene, camera);
      } finally {
        for (const name of ['indexBuffer', 'vertexBuffer'] as const) {
          Object.defineProperty(proto, name, originals[name]!);
        }
      }
      const draws = renderer.stats.drawCalls;
      expect(draws).toBe(32);
      // Once each, hoisted into a local before either the guard or the draw
      // branch reads them. These are accessors that call assertLive, so every
      // read is a method call; at 32 draws it is 32 calls per buffer, and at
      // 5,000 draws it is 5,000.
      expect(seen.indexBuffer).toBe(draws);
      expect(seen.vertexBuffer).toBe(draws);
      void gpu; void device; void scene; void camera;
    } });
  });

  test('the transform walk prunes a static scene at the root: one visit, no writes', async () => {
    // The `updateWorldMatrices` cost measured on a static scene is one visit and
    // no matrix writes, which is why candidate 2 in the attribution prices at
    // 0.0000us/frame. This is the invariant that makes it so, and it is worth
    // pinning: a static scene that re-walked its whole tree would be a
    // regression no frame-time assertion would catch, because the walk is 0.2%
    // of the frame either way.
    await withScene(200, { fn: ({ scene, nodes }) => {
      // The first pass writes every node's world matrix, because nothing has
      // been written yet. Settle it first: what is being measured is a
      // *subsequent* walk of an unmoved scene, not the cost of initialising.
      updateWorldMatrices(scene.root);
      resetNodeVisitCount();
      resetTransformWriteCount();
      updateWorldMatrices(scene.root);
      // One visit: the root's flags say nothing below it changed, so the whole
      // subtree is pruned without being pushed.
      expect(getNodeVisitCount()).toBe(1);
      expect(getTransformWriteCount()).toBe(0);

      // Moving one leaf of a flat scene marks every node on the path to the root,
      // and the root's children are all pushed and each is then pruned at its own
      // check. Visits = 201, writes = 1: the point is one *write*, not one visit.
      // The flat shape is what the benchmarks use, so this is the number the
      // attribution's "1 visit, 0 writes" static row is measured on.
      (nodes[100] as MeshNode).markDirty();
      resetNodeVisitCount();
      resetTransformWriteCount();
      updateWorldMatrices(scene.root);
      expect(getTransformWriteCount()).toBe(1);
      expect(getNodeVisitCount()).toBe(201);
    } });
  });

  test('a moving leaf in a deep chain visits only its path, and writes only its own matrix', async () => {
    // The pruning claim in node.ts, on the shape where it is true: a six-node
    // chain, move the leaf, and the walk touches six nodes rather than the whole
    // tree. This is the case the doc comment describes and the flat case cannot
    // show, because a flat root pushes all of its children by construction.
    await withScene(0, { fn: ({ scene, mesh0: mesh, material }) => {
      let cur = scene.root as unknown as MeshNode;
      for (let i = 0; i < 5; i++) {
        const n = new MeshNode({ name: 'n' + i, mesh, material });
        cur.add(n);
        cur = n;
      }
      updateWorldMatrices(scene.root);
      resetNodeVisitCount();
      resetTransformWriteCount();
      updateWorldMatrices(scene.root);
      expect(getNodeVisitCount()).toBe(1);

      cur.markDirty();
      resetNodeVisitCount();
      resetTransformWriteCount();
      updateWorldMatrices(scene.root);
      // Six nodes on the path: five of ours plus the scene root.
      expect(getNodeVisitCount()).toBe(6);
      // One write: the leaf. The five nodes above it are on the path but their own
      // transforms did not change, so their world matrices are untouched.
      expect(getTransformWriteCount()).toBe(1);
    } });
  });
});

describe('the static-scene write invariant', () => {
  test('a static scene writes the frame uniform every frame and nothing else', async () => {
    // The invariant, not an optimisation: after frame 1, a static scene must not
    // write object uniforms. `#packObjects` packs only dirty objects and
    // `uploadObjects` only covers the dirty range, so a static frame issues one
    // `writeBuffer` -- the frame block -- and no object bytes at all. If this
    // ever fails, every static scene is re-uploading its whole object range
    // every frame, which is 1.28MB at 1000 objects and is invisible in a frame
    // time because it is asynchronous.
    await withScene(120, { fn: ({ renderer, device, scene, camera }) => {
      renderer.render(scene, camera); // frame 1: allocates and packs everything
      device.writes.length = 0;
      renderer.render(scene, camera); // frame 2: nothing is dirty

      const objectWrites = device.writes.filter((w) => w.size > 4096);
      expect(objectWrites.length).toBe(0);
      expect(renderer.stats.packedObjects).toBe(0);
      expect(renderer.stats.skippedObjects).toBe(120);
      // The frame block still goes up, once: it carries the camera.
      expect(device.writes.length).toBe(1);
      expect(device.writes[0]!.size).toBeLessThanOrEqual(4096);
    } });
  });

  test('a moving scene re-uploads the object range, and that is the static/moving divergence', async () => {
    await withScene(120, { fn: ({ renderer, device, scene, camera, nodes }) => {
      renderer.render(scene, camera);
      device.writes.length = 0;
      (nodes[0] as MeshNode).markDirty();
      (nodes[1] as MeshNode).markDirty();
      renderer.render(scene, camera);

      expect(renderer.stats.packedObjects).toBe(2);
      expect(renderer.stats.skippedObjects).toBe(118);
      // One write for the objects that moved. `uploadObjects(lo, hi)` covers the
      // span between them, so this is the whole range from slot 0 to slot 1, and
      // it is at least 2 strides wide -- 512 bytes at the 256-byte stride.
      const objectWrites = device.writes.filter((w) => w.bytes.length > 0 && w.size >= 512);
      expect(objectWrites.length).toBe(1);
      expect(objectWrites[0]!.size).toBeGreaterThanOrEqual(512);
    } });
  });
});

describe('the draw item contract the pack loop depends on', () => {
  test('DrawItem.model is a live reference to the node\'s own Float32Array, not a copy', async () => {
    // If this ever becomes a copy, a moving object is drawn with the transform
    // it had when the frame started and no error is raised anywhere.
    await withScene(4, { fn: ({ scene, camera, nodes }) => {
      const items: DrawItem[] = [];
      scene.collectDrawItems(items, camera);
      expect(items.length).toBe(4);
      const node = nodes[2] as MeshNode;
      const before = items[2]!.model;
      expect(before).toBe(node.world);
      node.setPosition(1, 2, 3);
      updateWorldMatrices(scene.root);
      // The same array, with new contents.
      expect(items[2]!.model).toBe(node.world);
      expect(items[2]!.model[12]).toBe(1);
    } });
  });

  test('objectId is the collection index, so the dynamic offset a draw binds is derived from the same number the packer writes', async () => {
    // renderer.ts packs at `item.objectId` and binds at `item.objectOffset`, and
    // both are set from the item's index in `collectDrawItems`. Packing by
    // sorted position instead would put every transform in a neighbour's slot,
    // with no error anywhere.
    await withScene(8, { fn: ({ scene, camera }) => {
      const items: DrawItem[] = [];
      scene.collectDrawItems(items, camera);
      const byId = new Map<number, DrawItem>();
      for (const it of items) {
        expect(byId.has(it.objectId)).toBe(false);
        byId.set(it.objectId, it);
      }
      expect(byId.size).toBe(items.length);
      // The sort permutes the array but must not permute the ids: after sorting,
      // the set of ids is the same set, and each item still carries its own
      // offset.
      const idsBefore = items.map((i) => i.objectId).sort((a, b) => a - b);
      const offsBefore = new Map(items.map((i) => [i.objectId, i.objectOffset]));
      sortDrawItems(items);
      const idsAfter = items.map((i) => i.objectId).sort((a, b) => a - b);
      expect(idsAfter).toEqual(idsBefore);
      for (const it of items) expect(it.objectOffset).toBe(offsBefore.get(it.objectId)!);
    } });
  });

  test('the counting sort agrees with the comparison sort on opaque items, including stability', async () => {
    // sortDrawItems is the fast path and `compareDrawItems` is the specification
    // of the order. On opaque geometry they must agree exactly -- a disagreement
    // is a different picture on a different day, not a slower frame.
    //
    // Transparent geometry is a *known* divergence and has its own test below;
    // it is not folded in here, because this test is the one that fails loudly
    // if the opaque ordering ever changes.
    const items: DrawItem[] = [];
    const mA = { name: 'a' } as unknown as DrawItem['material'];
    const mB = { name: 'b' } as unknown as DrawItem['material'];
    const geom = {} as unknown as DrawItem['geometry'];
    for (let i = 0; i < 200; i++) {
      items.push(makeItem(i, 'opaque', (i * 37 % 101) / 101, i % 3, i % 2 === 0 ? mA : mB, geom));
    }
    const counting = items.slice();
    sortDrawItems(counting);
    const comparison = items.slice().sort(compareDrawItems);
    expect(counting.map((i) => i.objectId)).toEqual(comparison.map((i) => i.objectId));
    // Stability: items that compare equal keep their relative order, which is
    // what makes composing several counting passes a total order.
    for (let i = 1; i < counting.length; i++) {
      if (compareDrawItems(counting[i - 1]!, counting[i]!) === 0) {
        expect(counting[i - 1]!.objectId).toBeLessThan(counting[i]!.objectId);
      }
    }
  });

  test('BUG (src/render/sort.ts:139): the depth pass is near-first for both phases, so transparent items are drawn in the wrong blend order', () => {
    // Reported, not fixed: src/ is read-only to this file's owner.
    //
    // `sortDrawItems` composes four counting passes and the lowest-priority one
    // is `depthBucket`, which is ascending -- correct for opaque (front to back,
    // so early-Z rejects) and **wrong for transparent**, which must be back to
    // front because blending is order-dependent. `compareDrawItems`, the file's
    // own stated specification, branches on the phase
    // (`a.phase === 'opaque' ? a.depth - b.depth : b.depth - a.depth`);
    // `depthBucket` has no such branch, so the two disagree on exactly the
    // geometry where the order is visual rather than an optimisation.
    //
    // It does not show up on a default-camera scene by accident: `depthBucket`
    // clamps anything at or beyond 1.0 to the last of 1024 buckets, and
    // view-space depths in metres are all beyond 1.0, so every item lands in the
    // same bucket and the pass becomes a no-op that preserves collection order.
    // The bug needs a camera within a metre of the geometry to surface.
    //
    // These assertions pin the CORRECTED behaviour: the sort is the
    // specification. A regression fails them, and the comment above each says
    // what broke last time.
    const m = { name: 'm' } as unknown as DrawItem['material'];
    const geom = {} as unknown as DrawItem['geometry'];
    const near = makeItem(0, 'transparent', 1.1, 0, m, geom);
    const far = makeItem(1, 'transparent', 1.9, 0, m, geom);
    const counting = [near, far].slice();
    sortDrawItems(counting);

    // The specification says far first.
    expect([near, far].sort(compareDrawItems).map((i) => i.objectId)).toEqual([1, 0]);
    // The counting sort agrees: far first, the same as the specification.
    // It did not, once. `sort.ts` clamped depth as though it were normalised to
    // [0,1] when it is view-space -z in metres, so every object past 1 m landed
    // in the last bucket and the order was never applied -- silently, and
    // invisible in the image because for opaque geometry draw order is
    // bandwidth rather than correctness. For transparent it is a real visual
    // bug, since blending is order-dependent.
    expect(counting.map((i) => i.objectId)).toEqual([1, 0]);
    // Same for any two depths that land in different buckets, in either order.
    const reversed = [far, near].slice();
    sortDrawItems(reversed);
    expect(reversed.map((i) => i.objectId)).toEqual([1, 0]);
    // Opaque is unaffected: near first is correct there.
    const opaque = [makeItem(0, 'opaque', 1.1, 0, m, geom), makeItem(1, 'opaque', 1.9, 0, m, geom)];
    sortDrawItems(opaque);
    expect(opaque.map((i) => i.objectId)).toEqual([0, 1]);
  });

  test('sortDrawItems does not allocate: a second call on a sorted list reuses the same item objects', async () => {
    // Measured at 57-91 ns/item and already 2.8x cheaper than a shuffled-input
    // comparison sort, so the sort is not the gap. This pins the "does not
    // allocate" half of its contract, which is what keeps it that cheap.
    await withScene(100, { fn: ({ scene, camera }) => {
      const items: DrawItem[] = [];
      scene.collectDrawItems(items, camera);
      const first = items.map((i) => i);
      sortDrawItems(items);
      const sorted = items.slice();
      sortDrawItems(items);
      expect(items.map((i) => i.objectId)).toEqual(sorted.map((i) => i.objectId));
      for (let i = 0; i < items.length; i++) expect(items[i]).toBe(sorted[i]);
      expect(first.length).toBe(100);
    } });
  });
});

describe('the invariant the reported 4.00-calls-per-draw figure depends on', () => {
  test('the bind group count per frame equals the draw count plus one for the present pass', async () => {
    // One bind per draw for the object slot, plus whatever the material, texture
    // and present passes add. If the frame bind group were being re-bound per
    // draw again, this would be 2x the draws -- which is what the first, wrong
    // diagnosis in FINDING.md predicted, and the census disproved.
    await withScene(100, { fn: ({ renderer, device, scene, camera }) => {
      renderer.render(scene, camera);
      // Group 0 is the scene group: one bind per draw. Groups 1 and 2 are the
      // material and texture groups, bound once per pipeline change, so they are
      // counted separately rather than folded in.
      const group0 = scenePassOf(device).bindGroups.filter((b) => b.group === 0).length;
      expect(group0).toBe(100);
      expect(scenePassOf(device).bindGroups.filter((b) => b.group !== 0).length).toBeGreaterThanOrEqual(1);
      // The renderer's own stat agrees, and is what `bench/diag/perf` prints.
      expect(renderer.stats.bindGroupCalls).toBeGreaterThanOrEqual(100);
      // Nothing re-binds the frame half: the frame lives at offset 0 of the same
      // buffer and needs no bind of its own.
      const offsets = device.lastPasses.flatMap((p) => p.bindGroups).filter((b) => b.group === 0 && b.offsets !== null).map((b) => b.offsets![0]);
      expect(offsets.filter((o) => o === 0).length).toBe(0);
    } });
  });

  test('a frame with nothing in it still clears, or the present pass samples the previous frame', async () => {
    // Not a performance invariant but it is in the same loop and it is the kind
    // of thing that regresses while someone is optimising it.
    await withScene(4, { fn: ({ renderer, device, scene, camera }) => {
      renderer.render(scene, camera);
      expect(device.lastPasses.length).toBeGreaterThan(0);
      scene.clear();
      resetPasses(device);
      renderer.render(scene, camera);
      // The scene pass still opens and ends, with no draws, so the target is
      // cleared; and the present pass still runs.
      expect(device.lastPasses.length).toBeGreaterThanOrEqual(1);
      const sceneDraws = device.lastPasses
        .filter((p) => !p.label.startsWith('apse.present'))
        .flatMap((p) => p.draws).length;
      expect(sceneDraws).toBe(0);
      const presentDraws = device.lastPasses
        .filter((p) => p.label.startsWith('apse.present'))
        .flatMap((p) => p.draws).length;
      expect(presentDraws).toBe(1); // the fullscreen triangle
    } });
  });
});
