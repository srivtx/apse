/**
 * The encode loop's per-draw cost and, more importantly, its per-draw *shape*.
 *
 * ## Why the second half exists
 *
 * A call census — how many `setBindGroup` / `setVertexBuffer` / `drawIndexed`
 * a frame issues — is the natural thing to assert about an encode loop, and it
 * is what `bench/diag/perf/FINDING.md` is built on. It is also, on its own, a
 * trap. It counts calls; it cannot price an argument, and it cannot see a bind
 * of the *wrong object*.
 *
 * Both of those mattered here, in opposite directions:
 *
 *   1. The per-draw cost was dominated by the **argument** to `setBindGroup`,
 *      not by the call. The dynamic-offsets scratch was a `Uint32Array`, and
 *      Blink's `sequence<>` IDL conversion falls off its fast path for a typed
 *      array and walks the generic iterator protocol instead. Same call, same
 *      call count, ~0.9 us/draw more expensive. No census could ever have seen
 *      it, because the number of calls is identical either way.
 *   2. An attempt to cache the material bind groups per material put the
 *      re-read *after* the scene bind, so the first draw of every pass bound
 *      `null` for group 0 and the rest of the pass ran with no scene group
 *      bound. The call census was **byte-identical** to the working version —
 *      4.02 calls per draw either way. It only showed up as a 1.5 us/draw
 *      regression in a real browser, because `FakeFramePass` does not validate.
 *
 * So these tests assert three things a census cannot: that the object handed to
 * each `setBindGroup` is the right one, that it is non-null, and that the
 * per-draw scratch is the kind of value Blink converts cheaply.
 */

import { afterAll, beforeAll, describe, expect, test } from 'bun:test';

import { BIND_GROUP, FRAME_BLOCK, SCENE_FRAME_BYTES } from '../src/core/slot.ts';
import { Scene, MeshNode, PerspectiveCamera } from '../src/scene/index.ts';
import { box } from '../src/geometry/primitives/box.ts';
import { upload } from '../src/geometry/mesh.ts';
import { basicMaterial } from '../src/material/basic.ts';
import { diffuseMaterial } from '../src/material/diffuse.ts';
import { FakeFrameDevice, FakeFramePass, fakeWebGpuCanvas, installWebGpuBitmaps, withGpu } from './render-fakes.ts';
import { fakeLimits } from './fake-device.ts';
import { Renderer } from '../src/render/renderer.ts';
import type { Drawable } from '../src/render/types.ts';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

let uninstallBitmaps: (() => void) | null = null;

beforeAll(() => { uninstallBitmaps = installWebGpuBitmaps(); });
afterAll(() => { uninstallBitmaps?.(); uninstallBitmaps = null; });

interface Enc {
  readonly renderer: Renderer;
  readonly device: FakeFrameDevice;
  readonly scene: Scene;
  readonly camera: PerspectiveCamera;
  /** Scene passes, i.e. everything except the present pass. */
  scenePasses(): readonly FakeFramePass[];
}

/**
 * A renderer on a recording fake, with a `grid` of distinct meshes already
 * added and returned as the node list so a test can move them.
 *
 * **Distinct meshes, deliberately.** One shared mesh would make every
 * `lastVB` / `lastIB` guard hit, so the loop would take its cheap path and the
 * per-draw cost being measured would not be the one that is in dispute.
 */
async function withScene<T>(
  count: number,
  opts: { materials?: number; renderer?: Parameters<typeof Renderer.create>[1] },
  fn: (e: Enc & { nodes: MeshNode[]; materials: Drawable[] }) => Promise<T> | T,
): Promise<T> {
  const device = new FakeFrameDevice({ limits: fakeLimits(), features: [] });
  const { canvas } = fakeWebGpuCanvas(320, 200);
  return withGpu({ device: device as unknown as GPUDevice, preferredFormat: 'bgra8unorm' }, async () => {
    const renderer = await Renderer.create(canvas, opts.renderer ?? {});
    const gpu = device as unknown as GPUDevice;
    const camera = new PerspectiveCamera({ fov: 45, near: 0.1, far: 500, aspect: 320 / 200 });
    camera.lookAt([0, 0, 60], [0, 0, 0], [0, 1, 0]);
    const scene = new Scene('encode');

    // `materialCount` distinct materials, so the material-change path is real.
    const materials: Drawable[] = [];
    for (let m = 0; m < Math.max(1, opts.materials ?? 1); m++) {
      materials.push(m % 2 === 0
        ? await basicMaterial(gpu, { targetFormat: renderer.sceneFormat, name: `basic${m}` })
        : await diffuseMaterial(gpu, { targetFormat: renderer.sceneFormat, name: `diffuse${m}` }));
    }

    const nodes: MeshNode[] = [];
    const perRow = Math.ceil(Math.sqrt(count)) + 1;
    for (let i = 0; i < count; i++) {
      const node = new MeshNode({
        name: `m${i}`,
        mesh: upload(gpu, box({ width: 0.5 })),
        material: materials[i % materials.length]!,
        position: [(i % perRow) * 0.8 - perRow * 0.4, Math.floor(i / perRow) * 0.8 - perRow * 0.4, 0],
      });
      scene.add(node);
      nodes.push(node);
    }

    const enc: Enc & { nodes: MeshNode[]; materials: Drawable[] } = {
      renderer, device, scene, camera, nodes, materials,
      scenePasses: () => device.lastPasses.filter((p: FakeFramePass) => !p.label.startsWith('apse.present')),
    };
    try {
      return await fn(enc);
    } finally {
      renderer.dispose();
    }
  });
}

// ---------------------------------------------------------------------------
// The call census, which is the floor this loop must not fall below
// ---------------------------------------------------------------------------

describe('the encode loop — the per-draw call census', () => {
  test('500 distinct meshes are exactly 4.00 calls per draw', async () => {
    await withScene(500, {}, (e) => {
      e.renderer.render(e.scene, e.camera);

      // Every pass of the frame, the present pass included: the numbers below
      // are the ones FINDING.md quotes, and they include its one extra group-0
      // bind and its fullscreen draw.
      const counts = new Map<string, number>();
      for (const pass of e.device.lastPasses) {
        for (const call of pass.calls) counts.set(call, (counts.get(call) ?? 0) + 1);
      }
      const draws = e.renderer.stats.drawCalls;
      expect(draws).toBe(500);

      // One scene bind (the dynamic offset differs every draw, so it cannot be
      // suppressed), one vertex buffer, one index buffer, one draw. None is
      // redundant: the meshes are different meshes.
      expect(counts.get('setBindGroup:0')).toBe(501);
      expect(counts.get('setVertexBuffer:0')).toBe(501);
      expect(counts.get('drawIndexed')).toBe(500);
      expect(counts.get('setIndexBuffer:uint16')).toBe(500);

      // The frame's cost is these four calls, so the total is the number to
      // defend. 4.02 rather than 4.00 is the pass and the pipeline.
      const total = [...counts]
        .filter(([k]) => k !== 'beginRenderPass' && k !== 'end')
        .reduce((sum, [, n]) => sum + n, 0);
      expect(total / draws).toBeCloseTo(4.02, 2);
    });
  });

  test('the census is unchanged when a frame mixes several materials', async () => {
    // A caching bug that hid behind one material — a bind group read once and
    // never re-read — shows up here and nowhere else.
    await withScene(500, { materials: 4 }, (e) => {
      e.renderer.render(e.scene, e.camera);
      const draws = e.renderer.stats.drawCalls;
      expect(draws).toBe(500);
      // Every draw still binds group 0 with its own offset. If the offset were
      // hoisted per material, this would be 4 and every object in a run would be
      // drawn with the first one's matrix.
      const sceneBinds = e.scenePasses()
        .flatMap((p: FakeFramePass) => p.calls)
        .filter((c: string) => c === `setBindGroup:${BIND_GROUP.scene}`).length;
      expect(sceneBinds).toBe(draws);
    });
  });
});

// ---------------------------------------------------------------------------
// What a census cannot see
// ---------------------------------------------------------------------------

describe('the encode loop — the arguments, not just the calls', () => {
  test('every scene bind carries a real bind group, never null', async () => {
    // The regression this file exists for. Caching the bind groups per material
    // and reading them *after* the scene bind leaves the first draw of each pass
    // binding null — which is legal to issue, invalid to draw with, and
    // invisible to a call census.
    await withScene(50, { materials: 3 }, (e) => {
      e.renderer.render(e.scene, e.camera);
      for (const pass of e.scenePasses()) {
        for (const bind of pass.bindGroups) {
          expect(bind.bindGroup).toBeDefined();
          expect(bind.bindGroup).not.toBeNull();
        }
      }
    });
  });

  test('the group-0 bind is the material\'s own scene bind group', async () => {
    // Not "a bind group" but *the right one*: a stale group from a previous
    // material binds cleanly and draws another material's frame uniform.
    await withScene(30, { materials: 3 }, (e) => {
      e.renderer.render(e.scene, e.camera);
      const expected = new Set(e.materials.map((m) => m.sceneBindGroup));
      const sceneBinds = e.scenePasses().flatMap((p) => p.bindGroups)
        .filter((b) => b.group === BIND_GROUP.scene);
      expect(sceneBinds.length).toBeGreaterThan(0);
      for (const bind of sceneBinds) {
        expect(expected.has(bind.bindGroup)).toBe(true);
      }
    });
  });

  test('each scene bind carries a distinct, correctly strided dynamic offset', async () => {
    // 256 is `ObjectUniforms` stride, which must be a multiple of
    // `minUniformBufferOffsetAlignment`. A stride of 1 or of 3 is a validation
    // error at the driver and a wrong matrix in every frame that gets through.
    await withScene(200, {}, (e) => {
      e.renderer.render(e.scene, e.camera);
      const offsets = e.scenePasses().flatMap((p: FakeFramePass) => p.bindGroups)
        .filter((b: { group: number }) => b.group === BIND_GROUP.scene)
        .flatMap((b: { offsets: number[] | null }) => b.offsets ?? []);
      // One bind per draw, plus the one the present pass's own bind adds — which
      // is in a different pass, hence not counted here.
      expect(offsets.length).toBe(e.renderer.stats.drawCalls);
      for (const o of offsets) {
        expect(Number.isInteger(o)).toBe(true);
        // A multiple of the 256 stride, and past the frame region: object 0 sits
        // behind the frame block rather than on top of the camera.
        expect(o % 256).toBe(0);
        expect(o).toBeGreaterThanOrEqual(SCENE_FRAME_BYTES);
      }
      // And no two draws share one, or every object in the frame would be drawn
      // with a single neighbour's transform.
      expect(new Set(offsets).size).toBe(offsets.length);
    });
  });
});

// ---------------------------------------------------------------------------
// The argument-shape regression, asserted on the code that produces it
// ---------------------------------------------------------------------------

describe('the dynamic-offset scratch is the kind Blink converts cheaply', () => {
  test('it is a plain Array, not a typed array', async () => {
    // A census cannot see this, and it cost more than the entire per-draw gap
    // against three.js. The IDL is `sequence<GPUBufferDynamicOffset>`; both a
    // `Uint32Array` and an `Array` convert, but only the `Array` takes Blink's
    // `v8::Array` fast path. So the assertion is on the constructor, and it is
    // a source-level fact rather than a behavioural one because the difference
    // is in a conversion this fake never performs.
    const src = await Bun.file(new URL('../src/render/renderer.ts', import.meta.url)).text();
    const decl = src.match(/_dynamicOffsets[^=]*=\s*([^;]+);/);
    expect(decl).not.toBeNull();
    // A typed array would read `new Uint32Array(1)`, `new Int32Array(1)`, or
    // any other `new XArray(...)`. A plain array literal is what is wanted.
    expect(decl![1]!.trim()).not.toMatch(/new\s+\w*Array/);
    expect(decl![1]!.trim()).toMatch(/^\[/);
  });
});

// ---------------------------------------------------------------------------
// The invariants a cheaper encode loop must not break
// ---------------------------------------------------------------------------

describe('the encode loop — what must not change', () => {
  test('a static scene writes zero uniform bytes after its first frame', async () => {
    // The optimisation is to the encode loop, so the thing most at risk from it
    // is the pack step being short-circuited along with everything else. This is
    // a correctness property, not a speed one.
    await withScene(100, {}, (e) => {
      e.renderer.render(e.scene, e.camera);
      const afterFirst = e.device.writes.length;
      e.renderer.render(e.scene, e.camera);
      e.renderer.render(e.scene, e.camera);

      // The frame and the objects share one buffer and one label, so "wrote no
      // object bytes" is "wrote only the frame block's worth". Frame 1 writes
      // the whole scene region; later frames write the frame block alone. The
      // exact frame-block size is `FRAME_BLOCK.size`, imported here rather than
      // restated, so this cannot drift from the thing it is asserting about.
      const sceneLabel = e.device.writes[0]!.label;
      const firstSceneWrite = e.device.writes.find((w) => w.label === sceneLabel)!;
      const laterSceneWrites = e.device.writes.slice(afterFirst)
        .filter((w) => w.label === sceneLabel);
      expect(firstSceneWrite.size).toBeGreaterThan(FRAME_BLOCK.size);
      for (const w of laterSceneWrites) {
        expect(w.size).toBe(FRAME_BLOCK.size);
      }
      // And nothing at all is written for a material that did not change.
      expect(e.device.writes.slice(afterFirst).map((w) => w.label))
        .not.toContain('apse:basic:material');
    });
  });

  test('an unindexed mesh still calls draw, not drawIndexed', async () => {
    // The `ib` local replaced a second `geometry.indexBuffer` read at the draw
    // site. If it were ever wrong, an unindexed mesh would take the indexed path.
    await withScene(4, {}, (e) => {
      const unindexed = {
        layout: box().layout,
        vertexBuffer: e.device.createBuffer({ label: 'bare', size: 48, usage: 0x20 }) as unknown as GPUBuffer,
        indexBuffer: null,
        indexCount: 3,
        instanceCount: 1,
        firstInstance: 0,
        instanceBuffer: null,
      };
      e.scene.add(new MeshNode({ name: 'unindexed', mesh: unindexed, material: e.materials[0]! }));
      e.renderer.render(e.scene, e.camera);
      const calls = e.scenePasses().flatMap((p) => p.calls);
      expect(calls.filter((c) => c === 'draw').length).toBeGreaterThanOrEqual(1);
    });
  });

  test('an instanced mesh still binds slot 1 even when one instance is drawn', async () => {
    // The guard is `instanceBuffer !== null`, never `instanceCount > 1`, and
    // nothing in this change touches that — but the instanced branch shares the
    // buffer-getter locals with the fix, so it is asserted here.
    const device = new FakeFrameDevice({ limits: fakeLimits(), features: [] });
    const { canvas } = fakeWebGpuCanvas(320, 200);
    await withGpu({ device: device as unknown as GPUDevice, preferredFormat: 'bgra8unorm' }, async () => {
      const renderer = await Renderer.create(canvas, {});
      const gpu = device as unknown as GPUDevice;
      const camera = new PerspectiveCamera({ fov: 45, near: 0.1, far: 100, aspect: 1.6 });
      camera.lookAt([0, 0, 30], [0, 0, 0], [0, 1, 0]);
      const scene = new Scene('inst');
      const { uploadInstances, InstanceData } = await import('../src/geometry/instanced.ts');
      const { instancedMaterial } = await import('../src/material/instanced.ts');
      const matrices = new Float32Array(4 * 16);
      for (let i = 0; i < 4; i++) {
        matrices[i * 16] = 1; matrices[i * 16 + 5] = 1;
        matrices[i * 16 + 10] = 1; matrices[i * 16 + 15] = 1;
      }
      const material = await instancedMaterial(gpu, { targetFormat: renderer.sceneFormat });
      scene.add(new MeshNode({
        name: 'grid',
        mesh: upload(gpu, box(), {
          instances: uploadInstances(gpu, InstanceData.fromMatrices(matrices, { name: 'g' })),
          instanceCount: 1,
        }),
        material,
      }));
      renderer.render(scene, camera);
      const slot1 = device.lastPasses
        .flatMap((p) => p.vertexBuffers)
        .filter((v) => v.slot === 1);
      expect(slot1.length).toBeGreaterThanOrEqual(1);
      renderer.dispose();
    });
  });

  test('a disposed mesh still raises rather than silently yielding a dead buffer', async () => {
    // The `ib` local is a *saved* value, not a bypassed getter: it came from
    // `geometry.indexBuffer`, which checks liveness. This asserts that check is
    // still on the path the draw takes.
    const device = new FakeFrameDevice({ limits: fakeLimits(), features: [] });
    const { canvas } = fakeWebGpuCanvas(320, 200);
    await withGpu({ device: device as unknown as GPUDevice, preferredFormat: 'bgra8unorm' }, async () => {
      const renderer = await Renderer.create(canvas, {});
      const gpu = device as unknown as GPUDevice;
      const camera = new PerspectiveCamera({ fov: 45, near: 0.1, far: 100, aspect: 1.6 });
      camera.lookAt([0, 0, 30], [0, 0, 0], [0, 1, 0]);
      const scene = new Scene('dead');
      const material = await basicMaterial(gpu, { targetFormat: renderer.sceneFormat });
      const mesh = upload(gpu, box());
      scene.add(new MeshNode({ name: 'a', mesh, material }));
      renderer.render(scene, camera);
      mesh.dispose();
      expect(() => renderer.render(scene, camera)).toThrow();
      renderer.dispose();
    });
  });
});
