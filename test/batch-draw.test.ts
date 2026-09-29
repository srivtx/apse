import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { FakeFrameDevice, fakeWebGpuCanvas, withGpu, installWebGpuBitmaps } from './render-fakes.ts';
import { fakeLimits } from './fake-device.ts';
import { Renderer } from '../src/render/renderer.ts';
import { Scene } from '../src/scene/graph.ts';
import { MeshNode } from '../src/scene/node.ts';
import { PerspectiveCamera } from '../src/scene/camera.ts';
import { mergeMeshes, uploadBatch } from '../src/geometry/batch.ts';
import { box } from '../src/geometry/primitives/box.ts';
import { sphere as sph } from '../src/geometry/primitives/sphere.ts';
import { basicMaterial } from '../src/material/basic.ts';

let un: () => void = () => {};
beforeAll(() => { un = installWebGpuBitmaps(); });
afterAll(() => un());

describe('batched sub-range draws', () => {
  test('each sub-range draws its own geometry, not source 0 every time', async () => {
  const device = new FakeFrameDevice({ limits: fakeLimits(), features: [] });
  const { canvas } = fakeWebGpuCanvas(640, 480);
  await withGpu({ device: device as unknown as GPUDevice, preferredFormat: 'bgra8unorm' }, async () => {
    const renderer = await Renderer.create(canvas, {});
    const material = await basicMaterial(renderer.device.device, { targetFormat: renderer.sceneFormat });
    // Two sources with different vertex counts, so the second sub-range cannot
    // coincide with the first by accident.
    const batch = mergeMeshes([
      { mesh: box({ width: 1 }) },
      { mesh: sph({ widthSegments: 8, heightSegments: 8 }) },
    ]);
    const gpu = uploadBatch(renderer.device.device, batch);
    const scene = new Scene('b');
    const n_src = batch.ranges.length;
    for (let i = 0; i < n_src; i++) {
      const n = new MeshNode({ name: 'b' + i, mesh: gpu.sub(i), material });
      n.setPosition(i * 3 - 3, 0, 0);
      scene.add(n);
    }
    const cam = new PerspectiveCamera({ fov: 45, near: 0.1, far: 100, aspect: 640 / 480 });
    cam.lookAt([0, 0, 12], [0, 0, 0], [0, 1, 0]);
    renderer.camera = cam;
    renderer.render(scene, cam);
    const draws = device.lastPasses.flatMap((p) => p.calls).filter((c) => c.startsWith('drawIndexed'));
    // The regression: distinct sub-ranges must not all report 0/0, or the
    // renderer draws source 0 for every one of them.
    const zeros = Array.from({ length: n_src }, (_, i) => gpu.sub(i))
      .filter((s) => s.firstIndex === 0 && s.baseVertex === 0).length;
    expect(zeros).toBeLessThan(n_src);
    expect(draws.length).toBe(n_src);
  });
});
});
