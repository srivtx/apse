/**
 * The tree-shaking test, as code.
 *
 * A realistic minimal application: a renderer, a camera, one lit mesh, one
 * animation loop. Not a toy import — if the size gate measures `import { box }`
 * it proves nothing about what an actual page downloads.
 *
 * This file is bundled by `scripts/build.ts` and its gzip size is compared
 * against a hard ceiling. If tree-shaking ever silently stops working, the build
 * fails here rather than shipping a 300 KB renderer that claims to be 20.
 *
 * Everything imported below must be *used*. A dead import is not a size test.
 */

import {
  Renderer,
  PerspectiveCamera,
  Scene,
  MeshNode,
  box,
  upload,
  pbrMaterial,
  quat,
} from '../src/index.ts';

export async function start(canvas: HTMLCanvasElement): Promise<() => void> {
  const renderer = await Renderer.create(canvas, { budget: { cpu: 2 } });
  const mesh = upload(renderer.device.device, box({ width: 1.2 }));
  const material = await pbrMaterial(renderer.device.device, { targetFormat: renderer.sceneFormat });
  const scene = new Scene('spin');
  const camera = new PerspectiveCamera({ fov: 45, near: 0.1, far: 100, aspect: 1 });

  const node = new MeshNode({ name: 'cube', mesh, material });
  node.setPosition(0, 0, -3);
  scene.add(node);
  renderer.camera = camera;

  renderer.start((dt) => {
    quat.fromEulerXYZ(node.rotation, 0, node.rotation[1]! + dt * 0.5, 0);
    node.markDirty();
    camera.lookAt([0, 0, 0], [0, 0, -3], [0, 1, 0]);
  });

  return () => {
    renderer.stop();
    renderer.dispose();
    mesh.dispose();
    material.unref();
    scene.clear();
  };
}
