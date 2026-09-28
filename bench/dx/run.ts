/**
 * bench/dx — the harness that renders the same material in apse and in three.js
 * and counts what it cost to write.
 *
 *   bun run bench/dx/run.ts
 *
 * Two document types, two libraries, one browser process, one device.
 *
 *   /bench/dx/materials.html
 *     apse (WebGPU) and three.js `WebGLRenderer` (WebGL2), one sphere, one
 *     camera, one set of texture bytes, the same shading expressions written
 *     twice. The two images are read back and compared pixel by pixel. Also
 *     runs the silent-failure and error-message probes, and captures the exact
 *     GLSL three.js hands the driver by shadowing `gl.shaderSource`.
 *
 *   /bench/dx/nodes.html
 *     three.js `WebGPURenderer` with a `MeshStandardNodeMaterial`, in its own
 *     document: Chrome drops the first WebGPU instance when a second device is
 *     created in the same document, so the apse page and the node page cannot
 *     share one.
 *
 * The timing section re-opens the materials page `TIMING_SAMPLES` times, because
 * a shader program compiles once and a page that has already drawn measures a
 * cache hit.
 *
 * The line counts are read out of the two material files by this script, from
 * `// @count:<section>` … `// @count:end` markers, rather than asserted here.
 *
 * Nothing is written to disk. The report is stdout.
 */

import { serve } from 'bun';
import puppeteer from 'puppeteer';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

// The repo root. This file lives two levels down (bench/dx/), so one '..' is not enough.
const ROOT = new URL('../..', import.meta.url).pathname;
const PORT = 8791;
const TIMING_SAMPLES = 5;
const WIDTH = 640;
const HEIGHT = 360;

// ---------------------------------------------------------------------------
// The in-page harness
// ---------------------------------------------------------------------------

const PAGE_MATERIALS = /* html */ `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>apse dx</title>
<style>body{margin:0;background:#000}canvas{position:fixed;top:0;left:-99999px}</style>
</head><body><script type="module">
const g = globalThis;
const W = ${WIDTH}, H = ${HEIGHT};
const CLEAR = [0.02, 0.02, 0.03];

g.__dxError = null;
window.addEventListener('error', (e) => { g.__dxError = String(e.message); });
window.addEventListener('unhandledrejection', (e) => {
  const r = e.reason;
  g.__dxError = (r && r.stack) ? String(r.stack) : String(r);
  console.error('[dx] unhandledrejection: ' + g.__dxError);
});
console.info('[dx] script start');

const state = {};

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  // CanvasSizer derives the backing store from the layout box, so a canvas that
  // was merely shrunk to hide it renders into a 1x1 target and every measurement
  // still looks healthy. Same trap as bench/index.html.
  c.style.width = w + 'px';
  c.style.height = h + 'px';
  document.body.appendChild(c);
  Object.defineProperty(c, 'clientWidth', { value: w, configurable: true });
  Object.defineProperty(c, 'clientHeight', { value: h, configurable: true });
  c.width = w; c.height = h;
  return c;
}

/** De-interleaves apse's MeshData into three.js BufferAttributes. */
function toBufferGeometry(THREE, meshData) {
  const g = new THREE.BufferGeometry();
  const stride = meshData.layout.stride >> 2;
  const src = meshData.vertexData;
  const n = meshData.vertexCount;
  for (const [name, offset] of [['position', 0], ['normal', 3], ['uv', 6]]) {
    const size = name === 'uv' ? 2 : 3;
    const out = new Float32Array(n * size);
    for (let i = 0; i < n; i++) out.set(src.subarray(i * stride + offset, i * stride + offset + size), i * size);
    g.setAttribute(name, new THREE.BufferAttribute(out, size));
  }
  g.setIndex(new THREE.BufferAttribute(
    meshData.indexData instanceof Uint32Array
      ? new Uint32Array(meshData.indexData) : new Uint16Array(meshData.indexData), 1));
  g.boundingSphere = new THREE.Sphere(
    new THREE.Vector3(...meshData.boundingSphere.subarray(0, 3)), meshData.boundingSphere[3]);
  return g;
}

// --- shared inputs, one source of truth -------------------------------------
console.info('[dx] importing apse material');
const apseMat = await import('./apse-custom-material.ts');
console.info('[dx] importing three material');
const threeMat = await import('./three-custom-material.js');
console.info('[dx] materials loaded');
const P = apseMat.PARAMS;

// ---------------------------------------------------------------------------
g.__init = async () => {
  const apse = await import('/dist/index.js');
  const THREE = await import('/node_modules/three/build/three.module.js');

  const validationErrors = [];
  // apse routes uncaptured WebGPU errors through onValidationError. A raw
  // listener is added as well, so the report can say whether the browser itself
  // ever shows the underlying shader diagnostic or whether it is swallowed.
  const rawDeviceErrors = [];
  const listen = (d) => d.addEventListener('uncapturederror', (e) => {
    rawDeviceErrors.push(String(e.error && e.error.message ? e.error.message : e.error).slice(0, 900));
  });

  const canvas = makeCanvas(W, H);
  const renderer = await apse.Renderer.create(canvas, {
    sampleCount: 1,
    onValidationError: (m) => validationErrors.push(String(m)),
  });
  if (canvas.width < W || canvas.height < H) {
    throw new Error('apse canvas backing store is ' + canvas.width + 'x' + canvas.height);
  }

  const glCanvas = makeCanvas(W, H);
  const gl = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: false, alpha: false });
  gl.setPixelRatio(1);
  gl.setSize(W, H, false);
  // Compare in linear. three.js defaults to writing sRGB-encoded values into the
  // default framebuffer, which the generated prefix converts for you; apse's
  // capture target is a plain unorm surface. Leaving the OETF in on one side
  // only would make the diff a colour-space measurement.
  gl.outputColorSpace = THREE.LinearSRGBColorSpace;
  const clearColor = new THREE.Color().setRGB(CLEAR[0], CLEAR[1], CLEAR[2], THREE.LinearSRGBColorSpace);
  gl.setClearColor(clearColor, 1);
  const ctx = gl.getContext();
  const rt = new THREE.WebGLRenderTarget(W, H, {
    type: THREE.UnsignedByteType, format: THREE.RGBAFormat,
    colorSpace: THREE.NoColorSpace, depthBuffer: true, stencilBuffer: false,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  });

  // One geometry, uploaded to WebGPU and mirrored into WebGL from the very same
  // Float32Array, so no pixel difference can be blamed on the mesh.
  const meshData = apse.sphere({ radius: 1, widthSegments: 64, heightSegments: 48 });
  const gpuMesh = apse.upload(renderer.device.device, meshData);
  const geometry = toBufferGeometry(THREE, meshData);
  const albedoBytes = apseMat.ALBEDO_BYTES;
  const albedoView = apseMat.createAlbedoView(renderer.device.device);
  const albedoTex = threeMat.createAlbedoTexture(THREE);

  state.THREE = THREE; state.apse = apse; state.apseRenderer = renderer;
  state.gl = gl; state.rt = rt; state.gpuMesh = gpuMesh; state.geometry = geometry;
  state.albedoView = albedoView; state.albedoTex = albedoTex;
  listen(renderer.device.device);
  state.validationErrors = validationErrors;
  state.rawDeviceErrors = rawDeviceErrors;
  state.materials = {};

  return {
    apseAdapter: renderer.describeGpu(),
    apseFeatureLevel: renderer.featureLevel,
    apseBacking: [canvas.width, canvas.height],
    threeVersion: THREE.REVISION,
    threeContext: gl.capabilities.isWebGL2 ? 'webgl2' : 'webgl1',
    threeVendor: ctx.getParameter(ctx.VENDOR),
    threeRenderer: ctx.getParameter(ctx.RENDERER),
    mesh: { verts: meshData.vertexCount, indices: meshData.indexData.length, stride: meshData.layout.stride },
    albedoBytes: albedoBytes.length,
  };
};

// --- cameras: one view matrix, each library's own projection ---------------
function buildCameras() {
  const { apse, THREE } = state;
  const apseCam = new apse.PerspectiveCamera({ fov: P.fov, near: P.near, far: P.far, aspect: W / H });
  apseCam.lookAt(P.eye, P.target, [0, 1, 0]);
  apseCam.update(W / H);

  // apse computes world -> view; three.js wants view -> world, which is the
  // inverse. Both then derive the same eye position and the same shading inputs
  // from it. The projection stays each library's own, because the two use
  // different clip-space Z ranges and no honest comparison can share one.
  const view = apseCam.view;
  const inv = new THREE.Matrix4().fromArray(Array.from(view)).invert();

  const threeCam = new THREE.PerspectiveCamera(P.fov, W / H, P.near, P.far);
  threeCam.matrixAutoUpdate = false;
  threeCam.matrixWorldAutoUpdate = false;
  threeCam.matrixWorld.copy(inv);
  threeCam.matrixWorldInverse.fromArray(Array.from(view));
  threeCam.projectionMatrixInverse.copy(threeCam.projectionMatrix).invert();
  return { apseCam, threeCam };
}

function apseScene(mesh, material) {
  const { apse } = state;
  const scene = new apse.Scene({ name: 'dx' });
  const node = new apse.MeshNode({ name: 'sphere', mesh, material });
  scene.add(node);
  return scene;
}

function threeScene(mesh, material) {
  const { THREE } = state;
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(mesh, material));
  return scene;
}

// ---------------------------------------------------------------------------
// 1. Cold time to first drawn frame, for each library
// ---------------------------------------------------------------------------
g.__timing = async () => {
  const { apseRenderer, gl, rt, gpuMesh, geometry, albedoView, albedoTex } = state;
  const { apseCam, threeCam } = buildCameras();
  const out = {};

  // --- apse: from the awaited Material.create to the first readback --------
  {
    const t0 = performance.now();
    const material = await apseMat.createDxMaterial(apseRenderer.device.device, albedoView);
    const tCompiled = performance.now();
    const scene = apseScene(gpuMesh, material);
    const frame = await apseRenderer.capture(scene, apseCam);
    const tDrawn = performance.now();
    // Prove the frame is not blank before quoting a time for producing it.
    let drawn = 0;
    for (let i = 0; i < frame.data.length; i += 4) {
      if (Math.abs(frame.data[i] - 5) + Math.abs(frame.data[i + 1] - 5) + Math.abs(frame.data[i + 2] - 8) > 6) drawn++;
    }
    out.apse = {
      materialCompileMs: +(tCompiled - t0).toFixed(3),
      firstFrameReadbackMs: +(tDrawn - tCompiled).toFixed(3),
      totalMs: +(tDrawn - t0).toFixed(3),
      drawnPixels: drawn,
    };
  }

  // --- three.js: first render compiles, so it is one number ------------------
  {
    const material = threeMat.createShaderMaterial(state.THREE, albedoTex);
    const scene = threeScene(geometry, material);
    const px = new Uint8Array(W * H * 4);
    // The program is compiled exactly once, on this render, so this is the only
    // place the generated GLSL can be observed. Later renders reuse it.
    const gen = threeMat.captureGeneratedGLSL(gl, () => { gl.setRenderTarget(rt); gl.render(scene, threeCam); });
    state.generated = { threeV1: gen };
    const t0 = performance.now();
    gl.setRenderTarget(rt);
    gl.render(scene, threeCam);
    gl.readRenderTargetPixels(rt, 0, 0, W, H, px);
    const t1 = performance.now();
    let drawn = 0;
    for (let i = 0; i < px.length; i += 4) {
      if (Math.abs(px[i] - 5) + Math.abs(px[i + 1] - 5) + Math.abs(px[i + 2] - 8) > 6) drawn++;
    }
    out.three = {
      materialCreateMs: 0,
      firstFrameMs: +(t1 - t0).toFixed(3),
      totalMs: +(t1 - t0).toFixed(3),
      drawnPixels: drawn,
    };
    gl.setRenderTarget(null);
  }

  out.note = 'apse numbers exclude Renderer.create (device + swapchain). three.js numbers exclude new WebGLRenderer(). Both end at a complete pixel readback of the first drawn frame.';
  return out;
};

// ---------------------------------------------------------------------------
// 2. The pixel comparison
// ---------------------------------------------------------------------------
function diff(a, b) {
  const n = a.length / 4;
  const buckets = { gt0: 0, gt1: 0, gt2: 0, gt4: 0, gt8: 0, gt16: 0 };
  let max = 0, sum = 0, worst = null, covered = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const d = Math.max(
      Math.abs(a[o] - b[o]), Math.abs(a[o + 1] - b[o + 1]), Math.abs(a[o + 2] - b[o + 2]));
    sum += d;
    const bg = Math.abs(a[o] - 5) + Math.abs(a[o + 1] - 5) + Math.abs(a[o + 2] - 8) > 6
      || Math.abs(b[o] - 5) + Math.abs(b[o + 1] - 5) + Math.abs(b[o + 2] - 8) > 6;
    if (bg) covered++;
    if (d > 0) buckets.gt0++;
    if (d > 1) buckets.gt1++;
    if (d > 2) buckets.gt2++;
    if (d > 4) buckets.gt4++;
    if (d > 8) buckets.gt8++;
    if (d > 16) buckets.gt16++;
    if (d > max) { max = d; worst = { x: i % W, y: (i / W) | 0, apse: [a[o], a[o + 1], a[o + 2], a[o + 3]], three: [b[o], b[o + 1], b[o + 2], b[o + 3]] }; }
  }
  return { width: W, height: H, pixels: n, maxChannelDiff: max, meanAbsDiff: +(sum / n).toFixed(5), coveredPixels: covered, buckets, worst };
}

/**
 * Like diff(), but only over the pixels where mask says the subject is.
 *
 * The cache-key probe needs this: the two materials cannot both be at the same
 * place, so one of them is always somewhere else in the frame and a whole-image
 * difference would be dominated by a subject that is not under test.
 */
function diffMasked(a, b, mask) {
  const n = a.length / 4;
  const buckets = { gt0: 0, gt1: 0, gt2: 0, gt4: 0, gt8: 0, gt16: 0 };
  let max = 0, sum = 0, worst = null, compared = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i] === 0) continue;
    const o = i * 4;
    const d = Math.max(
      Math.abs(a[o] - b[o]), Math.abs(a[o + 1] - b[o + 1]), Math.abs(a[o + 2] - b[o + 2]));
    compared++; sum += d;
    if (d > 0) buckets.gt0++;
    if (d > 1) buckets.gt1++;
    if (d > 2) buckets.gt2++;
    if (d > 4) buckets.gt4++;
    if (d > 8) buckets.gt8++;
    if (d > 16) buckets.gt16++;
    if (d > max) { max = d; worst = { x: i % W, y: (i / W) | 0, withA: [a[o], a[o + 1], a[o + 2]], alone: [b[o], b[o + 1], b[o + 2]] }; }
  }
  return { pixelsCompared: compared, maxChannelDiff: max, meanAbsDiff: +(sum / Math.max(1, compared)).toFixed(5), buckets, worst };
}

/** 1 where the pixel is not the clear colour, 0 where it is. */
function coverageMask(px) {
  const n = px.length / 4;
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    m[i] = (Math.abs(px[o] - 5) + Math.abs(px[o + 1] - 5) + Math.abs(px[o + 2] - 8) > 6) ? 1 : 0;
  }
  return m;
}

function summary(px) {
  const n = px.length / 4;
  let covered = 0, luma = 0, min = 255, max = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const l = (px[o] + px[o + 1] + px[o + 2]) / 3;
    luma += l; if (l < min) min = l; if (l > max) max = l;
    if (Math.abs(px[o] - 5) + Math.abs(px[o + 1] - 5) + Math.abs(px[o + 2] - 8) > 6) covered++;
  }
  return { coveredPixels: covered, coverage: +(covered / n).toFixed(4), meanLuma: +(luma / n).toFixed(2), minLuma: min, maxLuma: max };
}

/**
 * WebGL's framebuffer origin is bottom-left, so readRenderTargetPixels returns
 * rows bottom-up. apse's capture() copies a WebGPU texture, whose origin is
 * top-left. Flipping is not optional: a vertically flipped sphere has identical
 * coverage, identical mean, identical min and identical max, so a summary-only
 * check passes while every pixel is wrong.
 */
function flipRows(px) {
  const out = new Uint8Array(px.length);
  const row = W * 4;
  for (let y = 0; y < H; y++) out.set(px.subarray((H - 1 - y) * row, (H - y) * row), y * row);
  return out;
}

/** Pulls an apse capture into a tight RGBA buffer, undoing any BGRA swizzle. */
function fromApseCapture(frame) {
  const out = new Uint8Array(W * H * 4);
  const bgra = frame.format.startsWith('bgra');
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const s = y * frame.bytesPerRow + x * 4, d = (y * W + x) * 4;
      out[d] = bgra ? frame.data[s + 2] : frame.data[s];
      out[d + 1] = frame.data[s + 1];
      out[d + 2] = bgra ? frame.data[s] : frame.data[s + 2];
      out[d + 3] = 255;
    }
  }
  return out;
}

async function renderApse(material) {
  const { apseRenderer, gpuMesh } = state;
  const { apseCam } = buildCameras();
  return apseRenderer.capture(apseScene(gpuMesh, material), apseCam);
}

function renderThree(material) {
  const { gl, rt, geometry, THREE } = state;
  const { threeCam } = buildCameras();
  const scene = threeScene(geometry, material);
  const px = new Uint8Array(W * H * 4);
  gl.setRenderTarget(rt);
  gl.render(scene, threeCam);
  gl.readRenderTargetPixels(rt, 0, 0, W, H, px);
  gl.setRenderTarget(null);
  return flipRows(px);
}

g.__images = async () => {
  let threePxRef = null;
  const { apseRenderer, albedoView, albedoTex, THREE } = state;
  const material = await apseMat.createDxMaterial(apseRenderer.device.device, albedoView);
  const apseFrame = await renderApse(material);
  const apsePx = fromApseCapture(apseFrame);
  // A fresh renderer is the only way to see the compile diagnostics again: by
  // now the program is cached, so a broken shader reports nothing at all.
  const diag = threeMat.captureDiagnostics(() => {
    threePxRef = renderThree(threeMat.createShaderMaterial(THREE, albedoTex));
  });
  const threePx = threePxRef;

  return {
    apse: { ...summary(apsePx), format: apseFrame.format, draws: apseRenderer.stats.drawCalls },
    three: { ...summary(threePx), draws: state.gl.info.render.calls, triangles: state.gl.info.render.triangles },
    threeConsoleErrors: diag.errors,
    threeThrown: diag.thrown,
    diff: diff(apsePx, threePx),
    validationErrors: state.validationErrors.slice(),
  };
};

// ---------------------------------------------------------------------------
// 3. Silent-failure and error probes
// ---------------------------------------------------------------------------

/** A camera at the shared pose, for the three.js-only probes. */
function plainCamera(THREE) {
  const cam = new THREE.PerspectiveCamera(P.fov, W / H, P.near, P.far);
  cam.position.set(...P.eye);
  cam.lookAt(0, 0, 0);
  cam.updateMatrixWorld();
  return cam;
}

function litScene(THREE, children) {
  const scene = new THREE.Scene();
  for (const c of children) scene.add(c);
  const key = new THREE.DirectionalLight(0xffffff, 3.0); key.position.set(1, 2, 3);
  const fill = new THREE.DirectionalLight(0xffffff, 1.2); fill.position.set(-2, 1, 2);
  scene.add(key, fill);
  return scene;
}

function twoLightScene(THREE, material) {
  return litScene(THREE, [new THREE.Mesh(state.geometry, material)]);
}

function renderThreeScene(THREE, scene) {
  const { gl, rt } = state;
  const px = new Uint8Array(W * H * 4);
  gl.setRenderTarget(rt);
  gl.render(scene, plainCamera(THREE));
  gl.readRenderTargetPixels(rt, 0, 0, W, H, px);
  gl.setRenderTarget(null);
  return flipRows(px);
}

g.__probes = async () => {
  const { apse, apseRenderer, albedoView, albedoTex, gl, THREE } = state;
  const out = {};

  // --- apse, SILENT: a declared varying that the body never assigns ----------
  {
    const entry = { library: 'apse', silent: true };
    try {
      const broken = await apse.Material.create(apseRenderer.device.device, apseMat.probeUnassignedVaryingSpec());
      broken.setTexture('albedoMap', albedoView);
      const brokenPx = fromApseCapture(await renderApse(broken));
      // The same material with out.worldPos actually written, for the contrast.
      const fixed = await apse.Material.create(apseRenderer.device.device, apseMat.probeWrittenVaryingSpec());
      fixed.setTexture('albedoMap', albedoView);
      const fixedPx = fromApseCapture(await renderApse(fixed));
      entry.rendered = true;
      entry.threw = null;
      entry.warningOrLog = null;
      entry.broken = summary(brokenPx);
      entry.written = summary(fixedPx);
      entry.brokenVsWritten = diff(brokenPx, fixedPx);
    } catch (e) {
      entry.rendered = false;
      entry.error = apseMat.describeError(e);
    }
    out.apse_unassigned_varying = entry;
  }

  // --- apse, DEFECT: obj.normalMatrix does not survive the round trip ---------
  {
    const specs = apseMat.probeNormalMatrixSpecs();
    const entry = { library: 'apse', silent: true,
      note: 'The mesh has an identity model matrix, so the correct answer for both materials is the vertex normal, unchanged. obj.normalMatrix is the identity on the CPU and should be the identity on the GPU.' };
    try {
      const a = await apse.Material.create(apseRenderer.device.device, specs.attribute);
      const b = await apse.Material.create(apseRenderer.device.device, specs.objNormalMatrix);
      const pa = fromApseCapture(await renderApse(a));
      const pb = fromApseCapture(await renderApse(b));
      entry.rendered = true;
      entry.threw = null;
      entry.warningOrLog = null;
      entry.fromAttribute = summary(pa);
      entry.fromObjNormalMatrix = summary(pb);
      entry.diff = diff(pa, pb);
    } catch (e) {
      entry.rendered = false;
      entry.error = apseMat.describeError(e);
    }
    out.apse_obj_normal_matrix = entry;
  }

  // --- apse, LOUD: a typo in the prelude -------------------------------------
  {
    const before = state.validationErrors.length;
    const rawBefore = state.rawDeviceErrors.length;
    const err = await apseMat.probePreludeTypo(apseRenderer.device.device);
    out.apse_prelude_typo = {
      library: 'apse', silent: false,
      thrownToTheAuthor: err,
      uncapturedDeviceErrorsViaApse: state.validationErrors.slice(before),
      uncapturedDeviceErrorsViaBrowserListener: state.rawDeviceErrors.slice(rawBefore),
      note: 'The first list is the onValidationError hook apse installs. The second is a raw uncapturederror listener on the same GPUDevice, i.e. what the browser console shows.',
    };
  }
  // --- apse, LOUD: is the WGSL diagnostic available but unsurfaced? ----------
  {
    const dev = apseRenderer.device.device;
    const bad = (await import('/dist/index.js')).generateScaffold({
      name: 'dx-compilation-info',
      slots: { tint: { type: 'vec3f', default: [1, 1, 1] } },
      prelude: 'fn dxLift(c : vec3f) -> vec3f { return c * 1.2; }',
      vertex: 'out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);',
      fragment: 'return vec4f(dxLifft(mat.tint), 1.0);',
    }).code;
    const mod = dev.createShaderModule({ code: bad });
    const info = await mod.getCompilationInfo();
    out.apse_compilation_info_available = {
      note: 'The same bad WGSL apse generated, handed straight to the device, and asked what the compiler says. If this has messages, the information exists and Material.create simply is not passing it on.',
      messages: [...info.messages].map((m) => [m.type, [m.lineNum, m.linePos], m.message]),
    };
  }

  // --- apse, LOUD: a typo in a body ------------------------------------------
  out.apse_body_typo = { library: 'apse', silent: false, error: apseMat.probeBodyTypo() };
  // --- apse, LOUD: a value type that does not match its declared varying -----
  out.apse_varying_value_type = { library: 'apse', silent: false, error: await apseMat.probeVaryingType(apseRenderer.device.device) };

  // --- three.js, SILENT: a .replace() whose target is not in the source ----
  out.three_missing_include_replace = { library: 'three.js', silent: true, ...threeMat.probeMissingIncludeReplacement(THREE) };

  // --- three.js, SILENT: onBeforeCompile with no customProgramCacheKey -------
  {
    // A and B are the same patch with a different constant baked in, and neither
    // overrides customProgramCacheKey, so if three.js keys the program only on
    // what getParameters can see, the two hash alike. Uniform *values* stay per
    // material, so the only thing that can be wrong is the baked constant.
    //
    // The control is B2, identical to B except that it *does* override the key.
    // Without a correct reference the experiment proves nothing: whichever
    // material compiles first wins, and the loser is drawn with the winner's
    // program, so a second render of the loser reproduces the same wrong result.
    //
    // A is placed nearer the camera than B, and three.js sorts opaque objects
    // front to back, so A compiles the shared program and B is the one that
    // inherits it. Both scenes hold two objects at identical positions; the only
    // difference is which B is in them.
    const a = threeMat.createPhongUncached(THREE, albedoTex, { gain: 1.0, tint: [1, 0, 0] });
    const b = threeMat.createPhongUncached(THREE, albedoTex, { gain: 0.2, tint: [0, 1, 0] });
    const b2 = threeMat.createPhongUncached(THREE, albedoTex, { gain: 0.2, tint: [0, 1, 0], cacheKey: 'dx-B-with-key' });

    const place = (mesh, x, z) => { mesh.scale.setScalar(0.45); mesh.position.set(x, 0, z); return mesh; };
    const meshA = place(new THREE.Mesh(state.geometry, a), -1.0, 1.2);
    const meshB = place(new THREE.Mesh(state.geometry, b), 1.0, 0);
    const meshB2 = place(new THREE.Mesh(state.geometry, b2), 1.0, 0);

    const programsBefore = gl.info.programs.length;
    renderThreeScene(THREE, litScene(THREE, [meshA]));                       // pins the shared key
    const withBrokenB = renderThreeScene(THREE, litScene(THREE, [meshA, meshB]));
    const withFixedB = renderThreeScene(THREE, litScene(THREE, [meshA, meshB2]));

    out.three_missing_cache_key = {
      library: 'three.js',
      silent: true,
      note: 'Both scenes hold A (baking DX_GAIN 1.0) and a B that bakes DX_GAIN 0.2, at identical positions. The only difference is that the second scene B overrides customProgramCacheKey and the first does not. A large masked difference means the first B inherited A program.',
      bWithoutCacheKeyVsBWith: diffMasked(withBrokenB, withFixedB, coverageMask(withFixedB)),
      programsCompiled: gl.info.programs.length - programsBefore,
      bWithKeyFrame: summary(withFixedB),
    };
  }

  // --- three.js, LOUD: an #include name that does not resolve ----------------
  {
    const material = threeMat.probeUnknownInclude(THREE);
    const d = threeMat.captureDiagnostics(() => renderThreeScene(THREE, twoLightScene(THREE, material)));
    out.three_unknown_include = { library: 'three.js', silent: false, thrown: d.thrown, consoleErrors: d.errors };
  }

  // --- three.js, LOUD: varying type mismatch between stages ------------------
  {
    const material = threeMat.probeVaryingTypeMismatch(THREE);
    const px = new Uint8Array(W * H * 4);
    const d = threeMat.captureDiagnostics(() => { px.set(renderThreeScene(THREE, twoLightScene(THREE, material))); });
    out.three_varying_type_mismatch = { library: 'three.js', silent: false, thrown: d.thrown, consoleErrors: d.errors, frame: summary(px) };
  }

  // --- three.js, LOUD: a value type error (vec2 into a vec3 varying) ---------
  {
    const material = threeMat.probeShaderTypeError(THREE);
    const d = threeMat.captureDiagnostics(() => renderThreeScene(THREE, twoLightScene(THREE, material)));
    out.three_value_type_error = { library: 'three.js', silent: false, thrown: d.thrown, consoleErrors: d.errors };
  }

  // --- the uniform read paths, side by side ---------------------------------
  {
    const patched = threeMat.createPhongPatched(THREE, albedoTex, { tint: [0.9, 0.1, 0.1] });
    const patchedScene = twoLightScene(THREE, patched);
    const { gl: glr, rt: rtr } = state;
    state.generated.threeV2 = threeMat.captureGeneratedGLSL(glr, () => {
      glr.setRenderTarget(rtr); glr.render(patchedScene, plainCamera(THREE)); glr.setRenderTarget(null);
    });
    const px = renderThreeScene(THREE, patchedScene);
    const shaderMat = threeMat.createShaderMaterial(THREE, albedoTex);
    out.uniform_read_paths = {
      note: 'v2 is a built-in MeshPhongMaterial patched by onBeforeCompile, with the tint injected into the patch. v1 is a ShaderMaterial built from the shared PARAMS. The two values differ because the materials differ, not because the read path does.',
      v2_builtInMaterialHasNoUniformsProperty: threeMat.builtInMaterialHasUniforms(patched),
      tintInjectedIntoThePatch: [0.9, 0.1, 0.1],
      v2_readBackVia_userData_shader_uniforms: threeMat.readOwnUniform(patched, 'tint').toArray(),
      v1_readBackVia_material_uniforms: threeMat.readOwnUniformShaderMaterial(shaderMat, 'tint').toArray(),
      apse_readBackVia: 'material.setSlot("tint", [...]) - a validated setter, no userData indirection',
      v2_frame: summary(px),
    };
  }

  out.apseWebGPUValidationErrors = state.validationErrors.slice();
  return out;
};

// ---------------------------------------------------------------------------
// 4. What each library generated
// ---------------------------------------------------------------------------
g.__generated = async () => {
  const { gl, rt, geometry, albedoTex, THREE } = state;
  const { threeCam } = buildCameras();
  const v1 = state.generated.threeV1;

  // V2 is compiled for the first time in the uniform-path probe, so the capture
  // is taken there. Re-rendering here would hit the program cache.
  const phong = threeMat.createPhongPatched(THREE, albedoTex, { tint: [0.9, 0.1, 0.1] });
  const phongScene = threeScene(geometry, phong);
  phongScene.add(new THREE.DirectionalLight(0xffffff, 3));
  const v2 = state.generated.threeV2 ?? { vertex: null, fragment: null };

  return {
    apseWGSL: apseMat.dxGeneratedWGSL(),
    threeV1Vertex: v1.vertex, threeV1Fragment: v1.fragment,
    threeV2Vertex: v2.vertex, threeV2Fragment: v2.fragment,
  };
};
console.info('[dx] script done');
g.__ready = true;
</script></body></html>`;

// --- the node-material page, in its own document -----------------------------

const PAGE_NODES = /* html */ `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>apse dx nodes</title>
<style>body{margin:0;background:#000}canvas{position:fixed;top:0;left:-99999px}</style>
<script type="importmap">{"imports":{
  "three/webgpu": "/node_modules/three/build/three.webgpu.js",
  "three/tsl": "/node_modules/three/build/three.tsl.js"
}}</script>
</head><body><script type="module">
const g = globalThis;
const W = ${WIDTH}, H = ${HEIGHT};
g.__ready = false;
g.__nodes = null;
g.__nodesError = null;
try {
  const three = await import('/node_modules/three/build/three.webgpu.js');
  const tsl = await import('/node_modules/three/build/three.tsl.js');
  const dx = await import('./three-custom-material.js');
  const apseMat = await import('./apse-custom-material.ts');
  const P = apseMat.PARAMS;

  const canvas = document.createElement('canvas');
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  document.body.appendChild(canvas);
  Object.defineProperty(canvas, 'clientWidth', { value: W, configurable: true });
  Object.defineProperty(canvas, 'clientHeight', { value: H, configurable: true });
  canvas.width = W; canvas.height = H;

  const renderer = new three.WebGPURenderer({ canvas, antialias: false });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  const isWebGPU = renderer.backend && renderer.backend.isWebGPUBackend === true;

  const albedo = dx.createAlbedoTexture(three);
  const material = dx.createNodeMaterial(three, tsl, albedo, { tint: P.tint, rimPower: P.rim.power });

  // The same sphere apse is using, from the same MeshData.
  const core = await import('/dist/index.js');
  const meshData = core.sphere({ radius: 1, widthSegments: 64, heightSegments: 48 });
  const geometry = new three.BufferGeometry();
  const stride = meshData.layout.stride >> 2, src = meshData.vertexData, n = meshData.vertexCount;
  for (const [name, off, size] of [['position', 0, 3], ['normal', 3, 3], ['uv', 6, 2]]) {
    const out = new Float32Array(n * size);
    for (let i = 0; i < n; i++) out.set(src.subarray(i * stride + off, i * stride + off + size), i * size);
    geometry.setAttribute(name, new three.BufferAttribute(out, size));
  }
  geometry.setIndex(new three.BufferAttribute(new Uint32Array(meshData.indexData), 1));

  const scene = new three.Scene();
  scene.add(new three.Mesh(geometry, material));
  for (const l of P.lights) {
    const dl = new three.DirectionalLight(0xffffff, 1);
    dl.position.set(l.dir[0], l.dir[1], l.dir[2]);
    scene.add(dl);
  }
  scene.add(new three.AmbientLight(0xffffff, 0.4));
  const camera = new three.PerspectiveCamera(P.fov, W / H, P.near, P.far);
  camera.position.set(...P.eye);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  const t0 = performance.now();
  await renderer.renderAsync(scene, camera);
  const t1 = performance.now();
  await new Promise((r) => requestAnimationFrame(r));
  // WebGPU has no preserveDrawingBuffer either; the node renderer exposes no
  // offscreen readback, so the check is coverage of the canvas via a 2D copy
  // taken in the same task as the draw is not possible. Report the draw
  // statistics instead of a pixel claim.
  // Draw into an offscreen target and read it back, so "did it draw" is a pixel
  // count rather than a claim. A WebGPU canvas has no preserveDrawingBuffer, and
  // three.js's node renderer exposes no synchronous readback of the swapchain.
  const rt = new three.RenderTarget(W, H, { depthBuffer: true, samples: 0 });
  let coverage = 0, maxLuma = 0, rows = 0;
  const t2 = performance.now();
  await renderer.setRenderTarget(rt);
  await renderer.renderAsync(scene, camera);
  const buf = await renderer.readRenderTargetPixelsAsync(rt, 0, 0, W, H);
  const readbackMs = +(performance.now() - t2).toFixed(3);
  rt.dispose();
  // copyTextureToBuffer pads rows to 256 bytes, so walk at the real stride.
  const readbackStride = Math.ceil((W * 4) / 256) * 256;
  rows = Math.floor(buf.length / readbackStride);
  for (let y = 0; y < Math.min(H, rows); y++) {
    for (let x = 0; x < W; x++) {
      const o = y * readbackStride + x * 4;
      const l = (buf[o] + buf[o + 1] + buf[o + 2]) / 3;
      if (l > maxLuma) maxLuma = l;
      if (l > 8) coverage++;
    }
  }

  g.__nodes = {
    threeRevision: three.REVISION,
    isWebGPU,
    backend: renderer.backend && renderer.backend.constructor ? renderer.backend.constructor.name : 'unknown',
    firstRenderMs: +(t1 - t0).toFixed(3),
    offscreenRenderPlusReadbackMs: readbackMs,
    calls: renderer.info.render.calls,
    triangles: renderer.info.render.triangles,
    pixelRatio: renderer.getPixelRatio(),
    canvasSize: [canvas.width, canvas.height],
    materialType: material.type,
    hasColorNode: material.colorNode !== undefined && material.colorNode !== null,
    hasEmissiveNode: material.emissiveNode !== undefined && material.emissiveNode !== null,
    lights: material.lights,
    pixelsCovered: coverage,
    coverage: +(coverage / (W * H)).toFixed(4),
    maxLuma,
    readback: { bufferBytes: buf.length, bytesPerRow: readbackStride, rows },
  };
} catch (e) {
  g.__nodesError = e && e.stack ? String(e.stack).slice(0, 1500) : String(e);
}
console.info('[dx] script done');
g.__ready = true;
</script></body></html>`;

// ---------------------------------------------------------------------------
// Line counting, read out of the material files
// ---------------------------------------------------------------------------

interface SectionCount { [section: string]: number }

function countSections(file: string): { sections: SectionCount; total: number; fileLines: number } {
  const raw = readFileSync(join(ROOT, file), 'utf8').split('\n');
  const sections: SectionCount = {};
  let open: string | null = null;
  let inBlock = false;
  let code = 0;
  for (const line of raw) {
    // Comments document the material; they are not material code. Stripping them
    // properly matters, because a JSDoc block is six lines and would otherwise be
    // charged to whichever section it happens to sit in.
    const t = inBlock
      ? (line.includes('*/') ? line.slice(line.indexOf('*/') + 2) : '')
      : line;
    if (inBlock && line.includes('*/')) inBlock = false;
    if (!inBlock && /\/\*/.test(t) && !t.includes('*/')) inBlock = true;
    const codeOnly = inBlock ? '' : t.replace(/\/\*.*?\*\//g, '').replace(/\/\/.*$/, '');
    if (codeOnly.trim().length > 0) code++;

    const trimmed = line.trim();
    if (open === null) {
      const m = /^\/\/ @count:([a-zA-Z0-9_-]+)$/.exec(trimmed);
      if (m !== null) { open = m[1]!; sections[open] = 0; }
      continue;
    }
    if (trimmed === '// @count:end') { open = null; continue; }
    if (codeOnly.trim().length === 0) continue;
    sections[open] = (sections[open] ?? 0) + 1;
  }
  const total = Object.values(sections).reduce((a, b) => a + b, 0);
  return { sections, total, fileLines: code };
}

// ---------------------------------------------------------------------------

const MIME: Record<string, string> = {
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
};

const transpiler = new Bun.Transpiler({ loader: 'ts' });

async function main(): Promise<void> {
  const server = serve({
    port: PORT,
    idleTimeout: 0,
    routes: {
      '/bench/dx/materials.html': () => new Response(PAGE_MATERIALS, { headers: { 'content-type': MIME['.html']! } }),
      '/bench/dx/nodes.html': () => new Response(PAGE_NODES, { headers: { 'content-type': MIME['.html']! } }),
      '/*': async (req) => {
        const path = new URL(req.url).pathname;
        const file = Bun.file(join(ROOT, path === '/' ? 'bench/dx/materials.html' : path.slice(1)));
        if (!(await file.exists())) return new Response('not found', { status: 404 });
        if (path.endsWith('.ts')) {
          // The apse material is TypeScript. Chrome will not run it and a 200
          // with the wrong MIME only produces a confusing import failure, so it
          // is transpiled here rather than relying on a browser flag.
          return new Response(transpiler.transformSync(await file.text()), { headers: { 'content-type': MIME['.js']! } });
        }
        const dot = path.slice(path.lastIndexOf('.'));
        return new Response(file, { headers: { 'content-type': MIME[dot] ?? 'application/octet-stream' } });
      },
    },
  });

  const browser = await puppeteer.launch({
    headless: true,
    executablePath: process.env['CHROME_PATH'] ?? undefined,
    channel: process.env['CHROME_PATH'] === undefined ? 'chrome' : undefined,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });

  const pageErrors: string[] = [];
  const openMaterialsPage = async (label: string): Promise<any> => {
    const page = await browser.newPage();
    page.on('pageerror', (e: unknown) => { const t = e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ''}` : String(e); pageErrors.push(`${label}: ${t}`); });
    page.on('requestfailed', (r) => console.log(`  [net] FAILED ${r.url()} ${r.failure()?.errorText ?? ''}`));
    page.on('response', (r) => { if (r.status() >= 400) console.log(`  [net] ${r.status()} ${r.url()}`); });
    await page.goto(`http://localhost:${PORT}/bench/dx/materials.html`, { waitUntil: 'load' });
    const ready = await page.waitForFunction(() => (globalThis as any).__ready === true, { timeout: 60_000 })
      .then(() => true).catch(() => false);
    if (!ready) {
      for (const e of pageErrors) console.log(`  PAGE ERROR: ${e}`);
      throw new Error(`page ${label} never became ready: ${await page.evaluate(() => String((globalThis as any).__dxError))}`);
    }
    return page;
  };

  try {
    // --- cold timing, on its own fresh document each time --------------------
    // A shader program compiles exactly once, so timing has to happen on a
    // document that has never drawn. Repeating it in one page would measure a
    // cache hit, and one sample on one page is not a number.
    const timingSamples: any[] = [];
    for (let i = 0; i < TIMING_SAMPLES; i++) {
      const t = await browser.newPage();
      await t.goto(`http://localhost:${PORT}/bench/dx/materials.html`, { waitUntil: 'load' });
      await t.waitForFunction(() => (globalThis as any).__ready === true, { timeout: 60_000 })
        .catch(() => { throw new Error('timing page never became ready'); });
      await t.evaluate(() => (globalThis as any).__init());
      timingSamples.push(await t.evaluate(() => (globalThis as any).__timing()));
      await t.close();
    }
    const timing = {
      samples: timingSamples,
      apseTotalMs: timingSamples.map((t) => t.apse.totalMs),
      threeTotalMs: timingSamples.map((t) => t.three.totalMs),
      note: timingSamples[0].note,
    };

    const page = await openMaterialsPage('main');
    const init = await page.evaluate(() => (globalThis as any).__init());
    console.log('\n  ═══ environment ═══');
    for (const [k, v] of Object.entries(init)) console.log(`  ${k.padEnd(18)} ${JSON.stringify(v)}`);

    // The generated-GLSL capture has to ride along with the first compile of
    // each program, so this step has to run before anything else renders.
    await page.evaluate(() => (globalThis as any).__timing());
    const images = await page.evaluate(() => (globalThis as any).__images());
    const probes = await page.evaluate(() => (globalThis as any).__probes());
    const generated = await page.evaluate(() => (globalThis as any).__generated());

    // --- page 2: the node-material competitor, its own document -------------
    const page2 = await browser.newPage();
    page2.on('pageerror', (e: unknown) => pageErrors.push(`page2: ${e instanceof Error ? e.message : String(e)}`));
    await page2.goto(`http://localhost:${PORT}/bench/dx/nodes.html`, { waitUntil: 'load' });
    const nodesReady = await page2.waitForFunction(() => (globalThis as any).__ready === true, { timeout: 90_000 })
      .then(() => true).catch(() => false);
    const nodes = nodesReady
      ? await page2.evaluate(() => (globalThis as any).__nodes)
      : { failed: await page2.evaluate(() => String((globalThis as any).__nodesError)) };
    const nodesError = await page2.evaluate(() => (globalThis as any).__nodesError ?? null);
    await page2.close();

    report({ init, timing, images, probes, generated, nodes, nodesError, pageErrors });
  } finally {
    await browser.close();
    server.stop(true);
  }
}

// ---------------------------------------------------------------------------
// Which lines of a generated program are the author's own
// ---------------------------------------------------------------------------

interface FileSections { [section: string]: string[] }

/** The source lines of every `// @count:<name>` … `// @count:end` block. */
function countedSections(file: string): FileSections {
  const out: FileSections = {};
  let open: string | null = null;
  for (const line of readFileSync(join(ROOT, file), 'utf8').split('\n')) {
    const t = line.trim();
    if (open === null) {
      const m = /^\/\/ @count:([a-zA-Z0-9_-]+)$/.exec(t);
      if (m !== null) { open = m[1]!; out[open] = []; }
      continue;
    }
    if (t === '// @count:end') { open = null; continue; }
    out[open]!.push(line);
  }
  return out;
}

/** Every backtick template literal in a set of source lines. */
function templateLiteralsIn(lines: readonly string[]): string[] {
  const out: string[] = [];
  const re = /`([\s\S]*?)`/g;
  const src = lines.join('\n');
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) out.push(m[1]!);
  return out;
}

/** Every single-quoted string literal in a set of source lines. */
function quotedStringsIn(lines: readonly string[]): string[] {
  const out: string[] = [];
  const re = /'([^'\n]*)'/g;
  const src = lines.join('\n');
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) out.push(m[1]!);
  return out;
}

const apseSections = countedSections('bench/dx/apse-custom-material.ts');
const threeSections = countedSections('bench/dx/three-custom-material.js');

/** The authored shader text each library was handed, taken from the counted sections. */
const apseBodies: string[] = [
  ...templateLiteralsIn(apseSections['prelude'] ?? []),
  ...templateLiteralsIn(apseSections['vertex'] ?? []),
  ...templateLiteralsIn(apseSections['fragment'] ?? []),
];
const threeV1Bodies: string[] = [
  ...templateLiteralsIn(threeSections['declarations'] ?? []),
  ...templateLiteralsIn(threeSections['uniformDeclarations'] ?? []),
  ...templateLiteralsIn(threeSections['vertex'] ?? []),
  ...templateLiteralsIn(threeSections['fragment'] ?? []),
];
/** V2 is a string-surgery patch, so its authored GLSL is a list of quoted fragments. */
const threeV2Bodies: string[] = [
  ...quotedStringsIn(threeSections['phong'] ?? []).filter((s) => s.includes('#include') || s.includes('uniform') || s.includes('outgoingLight') || s.includes('dx')),
];

/**
 * Counts the generated lines that are *not* one of the author's bodies.
 *
 * A generated program contains the authored bodies verbatim, so the total minus
 * the number of lines matching a body is what the library wrote on the author's
 * behalf. Approximate, because a body can be re-indented on the way in; the
 * exact total is printed alongside.
 */
function authoredLinesIn(generated: string, bodies: string[]): number {
  const lines = generated.split('\n');
  const owned = new Set<number>();
  for (const body of bodies) {
    const bodyLines = body.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
    for (let i = 0; i < lines.length; i++) {
      const t = lines[i]!.trim();
      if (t.length === 0) continue;
      if (bodyLines.includes(t) && !owned.has(i)) owned.add(i);
    }
  }
  return owned.size;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

function report(r: {
  init: Record<string, unknown>;
  timing: Record<string, any>;
  images: Record<string, any>;
  probes: Record<string, any>;
  generated: Record<string, string>;
  nodes: Record<string, any> | null;
  nodesError: string | null;
  pageErrors: string[];
}): void {
  const apseLines = countSections('bench/dx/apse-custom-material.ts');
  const threeLines = countSections('bench/dx/three-custom-material.js');

  console.log('\n  ═══ 1. do the two images agree ═══');
  console.log(`  apse   ${JSON.stringify(r.images.apse)}`);
  console.log(`  three  ${JSON.stringify(r.images.three)}`);
  if (r.images.threeConsoleErrors) console.log(`  three console errors: ${JSON.stringify(r.images.threeConsoleErrors, null, 2).split('\n').join('\n  ')}`);
  if (r.images.threeThrown) console.log(`  three threw: ${r.images.threeThrown}`);
  console.log(`  diff   ${JSON.stringify(r.images.diff, null, 2).split('\n').join('\n  ')}`);
  if (r.images.validationErrors.length > 0) {
    console.log(`  apse WebGPU validation errors: ${r.images.validationErrors.length}`);
  }

  console.log('\n  ═══ 2. lines of user-written material code ═══');
  const show = (label: string, c: ReturnType<typeof countSections>) => {
    const parts = Object.entries(c.sections).map(([k, v]) => `${k} ${v}`).join('  ');
    console.log(`  ${label.padEnd(6)} total ${String(c.total).padStart(3)}   [${parts}]   (file is ${c.fileLines} non-blank lines)`);
  };
  show('apse', apseLines);
  show('three', threeLines);
  console.log(`  apse  sections:  ${JSON.stringify(apseLines.sections)}`);
  console.log(`  three sections:  ${JSON.stringify(threeLines.sections)}`);
  const v1 = Object.keys(threeLines.sections)
    .filter((k) => k !== 'phong' && k !== 'nodes')
    .reduce((n, k) => n + (threeLines.sections[k] ?? 0), 0);
  console.log(`  three v1 (ShaderMaterial) subtotal ${v1}, v2 (onBeforeCompile) ${threeLines.sections['phong'] ?? 0}, v3 (node material) ${threeLines.sections['nodes'] ?? 0}`);

  console.log('\n  ═══ 3. time to first drawn frame, cold, ms ═══');
  const med = (xs: number[]): number => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0;
  const stats = (xs: number[]): string =>
    `min ${Math.min(...xs).toFixed(1)}  median ${med(xs).toFixed(1)}  max ${Math.max(...xs).toFixed(1)}`;
  console.log(`  apse   total ${stats(r.timing.apseTotalMs)}   [${r.timing.apseTotalMs.join(', ')}]`);
  console.log(`  three  total ${stats(r.timing.threeTotalMs)}   [${r.timing.threeTotalMs.join(', ')}]`);
  const parts = r.timing.samples.map((s: any) => `apse compile ${s.apse.materialCompileMs} + frame ${s.apse.firstFrameReadbackMs} | three first render ${s.three.firstFrameMs}`).join('\n      ');
  console.log(`  breakdown:\n      ${parts}`);
  console.log(`  ${r.timing.note}`);

  console.log('\n  ═══ 4. what each library generated for the author ═══');
  const wgslLines = r.generated.apseWGSL.split('\n').length;
  const apseAuthored = apseBodies.reduce((n, b) => n + b.split('\n').filter((l) => l.trim().length > 0).length, 0);
  console.log(`  apse generated WGSL        ${r.generated.apseWGSL.length} bytes, ${wgslLines} lines, of which ${apseAuthored} are the authored bodies and ${wgslLines - apseAuthored} were generated`);
  const bodies: Record<string, string[]> = {
    threeV1Vertex: [r.generated.threeV1Vertex ?? ''],
    threeV1Fragment: [r.generated.threeV1Fragment ?? ''],
    threeV2Vertex: [r.generated.threeV2Vertex ?? ''],
    threeV2Fragment: [r.generated.threeV2Fragment ?? ''],
  };
  for (const [key, src] of Object.entries(bodies)) {
    const v = src[0];
    if (v === null || v === '') { console.log(`  ${key.padEnd(25)} n/a`); continue; }
    const total = v.split('\n').length;
    const authored = authoredLinesIn(v, key === 'threeV1Vertex' || key === 'threeV1Fragment'
      ? threeV1Bodies : threeV2Bodies);
    console.log(`  ${key.padEnd(25)} ${v.length} bytes, ${total} lines, of which ~${authored} are the authored bodies and ${total - authored} were generated`);
  }

  console.log('\n  ═══ 4b. what the README claim asserts, checked against the generated source ═══');
  const g = r.generated;
  const declares = (src: string | null, name: string): boolean =>
    src !== null && new RegExp(`^\\s*(uniform|attribute|@group).*\\b${name}\\b`, 'm').test(src);
  for (const name of ['modelMatrix', 'viewMatrix', 'projectionMatrix', 'modelViewMatrix', 'normalMatrix', 'cameraPosition']) {
    const where = ['threeV1Vertex', 'threeV1Fragment', 'threeV2Vertex', 'threeV2Fragment']
      .filter((k) => declares(g[k], name));
    console.log(`  ${name.padEnd(18)} three.js declares it in: ${where.length === 0 ? 'NOWHERE' : where.join(', ')}`);
  }
  console.log(`  ${'normal attribute'.padEnd(18)} three.js declares it in: ${declares(g.threeV1Vertex, 'normal') ? 'threeV1Vertex' : 'NOWHERE'}`);
  console.log(`  ${'uv attribute'.padEnd(18)} three.js declares it in: ${declares(g.threeV1Vertex, 'uv') ? 'threeV1Vertex' : 'NOWHERE'}`);

  console.log('\n  ═══ 5. silent failures and loud errors, verbatim ═══');
  for (const [k, v] of Object.entries(r.probes)) {
    if (k === 'validationErrors') continue;
    console.log(`\n  ▸ ${k}`);
    console.log(`    ${JSON.stringify(v, null, 2).split('\n').join('\n    ')}`);
  }

  console.log('\n  ═══ 6. the node-material competitor (three.js WebGPURenderer) ═══');
  console.log(`  ${JSON.stringify(r.nodes, null, 2).split('\n').join('\n  ')}`);
  if (r.nodesError) console.log(`  error: ${r.nodesError}`);

  if (r.pageErrors.length > 0) {
    console.log('\n  ═══ uncaught page errors ═══');
    for (const e of r.pageErrors) console.log(`  ${e}`);
  }
}

main().catch((e) => {
  console.error(`\n  bench/dx failed: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}\n`);
  process.exit(1);
});
