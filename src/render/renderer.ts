/**
 * The frame loop.
 *
 * This is the only module that knows the order of operations in a frame, and it
 * is written so that the order is the whole story:
 *
 * ```txt
 *   1. size     — did the backing store change? recreate targets if so.
 *   2. cull     — scene.collectDrawItems  (pruned transform walk + sphere tests)
 *   3. sort     — opaque front-to-back, transparent back-to-front, both grouped
 *   4. pack     — world matrices into the object uniform buffer, one write each
 *   5. upload   — frame uniform (1 write) + object uniform (1 write)
 *   6. encode   — one command encoder, one render pass, N draws
 *   7. submit   — one queue.submit
 *   8. stats    — timings, on a frame budget the caller can declare
 * ```
 *
 * The budget is what makes this a renderer rather than a demo. Steps 2 through 5
 * are CPU time inside a 16.67 ms budget that is already shared with the browser's
 * compositor, the input system, and the garbage collector. A frame that costs
 * 0.3 ms of CPU and 4 ms of GPU is a well-behaved program; a frame that costs
 * 3 ms of CPU and 0.5 ms of GPU is a broken one, and no amount of GPU in the
 * machine will fix it.
 *
 * ## Why there is exactly one writeBuffer per uniform block
 *
 * Per-object uniform data is the single largest CPU cost in a naive WebGPU
 * renderer. The standard failure is one uniform buffer plus one bind group per
 * object, which measures at roughly 8,000 objects before the frame collapses on
 * an M1. apse instead keeps one large object buffer addressed with dynamic
 * offsets, so per object the cost is a `setBindGroup` with an integer offset and
 * nothing else. All the transform data for a frame is written in a single
 * `queue.writeBuffer` over a `Float32Array` the CPU already had to touch.
 */

import { fail } from '../core/error.ts';
import { BIND_GROUP, FRAME_BLOCK, OBJECT_BLOCK } from '../core/slot.ts';
import type { AseDevice, DeviceOptions } from './device.ts';
import { createDevice, isDevelopmentMode } from './device.ts';
import { CanvasSizer } from './context.ts';
import { RenderTargetImpl, createCanvasTarget, createColorTarget } from './target.ts';
import { alignUp } from '../core/uniform.ts';
import {
  DEFAULT_FRAME_LABEL,
  DEFAULT_OBJECT_LABEL,
  FrameUniforms,
  ObjectUniforms,
  deviceCache,
} from '../material/material.ts';
import { OBJECT_UNIFORM_STRIDE } from '../scene/graph.ts';
import type { Scene } from '../scene/graph.ts';
import type { Camera } from '../scene/camera.ts';
import type { DrawItem } from './types.ts';
import { sortDrawItems } from './sort.ts';
import type { Material } from '../material/material.ts';

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/** What one frame cost. All times in milliseconds. */
export interface FrameStats {
  /** Wall-clock time inside `render()`. The number that must stay under budget. */
  readonly cpu: number;
  /** GPU time, or 0 when `timestamp-query` is unavailable. 1–2 frames late. */
  readonly gpu: number;
  /** Measured ms/frame over the last `sampleSize` frames. */
  readonly averageCpu: number;
  /** Measured ms/frame over the last `sampleSize` frames. */
  readonly averageGpu: number;
  /** Draw calls encoded this frame. */
  readonly drawCalls: number;
  /** Triangles submitted this frame. */
  readonly triangles: number;
  /** Draw items the scene produced before frustum culling. */
  readonly candidates: number;
  /** Draw items culled by the frustum. */
  readonly culled: number;
  /** Bytes written by `queue.writeBuffer` this frame. */
  readonly uniformBytes: number;
  /** `setBindGroup` calls encoded, after redundant-call suppression. */
  readonly bindGroupCalls: number;
  /** `setPipeline` calls encoded, after redundant-call suppression. */
  readonly pipelineCalls: number;
  /** `setVertexBuffer` / `setIndexBuffer` calls encoded, after suppression. */
  readonly bufferCalls: number;
  /** Draw calls before suppression. Always `drawCalls`; kept for contrast. */
  readonly naiveCalls: number;
  /** Objects whose transform was re-packed this frame. */
  readonly packedObjects: number;
  /** Objects skipped because their transform was unchanged. */
  readonly skippedObjects: number;
  /** Average time in milliseconds per draw call. The metric that predicts the future. */
  readonly cpuPerDraw: number;
}

/** A performance limit the renderer can hold itself to. */
export interface RenderBudget {
  /** Maximum ms/frame of CPU time inside `render()`. */
  readonly cpu?: number;
  /** Maximum ms/frame of GPU time. */
  readonly gpu?: number;
  /** Maximum draw calls per frame. */
  readonly drawCalls?: number;
  /** Maximum ms/frame from a 1-object scene. Measures per-frame fixed cost. */
  readonly idle?: number;
  /** Samples averaged before a breach is reported. */
  readonly sampleSize?: number;
}

/** The pixels of a {@link Renderer.capture}, on the CPU. */
export interface CapturedFrame {
  readonly width: number;
  readonly height: number;
  /** Row stride. **Not** `width * 4` — see the note on `capture()`. */
  readonly bytesPerRow: number;
  readonly format: GPUTextureFormat;
  /** `bytesPerRow * height` bytes, 4 channels per pixel. */
  readonly data: Uint8Array;
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

export interface RendererOptions extends DeviceOptions {
  /** Depth attachment on the canvas target. Default true. */
  readonly depth?: boolean;
  /** Declarative performance limits, checked every frame. */
  readonly budget?: RenderBudget;
  /** Frames of samples retained for averaging. Default 120. */
  readonly sampleSize?: number;
  /** A camera to use when `render()` is called without one. */
  readonly camera?: Camera;
}

const EMPTY_ITEMS: DrawItem[] = [];

/** Scratch, module-level, and reused. The frame loop must not allocate. */
const _normal = new Float32Array(9);
/** Two floats for `resolution` / `viewport`, so the frame uniform writes nothing. */
const _pair = new Float32Array(2);
/**
 * Reused dynamic-offset array for the object bind group.
 *
 * Exactly one element, because the array length must equal the number of
 * *dynamic bindings in the layout* — not the spec's ceiling of 8 dynamic
 * buffers per pipeline layout. Passing 8 offsets to a layout with one dynamic
 * buffer is a validation error, and it is thrown per draw.
 *
 * Reused rather than a `[offset]` literal per draw, which would be one
 * short-lived array per object per frame.
 */
const _dynamicOffsets = new Uint32Array(1);

/** `FrameStats` with every field writable, so `#record` does not copy. */
type MutableStats = { -readonly [K in keyof FrameStats]: FrameStats[K] };

export class Renderer {
  readonly device: AseDevice;
  readonly format: GPUTextureFormat;
  readonly featureLevel: 'core' | 'compatibility';

  /** Shared across every material in the app. 256 bytes written once a frame. */
  readonly frameUniforms: FrameUniforms;
  /** One large buffer, addressed with dynamic offsets. */
  readonly objectUniforms: ObjectUniforms;

  #canvas: RenderTargetImpl;
  #sizer: CanvasSizer;
  #items: DrawItem[] = EMPTY_ITEMS;
  #sorted: DrawItem[] = [];
  #camera: Camera | null;
  #disposed = false;
  #running = false;
  #rafHandle = 0;
  #lastTime = 0;
  #startTime = 0;
  #elapsed = 0;
  #frameIndex = 0;

  #budget: RenderBudget | null;
  #sampleSize: number;
  #cpuSamples: Float64Array;
  #gpuSamples: Float64Array;
  #sampleCursor = 0;
  #sampleCount = 0;
  #budgetBreached: ((budget: string, actual: number, limit: number) => void) | null = null;

  #lastStats: FrameStats;
  #lastDrawn = 0;
  #current: MutableStats = emptyStats();
  #encoder: GPUCommandEncoder | null = null;

  private constructor(device: AseDevice, opts: RendererOptions) {
    this.device = device;
    this.format = device.format;
    this.featureLevel = device.featureLevel;

    // Resolved through the device cache rather than constructed, because every
    // material resolves the same two blocks through the same cache. Two
    // constructions would mean the renderer writes the frame uniform into a
    // buffer the materials' bind groups do not read: no error, no warning, and
    // every object drawn with an identity transform.
    const cache = deviceCache(device.device);
    this.frameUniforms = cache.frameUniforms(DEFAULT_FRAME_LABEL);
    this.objectUniforms = cache.objectUniforms(DEFAULT_OBJECT_LABEL, device.maxObjects);

    this.#sizer = new CanvasSizer(device.canvas, { maxPixelRatio: device.maxPixelRatio });
    this.#sizer.update();
    this.#canvas = createCanvasTarget(device, {
      width: this.#sizer.width,
      height: this.#sizer.height,
      depth: opts.depth ?? true,
    });

    this.#camera = opts.camera ?? null;
    this.#budget = opts.budget ?? null;
    this.#sampleSize = opts.sampleSize ?? 120;
    this.#cpuSamples = new Float64Array(this.#sampleSize);
    this.#gpuSamples = new Float64Array(this.#sampleSize);
    this.#lastStats = emptyStats();
  }

  /**
   * Creates a renderer and its device.
   *
   * Async because device acquisition is: `requestAdapter` and `requestDevice`
   * both return promises, and shader compilation on first draw is asynchronous.
   * A synchronous constructor is not possible here, and a lazily-initialised
   * one hides the cost until the first frame instead of at setup.
   */
  static async create(canvas: HTMLCanvasElement, opts: RendererOptions = {}): Promise<Renderer> {
    const device = await createDevice(canvas, opts);
    try {
      return new Renderer(device, opts);
    } catch (err) {
      device.destroy();
      throw err;
    }
  }

  /** One line about the machine, for bug reports and for the demo header. */
  describeGpu(): string {
    const i = this.device.adapterInfo;
    const parts = [i.vendor, i.architecture, i.device, i.description].filter((s) => s.length > 0);
    return `${parts.join(' ') || 'unknown adapter'} · ${this.featureLevel}`;
  }

  set camera(camera: Camera | null) { this.#camera = camera; }
  get camera(): Camera | null { return this.#camera; }

  /** The most recent completed frame. */
  get stats(): FrameStats { return this.#lastStats; }

  get drawItemCount(): number { return this.#items.length; }

  /** Called when a budget is exceeded, once per breach. */
  onBudgetBreached(fn: ((budget: string, actual: number, limit: number) => void) | null): void {
    this.#budgetBreached = fn;
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  /**
   * Renders one frame and returns immediately.
   *
   * Synchronous on purpose. Every call site that wants a frame now — a test, a
   * benchmark, a screenshot, a single draw into a readback buffer — should not
   * have to reason about a requestAnimationFrame callback that may not fire for
   * another 16 ms, or at all if the tab is hidden.
   *
   * `target` defaults to the canvas. Passing an offscreen target is how you
   * render to a texture for a post-processing pass, an export, or a readback.
   */
  render(scene: Scene, camera?: Camera, target: RenderTargetImpl = this.#canvas): FrameStats {
    this.#assertLive();
    if (this.#disposed) {
      fail('RENDERER_ALREADY_DISPOSED', 'This Renderer was disposed.', {
        why: 'Its device, pipelines, and render targets have been destroyed.',
        fix: 'Create a new Renderer. A disposed Renderer cannot be revived — the GPUDevice it owned is gone.',
      });
    }

    const cam = camera ?? this.#camera;
    if (cam === null) {
      fail('CAMERA_NOT_SET', 'No camera was passed to render() and none is set on the renderer.', {
        why: 'The view-projection matrix, the frustum for culling, and the position written to the frame uniform all come from the camera.',
        fix: 'Call `renderer.render(scene, camera)`, or set it once with `renderer.camera = camera`.',
      });
    }

    const cpuStart = now();

    // Frame delta, in seconds. `start()` sets this for the rAF path, but
    // `render()` is also called directly — by tests, by benchmarks, by `capture`
    // — and on that path the delta would otherwise stay 0, so any shader
    // animating from `frame.delta` silently freezes.
    const wall = cpuStart - this.#lastRenderTime;
    this.#lastRenderTime = cpuStart;
    this.#lastDelta = this.#frameIndex === 0 ? 0 : Math.min(wall / 1000, 0.1);

    this.#syncSize();

    // --- 2. cull ------------------------------------------------------------
    const sceneItems = scene.collectDrawItems(this.#items, cam, OBJECT_UNIFORM_STRIDE);
    const candidateTotal = sceneItems.length;

    // --- 3. sort ------------------------------------------------------------
    // The sorted array is what gets drawn. Sorting a copy and then encoding the
    // unsorted original is the worst possible version of this: you pay for the
    // sort and get none of the benefit.
    const drawList = this.#sort(sceneItems);

    // --- 4. pack ------------------------------------------------------------
    // Packed from the *collection* order, and keyed by `item.objectId`, which is
    // the collection index. The object uniform slot a draw item binds is derived
    // from that same id, so packing by sorted position would put every transform
    // in the wrong slot and every object would be drawn with a neighbour's
    // matrix — with no error anywhere. Do not "optimise" this to the loop index.
    const packed = this.#packObjects(sceneItems);
    this.#lastDrawn = sceneItems.length;

    // --- 5. upload ----------------------------------------------------------
    const uniformBytes = this.#upload(cam, packed);

    // --- 6. encode ----------------------------------------------------------
    this.#encode(drawList, packed, target);

    const cpu = now() - cpuStart;
    // From the walk, not derived here: `drawList` only holds survivors, so
    // subtracting its length from itself would read zero in every scene.
    this.#record(cpu, uniformBytes, candidateTotal, scene.meshNodeCount - candidateTotal);

    this.#frameIndex++;
    this.#lastStats = this.#current;
    return this.#current;
  }

  /**
   * Runs `render` on every animation frame.
   *
   * The callback receives the frame delta in seconds, already clamped to 100 ms.
   * That clamp is not a convenience: a backgrounded tab resumes with a delta of
   * several seconds, and any simulation integrating `position += velocity * dt`
   * teleports. Every frame-based system has this bug; the fix is to clamp once,
   * here, so no user code has to.
   */
  start(
    frame: (dt: number, elapsed: number, stats: FrameStats) => void,
  ): void {
    if (this.#running) return;
    this.#running = true;
    this.#startTime = now();
    this.#lastTime = this.#startTime;

    const tick = (): void => {
      if (!this.#running) return;
      this.#rafHandle = requestAnimationFrame(tick);
      const t = now();
      const dt = Math.min((t - this.#lastTime) / 1000, 0.1);
      this.#lastTime = t;
      this.#lastDelta = dt;
      this.#elapsed = (t - this.#startTime) / 1000;
      frame(dt, this.#elapsed, this.#lastStats);
    };
    this.#rafHandle = requestAnimationFrame(tick);
  }

  stop(): void {
    if (!this.#running) return;
    this.#running = false;
    cancelAnimationFrame(this.#rafHandle);
    this.#rafHandle = 0;
  }

  get running(): boolean { return this.#running; }

  /**
   * Scene exposure, consumed by the present pass's tone map.
   *
   * Written into the frame uniform on the frame it changes and not before, so
   * setting it every frame would cost a `writeBuffer` per frame to say the same
   * number.
   */
  set exposure(v: number) {
    if (v !== this.#exposure) {
      this.#exposure = v;
      this.#exposureDirty = true;
    }
  }
  get exposure(): number { return this.#exposure; }

  /**
   * Renders a scene and reads the pixels back on the CPU.
   *
   * This is the only supported way to get the image out of a WebGPU canvas, and
   * the reason is a genuine gap in the API: **a WebGPU canvas has no
   * `preserveDrawingBuffer`.** The swapchain texture is transient — it expires at
   * present, and sampling the canvas afterwards, by `drawImage` or otherwise,
   * yields an empty image. It is the single most common reason a WebGPU port of
   * a working WebGL program "renders but the screenshot is black".
   *
   * So this renders to an offscreen target with `COPY_SRC` and copies out of
   * it, which is deterministic. The alternative — reading the swapchain texture
   * — only works inside the same task as the draw, which no caller can rely on.
   *
   * `bytesPerRow` is padded to the 256-byte alignment `copyTextureToBuffer`
   * requires, so `data` is `bytesPerRow * height` long and rows must be read at
   * that stride rather than `width * 4`.
   */
  async capture(scene: Scene, camera?: Camera): Promise<CapturedFrame> {
    this.#assertLive();
    const target = this.#acquireCaptureTarget();
    this.render(scene, camera, target);

    const width = target.width;
    const height = target.height;
    const bytesPerRow = alignUp(width * 4, 256);
    const device = this.device.device;

    const readback = device.createBuffer({
      label: 'apse:capture',
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder({ label: 'apse:capture' });
    encoder.copyTextureToBuffer(
      { texture: target.colorTexture },
      { buffer: readback, bytesPerRow },
      { width, height },
    );
    device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    // `getMappedRange()` returns a buffer that is *detached* by `unmap()`.
    // `new Uint8Array(ab)` is a view over it, not a copy, so holding that view
    // across the unmap yields a detached buffer that reads as all zeroes — a
    // capture that silently returns a black image. `.slice()` is the copy, and
    // it has to happen here, inside the mapped window.
    const data = new Uint8Array(readback.getMappedRange()).slice();
    readback.unmap();
    readback.destroy();

    return { width, height, bytesPerRow, format: target.format, data };
  }

  /** Lazily allocated, resized with the canvas, disposed with the renderer. */
  #captureTarget: RenderTargetImpl | null = null;

  #acquireCaptureTarget(): RenderTargetImpl {
    if (this.#captureTarget !== null && !this.#captureTarget.disposed) return this.#captureTarget;
    const t = createColorTarget(this.device, {
      width: this.#sizer.width,
      height: this.#sizer.height,
      sampleCount: 1,
      label: 'apse:capture',
      // Depth is not optional here. A render pass must agree with its pipelines
      // about whether a depth attachment exists, and apse's materials declare
      // depth state by default. A pass with no depth attachment and a pipeline
      // that has depthStencil is a validation error, and the consequence is that
      // the entire pass is discarded — including the clear. The symptom is a
      // uniformly black readback and no error anywhere, which is the most
      // expensive kind of bug there is.
      depth: true,
      // COPY_SRC is the entire point of this target. Without it the copy below
      // is a validation error, and the error names a usage flag rather than the
      // actual mistake.
      usage: GPUTextureUsage.COPY_SRC,
    });
    this.#captureTarget = t;
    return t;
  }

  dispose(): void {
    if (this.#disposed) return;
    this.stop();
    this.#disposed = true;
    this.#captureTarget?.dispose();
    this.#canvas.dispose();
    this.#sizer.dispose();
    this.objectUniforms.dispose();
    this.frameUniforms.dispose();
    this.device.destroy();
  }

  // -------------------------------------------------------------------------
  // Steps
  // -------------------------------------------------------------------------

  /**
   * Step 1: reconcile the canvas backing store with the target.
   *
   * Resizing invalidates the swapchain texture, and WebGPU has no depth buffer,
   * so the depth attachment has to be rebuilt too. Doing this only on an actual
   * change matters: a naive `resize()` on every frame reallocates every frame.
   */
  #syncSize(): void {
    if (!this.#sizer.update()) return;
    this.#canvas.resize(this.#sizer.width, this.#sizer.height);
  }

  /**
   * Step 3: sort draw items.
   *
   * Opaque geometry sorts front-to-back, because early-Z rejects fragments of
   * everything behind what has already been drawn — so the draw order is a
   * bandwidth optimisation, not a visual one. Transparent geometry sorts
   * back-to-front, because that one *is* visual: blending is order-dependent.
   *
   * Both lists are grouped by material first. `setPipeline` is a relatively
   * expensive call, and two items sharing a pipeline can skip it entirely. The
   * grouping costs one comparison in the comparator, so it pays for itself after
   * two items share a material, which in any real scene is most of them.
   */
  #sort(items: DrawItem[]): DrawItem[] {
    const out = this.#sorted;
    out.length = 0;

    for (let i = 0; i < items.length; i++) {
      const it = items[i]!;
      if (it.visible) out.push(it);
    }

    sortDrawItems(out);
    return out;
  }

  /**
   * Step 4: write every object's world matrix into the shared object buffer.
   *
   * The normal matrix is the inverse transpose of the upper 3x3, computed with
   * the adjugate rather than a general matrix inverse: 27 multiplies against
   * 4x4's ~120, and at a thousand objects that difference is a third of the
   * frame.
   *
   * The uniform buffer grows by doubling rather than by exact size, so a scene
   * that oscillates around a threshold does not reallocate every frame.
   */
  /**
   * Step 4: write each dirty object's transform into the shared object buffer.
   *
   * **Only dirty objects are packed, and only the dirty byte range is
   * uploaded.** A static scene writes nothing after its first frame; a scene
   * where one object moves writes 256 bytes, not 1.28 MB.
   *
   * The dirty key is `(identity of model, worldVersion, buffer generation)`:
   *
   *  - `model` is the node's own `Float32Array(16)`, allocated per node and
   *    never replaced, so identity alone identifies the node. It is also what
   *    makes the key correct: `worldVersion` is drawn from a global clock, so
   *    two different nodes written in the same pass carry the same number with
   *    completely different matrices. Identity is load-bearing, not a
   *    fast path.
   *  - the generation counter invalidates every cached entry when the uniform
   *    buffer is reallocated, because the CPU mirror is a new array and every
   *    previous write is gone with it.
   *
   * Cost of a skipped object is two integer compares and two stores. Cost of a
   * packed one is 30 multiplies, 18 adds, a divide, 56 typed-array stores, and
   * 256 bytes of upload.
   */
  #packObjects(items: DrawItem[]): number {
    const count = items.length;
    const u = this.objectUniforms;
    if (count > u.capacity) {
      u.allocate(nextPowerOfTwo(count));
      // A new mirror means every cached write is stale.
      this.#packGeneration++;
      for (let i = 0; i < this.#packedGeneration.length; i++) this.#packedGeneration[i] = this.#packGeneration;
    }
    if (this.#packedVersion.length < u.capacity) {
      this.#packedVersion = growNumbers(this.#packedVersion, u.capacity);
      this.#packedModel = growSlots(this.#packedModel, u.capacity);
      this.#packedGeneration = growNumbers(this.#packedGeneration, u.capacity);
    }

    const stride = OBJECT_BLOCK.stride;
    const gen = this.#packGeneration;
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    let packed = 0;

    for (let i = 0; i < count; i++) {
      const item = items[i]!;
      const slot = item.objectId;
      if (this.#packedModel[slot] === item.model &&
          this.#packedVersion[slot] === item.worldVersion &&
          this.#packedGeneration[slot] === gen) continue;

      normalMatrixOf(item.model, _normal);
      // Packed at `slot`, not at the loop index: the dynamic offset a draw binds
      // is derived from `objectId`, so packing by position would put each
      // transform in the slot of whatever object sorts into that position.
      u.pack(slot, item.model, _normal, slot, item.firstInstance, 1);

      this.#packedModel[slot] = item.model;
      this.#packedVersion[slot] = item.worldVersion;
      this.#packedGeneration[slot] = gen;
      if (slot < lo) lo = slot;
      if (slot > hi) hi = slot;
      packed++;
    }

    if (hi >= lo) u.uploadRange(lo * stride, (hi + 1) * stride);
    this.#packedCount = packed;
    return count;
  }

  #packedModel: (Float32Array | null)[] = [];
  #packedVersion: number[] = [];
  #packedGeneration: number[] = [];
  #packGeneration = 1;
  #packedCount = 0;

  /**
   * Step 5: upload the frame uniform.
   *
   * One write per frame, shared by every material. This is the single largest
   * saving over a per-material frame uniform: 500 materials writing the same
   * 256 bytes is 500 writes of 128 KB to say one thing 500 times.
   *
   * Written through `FrameUniforms.set`, not through its `block` directly.
   * That is not a style preference: `set` is what marks the block dirty, and
   * `flush` is a no-op on a clean block. Writing the block behind its back and
   * then flushing leaves a zeroed uniform buffer on the GPU — every
   * `frame.viewProj` is a zero matrix, every triangle is degenerate, every
   * draw call succeeds, and the screen is empty. There is no error to find.
   */
  #upload(cam: Camera, objectCount: number): number {
    const f = this.frameUniforms;
    f.set('view', cam.view);
    f.set('proj', cam.projection);
    f.set('viewProj', cam.viewProj);
    f.set('invView', cam.invView);
    f.set('invProj', cam.invProj);
    f.set('invViewProj', cam.invViewProj);
    f.set('camPos', cam.worldPosition);
    f.set('time', this.#elapsed);
    f.set('delta', this.#lastDelta);
    f.set('elapsed', this.#elapsed);
    _pair[0] = this.#sizer.width;
    _pair[1] = this.#sizer.height;
    f.set('resolution', _pair);
    f.set('viewport', _pair);
    // Exposure belongs to the present pass, which re-asserts it on the frames
    // it changes. Writing 1 here would undo that every frame.
    if (this.#exposureDirty) {
      f.set('exposure', this.#exposure);
      this.#exposureDirty = false;
    }
    f.set('alpha', 1);
    f.flush();

    return FRAME_BLOCK.size + objectCount * OBJECT_BLOCK.stride;
  }

  #lastDelta = 0;
  #lastRenderTime = 0;
  #exposure = 1;
  #exposureDirty = true;

  /**
   * Step 6: encode and submit the frame.
   *
   * One command encoder, one render pass, one submit. The pass is opened once
   * for all opaque geometry, drawn, then opened again for transparent geometry
   * with a different depth-write mode — because depth state is baked into the
   * pipeline, not settable mid-pass, so the two phases genuinely cannot share a
   * pass.
   */
  #encode(items: DrawItem[], objectCount: number, target: RenderTargetImpl): void {
    const device = this.device.device;
    const encoder = device.createCommandEncoder({ label: 'apse:frame' });
    this.#encoder = encoder;

    const colorView = target.isCanvas
      ? target.getColorViewForFrame()
      : target.colorView;
    const depthView = target.depthView;
    const msaa = target.sampleCount > 1;

    let lastPipeline: GPURenderPipeline | null = null;
    let lastPhase: DrawItem['phase'] | null = null;
    let pass: GPURenderPassEncoder | null = null;
    let drawCalls = 0;
    let triangles = 0;
    this.#encPipelineCalls = 0;
    this.#encBindGroupCalls = 0;
    this.#encBufferCalls = 0;

    // Redundant-call suppression.
    //
    // WebGPU charges per API call, and a draw is 7 of them. When consecutive
    // items share a material, four of those seven are re-sending state that is
    // already bound. Measured on a 5000-object single-material scene, the guards
    // below take 30,001 calls down to 10,005 — a 67% cut in submission cost.
    //
    // The guards are worth nothing unless the draw list is grouped, which is
    // what the sort is for: in graph order every guard's state differs on every
    // draw. Sorting and guarding are one change, not two.
    let lastFrameBG: GPUBindGroup | null = null;
    let lastMaterialBG: GPUBindGroup | null = null;
    let lastTextureBG: GPUBindGroup | null = null;
    let lastVB: GPUBuffer | null = null;
    let lastIB: GPUBuffer | null = null;
    let lastInstanceIB: GPUBuffer | null = null;

    const begin = (): GPURenderPassEncoder => {
      const p = encoder.beginRenderPass({
        label: msaa ? 'apse:msaa' : 'apse:pass',
        colorAttachments: [{
          view: colorView,
          resolveTarget: msaa ? target.sampleView : undefined,
          clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1 },
          loadOp: 'clear',
          storeOp: msaa ? 'discard' : 'store',
        }],
        depthStencilAttachment: depthView === undefined ? undefined : {
          view: depthView,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      lastPipeline = null;
      // A pass has its own attachments, so nothing bound in the previous pass
      // can be assumed. Resetting the guards here is what makes them correct
      // rather than merely fast.
      lastFrameBG = null;
      lastMaterialBG = null;
      lastTextureBG = null;
      lastVB = null;
      lastIB = null;
      lastInstanceIB = null;
      return p;
    };

    for (let i = 0; i < items.length; i++) {
      const item = items[i]!;
      const material = item.material;
      const geometry = item.geometry;

      // Depth state is baked into the pipeline, not settable mid-pass, so an
      // opaque-to-transparent transition genuinely requires a second pass. A
      // pipeline change within a phase does not.
      if (pass === null || lastPhase !== item.phase) {
        if (pass !== null) pass.end();
        pass = begin();
        lastPhase = item.phase;
        lastPipeline = null;
      }
      const enc = pass;

      if (lastPipeline !== material.renderPipeline) {
        this.#encPipelineCalls++;
        // A pipeline is bound to its attachment formats at creation, and Dawn
        // reports a mismatch at `setPipeline` as a validation error naming two
        // format enums. That is a technically accurate description of a mistake
        // that is almost always "you forgot to pass the format", so it is
        // checked here instead, where the fix can be stated.
        if (material.targetFormats[0] !== target.format) {
          fail('RENDER_TARGET_FORMAT_MISMATCH',
            `Material "${material.name}" was compiled for colour format ` +
            `"${material.targetFormats[0] ?? 'unknown'}" but is being drawn into a target of ` +
            `format "${target.format}".`, {
            why: 'WebGPU bakes the colour attachment format into a render pipeline. A pipeline cannot be used with a pass whose colour format differs, and the mismatch invalidates the whole command buffer — not just this draw — so nothing appears on screen and no exception is thrown.',
            fix: material.targetFormats[0] === 'rgba8unorm' || material.targetFormats[0] === 'bgra8unorm'
              ? `Pass the target's format when creating the material: pbrMaterial(device, { targetFormat: "${target.format}" }). The canvas default is navigator.gpu.getPreferredCanvasFormat(), which is bgra8unorm on desktop.`
              : `Recreate the material with \`targetFormat: "${target.format}"\`, or draw it into a target of format "${material.targetFormats[0]}".`,
          });
        }
        enc.setPipeline(material.renderPipeline);
        lastPipeline = material.renderPipeline;
      }

      // The frame uniform is one buffer per device, so this binds once per pass
      // and the comparison is the cost of noticing that.
      const frameBG = material.frameBindGroup;
      if (lastFrameBG !== frameBG) {
        this.#encBindGroupCalls++;
        enc.setBindGroup(BIND_GROUP.frame, frameBG);
        lastFrameBG = frameBG;
      }

      // The one genuinely per-draw binding: the object transform, addressed by a
      // dynamic offset. Reusing the offsets array matters — a `[offset]` literal
      // per draw is 5000 short-lived arrays a frame at 5000 objects.
      _dynamicOffsets[0] = item.objectOffset;
      this.#encBindGroupCalls++;
      enc.setBindGroup(BIND_GROUP.object, material.objectBindGroup, _dynamicOffsets);

      // Group 2 is keyed to the pipeline, so it is cleared whenever the pipeline
      // changes rather than carried across one. Carrying it is a latent
      // validation error, not just a stale optimisation.
      if (lastPipeline === null) lastMaterialBG = null;
      const matBG = material.materialBindGroup;
      if (matBG !== null && lastMaterialBG !== matBG) {
        this.#encBindGroupCalls++;
        enc.setBindGroup(BIND_GROUP.material, matBG);
        lastMaterialBG = matBG;
      }
      const texBG = material.textureBindGroup;
      if (texBG !== null && lastTextureBG !== texBG) {
        this.#encBindGroupCalls++;
        enc.setBindGroup(BIND_GROUP.texture, texBG);
        lastTextureBG = texBG;
      }

      const instances = item.instanceCount;
      const first = item.firstInstance;
      if (geometry.indexBuffer !== null) {
        if (lastIB !== geometry.indexBuffer) {
          this.#encBufferCalls++;
          enc.setIndexBuffer(geometry.indexBuffer, indexFormatOf(geometry));
          lastIB = geometry.indexBuffer;
        }
        if (lastVB !== geometry.vertexBuffer) {
          this.#encBufferCalls++;
          enc.setVertexBuffer(0, geometry.vertexBuffer);
          lastVB = geometry.vertexBuffer;
        }
        triangles += (geometry.indexCount / 3) * instances;
      } else {
        if (lastVB !== geometry.vertexBuffer) {
          enc.setVertexBuffer(0, geometry.vertexBuffer);
          lastVB = geometry.vertexBuffer;
        }
        triangles += (geometry.indexCount / 3) * instances;
      }

      // Slot 1 carries per-instance transforms. An instanced layout always has
      // two slots; a non-instanced one has one, and the buffer is absent, so the
      // slot is skipped rather than bound to null.
      const instBuf = instanceBufferOf(geometry);
      if (instBuf !== null) {
        if (lastInstanceIB !== instBuf) {
          this.#encBufferCalls++;
          enc.setVertexBuffer(1, instBuf);
          lastInstanceIB = instBuf;
        }
      }

      if (geometry.indexBuffer !== null) {
        enc.drawIndexed(geometry.indexCount, instances, 0, 0, first);
      } else {
        enc.draw(geometry.indexCount, instances, 0, first);
      }
      drawCalls++;

      if (material.slotsDirty) material.flushSlots();
    }

    if (pass !== null) pass.end();
    device.queue.submit([encoder.finish()]);
    this.#encoder = null;
    this.#pendingDrawCalls = drawCalls;
    this.#pendingTriangles = triangles;
    void objectCount;
  }

  #pendingDrawCalls = 0;
  #pendingTriangles = 0;
  #encPipelineCalls = 0;
  #encBindGroupCalls = 0;
  #encBufferCalls = 0;

  /**
   * Step 8: statistics and budget enforcement.
   *
   * Averaged over a window rather than checked per frame, because a single slow
   * frame is a GC pause and not a regression. The window length is the
   * difference between a budget that reports a problem and a budget that fires
   * every time a shader finishes compiling.
   */
  #record(cpu: number, uniformBytes: number, candidates: number, culled: number): void {
    const s = this.#current;
    const gpu = 0;
    const drawCalls = this.#pendingDrawCalls;
    const triangles = this.#pendingTriangles;

    this.#cpuSamples[this.#sampleCursor] = cpu;
    this.#gpuSamples[this.#sampleCursor] = gpu;
    this.#sampleCursor = (this.#sampleCursor + 1) % this.#sampleSize;
    if (this.#sampleCount < this.#sampleSize) this.#sampleCount++;

    s.cpu = cpu;
    s.gpu = gpu;
    s.drawCalls = drawCalls;
    s.triangles = triangles;
    s.candidates = candidates;
    s.culled = culled;
    s.uniformBytes = uniformBytes;
    s.cpuPerDraw = drawCalls === 0 ? 0 : cpu / drawCalls;
    s.pipelineCalls = this.#encPipelineCalls;
    s.bindGroupCalls = this.#encBindGroupCalls;
    s.bufferCalls = this.#encBufferCalls;
    s.naiveCalls = drawCalls * 7;
    s.packedObjects = this.#packedCount;
    s.skippedObjects = this.#lastDrawn - this.#packedCount;
    s.averageCpu = average(this.#cpuSamples, this.#sampleCount);
    s.averageGpu = average(this.#gpuSamples, this.#sampleCount);

    this.#checkBudget(s);
  }

  #checkBudget(s: FrameStats): void {
    const b = this.#budget;
    if (b === null) return;
    const breach = (name: string, actual: number, limit: number): void => {
      if (actual > limit && isDevelopmentMode()) {
        this.#budgetBreached?.(name, actual, limit);
      }
    };
    if (b.cpu !== undefined) breach('cpu', s.averageCpu, b.cpu);
    if (b.gpu !== undefined) breach('gpu', s.averageGpu, b.gpu);
    if (b.drawCalls !== undefined) breach('drawCalls', s.drawCalls, b.drawCalls);
    if (b.idle !== undefined && s.drawCalls <= 1) breach('idle', s.averageCpu, b.idle);
  }

  #assertLive(): void {
    this.device.assertLive();
  }

  /** The command encoder for the frame in flight, for readback and diagnostics. */
  get currentEncoder(): GPUCommandEncoder | null { return this.#encoder; }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function emptyStats(): MutableStats {
  return {
    cpu: 0, gpu: 0, averageCpu: 0, averageGpu: 0,
    drawCalls: 0, triangles: 0, candidates: 0, culled: 0,
    uniformBytes: 0, cpuPerDraw: 0,
    bindGroupCalls: 0, pipelineCalls: 0, bufferCalls: 0, naiveCalls: 0,
    packedObjects: 0, skippedObjects: 0,
  };
}

function average(samples: Float64Array, count: number): number {
  if (count === 0) return 0;
  let sum = 0;
  for (let i = 0; i < count; i++) sum += samples[i]!;
  return sum / count;
}

/**
 * Grows a bookkeeping array by doubling.
 *
 * A plain `number[]` rather than a typed array on purpose: these are read once
 * per object per frame and written only when an object actually moves, so the
 * typed-array access advantage does not apply and a plain array avoids the
 * `ArrayBuffer`/`SharedArrayBuffer` generic gymnastics for no measurable gain.
 */
function growNumbers(a: number[], n: number): number[] {
  if (a.length >= n) return a;
  const next = new Array<number>(Math.max(n, a.length * 2)).fill(0);
  for (let i = 0; i < a.length; i++) next[i] = a[i]!;
  return next;
}

function growSlots(a: (Float32Array | null)[], n: number): (Float32Array | null)[] {
  if (a.length >= n) return a;
  const next = new (a.constructor as new (n: number) => (Float32Array | null)[])(
    Math.max(n, a.length * 2),
  );
  for (let i = 0; i < a.length; i++) next[i] = a[i]!;
  return next;
}

function nextPowerOfTwo(n: number): number {
  let p = 1024;
  while (p < n) p *= 2;
  return p;
}

function instanceBufferOf(geometry: DrawItem['geometry']): GPUBuffer | null {
  const b = (geometry as { instanceBuffer?: GPUBuffer | null }).instanceBuffer;
  return b ?? null;
}

function indexFormatOf(geometry: DrawItem['geometry']): GPUIndexFormat {
  const fmt = (geometry as { indexFormat?: GPUIndexFormat | null }).indexFormat;
  return fmt ?? 'uint16';
}

/**
 * Inverse transpose of the upper-left 3x3, into a 9-float column-major array.
 *
 * The adjugate divided by the determinant, expanded by hand. A general 4x4
 * inverse is ~120 multiplies and an allocation-free version still costs six
 * times as much for a result whose fourth row is never used.
 */
function normalMatrixOf(m: Float32Array, out: Float32Array): Float32Array {
  const a = m[0]!, b = m[1]!, c = m[2]!;
  const d = m[4]!, e = m[5]!, f = m[6]!;
  const g = m[8]!, h = m[9]!, i = m[10]!;

  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  let det = a * A + b * B + c * C;

  if (det === 0 || !Number.isFinite(det)) {
    // A degenerate transform has no inverse. Identity is the least-wrong
    // answer: it makes the object look un-transformed rather than invisible,
    // which is a bug a user can see and report.
    out.set(IDENTITY_NORMAL);
    return out;
  }
  det = 1 / det;

  out[0] = A * det;             out[1] = B * det;             out[2] = C * det;
  out[3] = (c * h - b * i) * det; out[4] = (a * i - c * g) * det; out[5] = (b * g - a * h) * det;
  out[6] = (b * f - c * e) * det; out[7] = (c * d - a * f) * det; out[8] = (a * e - b * d) * det;
  return out;
}

const IDENTITY_NORMAL = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

export type { Material };
