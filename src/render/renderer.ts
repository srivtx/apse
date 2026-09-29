/**
 * The frame loop.
 *
 * This is the only module that knows the order of operations in a frame, and it
 * is written so that the order is the whole story:
 *
 * ```txt
 *   1. size     — did the backing store change? recreate targets if so
 *   2. cull     — scene.collectDrawItems  (pruned transform walk + sphere tests)
 *   3. sort     — opaque front-to-back, transparent back-to-front, both grouped
 *   4. pack     — world matrices into the object uniform buffer, one write each
 *   5. upload   — frame uniform (1 write) + object uniform (1 write)
 *   6. encode   — one command encoder, one render pass, N draws
 *   6b. present — one more pass on the same encoder: the fullscreen tone map
 *   7. submit   — one queue.submit
 *   8. stats    — timings, on a frame budget the caller can declare
 * ```
 *
 * Step 6b is the present pass, and it is on by default. `renderer.sceneTarget`
 * is therefore an `rgba16float` intermediate rather than the canvas, and every
 * material in the scene has to be compiled for `renderer.sceneFormat` — a
 * material built for `getPreferredCanvasFormat()` fails through
 * `assertDrawable`, naming both formats, instead of invalidating the whole
 * command buffer with no exception. `toneMapping: null` and `hdr: false` are the
 * two ways out, and both are slower.
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
import { isErr } from '../core/result.ts';
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
import type { Scene } from '../scene/graph.ts';
import type { Camera } from '../scene/camera.ts';
import type { DrawItem, FrameTimingStats, RenderTarget } from './types.ts';
import { sortDrawItems } from './sort.ts';
import { GpuTimer } from './timing.ts';
import { DEFAULT_TONE_MAPPING, PresentPass } from './present.ts';
import type { TonemapOptions } from '../material/tonemap.ts';
import type { Material } from '../material/material.ts';

/**
 * `(beginning, end)` query pairs one frame can carry.
 *
 * Three, because three is every pass a frame can open: opaque, transparent, and
 * the present. A pair may be written **once per submission**, so a pair cannot
 * be opened on one pass and closed on the next — reusing a write index is a
 * validation error, and validation errors on a render pass discard the pass.
 * One pair would time whichever single pass happened to be stamped, which on a
 * transparent-heavy frame is the wrong one.
 */
const TIMESTAMP_PAIRS = 3;

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/**
 * What one frame cost. All times in milliseconds.
 *
 * The three GPU fields come from {@link FrameTimingStats} rather than being
 * declared here, so there is one definition of "a missing measurement is
 * `null`" and not two that can drift.
 */
export interface FrameStats extends FrameTimingStats {
  /** Wall-clock time inside `render()`. The number that must stay under budget. */
  readonly cpu: number;
  /** Measured ms/frame over the last `sampleSize` frames. */
  readonly averageCpu: number;
  /** Draw calls encoded this frame. */
  readonly drawCalls: number;
  /** Triangles submitted this frame, counting every instance. */
  readonly triangles: number;
  /** Mesh nodes the scene walk found, before frustum culling. */
  readonly candidates: number;
  /** Mesh nodes the frustum rejected. Never counts a node that was not tested. */
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
  /** Maximum ms/frame of draw calls. */
  readonly drawCalls?: number;
  /**
   * Maximum ms/frame of GPU time.
   *
   * Checked only when the device can be timed — see
   * {@link FrameTimingStats.gpuTimingAvailable}. A budget against a number the
   * renderer never measured has nothing to compare, and reporting a breach of it
   * would send the reader to the GPU for a problem the CPU has.
   */
  readonly gpu?: number;
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
  /**
   * The tone map, or `null` for no present pass at all. Default
   * {@link DEFAULT_TONE_MAPPING}.
   *
   * **On by default, and that is the point.** A material writes linear values
   * and a canvas in `bgra8unorm` stores them verbatim, so a straight-to-canvas
   * frame is a linear image displayed as if it were sRGB: too dark, with a lit
   * surface reading as unlit and nothing reporting an error. That is why apse
   * rendered darker than three.js out of the box.
   *
   * `null` restores the direct path — no intermediate, no fullscreen pass, no
   * pipeline. It is a real cost, not a free setting: see `hdr`.
   */
  readonly toneMapping?: TonemapOptions | null;
  /**
   * Render the scene into an `rgba16float` intermediate and tone map that.
   * Default true.
   *
   * **A tone map fed already-clipped LDR is nearly a no-op.** Without this, the
   * intermediate carries the destination's own 8-bit format, the highlights were
   * discarded before the curve saw them, and what is left is a full-screen pass
   * to apply a curve to a clamped image.
   *
   * The consequence is that the scene renders into `rgba16float` instead of the
   * canvas format, so **every material must be built for
   * {@link Renderer.sceneFormat}**. One built for `getPreferredCanvasFormat()`
   * now fails loudly, through `assertDrawable`, naming both formats — which is
   * the intended trade: the alternative is a command buffer invalidated whole,
   * with no exception. `toneMapping: null` and `hdr: false` are the two escape
   * hatches, and they are the only way to keep a material written for the
   * canvas format.
   */
  readonly hdr?: boolean;
  /** Declarative performance limits, checked every frame. */
  readonly budget?: RenderBudget;
  /** Frames of samples retained for averaging. Default 120. */
  readonly sampleSize?: number;
  /** A camera to use when `render()` is called without one. */
  readonly camera?: Camera;
}

const EMPTY_ITEMS: DrawItem[] = [];

/** Scratch, module-level, and reused. The frame loop must not allocate. */
const _normal = new Float32Array(12);
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
 *
 * **A plain `Array`, not a `Uint32Array`.** The two are equally correct — the
 * IDL type is `sequence<GPUBufferDynamicOffset>`, so both convert — but Blink's
 * `sequence<>` conversion has a fast path that requires a real `v8::Array` and
 * falls off it for a typed array, walking the generic iterator protocol instead.
 * Measured on Apple M2 / Chrome / headless with the queue drained each iteration,
 * one `setBindGroup` with a reused 1-element `Uint32Array` costs 1.22 us against
 * 0.28 us for the same reused 1-element `Array`, in core and in compatibility
 * alike. At one such bind per draw that is ~0.9 us per draw — several times the
 * entire per-draw gap this renderer had against three.js — and it was invisible
 * to every call census, because the number of calls is identical either way.
 * A call census counts calls; it cannot price an argument.
 */
const _dynamicOffsets: number[] = [0];

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
  /** Null when `toneMapping: null`, in which case the scene draws to the canvas. */
  #present: PresentPass | null = null;
  /** Null when the device has no `timestamp-query`. Not an error: see FrameTimingStats. */
  #timer: GpuTimer | null = null;
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

    // Three pairs, because a frame can open an opaque pass, a transparent pass
    // and the present pass, and one write index may only be written once per
    // submission. `isErr`, not a throw: a device without `timestamp-query` is a
    // fact about the machine, and it is roughly half of them. Carrying on with
    // `stats.gpu === null` is the whole contract.
    const timing = GpuTimer.create(device, { pairs: TIMESTAMP_PAIRS });
    this.#timer = isErr(timing) ? null : timing.value;
  }

  /**
   * Creates a renderer and its device.
   *
   * Async because device acquisition is: `requestAdapter` and `requestDevice`
   * both return promises, and shader compilation on first draw is asynchronous.
   * A synchronous constructor is not possible here, and a lazily-initialised
   * one hides the cost until the first frame instead of at setup.
   *
   * It is also where the present pass is built, which is the second reason to
   * await: a tone map is a compiled pipeline, and compiling one on the first
   * frame is a two-to-five second stall on a cold shader cache.
   */
  static async create(canvas: HTMLCanvasElement, opts: RendererOptions = {}): Promise<Renderer> {
    const device = await createDevice(canvas, opts);
    try {
      const renderer = new Renderer(device, opts);
      renderer.#present = await renderer.#createPresentPass(opts);
      // The pass may have been built with an exposure of its own, and this is the
      // only object that reports one. Adopting it here is what keeps
      // `renderer.exposure` from answering 1 while the screen is at 0.5.
      if (renderer.#present !== null) renderer.#exposure = renderer.#present.exposure;
      return renderer;
    } catch (err) {
      // Destroying the device frees every resource derived from it, so this is
      // the whole teardown even though the partially built renderer's own
      // objects are left to the collector.
      device.destroy();
      throw err;
    }
  }

  /**
   * The present pass, or `null` when the caller asked for the direct path.
   *
   * Built after the canvas target because the pass's destination *is* that
   * target, and its intermediate is sized from it. The resolved options are kept
   * because {@link Renderer.capture} builds a second, identical pass over its own
   * target, and a capture that tone mapped differently from the screen would be
   * worse than no capture at all.
   */
  async #createPresentPass(opts: RendererOptions): Promise<PresentPass | null> {
    this.#toneOptions = opts.toneMapping === undefined ? DEFAULT_TONE_MAPPING : opts.toneMapping;
    this.#hdr = opts.hdr ?? true;
    if (this.#toneOptions === null) return null;
    return PresentPass.create(this.device, {
      target: this.#canvas,
      toneMapping: this.#toneOptions,
      hdr: this.#hdr,
      // Forwarded so `sampleCount: 4` is real MSAA rather than an option that
      // quietly does nothing — a canvas texture is never multisampled, so the
      // intermediate is the only place 4x can live. Ignored by the direct path,
      // where nothing is multisampled at all.
      sampleCount: this.device.sampleCount,
      label: 'apse.present',
    });
  }

  /** Resolved once, so `capture()`'s pass cannot differ from the screen's. */
  #toneOptions: TonemapOptions | null = null;
  #hdr = true;

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

  /**
   * Where the scene is drawn: the present pass's intermediate, or the canvas.
   *
   * This is the target a material's `targetFormat` has to match, and
   * {@link Renderer.sceneFormat} is the number to read rather than guessing.
   */
  get sceneTarget(): RenderTargetImpl {
    return this.#resolveSceneTarget();
  }

  /**
   * The colour format the scene is rendered into, which is what every material
   * in the scene has to be compiled for.
   *
   * `rgba16float` by default, because the present pass tone maps an HDR
   * intermediate. It is the canvas's preferred format only in the direct path.
   */
  get sceneFormat(): GPUTextureFormat {
    return this.#resolveSceneTarget().format;
  }

  #resolveSceneTarget(): RenderTargetImpl {
    const pass = this.#present;
    if (pass === null) return this.#canvas;
    return asImpl(pass.sceneTarget);
  }

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
   * `target` defaults to the **scene** target: the present pass's intermediate
   * when there is one, the canvas when there is not. That is the only correct
   * default, because the present pass is built around sampling an intermediate —
   * a canvas texture has no `TEXTURE_BINDING`, so there is nothing for it to
   * read. Passing the canvas explicitly would draw into a texture the tone map
   * never sees, and the user would get a black frame from a renderer that
   * reports no error.
   *
   * Passing an offscreen target is how you render to a texture for a later
   * post-processing pass, an export, or a readback. **The present pass still
   * presents into the canvas**, from its own intermediate, so a custom target
   * receives the scene and not the tone-mapped image. See {@link Renderer.capture}
   * for what that means for screenshots.
   */
  render(scene: Scene, camera?: Camera, target: RenderTargetImpl = this.#resolveSceneTarget()): FrameStats {
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
    const sceneItems = scene.collectDrawItems(this.#items, cam);

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
    // Both counters come from the walk, not from `drawList` and not from
    // arithmetic on it: `drawList` only holds survivors, so subtracting its
    // length from itself would read zero in every scene, and
    // `meshNodeCount - candidates` is only equal to the culled count until the
    // walk starts dropping a node for a reason that is not culling.
    this.#record(cpu, uniformBytes, scene.meshNodeCount, scene.culledCount);

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
      // Forwarded, because the frame block's copy is not what the image comes
      // from: the tone map re-asserts its own value every render, so setting it
      // on the renderer alone would leave the screen at the old exposure.
      this.#present?.setExposure(v);
      this.#capturePass?.setExposure(v);
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
   *
   * **With the present pass on, the scene is not rendered into that target.** It
   * cannot be: a material's pipeline is compiled for the scene format, which is
   * `rgba16float`, and an 8-bit capture target is a different format — so the
   * mismatch is a hard error rather than a dark image. The capture therefore goes
   * through a second present pass whose *destination* is the capture target: the
   * scene renders into an intermediate of the format it was compiled for, the
   * tone map runs, and the result lands in the 8-bit target. Those are the same
   * pixels that are on screen, which is the whole point of a screenshot — and
   * copying the pre-tone-map intermediate instead would hand back linear values
   * that look far too dark for exactly the reason the un-tone-mapped canvas did.
   *
   * The cost is one extra pipeline, one fullscreen mesh and one intermediate,
   * built on the first `capture()` and never again. The pipeline is compiled
   * rather than shared because {@link PresentPass} owns its own; the layout it
   * compiles against is the shared per-device one, so the driver reuses the
   * compiled program rather than building a second one.
   */
  async capture(scene: Scene, camera?: Camera): Promise<CapturedFrame> {
    this.#assertLive();
    const target = this.#acquireCaptureTarget();
    if (this.#present === null) {
      // No tone map anywhere, so the scene's own output *is* the image and the
      // capture target is simply a second canvas.
      this.render(scene, camera, target);
    } else {
      const pass = await this.#capturePresentPass(target);
      this.render(scene, camera, asImpl(pass.sceneTarget));
      // Its own encoder and its own submit. A screenshot is worth a second
      // submit; the frame loop is not, which is why the present pass inside
      // `render()` shares the frame's encoder instead.
      //
      // The screen's own present pass still runs inside that `render()` call —
      // one wasted fullscreen pass, on a path that already awaits a buffer map.
      // Suppressing it would mean threading a "do not present" flag through
      // `render()` for the sake of one draw a screenshot performs.
      pass.render();
    }

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
  #capturePass: PresentPass | null = null;
  /** In-flight build, so two concurrent `capture()` calls share one pass. */
  #capturePassPending: Promise<PresentPass> | null = null;

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

  /**
   * The present pass that presents into the capture target. Built once.
   *
   * Same tone map and same intermediate format as the screen's pass, so the
   * captured pixels are the displayed pixels rather than a second opinion about
   * them. The tone map's own sRGB decision comes from the destination it is given
   * — the capture target's format, which is the canvas's — so the two passes
   * cannot disagree about it.
   */
  async #capturePresentPass(target: RenderTargetImpl): Promise<PresentPass> {
    if (this.#capturePass !== null) return this.#capturePass;
    this.#capturePassPending ??= PresentPass.create(this.device, {
      target,
      toneMapping: this.#toneOptions,
      hdr: this.#hdr,
      sampleCount: 1,
      label: 'apse.capture.present',
    });
    try {
      const pass = await this.#capturePassPending;
      pass.setExposure(this.#exposure);
      this.#capturePass = pass;
      return pass;
    } finally {
      this.#capturePassPending = null;
    }
  }

  dispose(): void {
    if (this.#disposed) return;
    this.stop();
    this.#disposed = true;
    // The present pass owns the scene target, its fullscreen mesh, and the tone
    // map material, and it disposes them itself. The order matters only in that
    // the pass must go before the canvas it presents into.
    this.#present?.dispose();
    this.#present = null;
    // Before the capture target it presents into, same as the canvas above.
    this.#capturePass?.dispose();
    this.#capturePass = null;
    // A pending timestamp map rejects on destroy; GpuTimer absorbs it. Nothing
    // here is awaited, because teardown does not block.
    this.#timer?.dispose();
    this.#timer = null;
    this.#captureTarget?.dispose();
    this.#canvas.dispose();
    this.#sizer.dispose();
    // The renderer allocated the shared scene buffer, so the renderer frees it.
    // `FrameUniforms.dispose` and `ObjectUniforms.dispose` deliberately do not:
    // they are two faces of one per-device allocation, and a `PresentPass` that
    // disposed its face would take every other material's camera with it.
    deviceCache(this.device.device).sceneUniforms().dispose();
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
   *
   * The present pass's intermediate is resized in the same step, and it is not
   * optional. It is the scene target now, so leaving it at the old extent would
   * render this frame at one size and tone map it at another — a stretched
   * image with no error anywhere, on the one frame after every window resize.
   */
  #syncSize(): void {
    if (!this.#sizer.update()) return;
    this.#canvas.resize(this.#sizer.width, this.#sizer.height);
    this.#present?.resize(this.#sizer.width, this.#sizer.height);
    // The capture target and its pass are resized together, and only if they
    // exist. `capture()` reads the canvas's size when it first runs, so without
    // this a screenshot taken after a window resize would be the old size — and
    // the pass's intermediate would not match the target it presents into.
    if (this.#captureTarget !== null) {
      this.#captureTarget.resize(this.#sizer.width, this.#sizer.height);
      this.#capturePass?.resize(this.#sizer.width, this.#sizer.height);
    }
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
      //
      // **Once per draw item, never once per instance.** An instanced draw
      // binds a single object slot and reads each instance's transform from
      // `instanceBuffer` as a vertex attribute at `@location(3..6)` — which is
      // a vertex buffer rather than a storage buffer precisely so this works in
      // compatibility mode. Looping over instances here would write the same
      // 256 bytes N times into N slots, and N − 1 of them would be read by
      // nobody.
      u.pack(slot, item.model, _normal, slot, item.firstInstance, 1);

      this.#packedModel[slot] = item.model;
      this.#packedVersion[slot] = item.worldVersion;
      this.#packedGeneration[slot] = gen;
      if (slot < lo) lo = slot;
      if (slot > hi) hi = slot;
      packed++;
    }

    // Slot indices, not byte offsets. `uploadRange` took bytes and would now
    // start at the frame region, overwriting the camera with an object matrix.
    if (hi >= lo) u.uploadObjects(lo, hi);
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
   *
   * The present pass is step 6b, recorded into the *same* encoder just before
   * the submit. One submit rather than two: two submits would let the browser
   * present the scene target's frame as its own, which is a frame with no tone
   * map on screen and a compositor sync in the middle of it.
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
    // Reset per frame, and read by #record after the submit. It is a field
    // rather than a local because the reading is taken after this method has
    // returned, from the pairs the readback in this frame will resolve.
    this.#passCount = 0;

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
    let lastMaterialBG: GPUBindGroup | null = null;
    let lastTextureBG: GPUBindGroup | null = null;
    let lastVB: GPUBuffer | null = null;
    let lastIB: GPUBuffer | null = null;
    let lastInstanceIB: GPUBuffer | null = null;

    /**
     * The `timestampWrites` for the next pass, or `undefined` for none.
     *
     * Each pass takes the next pair, and a pass past the last pair gets none
     * rather than a shared one: a `(beginning, end)` write index may be written
     * **once per submission**, and writing it twice is a validation error that
     * discards the pass. Four passes cannot be timed with three pairs, and
     * timing three of them honestly beats invalidating the frame.
     */
    const nextStamp = (): GPURenderPassTimestampWrites | undefined => {
      const timer = this.#timer;
      const pair = this.#passCount++;
      if (timer === null || pair >= TIMESTAMP_PAIRS) return undefined;
      return timer.writes(pair);
    };

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
        // Only ever present when a timer exists. Attaching a write index on a
        // device without `timestamp-query` is a validation error, not a no-op,
        // and it would invalidate every frame on exactly the devices that
        // cannot report their GPU time.
        timestampWrites: nextStamp(),
      });
      lastPipeline = null;
      // A pass has its own attachments, so nothing bound in the previous pass
      // can be assumed. Resetting the guards here is what makes them correct
      // rather than merely fast.
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
        // A pipeline is bound to its attachment formats at creation, and using
        // it with a pass whose colour format differs invalidates the whole
        // command buffer with no exception anywhere. `assertDrawable` is the one
        // place that already knows how to say that in terms of the mistake, and
        // it costs one array length and one string identity per pipeline change.
        target.assertDrawable(material);
        enc.setPipeline(material.renderPipeline);
        lastPipeline = material.renderPipeline;
      }

      // One bind group, one bind, per draw. Frame and object share a buffer, so
      // this used to be two calls and the frame half was pure overhead: it
      // carried the same value on every draw of the pass.
      //
      // That second call was the whole per-draw gap against three.js. 1000
      // objects meant 1000 object binds plus 2 frame binds, and this loop is
      // 85-100% of the frame -- see bench/diag/perf/FINDING.md. `_dynamicOffsets`
      // is reused rather than a `[offset]` literal, which would be 5000
      // short-lived arrays a frame at 5000 objects.
      _dynamicOffsets[0] = item.objectOffset;
      this.#encBindGroupCalls++;
      enc.setBindGroup(BIND_GROUP.scene, material.sceneBindGroup, _dynamicOffsets);

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
      // Read each buffer once. These are accessors, not fields: `GpuMesh` checks
      // liveness on every read, so `lastVB !== geometry.vertexBuffer` written as
      // `enc.setVertexBuffer(0, geometry.vertexBuffer)` costs two calls and two
      // checks where a local costs one of each. At 1,000 draws that is 2,000
      // redundant liveness checks per frame, and this loop is 85-100% of it.
      const ib = geometry.indexBuffer;
      const vb = geometry.vertexBuffer;
      if (ib !== null) {
        if (lastIB !== ib) {
          this.#encBufferCalls++;
          enc.setIndexBuffer(ib, indexFormatOf(geometry));
          lastIB = ib;
        }
        if (lastVB !== vb) {
          this.#encBufferCalls++;
          enc.setVertexBuffer(0, vb);
          lastVB = vb;
        }
      } else if (lastVB !== vb) {
        this.#encBufferCalls++;
        enc.setVertexBuffer(0, vb);
        lastVB = vb;
      }
      // Counted once, after the branch: an unindexed mesh's `indexCount` is its
      // vertex count, so both forms submit the same number of triangles.
      triangles += (geometry.indexCount / 3) * instances;

      // Slot 1 carries the per-instance transforms, `stepMode: 'instance'`.
      //
      // **Non-null is the only correct test for "this is instanced".** The count
      // is not a substitute in either direction. `instanceCount > 1` skips the
      // slot on a mesh that was uploaded with a thousand instances and asked to
      // draw one of them — an ordinary call, and the pipeline still declares
      // slot 1, so the vertex stage reads whatever happens to be bound there: a
      // wrong image and no error. `instanceCount > 0` is a scene-level test and
      // says nothing about what the layout declares.
      //
      // The whole of instancing is this one call plus the count in the draw
      // below: N copies, one `drawIndexed`, one object-uniform slot.
      const instBuf = item.geometry.instanceBuffer;
      if (instBuf !== null && lastInstanceIB !== instBuf) {
        this.#encBufferCalls++;
        enc.setVertexBuffer(1, instBuf);
        lastInstanceIB = instBuf;
      }

      // `ib`, not `geometry.indexBuffer`: the branch above already established
      // which of the two this is, and the getter checks liveness, so reading it
      // again is a third liveness check per draw to learn what a local in scope
      // already says. Measured in Chrome at ~16 ns/draw for the loop body.
      if (ib !== null) {
        enc.drawIndexed(geometry.indexCount, instances, 0, 0, first);
      } else {
        enc.draw(geometry.indexCount, instances, 0, first);
      }
      drawCalls++;

      if (material.slotsDirty) material.flushSlots();
    }

    if (pass !== null) pass.end();

    // A frame with nothing in it still has to clear. With the present pass on,
    // the scene target is an intermediate that outlives the frame, and the tone
    // map samples whatever is in it — so without this an empty scene would
    // present the *previous* frame's pixels indefinitely. One pass, no draws,
    // and only on frames that would otherwise have encoded nothing at all.
    if (items.length === 0) begin().end();

    const present = this.#present;
    if (present !== null) {
      const pair = this.#passCount++;
      // Same rule as `nextStamp`: a pair past the end is skipped rather than
      // shared, because a write index may only be written once per submission.
      const stamp = this.#timer === null || pair >= TIMESTAMP_PAIRS ? undefined : this.#timer.writes(pair);
      present.render(encoder, stamp);
    }

    // After every stamped pass has ended, before `finish()`. `resolveQuerySet` is
    // a queue command: encoded after the passes, it observes their timestamps,
    // and encoded into the same command buffer, the result never travels through
    // the CPU on its way to the readback.
    if (this.#timer !== null && this.#passCount > 0) this.#timer.encodeReadback(encoder);
    device.queue.submit([encoder.finish()]);
    // Never awaited. The reading lands a frame or two later, which is what
    // `#measureGpu` reports; blocking here would stall every frame for a number
    // the caller cannot act on until the frame after anyway.
    this.#timer?.poll();
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
  /** Render passes opened by the frame being encoded. See `nextStamp`. */
  #passCount = 0;

  /**
   * Milliseconds of GPU time for the frame just encoded, or `null`.
   *
   * Read from the timer *after* the submit, so it is a reading from an earlier
   * frame: the timestamp for this frame's work has been encoded but the copy
   * that brings it to the CPU has not been mapped yet. That one-to-two-frame lag
   * is why this is a trend and not a verdict — and it is also why the previous
   * behaviour of reporting a hardcoded `0` was not a simplification but a lie
   * on every frame.
   */
  #measureGpu(): number | null {
    const timer = this.#timer;
    if (timer === null) return null;
    // No pass carried a timestamp, so nothing was measured. A 0 here would say
    // the GPU was idle, which is a different fact about a different machine.
    if (this.#passCount === 0) return null;
    // Pairs past the ones this frame stamped are excluded rather than summed:
    // a frame that opened one pass cannot have a present pass in it, and
    // including a stale pair would attribute another frame's time to this one.
    return timer.sumMs(Math.min(this.#passCount, TIMESTAMP_PAIRS));
  }

  /**
   * Step 8: statistics and budget enforcement.
   *
   * Averaged over a window rather than checked per frame, because a single slow
   * frame is a GC pause and not a regression. The window length is the
   * difference between a budget that reports a problem and a budget that fires
   * every time a shader finishes compiling.
   *
   * `candidates` is the pre-cull count the walk saw and `culled` is what the
   * frustum rejected. Both are passed in rather than derived: the draw list holds
   * only survivors, so anything computed from it reads zero in every scene, and
   * `meshNodeCount - survivors` silently counts a node that was not submitted for
   * a non-cull reason — a hidden layer, an empty geometry — as though the
   * frustum had thrown it away.
   */
  #record(cpu: number, uniformBytes: number, candidates: number, culled: number): void {
    const s = this.#current;
    const gpu = this.#measureGpu();
    const drawCalls = this.#pendingDrawCalls;
    const triangles = this.#pendingTriangles;

    this.#cpuSamples[this.#sampleCursor] = cpu;
    // NaN rather than 0 for "not measured", because the ring cannot shrink. A
    // zero here would be averaged in as though the GPU had been timed at zero,
    // and the mean of nine measured frames and one unmeasured frame is not a
    // tenth of anything.
    this.#gpuSamples[this.#sampleCursor] = gpu ?? Number.NaN;
    this.#sampleCursor = (this.#sampleCursor + 1) % this.#sampleSize;
    if (this.#sampleCount < this.#sampleSize) this.#sampleCount++;

    s.cpu = cpu;
    s.gpu = gpu;
    s.gpuTimingAvailable = this.#timer !== null;
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
    s.averageGpu = averageMeasured(this.#gpuSamples, this.#sampleCount);

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
    // Only with a number to compare. A `gpu` budget on a device that cannot be
    // timed has to stay silent: reporting a breach of a measurement that was
    // never taken is the one thing a budget must not do, because it sends the
    // reader to the GPU for a problem the CPU has.
    if (b.gpu !== undefined && s.averageGpu !== null) breach('gpu', s.averageGpu, b.gpu);
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
    cpu: 0, gpu: null, averageCpu: 0, averageGpu: null, gpuTimingAvailable: false,
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
 * Mean of the samples that are numbers, or `null` when none of them are.
 *
 * Separate from {@link average} because the two windows hold different things.
 * The CPU sample is always a number; a GPU sample is `NaN` on every frame the
 * device could not be timed, and dividing by the full window would report an
 * average of a measurement that was partly fabricated. Skipping the gaps and
 * dividing by the readings that exist is the honest mean, and returning `null`
 * for "none yet" is what keeps `averageGpu: 0` from reading as an idle GPU.
 */
function averageMeasured(samples: Float64Array, count: number): number | null {
  let sum = 0;
  let readings = 0;
  for (let i = 0; i < count; i++) {
    const value = samples[i]!;
    if (!Number.isFinite(value)) continue;
    sum += value;
    readings++;
  }
  return readings === 0 ? null : sum / readings;
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

function indexFormatOf(geometry: DrawItem['geometry']): GPUIndexFormat {
  const fmt = (geometry as { indexFormat?: GPUIndexFormat | null }).indexFormat;
  return fmt ?? 'uint16';
}

/**
 * Narrows a present pass's intermediate to its concrete class.
 *
 * `PresentPass.sceneTarget` is typed as the `RenderTarget` interface, and its own
 * comment argues for that: a renderer's `target` parameter should be the
 * interface too. The renderer cannot take it, because `#encode` calls
 * `assertDrawable`, which the interface does not declare — and a check that only
 * some targets get is the exact failure this class of mistake produces.
 *
 * Narrowed rather than cast, so that if a future pass ever hands back something
 * else this says so at the point of use instead of skipping a format check on
 * every draw of every frame.
 */
function asImpl(target: RenderTarget): RenderTargetImpl {
  if (target instanceof RenderTargetImpl) return target;
  fail('INTERNAL_INVARIANT',
    `A present pass handed back a scene target of type ${target.constructor.name}.`, {
    why: 'PresentPass builds its intermediate with createColorTarget, so it is always a RenderTargetImpl. The renderer needs assertDrawable from it, and a target that is not one would silently skip the colour-format check that stands between a user and an invalidated command buffer.',
    fix: 'This is a bug in apse. Please report it with your RendererOptions.',
  });
}

/**
 * Inverse transpose of the upper-left 3x3, into a 9-float column-major array.
 *
 * The adjugate divided by the determinant, expanded by hand. A general 4x4
 * inverse is ~120 multiplies and an allocation-free version still costs six
 * times as much for a result whose fourth row is never used.
 */
/**
 * Inverse transpose of the upper-left 3x3, written **in WGSL `mat3x3<f32>`
 * layout**: three columns, each 16-byte aligned, so 12 floats with 3 of them
 * padding.
 *
 * The padding is the whole point. A `mat3x3<f32>` is not nine contiguous floats
 * — each of its three columns starts at a 16-byte boundary, because `vec3<f32>`
 * is 16-byte aligned. Writing a tight 3x3 and letting the packer place the
 * columns therefore reads three floats that were never written, and the third
 * column arrives as zeros. The result is a normal matrix that is silently wrong
 * for every mesh in the scene, which shades as though every face were lit
 * head-on. No error is raised, anywhere, at any point.
 *
 * The adjugate divided by the determinant is `transpose(inverse)` directly, so
 * no transpose step is needed. A general 4x4 inverse would be ~120 multiplies
 * against this one's ~30, for a result whose fourth row is never read.
 */
function normalMatrixOf(m: Float32Array, out: Float32Array): Float32Array {
  const a = m[0]!, b = m[1]!, c = m[2]!;
  const d = m[4]!, e = m[5]!, f = m[6]!;
  const g = m[8]!, h = m[9]!, i = m[10]!;

  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;

  if (det === 0 || !Number.isFinite(det)) {
    // A degenerate transform has no inverse. Identity is the least-wrong
    // answer: the object looks un-transformed rather than invisible, which is a
    // bug a user can see and report.
    for (let k = 0; k < 12; k++) out[k] = IDENTITY_NORMAL[k];
    return out;
  }
  const s = 1 / det;

  // Column 0 at 0..2, column 1 at 4..6, column 2 at 8..10. Indices 3, 7 and 11
  // are the vec3 padding and stay zero.
  out[0] = A * s;                 out[1] = B * s;                 out[2] = C * s;
  out[4] = (c * h - b * i) * s;   out[5] = (a * i - c * g) * s;   out[6] = (b * g - a * h) * s;
  out[8] = (b * f - c * e) * s;   out[9] = (c * d - a * f) * s;   out[10] = (a * e - b * d) * s;
  out[3] = 0; out[7] = 0; out[11] = 0;
  return out;
}

/** Identity in the same padded 12-float layout. */
const IDENTITY_NORMAL = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);


export type { Material };
