/**
 * Render targets: the colour, depth, and MSAA textures a pass draws into.
 *
 * **There is no default framebuffer.** WebGL handed you a colour buffer, a
 * depth buffer, and MSAA for the price of one context attribute. WebGPU hands
 * you nothing. Every attachment is a texture you allocate, configure, attach,
 * and destroy yourself, and forgetting one of them is the single most common
 * reason a ported WebGL sample "runs but looks wrong" — depth testing silently
 * off, or antialiasing silently off, or a resolve target that never resolves.
 *
 * Three rules run through this file.
 *
 * 1. **`colorView` is the MSAA view; `sampleView` is the one you sample.**
 *    A multisampled texture cannot be bound as a sampled texture at all — it
 *    is a validation error, not a blurry result. So a 4x target allocates two
 *    colour textures: a multisampled one you render into, and a single-sample
 *    one that the resolve writes into, which is the only one a later pass can
 *    read. With `sampleCount: 1` there is nothing to resolve and the two views
 *    are the same object.
 *
 * 2. **The canvas backbuffer is single-sampled, always.** `getCurrentTexture()`
 *    never returns a multisampled texture, so `antialias: true` has to be built
 *    by hand: an offscreen 4x target whose resolve target is the canvas. This is
 *    why `createCanvasTarget` refuses `sampleCount: 4` rather than quietly
 *    ignoring it.
 *
 * 3. **A canvas view expires at present.** `getCurrentTexture()` returns a
 *    texture that is only valid for the current frame; the next call after a
 *    present returns a different one. A cached canvas view is therefore a view
 *    of a dead texture, and the resulting error is opaque. See
 *    {@link RenderTargetImpl.getColorViewForFrame}.
 */

import { fail } from '../core/error.ts';
import { Resource } from '../core/resource.ts';
import { isDevelopmentMode, TEXTURE_USAGE } from './device.ts';
import type { AseDevice, AseLimits } from './device.ts';
import type { RenderTarget } from './types.ts';

/** MSAA levels apse supports. Anything else needs a pipeline change too. */
export type RenderTargetSampleCount = 1 | 4;

/**
 * How many canvas colour views one frame may acquire before apse complains.
 *
 * A frame should call `getColorViewForFrame()` once. Reaching for `colorView`
 * inside a per-object loop is a real mistake and it is invisible until the
 * driver starts rejecting the third call, so a debug build says something.
 */
const CANVAS_VIEW_TRIPWIRE = 4;

/** Depth format used when none is given. */
export const DEFAULT_DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

/**
 * Depth formats apse accepts, with the portability note that matters.
 *
 * `depth24plus` is the default because it is renderable as a depth-stencil
 * attachment everywhere and costs 3 bytes per sample. `depth32float` gives a
 * full 32 bits of range, which matters for a reversed-Z far plane or a large
 * outdoor scene, but it is *not* universally renderable: it is gated on the
 * `depth32float-stencil8` feature family on some implementations, and asking
 * for it where it is unsupported is a validation error at texture creation, on
 * a device you have already committed to. Choose it deliberately.
 */
export const DEPTH_FORMATS: readonly GPUTextureFormat[] = [
  'depth24plus',
  'depth32float',
  'depth16unorm',
];

/**
 * The part of an {@link AseDevice} a render target actually needs.
 *
 * Narrow on purpose. `AseDevice` satisfies it structurally, and a test can pass
 * a three-field fake — which is how the texture accounting in
 * `test/device.test.ts` is proven without a GPU.
 */
export interface RenderTargetDevice {
  readonly device: GPUDevice;
  readonly limits: AseLimits;
  assertLive(): void;
}

/** What the canvas target needs, on top of {@link RenderTargetDevice}. */
export interface CanvasTargetDevice extends RenderTargetDevice {
  readonly canvas: HTMLCanvasElement;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;
}

/** Options for {@link createCanvasTarget}. */
export interface CanvasTargetOptions {
  /**
   * Backing-store size. Defaults to the canvas's current `width`/`height`.
   *
   * The canvas target writes `canvas.width`/`canvas.height` itself, so pass the
   * `CanvasSizer` output here. It is the only thing in apse that writes those
   * properties, which is what keeps `target.width === canvas.width` an
   * invariant rather than a hope.
   */
  width?: number;
  height?: number;
  /** Allocate a depth attachment. Default false. */
  depth?: boolean;
  /** Defaults to `depth24plus`. */
  depthFormat?: GPUTextureFormat;
  /** Add TEXTURE_BINDING to the depth texture, for sampling it later. */
  sampleDepth?: boolean;
  /** Prefix for every label apse creates. Default 'apse.canvas'. */
  label?: string;
}

/** Options shared by the offscreen target factories. */
export interface ColorTargetOptions {
  width: number;
  height: number;
  /**
   * Colour format. Defaults to the device's canvas format.
   *
   * That default is deliberate: the canvas format is guaranteed renderable on
   * this device, and it means a post-process target that ends up blitted to the
   * canvas needs no conversion. Use `rgba16float` for an HDR intermediate and
   * accept that you now own the tone map.
   */
  format?: GPUTextureFormat;
  /** 1 or 4. Default 1. */
  sampleCount?: RenderTargetSampleCount;
  /** Allocate a depth attachment. Default false. */
  depth?: boolean;
  /** Defaults to `depth24plus`. */
  depthFormat?: GPUTextureFormat;
  /** Add TEXTURE_BINDING to the depth texture, for sampling it later. */
  sampleDepth?: boolean;
  /** Prefix for every label apse creates. Default 'apse.target'. */
  label?: string;
  /**
   * Extra texture usage flags on the colour attachment, ORed with
   * `RENDER_ATTACHMENT`.
   *
   * The default is `RENDER_ATTACHMENT` alone, because that is all a target
   * needs to be drawn into. `COPY_SRC` is what makes a target readable back to
   * the CPU — the only way to get pixels out of a WebGPU canvas, since it has no
   * `preserveDrawingBuffer`. `TEXTURE_BINDING` is what makes it sampleable by a
   * later pass. Both cost bandwidth on some hardware, which is why neither is on
   * by default: a flag nobody asked for is a flag nobody is paying for.
   */
  usage?: GPUTextureUsageFlags;
}

// ---------------------------------------------------------------------------
// The implementation
// ---------------------------------------------------------------------------

/**
 * A render target. One class for all three shapes — canvas, offscreen colour,
 * depth — because the differences are a flag and a couple of nulls, and three
 * near-identical classes would be three places for the resize-leak bug to hide.
 */
export class RenderTargetImpl extends Resource implements RenderTarget {
  readonly #device: RenderTargetDevice;
  readonly #label: string;
  readonly #format: GPUTextureFormat;
  readonly #depthFormat: GPUTextureFormat | undefined;
  readonly #sampleCount: RenderTargetSampleCount;
  readonly #isCanvas: boolean;
  readonly #canvasContext: GPUCanvasContext | null;
  readonly #canvas: HTMLCanvasElement | null;
  readonly #sampleDepth: boolean;
  readonly #wantsDepth: boolean;
  /** False for a depth-only target, which owns no colour texture at all. */
  readonly #ownsColor: boolean;
  /** Extra colour-texture usage requested by the caller. */
  readonly #usage: GPUTextureUsageFlags;

  #width = 1;
  #height = 1;

  // Owned textures. Every one of these is destroyed on resize and on dispose;
  // a texture that is not in this list is not this class's to free.
  #color: GPUTexture | null = null;
  #resolve: GPUTexture | null = null;
  #depth: GPUTexture | null = null;
  /** Lazily created 1x1 colour view for a depth-only target. See colorView. */
  #placeholderColor: GPUTexture | null = null;

  #colorView: GPUTextureView | null = null;
  #sampleView: GPUTextureView | null = null;
  #depthView: GPUTextureView | null = null;

  /** Canvas view acquisitions since the last microtask checkpoint. */
  #acquisitions = 0;
  #tripwireFired = false;

  /** Not `private`: the three factories below construct it directly. */
  constructor(init: {
    device: RenderTargetDevice;
    width: number;
    height: number;
    format: GPUTextureFormat;
    depthFormat: GPUTextureFormat | undefined;
    sampleCount: RenderTargetSampleCount;
    sampleDepth: boolean;
    ownsColor: boolean;
    usage?: GPUTextureUsageFlags;
    label: string;
    canvas: HTMLCanvasElement | null;
    canvasContext: GPUCanvasContext | null;
  }) {
    super();
    assertSize(init.width, init.height, init.device.limits, init.label);
    this.#device = init.device;
    this.#label = init.label;
    this.#format = init.format;
    this.#depthFormat = init.depthFormat;
    this.#sampleCount = init.sampleCount;
    this.#wantsDepth = init.depthFormat !== undefined;
    this.#sampleDepth = init.sampleDepth;
    this.#ownsColor = init.ownsColor;
    this.#usage = init.usage ?? 0;
    this.#canvas = init.canvas;
    this.#canvasContext = init.canvasContext;
    this.#isCanvas = init.canvas !== null;

    this.#width = init.width;
    this.#height = init.height;
    if (this.#isCanvas) this.#applyCanvasSize();
    this.#allocate();
  }

  // --- RenderTarget -------------------------------------------------------

  get width(): number {
    return this.#width;
  }

  get height(): number {
    return this.#height;
  }

  get format(): GPUTextureFormat {
    return this.#format;
  }

  get depthFormat(): GPUTextureFormat | undefined {
    return this.#depthFormat;
  }

  get sampleCount(): RenderTargetSampleCount {
    return this.#sampleCount;
  }

  get isCanvas(): boolean {
    return this.#isCanvas;
  }

  /**
   * The colour attachment for a render pass.
   *
   * For an offscreen target this is the stable, owned view. For the canvas it
   * is **not stable**: it re-acquires from `getCurrentTexture()` on every
   * access, because the canvas texture expires at present. Call
   * {@link getColorViewForFrame} once per frame and hold the result; a value
   * read into a variable at the top of `render()` is correct, a value read from
   * this property inside a loop is a new view per iteration and will mislead
   * anything that compares it for identity.
   */
  get colorView(): GPUTextureView {
    this.assertUsable('colorView');
    if (this.#isCanvas) return this.getColorViewForFrame();
    const view = this.#colorView;
    return view ?? this.#createPlaceholderColorView();
  }

  /**
   * The underlying colour texture, for `copyTextureToBuffer` and friends.
   *
   * Only valid on an offscreen target that requested `COPY_SRC`. A view is what
   * you draw with; the texture is what you copy out of, and the two are
   * different objects with different lifetimes — the view is recreated on
   * resize, the texture is destroyed and replaced.
   */
  get colorTexture(): GPUTexture {
    this.assertUsable('colorTexture');
    const t = this.#color;
    if (t === null) {
      fail('INTERNAL_INVARIANT',
        `Render target "${this.#label}" has no colour texture.`, {
        why: 'A canvas target does not own its colour texture — the browser does — and a depth-only target never had one. Neither can be copied from.',
        fix: 'Use createColorTarget(device, { usage: GPUTextureUsage.COPY_SRC }) for anything that has to be read back.',
      });
    }
    return t;
  }

  get depthView(): GPUTextureView | undefined {
    this.assertUsable('depthView');
    return this.#depthView ?? undefined;
  }

  /**
   * The view a later pass samples.
   *
   * With `sampleCount: 4` this is the resolve texture's view, **not** the
   * multisampled one. A multisampled texture cannot be bound as a sampled
   * texture; `createView()` on it produces an object that is only legal as a
   * render attachment, and using it in a bind group is a validation error
   * naming a `sampled texture` binding. This property exists so that mistake has
   * exactly one correct answer to point at.
   *
   * `undefined` for the canvas: the swapchain texture is configured with
   * `RENDER_ATTACHMENT | COPY_SRC` and no `TEXTURE_BINDING`, because granting
   * texture binding to a swapchain image forces the compositor path to keep a
   * resolvable copy of every frame. To sample the presented image, render to an
   * offscreen target and blit it — which is also the only way to get MSAA.
   */
  get sampleView(): GPUTextureView | undefined {
    this.assertUsable('sampleView');
    if (this.#isCanvas) return undefined;
    if (this.#sampleView !== null) return this.#sampleView;
    return this.colorView;
  }

  /**
   * The canvas colour view for the frame being encoded.
   *
   * **Call this exactly once per frame.** `getCurrentTexture()` returns the
   * texture that will be composited, and the browser expires it when the frame
   * is presented. A view captured last frame points at a texture that no longer
   * exists; using it produces a WebGPU error with no useful text, because there
   * is no API that can tell a stale view from a live one. Re-acquiring every
   * frame is the only correct answer, and it is cheap.
   *
   * The size of the returned texture follows `canvas.width`/`canvas.height`, so
   * a resize between frames is picked up automatically — but the render pass
   * `loadOp` and the viewport have to match, which is what
   * {@link resize} is for.
   */
  getColorViewForFrame(): GPUTextureView {
    this.assertUsable('getColorViewForFrame');
    if (this.#canvasContext === null) {
      fail('INTERNAL_INVARIANT',
        'getColorViewForFrame() was called on a target that has no canvas context.', {
        why: 'Only a canvas target has a per-frame texture. Every other kind of render target owns a texture that is valid until it is destroyed, so this method is meaningless for it.',
        fix: 'Use `colorView` for an offscreen target, or `sampleView` if a later pass is going to read it. Reach for `getColorViewForFrame()` only on the value returned by `createCanvasTarget()`.',
      });
    }
    this.#acquisitions++;
    if (this.#acquisitions === 1) {
      // A microtask runs at the end of the current task, which is exactly the
      // boundary of "one frame": the frame is encoded inside a single task.
      queueMicrotask(() => { this.#acquisitions = 0; });
    }
    if (isDevelopmentMode() && !this.#tripwireFired && this.#acquisitions > CANVAS_VIEW_TRIPWIRE) {
      this.#tripwireFired = true;
      console.warn(
        `apse: "${this.#label}" acquired ${this.#acquisitions} canvas colour views in one frame. ` +
        'One getColorViewForFrame() per frame is the contract; a view read inside a loop ' +
        'allocates a new GPUTextureView per iteration and will be wrong the moment the ' +
        'frame is presented.',
      );
    }
    return this.#canvasContext.getCurrentTexture()
      .createView({ label: `${this.#label}:canvasCurrent` });
  }

  /**
   * Recreates at a new size. No-op when the size is unchanged.
   *
   * Destroys the old textures *before* creating the new ones, in that order, on
   * purpose. The other order peaks at two full targets of memory, which on a
   * 4x MSAA 1080p target is the difference between allocating and failing to
   * allocate on a mid-range phone. A resize leak — old textures kept alive by a
   * field nobody nulled — is the most common bug in this file's history, and
   * the order here is what makes it impossible to write one.
   */
  resize(width: number, height: number): void {
    this.assertUsable('resize');
    this.#assertDeviceLive('resize');
    assertSize(width, height, this.#device.limits, this.#label);
    if (width === this.#width && height === this.#height) return;

    this.#freeTextures();
    this.#width = width;
    this.#height = height;
    if (this.#isCanvas) this.#applyCanvasSize();
    this.#allocate();
  }

  /**
   * Releases every texture this target owns. Idempotent.
   *
   * Safe to call on a lost device, and deliberately not gated on
   * `assertLive()`: after a driver reset there is nothing to talk to, but the
   * wrappers still exist and still hold their references, and the next
   * `recover()` would otherwise inherit them. Freeing is a no-op on a dead
   * device and a real free on a live one, so it is always worth calling.
   */
  protected override onDispose(): void {
    this.#freeTextures();
  }

  // --- internals ----------------------------------------------------------

  private assertUsable(member: string): void {
    if (!this.disposed) return;
    // There is no dedicated catalog code for "used a disposed target" — the
    // closest is RENDERER_ALREADY_DISPOSED, which is about the renderer and
    // owned by another module. INTERNAL_INVARIANT with an explicit message is
    // the honest choice, and adding a code here would change a shared contract.
    fail('INTERNAL_INVARIANT',
      `RenderTargetImpl.${member} was used after dispose().`, {
      why: 'dispose() called texture.destroy() on every attachment. The GPUTextureView objects this target handed out are invalid from that moment, and in WebGPU an invalid object does not throw when used — it silently does nothing, which is how a "fixed" render target keeps rendering to last frame\'s pixels.',
      fix: 'Check `.disposed` before reusing a target, or hold a reference with `.ref()` for as long as a second owner still uses it. The renderer should recreate the target on resize rather than reusing one it has already released.',
      detail: { kind: 'lifecycle', resource: 'RenderTargetImpl', state: 'destroyed' },
    });
  }

  #assertDeviceLive(operation: string): void {
    try {
      this.#device.assertLive();
    } catch (thrown) {
      fail('DEVICE_LOST',
        `Cannot ${operation} a render target: the device it was created on is gone.`, {
        why: 'Textures are owned by the device that created them. After a device loss every texture, view and bind group derived from it is invalid, and creating a replacement texture on the dead device produces another invalid one — silently.',
        fix: 'Call `device.recover()`, then rebuild the render target on the new device. Target objects are cheap; keep no target across a recovery.',
        cause: thrown,
      });
    }
  }

  /**
   * For a depth-only target, `RenderTarget.colorView` is non-optional, so
   * something has to be there.
   *
   * Rather than return the depth texture's view — which would be a colour
   * attachment that is silently the wrong type — this allocates a 1x1
   * single-sample colour texture on first access, and never before. A depth
   * target nobody reads `colorView` on costs nothing.
   */
  #createPlaceholderColorView(): GPUTextureView {
    if (this.#ownsColor) {
      fail('INTERNAL_INVARIANT',
        'A colour target had no colour view.', {
        why: 'createColorTarget always allocates its colour texture, so #colorView is set from the constructor onwards. Reaching the placeholder path with one means allocation was skipped or cleared without a resize, and the view handed back would be a 1x1 texture the size of a postage stamp.',
        fix: 'This is a bug in apse. Please report it with the target options you passed.',
      });
    }
    this.#device.assertLive();
    this.#placeholderColor = this.#device.device.createTexture({
      label: `${this.#label}:colorPlaceholder`,
      size: { width: 1, height: 1 },
      format: this.#format,
      sampleCount: 1,
      usage: TEXTURE_USAGE.RENDER_ATTACHMENT,
    });
    const view = this.#placeholderColor.createView({ label: `${this.#label}:colorPlaceholder` });
    this.#colorView = view;
    this.#sampleView = view;
    return view;
  }

  /** The canvas target owns the element's backing store; nobody else writes it. */
  #applyCanvasSize(): void {
    const canvas = this.#canvas;
    if (canvas === null) return;
    if (canvas.width !== this.#width) canvas.width = this.#width;
    if (canvas.height !== this.#height) canvas.height = this.#height;
  }

  #allocate(): void {
    const gpu = this.#device.device;
    const { width, height } = this;

    if (this.#ownsColor && !this.#isCanvas) {
      // The colour attachment. Multisampled targets get RENDER_ATTACHMENT
      // only: a multisampled texture cannot be sampled, so TEXTURE_BINDING on
      // it is not merely useless, it is a validation error on some drivers.
      this.#color = gpu.createTexture({
        label: `${this.#label}:color`,
        size: { width, height },
        format: this.#format,
        sampleCount: this.#sampleCount,
        usage: TEXTURE_USAGE.RENDER_ATTACHMENT | this.#usage,
      });
      this.#colorView = this.#color.createView({ label: `${this.#label}:color` });

      if (this.#sampleCount > 1) {
        // The resolve destination. This is the texture a later pass samples,
        // and the only reason the 4x path costs two colour textures instead of
        // one. It is also the thing that makes MSAA + post-processing possible
        // at all: there is no way to sample the multisampled attachment.
        //
        // No COPY_SRC: a resolve target needs RENDER_ATTACHMENT to be written by
        // the resolve, and readback would need COPY_SRC, but adding COPY_SRC to
        // a colour attachment can force a slower path on tiled GPUs. A
        // screenshot of an offscreen target should copy from the canvas, which
        // already has it.
        this.#resolve = gpu.createTexture({
          label: `${this.#label}:colorResolve`,
          size: { width, height },
          format: this.#format,
          sampleCount: 1,
          usage: TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING,
        });
        this.#sampleView = this.#resolve.createView({ label: `${this.#label}:colorResolve` });
      } else {
        this.#sampleView = this.#colorView;
      }
    }

    if (this.#wantsDepth && this.#depthFormat !== undefined) {
      // Depth follows the colour sample count: a render pass requires every
      // attachment to agree, and there is no way to multisample depth
      // independently. depth24plus rather than depth32float: it is renderable
      // as a depth-stencil attachment on every device, and 32-bit float depth
      // is not.
      this.#depth = gpu.createTexture({
        label: `${this.#label}:depth`,
        size: { width, height },
        format: this.#depthFormat,
        sampleCount: this.#isCanvas ? 1 : this.#sampleCount,
        usage: TEXTURE_USAGE.RENDER_ATTACHMENT
          | (this.#sampleDepth ? TEXTURE_USAGE.TEXTURE_BINDING : 0),
      });
      this.#depthView = this.#depth.createView({ label: `${this.#label}:depth` });
    }
  }

  /** Destroys every owned texture and clears every derived view. */
  #freeTextures(): void {
    for (const texture of [this.#color, this.#resolve, this.#depth, this.#placeholderColor]) {
      texture?.destroy();
    }
    this.#color = null;
    this.#resolve = null;
    this.#depth = null;
    this.#placeholderColor = null;
    this.#colorView = null;
    this.#sampleView = null;
    this.#depthView = null;
  }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * Rejects an unusable size, with the actual number and the limit in the text.
 *
 * Note that a size over `maxTextureDimension2D` is an *error*, not a clamp. A
 * silent clamp is worse than a failure: a 5000-pixel canvas quietly rendered
 * into a 4096-pixel target is a cropped image with no symptom anywhere, and
 * `resize()` is usually called from a resize handler where a thrown error is
 * far easier to trace than a missing edge.
 */
function assertSize(width: number, height: number, limits: AseLimits, label: string): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    fail('RENDER_TARGET_SIZE_INVALID',
      `Render target "${label}" was given the size ${width}x${height}.`, {
      why: 'A GPUTexture must have positive integer extents in both dimensions. A zero-sized target is the normal result of measuring a canvas that is display:none, or of measuring before the first layout pass.',
      fix: 'Give the target a positive integer size, and make sure a layout pass has run. If the size came from a canvas, `CanvasSizer.update()` already clamps a zero to 1 so that a hidden canvas does not throw.',
    });
  }
  const max = limits.maxTextureDimension2D;
  if (width > max || height > max) {
    fail('CANVAS_SIZE_INVALID',
      `Render target "${label}" is ${width}x${height}, over this device's maxTextureDimension2D of ${max}.`, {
      why: 'The same limit that caps the canvas backbuffer caps every 2D texture: a WebGPU surface cannot exceed maxTextureDimension2D. On a compatibility-mode device that limit is 4096, not the 8192 of a core device, so a 5120x2880 canvas that works on a laptop fails on a phone.',
      fix: `Clamp the backing store to ${max} in each dimension. On a canvas, that means lowering maxPixelRatio — and if even 1x does not fit, render at a lower internal resolution and let the compositor scale it up. On an offscreen target, lower the size you pass in.`,
      detail: { kind: 'numeric', field: 'size', value: Math.max(width, height), min: 1, max },
    });
  }
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

/**
 * Wraps the canvas backbuffer as a render target.
 *
 * The special part is what this does *not* own: the colour texture belongs to
 * the browser, so no `createTexture` call is made for it, no `destroy()` is ever
 * called on it, and its view is re-acquired every frame. Everything else — the
 * depth attachment, the size, the labels — is a normal target.
 *
 * **There is no `sampleCount` option here, and that is deliberate.** A canvas
 * texture is never multisampled: `getCurrentTexture()` always returns a
 * single-sample attachment, so an `antialias: true` equivalent has to be built
 * by hand as an offscreen 4x target whose resolve destination is the canvas.
 * Offering the option and rejecting it would be worse than not offering it —
 * `createColorTarget(device, { sampleCount: 4 })` followed by a blit to
 * `createCanvasTarget(device)` is the whole of how you do MSAA in WebGPU.
 */
export function createCanvasTarget(
  device: AseDevice | CanvasTargetDevice,
  opts: CanvasTargetOptions = {},
): RenderTargetImpl {
  const canvas = device.canvas;
  const label = opts.label ?? 'apse.canvas';
  const width = opts.width ?? canvas.width;
  const height = opts.height ?? canvas.height;
  assertSize(width, height, device.limits, label);

  const depthFormat = opts.depth === true
    ? (opts.depthFormat ?? DEFAULT_DEPTH_FORMAT)
    : undefined;
  if (depthFormat !== undefined && !DEPTH_FORMATS.includes(depthFormat)) {
    fail('OPTION_UNKNOWN',
      `Option depthFormat was given the value "${depthFormat}", which is not one of: ${DEPTH_FORMATS.join(', ')}.`, {
      why: 'apse only offers depth formats it has reasoned about. A stencil-capable format is available through depth24plus, and depth32float is offered as an explicit portability trade rather than a default.',
      fix: `Use one of: ${DEPTH_FORMATS.join(', ')}. depth24plus is the default and is renderable on every device; depth32float costs portability.`,
    });
  }

  return new RenderTargetImpl({
    device,
    width,
    height,
    format: device.format,
    depthFormat,
    sampleCount: 1,
    sampleDepth: opts.sampleDepth === true,
    ownsColor: true,
    label,
    canvas,
    canvasContext: device.context,
  });
}

/**
 * An offscreen colour target.
 *
 * With `sampleCount: 4` this allocates two colour textures — the multisampled
 * attachment and the single-sample resolve — plus depth if asked. Use
 * {@link RenderTargetImpl.colorView} as the render pass attachment and
 * {@link RenderTargetImpl.sampleView} as the bind group entry.
 */
export function createColorTarget(
  device: AseDevice | RenderTargetDevice,
  opts: ColorTargetOptions,
): RenderTargetImpl {
  const label = opts.label ?? 'apse.target';
  const sampleCount = opts.sampleCount ?? 1;
  assertSampleCount(sampleCount, label);
  const depthFormat = resolveDepthFormat(opts.depth, opts.depthFormat, label);

  return new RenderTargetImpl({
    device,
    width: opts.width,
    height: opts.height,
    format: opts.format ?? formatOf(device),
    depthFormat,
    sampleCount,
    sampleDepth: opts.sampleDepth === true,
    ownsColor: true,
    usage: opts.usage,
    label,
    canvas: null,
    canvasContext: null,
  });
}

/**
 * A depth-only target, for a pass that has nothing to write colour to.
 *
 * `RenderTarget.colorView` is a non-optional `GPUTextureView`, so something has
 * to be behind it. It is a 1x1 texture created on first access, never before:
 * reading `colorView` on a depth target is almost certainly a mistake, and
 * doing nothing about it until then means the mistake is free if you never make
 * it. `depthView` is the real attachment.
 */
export function createDepthTarget(
  device: AseDevice | RenderTargetDevice,
  opts: ColorTargetOptions,
): RenderTargetImpl {
  const label = opts.label ?? 'apse.depth';
  const sampleCount = opts.sampleCount ?? 1;
  assertSampleCount(sampleCount, label);
  const depthFormat = opts.depthFormat ?? DEFAULT_DEPTH_FORMAT;
  if (!DEPTH_FORMATS.includes(depthFormat)) {
    fail('OPTION_UNKNOWN',
      `Option depthFormat was given the value "${depthFormat}", which is not one of: ${DEPTH_FORMATS.join(', ')}.`, {
      why: 'apse only offers depth formats it has reasoned about, and a depth-only target has to have one.',
      fix: `Use one of: ${DEPTH_FORMATS.join(', ')}, or omit it for the depth24plus default.`,
    });
  }

  return new RenderTargetImpl({
    device,
    width: opts.width,
    height: opts.height,
    format: opts.format ?? formatOf(device),
    depthFormat,
    sampleCount,
    sampleDepth: opts.sampleDepth === true,
    ownsColor: false,
    label,
    canvas: null,
    canvasContext: null,
  });
}

function resolveDepthFormat(
  depth: boolean | undefined,
  depthFormat: GPUTextureFormat | undefined,
  label: string,
): GPUTextureFormat | undefined {
  if (depth !== true) return undefined;
  const format = depthFormat ?? DEFAULT_DEPTH_FORMAT;
  if (!DEPTH_FORMATS.includes(format)) {
    fail('OPTION_UNKNOWN',
      `Render target "${label}" was given depthFormat "${format}", which is not one of: ${DEPTH_FORMATS.join(', ')}.`, {
      why: 'apse only offers depth formats it has reasoned about. depth24plus is renderable as a depth-stencil attachment everywhere; depth32float gives full 32-bit range but is not universally renderable.',
      fix: `Use one of: ${DEPTH_FORMATS.join(', ')}. Pass depth: true to get the depth24plus default.`,
    });
  }
  return format;
}

function assertSampleCount(sampleCount: number, label: string): void {
  if (sampleCount === 1 || sampleCount === 4) return;
  fail('OPTION_UNKNOWN',
    `Render target "${label}" was given sampleCount ${sampleCount}, which is not one of: 1, 4.`, {
    why: 'The sample count is baked into the render pipeline as well as the attachment, and apse builds exactly two shapes: single-sampled, and 4x with a resolve target. 2x exists in the hardware and not in WebGPU, and 8x is not portable.',
    fix: 'Use 1, or 4. If you need more than 4x, supersample: render at 2x the resolution and downsample in a post pass, which is what 8x MSAA amounts to anyway.',
  });
}

/**
 * Colour format for an offscreen target when none is given.
 *
 * `rgba8unorm` is a safe floor: it is renderable and sampleable on every
 * conformant device, in every feature level, with no optional feature. An
 * `AseDevice` carries its canvas format and prefers that instead, which is
 * usually the better choice — see {@link ColorTargetOptions.format}.
 */
const DEFAULT_TARGET_FORMAT: GPUTextureFormat = 'rgba8unorm';

function formatOf(device: AseDevice | RenderTargetDevice): GPUTextureFormat {
  return (device as { format?: GPUTextureFormat }).format ?? DEFAULT_TARGET_FORMAT;
}
