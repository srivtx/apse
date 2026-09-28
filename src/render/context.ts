/**
 * Canvas sizing: the arithmetic between CSS pixels and GPU pixels.
 *
 * This module exists because the two are not the same number, and getting it
 * wrong is silent. A canvas has three sizes:
 *
 *   CSS size      what the element occupies in layout. Author-controlled.
 *   backing size  `canvas.width` x `canvas.height`. What the GPU allocates.
 *   display size  backing / devicePixelRatio. What the compositor scales to.
 *
 * Set backing = CSS and the render is soft on every HiDPI screen. Set backing
 * to CSS x DPR and forget to cap it and a 3x phone allocates nine times the
 * memory and renders at 9x the fragment cost for a difference nobody can see.
 *
 * Two rules make this safe, and both are enforced in {@link backingSize}:
 *
 *   - **A backing store is never zero.** A `display: none` canvas, a detached
 *     element in a hidden iframe, and a flex item mid-layout all report a zero
 *     client rect, and `devicePixelRatio` itself can be 0 in a hidden iframe.
 *     WebGPU rejects a zero-sized texture, and it does so with a validation
 *     error about texture dimensions rather than about the canvas.
 *   - **DPR is capped, not trusted.** apse caps at 2 by default. The jump from
 *     1x to 2x is the one people can see; 2x to 3x costs 2.25x the fragments for
 *     a difference most viewers cannot, and on a 6.7" phone at 3x it is the
 *     difference between hitting 60fps and not.
 *
 * Everything here is pure arithmetic plus two DOM reads. `CanvasSizer.backingSize`
 * is a static function specifically so the interesting part can be tested
 * exhaustively without a DOM, a GPU, or a browser.
 */

import { fail } from '../core/error.ts';

export interface CanvasSizerOptions {
  /** Pixel ratio cap. Default 2. Values <= 0 are treated as 1. */
  maxPixelRatio?: number;
  /**
   * The device's `maxTextureDimension2D`, or `undefined` for no ceiling.
   *
   * Passed by the renderer, which knows the device. With it, a backing store
   * that would exceed the limit is **clamped** rather than left to fail at
   * texture creation. That is the opposite of what `RenderTargetImpl` does with
   * an explicit size, and the difference is the point:
   *
   *   - An offscreen target has a *requested* size. Quietly handing back 4096
   *     instead of 5000 is a cropped image with no symptom anywhere, so
   *     `assertSize` raises `CANVAS_SIZE_INVALID` with both numbers.
   *   - A canvas has a *CSS* size, and its backing store is a choice the sizer
   *     makes on the user's behalf. The compositor scales the backing store to
   *     the CSS box, so clamping costs resolution nobody asked for and cannot
   *     crop: a 5120-wide window on a 4096 compatibility-mode device gets 4096
   *     pixels of supersampling and the image is still entirely on screen.
   *     Failing there would take down the frame loop on a large monitor.
   *
   * The floor is applied after the cap, so a limit below 1 — which no device
   * reports — still yields a 1x1 target rather than a zero-sized texture.
   */
  maxTextureDimension2D?: number;
}

/**
 * Rounds to a whole pixel, then caps and floors, in that order.
 *
 * The order matters and each step exists because of a value the previous one can
 * produce: `Math.round(-0.5)` is `-0` and `Math.round(0.2)` is `0`, so the floor
 * has to be explicit rather than a `|| 1`; and a cap of 1 on a 1-pixel extent is
 * still 1, but a cap of 0 — which no device reports and a test will pass — is
 * why the floor is applied *after* the cap as well as before.
 */
function quantize(cssExtent: number, ratio: number, ceiling: number): number {
  if (!Number.isFinite(cssExtent) || cssExtent <= 0) return 1;
  const scaled = Math.round(cssExtent * ratio);
  const capped = scaled > ceiling ? ceiling : scaled;
  return capped > 0 ? capped : 1;
}

/** Coerces a pixel ratio to a usable positive finite number. */
function sanitizeRatio(value: number, fallback: number): number {
  if (!Number.isFinite(value) || value <= 0) return fallback;
  return value;
}

/** A device limit, or no limit at all. A limit below 1 is treated as absent. */
function sanitizeCeiling(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value) || value < 1) return Number.POSITIVE_INFINITY;
  return Math.floor(value);
}

/**
 * The backing-store size for a given CSS size, at a capped pixel ratio.
 *
 * Pure, total, and the only place this decision is made. Everything else in
 * this module exists to feed it the right four numbers.
 *
 * The rules, in order:
 *   1. a non-finite or non-positive `dpr` or `maxDpr` becomes 1 — `devicePixelRatio`
 *      is 0 in a hidden iframe, and `NaN` in a detached document in some engines;
 *   2. the effective ratio is `min(dpr, maxDpr)`;
 *   3. each extent is `round(css * ratio)`;
 *   4. when `maxDim` is a positive number, each extent is then capped at it;
 *   5. any extent that is not a positive integer becomes 1.
 *
 * Step 5 comes last on purpose: both the multiply and the cap can produce 0, and
 * a zero-sized texture is a validation error about texture dimensions that names
 * nothing about the canvas. With no `maxDim` — the four-argument form, which is
 * what a caller with no device has — the result is exactly the arithmetic above,
 * unclamped, so an explicit oversize size still reaches `RenderTargetImpl` and
 * fails there with `CANVAS_SIZE_INVALID` and both numbers.
 */
export class CanvasSizer {
  static backingSize(
    cssW: number,
    cssH: number,
    dpr: number,
    maxDpr: number,
    maxDim: number = Number.POSITIVE_INFINITY,
  ): { width: number; height: number } {
    const ratio = Math.min(sanitizeRatio(dpr, 1), sanitizeRatio(maxDpr, 1));
    const ceiling = Number.isFinite(maxDim) && maxDim > 0 ? Math.floor(maxDim) : Number.POSITIVE_INFINITY;
    return { width: quantize(cssW, ratio, ceiling), height: quantize(cssH, ratio, ceiling) };
  }

  readonly #canvas: HTMLCanvasElement;
  readonly #maxPixelRatio: number;
  readonly #maxDim: number;
  #observer: ResizeObserver | null = null;
  #cssWidth = 0;
  #cssHeight = 0;
  #width = 0;
  #height = 0;
  #dpr = -1;
  /** Set by the ResizeObserver: the element may have moved. */
  #needsMeasure = true;
  #disposed = false;
  /** True when the last measure was capped by the texture-dimension ceiling. */
  #clamped = false;

  constructor(canvas: HTMLCanvasElement, opts: CanvasSizerOptions = {}) {
    this.#canvas = canvas;
    this.#maxPixelRatio = sanitizeRatio(opts.maxPixelRatio ?? 2, 2);
    this.#maxDim = sanitizeCeiling(opts.maxTextureDimension2D);

    // A ResizeObserver, not a `window.resize` listener. `window.resize` only
    // fires for the *viewport*, so it misses every element-level resize that
    // actually matters for a canvas: a CSS grid column reflowing, a sidebar
    // being dragged, `display: none` being toggled off, a flex sibling growing.
    // Those are the common case in an app shell, and with a window listener the
    // canvas keeps rendering at the old resolution until the next full-window
    // resize — which on a page that never resizes the window is never.
    if (typeof ResizeObserver !== 'undefined') {
      this.#observer = new ResizeObserver(() => {
        // Set a flag; measure in update(). Reading layout inside the observer
        // callback forces a synchronous reflow, and the callback fires before
        // layout has necessarily settled.
        this.#needsMeasure = true;
      });
      this.#observer.observe(canvas);
    }
    this.update();
  }

  /** CSS size in CSS pixels, from the element's client rect. */
  get cssWidth(): number {
    return this.#cssWidth;
  }

  get cssHeight(): number {
    return this.#cssHeight;
  }

  /** Backing store size, DPR-capped. */
  get width(): number {
    return this.#width;
  }

  get height(): number {
    return this.#height;
  }

  /** The cap in force. */
  get maxPixelRatio(): number {
    return this.#maxPixelRatio;
  }

  /** The texture-dimension ceiling, or `Infinity` when none was given. */
  get maxTextureDimension2D(): number {
    return this.#maxDim;
  }

  /**
   * True when the last measure hit the ceiling.
   *
   * Worth exposing rather than swallowing: a clamped canvas is *correct* — the
   * image is whole and merely less sharp — but a user reporting "my canvas is
   * blurry" is asking a question only this answers, and the answer should not
   * require reading the source.
   */
  get clamped(): boolean {
    return this.#clamped;
  }

  /**
   * Call once per frame. Returns true when the backing size changed.
   *
   * The return value is the whole point. A renderer that recreates its colour
   * and depth textures on every resize notification reallocates several
   * megabytes every time a scrollbar appears, and a ResizeObserver fires
   * mid-interaction often enough that this becomes every frame. Returning false
   * on a no-op means the renderer does the *cheap* thing in the common case.
   *
   * Note the distinction this draws: "something may have moved" is tracked
   * separately from "the size changed", and only the second one is reported. A
   * ResizeObserver that fires for a transform, a scrollbar, or a re-layout that
   * happens to produce the same integer pixel count costs one layout read and
   * nothing else.
   *
   * The comparison is on backing sizes, not CSS sizes: a 1px CSS change that
   * rounds away to the same backing size is correctly reported as no change.
   */
  update(): boolean {
    if (this.#disposed) {
      fail('INTERNAL_INVARIANT',
        'CanvasSizer.update() was called after dispose().', {
        why: 'dispose() disconnects the ResizeObserver, so the sizer can no longer see a resize. Sizes freeze at their last values and the canvas silently stops tracking its element.',
        fix: 'Do not call update() after dispose(), or build a new CanvasSizer. The renderer should dispose its sizer exactly once, at teardown, alongside the device.',
      });
    }
    // `devicePixelRatio` is a global property read, not a layout read, so it is
    // cheap enough for every frame — and it is the *only* way to notice a zoom
    // or a monitor change. A ResizeObserver does not report those: the element's
    // CSS box is unchanged when the window zooms, so the observer stays quiet
    // while the correct backing store triples.
    const dpr = this.devicePixelRatio;
    const dprChanged = dpr !== this.#dpr;
    if (dprChanged) this.#dpr = dpr;

    // The layout box is only read when something says it may have moved, or on
    // every frame when there is no observer to say so. Reading `clientWidth` is
    // a forced synchronous layout, and doing it per frame is the single largest
    // avoidable cost in a render loop.
    if (!dprChanged && !this.#needsMeasure && this.#observer !== null) return false;
    this.#needsMeasure = false;
    return this.#measure();
  }

  /**
   * Reads the element's layout box and reports whether the backing size moved.
   *
   * `clientWidth`/`clientHeight` rather than `getBoundingClientRect()` on
   * purpose. The bounding rect is affected by CSS transforms and can be
   * fractional, so a canvas scaled by `transform: scale(1.5)` would report a
   * backing store 1.5x too large; `clientWidth` is the untransformed integer
   * layout box, which is the number of CSS pixels the element actually occupies.
   */
  #measure(): boolean {
    const canvas = this.#canvas;
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    this.#cssWidth = Number.isFinite(cssWidth) ? cssWidth : 0;
    this.#cssHeight = Number.isFinite(cssHeight) ? cssHeight : 0;

    const size = CanvasSizer.backingSize(
      this.#cssWidth, this.#cssHeight, this.#dpr, this.#maxPixelRatio, this.#maxDim,
    );
    // Recorded even on a no-change frame: the ceiling can be hit by a size that
    // has not moved, and "clamped" is a property of the arithmetic, not of a
    // transition.
    this.#clamped = Number.isFinite(this.#maxDim) &&
      (this.#cssWidth * Math.min(this.#dpr, this.#maxPixelRatio) > this.#maxDim ||
        this.#cssHeight * Math.min(this.#dpr, this.#maxPixelRatio) > this.#maxDim);
    if (size.width === this.#width && size.height === this.#height) return false;
    this.#width = size.width;
    this.#height = size.height;
    return true;
  }

  /**
   * The current DPR, sanitised. Exposed so a caller that also needs to compute
   * a non-canvas target's size can use the same number the canvas used.
   */
  get devicePixelRatio(): number {
    const dpr = (globalThis as { devicePixelRatio?: number }).devicePixelRatio ?? 1;
    return sanitizeRatio(dpr, 1);
  }

  /**
   * Stops observing. Safe to call more than once.
   *
   * Does not reset `canvas.width`/`canvas.height`: shrinking the backing store
   * here would blank the canvas, and the caller is tearing down anyway.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#observer?.disconnect();
    this.#observer = null;
  }
}
