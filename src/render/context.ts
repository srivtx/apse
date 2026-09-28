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
}

/** Rounds to a whole pixel, never returning 0 or a negative extent. */
function quantize(cssExtent: number, ratio: number): number {
  if (!Number.isFinite(cssExtent) || cssExtent <= 0) return 1;
  const scaled = Math.round(cssExtent * ratio);
  // Math.round(-0.5) is -0 and Math.round of a tiny positive number is 0, so
  // the clamp has to be explicit rather than a `|| 1`.
  return scaled > 0 ? scaled : 1;
}

/** Coerces a pixel ratio to a usable positive finite number. */
function sanitizeRatio(value: number, fallback: number): number {
  if (!Number.isFinite(value) || value <= 0) return fallback;
  return value;
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
 *   3. each extent is `round(css * ratio)`, and any result that is not a
 *      positive integer becomes 1.
 *
 * Note the result is *not* clamped to `maxTextureDimension2D`. This function has
 * no device to clamp against, and a silent clamp is worse than a clear error:
 * a 5000-pixel canvas quietly turned into a 4096-pixel render target is a
 * cropped image, not a diagnosable one. `RenderTargetImpl` raises
 * `CANVAS_SIZE_INVALID` with both numbers instead.
 */
export class CanvasSizer {
  static backingSize(
    cssW: number,
    cssH: number,
    dpr: number,
    maxDpr: number,
  ): { width: number; height: number } {
    const ratio = Math.min(sanitizeRatio(dpr, 1), sanitizeRatio(maxDpr, 1));
    return { width: quantize(cssW, ratio), height: quantize(cssH, ratio) };
  }

  readonly #canvas: HTMLCanvasElement;
  readonly #maxPixelRatio: number;
  #observer: ResizeObserver | null = null;
  #cssWidth = 0;
  #cssHeight = 0;
  #width = 0;
  #height = 0;
  #dpr = -1;
  /** Set by the ResizeObserver: the element may have moved. */
  #needsMeasure = true;
  #disposed = false;

  constructor(canvas: HTMLCanvasElement, opts: CanvasSizerOptions = {}) {
    this.#canvas = canvas;
    this.#maxPixelRatio = sanitizeRatio(opts.maxPixelRatio ?? 2, 2);

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
      this.#cssWidth, this.#cssHeight, this.#dpr, this.#maxPixelRatio,
    );
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
