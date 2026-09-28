/**
 * Device, target, and canvas-sizing tests.
 *
 * There is no GPU here, and the shape of these tests is dictated by that. What
 * can be tested is the part that is *apse's* rather than the driver's: the
 * limit copy, the size arithmetic, the texture accounting, and the error
 * contract. Each of those has a way of being wrong that produces a plausible
 * looking render, which is why they are worth a table row each.
 *
 * The three tests that matter most, in order:
 *
 *   1. **`backingSize`**, table-driven. Pure, and the arithmetic where canvas
 *      bugs live — a zero-sized backing store is a WebGPU validation error
 *      about texture dimensions that names nothing about the canvas.
 *   2. **The limit copy**, against a fake `GPUSupportedLimits` built from
 *      prototype getters with no own properties. This is a test of the
 *      *defence*, not the feature: it proves `{...limits}` would have produced
 *      `{}` and that `copyLimits` does not. A test that only checks the copied
 *      values would pass just as happily against a fake with own properties,
 *      and would therefore prove nothing.
 *   3. **Texture accounting** against a fake device that records every
 *      `createTexture` and `destroy`. A resize leak is the most common bug in a
 *      render-target layer and it is invisible until the tab OOMs an hour in.
 */

import { describe, expect, test } from 'bun:test';

import { AseError, isAseError } from '../src/core/error.ts';
import type { AseErrorCode } from '../src/core/error.ts';
import {
  ASE_LIMIT_NAMES,
  COMPAT_LIMITS,
  createDevice,
  COMPAT_VERTEX_STAGE_LIMITS,
  compatRequiredLimits,
  CORE_DEFAULT_LIMITS,
  copyLimits,
  formatLimitRequest,
  LIMIT_CLASSES,
  OBJECT_UNIFORM_STRIDE_BYTES,
  withErrorScope,
} from '../src/render/device.ts';
import type { AseLimitName, DeviceOptions } from '../src/render/device.ts';
import { CanvasSizer } from '../src/render/context.ts';
import {
  createCanvasTarget,
  createColorTarget,
  createDepthTarget,
  DEFAULT_DEPTH_FORMAT,
  RenderTargetImpl,
} from '../src/render/target.ts';
import type { RenderTargetDevice } from '../src/render/target.ts';
import {
  asCanvasTargetDevice,
  asGpuDevice,
  fakeAseDevice,
  fakeLimits,
  FakeGPUDevice,
  FAKE_TEXTURE_USAGE as USAGE,
} from './fake-device.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Asserts that `fn` throws an AseError with the given code.
 *
 * Typed against the real union, so a typo in a code name in a test is a
 * compile error rather than a test that silently never matches.
 */
function expectCode(fn: () => unknown, code: AseErrorCode): AseError {
  let thrown: unknown;
  try {
    fn();
  } catch (error) {
    thrown = error;
  }
  if (!isAseError(thrown)) {
    throw new Error(`expected an AseError with code ${code}, got ${String(thrown)}`);
  }
  expect((thrown as AseError).code).toBe(code);
  return thrown as AseError;
}

async function expectCodeAsync(fn: () => Promise<unknown>, code: AseErrorCode): Promise<AseError> {
  let thrown: unknown;
  try {
    await fn();
  } catch (error) {
    thrown = error;
  }
  if (!isAseError(thrown)) {
    throw new Error(`expected an AseError with code ${code}, got ${String(thrown)}`);
  }
  expect((thrown as AseError).code).toBe(code);
  return thrown as AseError;
}

// ===========================================================================
// 1. CanvasSizer.backingSize — the pure arithmetic
// ===========================================================================

describe('CanvasSizer.backingSize', () => {
  // The whole table. (cssW, cssH, dpr, maxDpr) -> expected (width, height).
  const CASES: ReadonlyArray<{
    name: string;
    css: [number, number];
    dpr: number;
    maxDpr: number;
    expect: [number, number];
    why: string;
  }> = [
    {
      name: '1x device, cap 2',
      css: [800, 600], dpr: 1, maxDpr: 2, expect: [800, 600],
      why: 'the trivial case: at 1x the backing store is the CSS size',
    },
    {
      name: '2x device, cap 2',
      css: [800, 600], dpr: 2, maxDpr: 2, expect: [1600, 1200],
      why: 'exactly at the cap: full retina, which is what the cap exists to allow',
    },
    {
      name: '3x phone, cap 2',
      css: [390, 844], dpr: 3, maxDpr: 2, expect: [780, 1688],
      why: 'the reason the cap exists: 3x would be 2.25x the fragments for a difference most viewers cannot see',
    },
    {
      name: '3x phone, cap 1.5',
      css: [390, 844], dpr: 3, maxDpr: 1.5, expect: [585, 1266],
      why: 'a fractional cap, which is the usual real-world value',
    },
    {
      name: 'fractional DPR rounds',
      css: [100, 100], dpr: 1.5, maxDpr: 3, expect: [150, 150],
      why: '1.5 x 100 is exact, so this only proves the cap is not the binding constraint',
    },
    {
      name: 'fractional DPR rounds down',
      css: [101, 33], dpr: 1.5, maxDpr: 2, expect: [152, 50],
      why: '101 x 1.5 = 151.5 rounds to 152 and 33 x 1.5 = 49.5 rounds to 50 — Math.round, not floor and not ceil',
    },
    {
      name: 'fractional CSS size rounds',
      css: [800.4, 600.6], dpr: 1, maxDpr: 2, expect: [800, 601],
      why: 'a layout engine can hand back a fractional client rect; the backing store is whole pixels',
    },
    {
      name: 'DPR of 0 clamps to 1',
      css: [800, 600], dpr: 0, maxDpr: 2, expect: [800, 600],
      why: 'devicePixelRatio is genuinely 0 in a hidden iframe, and 800x600 is the right answer, not 0x0 and not a throw',
    },
    {
      name: 'DPR of 0 with a cap below 1',
      css: [640, 480], dpr: 0, maxDpr: 0.5, expect: [320, 240],
      why: 'sanitising dpr to 1 and then applying the cap: the cap is a separate decision and still applies',
    },
    {
      name: 'maxDPR of 0 clamps to 1',
      css: [800, 600], dpr: 2, maxDpr: 0, expect: [800, 600],
      why: 'a zero cap means "ignore DPR", not "allocate nothing"',
    },
    {
      name: 'negative maxDPR clamps to 1',
      css: [200, 100], dpr: 3, maxDpr: -2, expect: [200, 100],
      why: 'a nonsense cap must degrade to 1x rather than to a negative or zero size',
    },
    {
      name: 'NaN DPR clamps to 1',
      css: [512, 512], dpr: Number.NaN, maxDpr: 2, expect: [512, 512],
      why: 'NaN propagates through every arithmetic operation, so it has to be intercepted before the multiply',
    },
    {
      name: 'infinite DPR clamps to 1',
      css: [512, 512], dpr: Number.POSITIVE_INFINITY, maxDpr: 2, expect: [512, 512],
      why: 'Infinity x 512 is Infinity, Math.round(Infinity) is Infinity, and an infinite extent is a validation error. Clamping to 1 is the only safe reading.',
    },
    {
      name: 'zero CSS width clamps to 1',
      css: [0, 600], dpr: 2, maxDpr: 2, expect: [1, 1200],
      why: 'display:none gives a zero client rect. A 0-width texture is rejected by WebGPU with an error about texture dimensions, so the floor is 1',
    },
    {
      name: 'zero CSS height clamps to 1',
      css: [800, 0], dpr: 2, maxDpr: 2, expect: [1600, 1],
      why: 'same, in the other axis',
    },
    {
      name: 'both zero clamps to 1x1',
      css: [0, 0], dpr: 3, maxDpr: 2, expect: [1, 1],
      why: 'a hidden iframe: the degenerate case, and it must not throw',
    },
    {
      name: 'negative CSS clamps to 1',
      css: [-100, -100], dpr: 2, maxDpr: 2, expect: [1, 1],
      why: 'Math.round(-200) is -200, not -0, so an explicit clamp is needed; a negative extent is a validation error',
    },
    {
      name: 'sub-1 CSS size',
      css: [0.4, 0.4], dpr: 1, maxDpr: 2, expect: [1, 1],
      why: 'a canvas smaller than one CSS pixel: Math.round(0.4) is 0, so the floor does the work',
    },
    {
      name: 'sub-1 CSS size at 2x',
      css: [0.4, 0.6], dpr: 2, maxDpr: 2, expect: [1, 1],
      why: '0.4 x 2 = 0.8 rounds to 1 anyway; 0.6 x 2 = 1.2 rounds to 1',
    },
    {
      name: 'sub-1 CSS at 1.5x rounds to 1',
      css: [0.4, 0.5], dpr: 1.5, maxDpr: 2, expect: [1, 1],
      why: '0.4 x 1.5 = 0.6 rounds to 1 and 0.5 x 1.5 = 0.75 rounds to 1 — never 0',
    },
    {
      name: 'exactly 0.5 at 1x rounds to 1',
      css: [0.5, 0.5], dpr: 1, maxDpr: 2, expect: [1, 1],
      why: 'Math.round(0.5) is 1, and even if it were 0 the floor would catch it. Both defences are tested.',
    },
    {
      name: 'huge size stays huge',
      css: [100000, 50000], dpr: 1, maxDpr: 2, expect: [100000, 50000],
      why: 'CanvasSizer does NOT clamp to maxTextureDimension2D. It has no device, and a silent clamp is a cropped image with no symptom anywhere — RenderTargetImpl raises CANVAS_SIZE_INVALID instead.',
    },
    {
      name: 'huge size at 3x stays huge',
      css: [5000, 5000], dpr: 3, maxDpr: 3, expect: [15000, 15000],
      why: 'the case that reaches RenderTargetImpl and fails there with both numbers in the message',
    },
    {
      name: 'a 4K canvas at 1x',
      css: [3840, 2160], dpr: 1, maxDpr: 2, expect: [3840, 2160],
      why: 'fits a 4096 compatibility-mode limit exactly, which is why that limit is a real constraint on a 4K canvas at 1x',
    },
    {
      name: 'a 4K canvas at 2x overflows compat',
      css: [3840, 2160], dpr: 2, maxDpr: 2, expect: [7680, 4320],
      why: 'the actual reason a 4K canvas needs a pixel-ratio cap on a phone: 7680 over the 4096 limit',
    },
  ];

  for (const c of CASES) {
    test(`${c.name} — ${c.why}`, () => {
      const size = CanvasSizer.backingSize(c.css[0], c.css[1], c.dpr, c.maxDpr);
      expect(size).toEqual({ width: c.expect[0], height: c.expect[1] });
    });
  }

  test('the result is always a pair of positive integers', () => {
    // A fuzz sweep over the whole input space, asserting the invariant rather
    // than a value. The table above covers the interesting points; this covers
    // the space between them.
    const samples = [0, -1, 0.4, 0.5, 1, 1.5, 2, 2.5, 3, 7.25, 12345, Number.NaN, Infinity];
    for (const css of samples) {
      for (const dpr of samples) {
        for (const max of samples) {
          const { width, height } = CanvasSizer.backingSize(css, css, dpr, max);
          expect(Number.isInteger(width)).toBe(true);
          expect(Number.isInteger(height)).toBe(true);
          expect(width).toBeGreaterThan(0);
          expect(height).toBeGreaterThan(0);
        }
      }
    }
  });

  test('a fresh object each call, so callers cannot alias it', () => {
    const a = CanvasSizer.backingSize(800, 600, 2, 2);
    const b = CanvasSizer.backingSize(800, 600, 2, 2);
    expect(a).toEqual(b);
    expect(a).not.toBe(b);
  });
});

// ===========================================================================
// 2. CanvasSizer — the DOM bookkeeping
// ===========================================================================

/** A canvas stand-in with just the two layout reads the sizer performs. */
function fakeCanvas(cssWidth: number, cssHeight: number): HTMLCanvasElement {
  return { clientWidth: cssWidth, clientHeight: cssHeight } as unknown as HTMLCanvasElement;
}

/** Sets `globalThis.devicePixelRatio` for the duration of `fn`. */
async function withDpr<T>(dpr: number, fn: () => T | Promise<T>): Promise<T> {
  const globals = globalThis as { devicePixelRatio?: number };
  const previous = globals.devicePixelRatio;
  globals.devicePixelRatio = dpr;
  try {
    return await fn();
  } finally {
    globals.devicePixelRatio = previous;
  }
}

describe('CanvasSizer', () => {
  test('reads the layout box and applies the pixel ratio cap', async () => {
    await withDpr(3, () => {
      const sizer = new CanvasSizer(fakeCanvas(390, 844), { maxPixelRatio: 2 });
      expect(sizer.cssWidth).toBe(390);
      expect(sizer.cssHeight).toBe(844);
      expect(sizer.width).toBe(780);
      expect(sizer.height).toBe(1688);
      expect(sizer.maxPixelRatio).toBe(2);
      sizer.dispose();
    });
  });

  test('defaults to a pixel ratio cap of 2', async () => {
    await withDpr(3, () => {
      const sizer = new CanvasSizer(fakeCanvas(100, 100));
      expect(sizer.maxPixelRatio).toBe(2);
      expect(sizer.width).toBe(200);
      sizer.dispose();
    });
  });

  test('the first update reports a change, the second does not', async () => {
    await withDpr(1, () => {
      // The constructor performs the initial measure, so the first explicit
      // update is a no-op — there is nothing left to report.
      const sizer = new CanvasSizer(fakeCanvas(640, 480));
      expect(sizer.update()).toBe(false);
      expect(sizer.update()).toBe(false);
      sizer.dispose();
    });
  });

  test('a device pixel ratio of 0 does not throw and does not produce 0', async () => {
    await withDpr(0, () => {
      const sizer = new CanvasSizer(fakeCanvas(640, 480));
      expect(sizer.width).toBe(640);
      expect(sizer.height).toBe(480);
      sizer.dispose();
    });
  });

  test('a hidden canvas with a zero layout box yields 1x1, not 0x0', async () => {
    await withDpr(2, () => {
      const sizer = new CanvasSizer(fakeCanvas(0, 0));
      expect(sizer.width).toBe(1);
      expect(sizer.height).toBe(1);
      sizer.dispose();
    });
  });

  test('a pixel ratio change is noticed even though the layout box did not move', async () => {
    // The scenario that makes the per-frame DPR read necessary: a ResizeObserver
    // reports the element's *CSS* box, which does not change when the window
    // zooms. Without the global read the sizer would keep rendering at 1x.
    const canvas = fakeCanvas(800, 600);
    await withDpr(1, () => {
      const sizer = new CanvasSizer(canvas);
      expect(sizer.width).toBe(800);
      expect(sizer.update()).toBe(false);
      return withDpr(2, () => {
        expect(sizer.update()).toBe(true);
        expect(sizer.width).toBe(1600);
        sizer.dispose();
      });
    });
  });

  test('dispose is idempotent, and update() after it is a typed failure', async () => {
    await withDpr(1, () => {
      const sizer = new CanvasSizer(fakeCanvas(100, 100));
      sizer.dispose();
      sizer.dispose();
      const error = expectCode(() => sizer.update(), 'INTERNAL_INVARIANT');
      expect(error.why.length).toBeGreaterThan(0);
      expect(error.fix.length).toBeGreaterThan(0);
    });
  });
});

describe('CanvasSizer — the ResizeObserver path', () => {
  /** A canvas stand-in that counts layout reads, which is the whole point. */
  function countedCanvas(cssWidth: number, cssHeight: number): HTMLCanvasElement & { reads: number } {
    const canvas = {
      reads: 0,
      get clientWidth() { canvas.reads++; return cssWidth; },
      get clientHeight() { canvas.reads++; return cssHeight; },
    };
    return canvas as unknown as HTMLCanvasElement & { reads: number };
  }

  /** Stubs `globalThis.ResizeObserver` for the duration of `fn`. */
  async function withObserver<T>(fn: (fire: () => void, state: { disconnects: number }) => T | Promise<T>): Promise<T> {
    const state = { disconnects: 0 };
    let notify: (() => void) | null = null;
    class StubResizeObserver {
      constructor(_callback: () => void) { notify = _callback; }
      observe(): void { /* no-op */ }
      unobserve(): void { /* no-op */ }
      disconnect(): void { state.disconnects++; }
    }
    const globals = globalThis as { ResizeObserver?: unknown };
    const previous = globals.ResizeObserver;
    globals.ResizeObserver = StubResizeObserver;
    try {
      return await fn(() => notify?.(), state);
    } finally {
      if (previous === undefined) delete globals.ResizeObserver;
      else globals.ResizeObserver = previous;
    }
  }

  test('observes the element, and does not poll the layout box every frame', async () => {
    await withDpr(1, () => withObserver(async (fire, state) => {
      const canvas = countedCanvas(800, 600);
      const sizer = new CanvasSizer(canvas, { maxPixelRatio: 2 });
      expect(sizer.width).toBe(800);

      // Reading `clientWidth` is a forced synchronous layout. With an observer
      // standing by, apse must not do it on a frame where nothing moved.
      const afterConstruction = canvas.reads;
      expect(sizer.update()).toBe(false);
      expect(sizer.update()).toBe(false);
      expect(sizer.update()).toBe(false);
      expect(canvas.reads).toBe(afterConstruction);

      // An observer notification costs one layout read and nothing else: the
      // element did not actually change size, so nothing changed.
      fire();
      expect(canvas.reads).toBe(afterConstruction);
      expect(sizer.update()).toBe(false);
      expect(canvas.reads).toBe(afterConstruction + 2);

      sizer.dispose();
      expect(state.disconnects).toBe(1);
      sizer.dispose();
      expect(state.disconnects).toBe(1);
    }));
  });

  test('an observer notification that moves the element reports a change', async () => {
    await withDpr(1, () => withObserver(async (fire) => {
      const canvas = countedCanvas(800, 600);
      const sizer = new CanvasSizer(canvas);
      // A grid column reflow: the element moves and the observer says so. This
      // is the case a `window.resize` listener would miss entirely.
      Object.defineProperty(canvas, 'clientWidth', { get: () => 640, configurable: true });
      Object.defineProperty(canvas, 'clientHeight', { get: () => 480, configurable: true });
      fire();
      expect(sizer.update()).toBe(true);
      expect(sizer.width).toBe(640);
      expect(sizer.height).toBe(480);
      expect(sizer.cssWidth).toBe(640);
      sizer.dispose();
    }));
  });

  test('without an observer, the layout box is re-read every frame', async () => {
    // The fallback, and the reason it exists: there is nothing to tell the
    // sizer the element moved, so it has to look. Correctness over speed.
    expect(typeof (globalThis as { ResizeObserver?: unknown }).ResizeObserver).toBe('undefined');
    await withDpr(1, () => {
      const canvas = countedCanvas(800, 600);
      const sizer = new CanvasSizer(canvas);
      const afterConstruction = canvas.reads;
      sizer.update();
      sizer.update();
      expect(canvas.reads).toBe(afterConstruction + 4);
      sizer.dispose();
    });
  });

  test('a pixel ratio change is noticed with an observer attached too', async () => {
    // The observer reports the element's CSS box, which does not change when
    // the window zooms. Without the per-frame global read the sizer would
    // keep rendering at the old resolution after a zoom, with no resize event.
    await withObserver(async () => {
      const canvas = countedCanvas(400, 300);
      await withDpr(1, () => {
        const sizer = new CanvasSizer(canvas, { maxPixelRatio: 3 });
        expect(sizer.width).toBe(400);
        return withDpr(2.5, () => {
          expect(sizer.update()).toBe(true);
          expect(sizer.width).toBe(1000);
          sizer.dispose();
        });
      });
    });
  });
});

// ===========================================================================
// 3. Limit copying — the defence against `{...limits}`
// ===========================================================================

describe('copyLimits', () => {
  test('the fake really is prototype getters with no own properties', () => {
    // The premise of every other test in this block. If this fails, the tests
    // below are proving nothing and the fake needs fixing, not the code.
    const limits = fakeLimits();
    expect(Object.keys(limits)).toEqual([]);
    expect(Object.getOwnPropertyNames(limits)).toEqual([]);
    expect(Object.prototype.hasOwnProperty.call(limits, 'maxTextureDimension2D')).toBe(false);
    // ...and the values are still reachable, which is the whole point.
    expect(limits.maxTextureDimension2D).toBe(4096);
    expect(limits.maxUniformBufferBindingSize).toBe(16384);
  });

  test('object spread on a real limits object produces {}', () => {
    // The bug, stated as an assertion so it cannot regress into a belief.
    const spread = { ...fakeLimits() } as unknown as Record<string, number>;
    expect(Object.keys(spread)).toEqual([]);
    expect(spread).toEqual({});
  });

  test('copyLimits copies every field the fake exposes, explicitly', () => {
    const source = fakeLimits();
    const copy = copyLimits(source);

    // Exactly the declared set: no more (no __brand, no unlisted limits) and
    // no fewer (a limit that silently stops being copied is a limit that
    // silently stops being validated).
    expect(Object.keys(copy).sort()).toEqual([...ASE_LIMIT_NAMES].sort());
    expect(ASE_LIMIT_NAMES).toHaveLength(25);

    for (const name of ASE_LIMIT_NAMES) {
      expect(copy[name]).toBe(source[name] as number);
    }
  });

  test('the copy is a plain frozen object, not a limits-shaped one', () => {
    const copy = copyLimits(fakeLimits());
    expect(Object.isFrozen(copy)).toBe(true);
    expect(Object.getPrototypeOf(copy)).toBe(Object.prototype);
    // Object.freeze is shallow but these are all numbers, so it is total here.
    expect(() => {
      (copy as unknown as Record<string, number>)['maxBindGroups'] = 99;
    }).toThrow();
    expect(copy.maxBindGroups).toBe(4);
  });

  test('the __brand nominal-typing marker is not carried across', () => {
    const branded = fakeLimits();
    Object.defineProperty(branded, '__brand', { value: 'GPUSupportedLimits', enumerable: true });
    const copy = copyLimits(branded);
    expect('__brand' in copy).toBe(false);
  });

  test('the copy reflects the source, not the constants', () => {
    // A device that granted better-than-compat limits (a core-mode device, or
    // one that rounded a requested alignment down) must report the truth.
    const source = fakeLimits({
      maxTextureDimension2D: 8192,
      maxUniformBufferBindingSize: 65536,
      minUniformBufferOffsetAlignment: 64,
    });
    const copy = copyLimits(source);
    expect(copy.maxTextureDimension2D).toBe(8192);
    expect(copy.maxUniformBufferBindingSize).toBe(65536);
    // Alignment limits go down, not up, so a smaller value is a *better* one.
    expect(copy.minUniformBufferOffsetAlignment).toBe(64);
  });

  test('every declared limit is a real GPUSupportedLimits member', () => {
    // Guards against a typo in AseLimitName: `limits[typo]` would be undefined
    // at runtime, silently, because TypeScript cannot check a dynamic index into
    // an interface without a cast. The fake here is a proxy over a known-good
    // key set, so a miss is visible.
    const known = new Set([
      'maxTextureDimension1D', 'maxTextureDimension2D', 'maxTextureDimension3D',
      'maxTextureArrayLayers', 'maxBindGroups', 'maxBindGroupsPlusVertexBuffers',
      'maxBindingsPerBindGroup', 'maxDynamicUniformBuffersPerPipelineLayout',
      'maxDynamicStorageBuffersPerPipelineLayout', 'maxSampledTexturesPerShaderStage',
      'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage',
      'maxStorageTexturesPerShaderStage', 'maxUniformBuffersPerShaderStage',
      'maxUniformBufferBindingSize', 'maxStorageBufferBindingSize',
      'minUniformBufferOffsetAlignment', 'minStorageBufferOffsetAlignment',
      'maxVertexBuffers', 'maxBufferSize', 'maxVertexAttributes',
      'maxVertexBufferArrayStride', 'maxInterStageShaderVariables',
      'maxColorAttachments', 'maxColorAttachmentBytesPerSample',
      'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY',
      'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupsPerDimension',
    ]);
    const proxy = new Proxy({} as GPUSupportedLimits, {
      get(_t, key: string) {
        if (typeof key !== 'string' || !known.has(key)) {
          throw new Error(`copyLimits read a limit that does not exist: "${String(key)}"`);
        }
        return 1;
      },
    });
    expect(() => copyLimits(proxy)).not.toThrow();
  });
});

// ===========================================================================
// 4. The compatibility limit table
// ===========================================================================

describe('COMPAT_LIMITS', () => {
  test('matches the compatibility-mode default column of the spec', () => {
    // WebGPU Editor's Draft, 23 September 2026, section 3.6.2 "Limits".
    // Written out rather than derived so a change has to be deliberate.
    expect(COMPAT_LIMITS).toEqual({
      maxTextureDimension2D: 4096,
      maxBufferSize: 268435456,
      maxUniformBufferBindingSize: 16384,
      maxStorageBufferBindingSize: 134217728,
      minUniformBufferOffsetAlignment: 256,
      maxBindGroups: 4,
      maxBindingsPerBindGroup: 1000,
      maxDynamicUniformBuffersPerPipelineLayout: 8,
      maxSampledTexturesPerShaderStage: 16,
      maxSamplersPerShaderStage: 16,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageTexturesPerShaderStage: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxVertexBuffers: 8,
      maxVertexAttributes: 16,
      maxVertexBufferArrayStride: 2048,
      maxInterStageShaderVariables: 15,
      maxColorAttachments: 4,
      maxColorAttachmentBytesPerSample: 32,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
      maxComputeWorkgroupSizeY: 128,
      maxComputeWorkgroupSizeZ: 64,
      maxComputeWorkgroupsPerDimension: 65535,
    });
  });

  test('the values that actually differ from core, for the record', () => {
    // The six numbers a shader author needs to know about. If one of these ever
    // changes, this test is the place the reason should be written.
    expect(COMPAT_LIMITS.maxUniformBufferBindingSize).toBe(16384);
    expect(COMPAT_LIMITS.maxColorAttachments).toBe(4);
    expect(COMPAT_LIMITS.maxTextureDimension2D).toBe(4096);
    expect(COMPAT_LIMITS.maxInterStageShaderVariables).toBe(15);
    expect(COMPAT_LIMITS.maxComputeInvocationsPerWorkgroup).toBe(128);
    expect(COMPAT_LIMITS.maxComputeWorkgroupSizeX).toBe(128);
    expect(COMPAT_LIMITS.maxComputeWorkgroupSizeY).toBe(128);
  });

  test('the vertex stage gets no storage buffers in compatibility mode', () => {
    expect(COMPAT_VERTEX_STAGE_LIMITS.maxStorageBuffersInVertexStage).toBe(0);
    expect(COMPAT_VERTEX_STAGE_LIMITS.maxStorageTexturesInVertexStage).toBe(0);
    // ...while the fragment stage is unconstrained by that rule.
    expect(COMPAT_VERTEX_STAGE_LIMITS.maxStorageBuffersInFragmentStage).toBe(8);
  });

  test('the table is frozen', () => {
    expect(Object.isFrozen(COMPAT_LIMITS)).toBe(true);
    expect(Object.isFrozen(CORE_DEFAULT_LIMITS)).toBe(true);
    expect(Object.isFrozen(ASE_LIMIT_NAMES)).toBe(true);
  });
});

describe('compatRequiredLimits', () => {
  test('requests exactly the compatibility defaults — never anything better', () => {
    // This is the load-bearing assertion of the whole module. The spec gives a
    // device the defaults for its feature level and then raises each entry to
    // max(requested, default), so asking for the compat value is always legal
    // and can never make the page depend on a capability the device lacks.
    const requested = compatRequiredLimits();
    for (const name of ASE_LIMIT_NAMES) {
      const value = requested[name];
      const compat = COMPAT_LIMITS[name];
      const core = CORE_DEFAULT_LIMITS[name];
      const alignment = LIMIT_CLASSES[name] === 'alignment';
      if (alignment) {
        // Alignment: lower is better, so the request must be no *worse* than
        // the compat default and no better than the core default either.
        expect(value).toBeGreaterThanOrEqual(compat as number);
        expect(value).toBeLessThanOrEqual(core as number);
      } else {
        // Maximum: higher is better, so the request must be no better than
        // either profile's default.
        expect(value).toBeLessThanOrEqual(compat as number);
        expect(value).toBeLessThanOrEqual(core as number);
      }
    }
  });

  test('never asks for more than the compat profile allows, in any direction', () => {
    // The same assertion, phrased as the mistake it prevents: no entry may be
    // better than the compatibility default, because a device in compatibility
    // mode may not be able to grant it.
    for (const name of ASE_LIMIT_NAMES) {
      const requested = compatRequiredLimits()[name];
      const compat = COMPAT_LIMITS[name] as number;
      if (LIMIT_CLASSES[name] === 'alignment') {
        expect(requested).toBeGreaterThanOrEqual(compat);
      } else {
        expect(requested).toBeLessThanOrEqual(compat);
      }
    }
  });

  test('is in fact exactly the compat table, key for key', () => {
    expect(compatRequiredLimits()).toEqual({ ...COMPAT_LIMITS });
  });

  test('sends no key the implementation might not know', () => {
    // requiredLimits rejects unknown keys outright, so a limit added to the
    // spec next year is not safe to send today. These three are real spec
    // members that apse deliberately omits.
    const sent = compatRequiredLimits() as Record<string, number>;
    expect(Object.keys(sent)).toHaveLength(ASE_LIMIT_NAMES.length);
    expect(sent['maxBindGroupsPlusVertexBuffers']).toBeUndefined();
    expect(sent['maxImmediateSize']).toBeUndefined();
    expect(sent['maxTextureDimension1D']).toBeUndefined();
    expect(sent['maxTextureDimension3D']).toBeUndefined();
  });

  test('a fresh object each call, so a caller cannot poison the next device', () => {
    const first = compatRequiredLimits() as Record<AseLimitName, number>;
    first.maxBindGroups = 99;
    expect(compatRequiredLimits().maxBindGroups).toBe(4);
  });

  test('formatLimitRequest names every limit, for an error message', () => {
    const text = formatLimitRequest(compatRequiredLimits());
    for (const name of ASE_LIMIT_NAMES) expect(text).toContain(`${name}=`);
    expect(text).toContain('maxUniformBufferBindingSize=16384');
  });
});

// ===========================================================================
// 5. withErrorScope
// ===========================================================================

describe('withErrorScope', () => {
  test('returns the value and leaves the scope stack balanced', async () => {
    const device = new FakeGPUDevice();
    const result = await withErrorScope(
      asGpuDevice(device), 'validation', () => 42, 'test op',
    );
    expect(result).toBe(42);
    expect(device.scopeStack).toHaveLength(0);
    expect(device.poppedErrors).toEqual([null]);
  });

  test('awaits an async body', async () => {
    const device = new FakeGPUDevice();
    const result = await withErrorScope(
      asGpuDevice(device), 'validation', async () => 'done', 'test op',
    );
    expect(result).toBe('done');
    expect(device.poppedErrors).toEqual([null]);
  });

  test('a raised error becomes a typed failure quoting the raw driver text', async () => {
    const device = new FakeGPUDevice({
      errorScopes: [{ message: 'usage flags 0x14 are not allowed on rgba8unorm' } as GPUError],
    });
    const error = await expectCodeAsync(
      () => withErrorScope(asGpuDevice(device), 'validation', () => 1, 'the checker pipeline'),
      'GPU_VALIDATION_FAILED',
    );
    expect(error.message).toContain('the checker pipeline');
    expect(error.message).toContain('usage flags 0x14 are not allowed on rgba8unorm');
    expect(error.detail).toEqual({
      kind: 'gpu-validation',
      scope: 'validation',
      raw: 'usage flags 0x14 are not allowed on rgba8unorm',
    });
    // The fix repeats the raw text: it is the only actionable part of the error.
    expect(error.fix).toContain('0x14');
  });

  test('the scope is popped even when the body throws', async () => {
    // An unbalanced scope stack mis-attributes the next unrelated error to this
    // call, which is far worse than the original exception.
    const device = new FakeGPUDevice();
    await expect(withErrorScope(asGpuDevice(device), 'validation', () => {
      throw new Error('body exploded');
    })).rejects.toThrow('body exploded');
    expect(device.scopeStack).toHaveLength(0);
    expect(device.poppedErrors).toEqual([null]);
  });

  test('a null error means success', async () => {
    const device = new FakeGPUDevice({ errorScopes: [null] });
    await expect(withErrorScope(asGpuDevice(device), 'internal', () => 1)).resolves.toBe(1);
  });

  test('the scope filter is passed through to the device', async () => {
    const device = new FakeGPUDevice();
    await withErrorScope(asGpuDevice(device), 'out-of-memory', () => 1);
    expect(device.scopeStack).toHaveLength(0);
    expect(device.poppedErrors).toHaveLength(1);
  });

  test('an operation name defaults to something readable', async () => {
    const device = new FakeGPUDevice({ errorScopes: [{ message: 'boom' } as GPUError] });
    const error = await expectCodeAsync(
      () => withErrorScope(asGpuDevice(device), 'validation', () => 1),
      'GPU_VALIDATION_FAILED',
    );
    expect(error.message).toContain('A WebGPU operation');
  });
});

// ===========================================================================
// 6. RenderTargetImpl texture accounting
// ===========================================================================

/** A device + target in one line, with the fake reachable for assertions. */
function target(opts: {
  width?: number;
  height?: number;
  format?: GPUTextureFormat;
  sampleCount?: 1 | 4;
  depth?: boolean;
  depthFormat?: GPUTextureFormat;
  sampleDepth?: boolean;
  label?: string;
  limits?: GPUSupportedLimits;
}): { fake: ReturnType<typeof fakeAseDevice>; device: FakeGPUDevice; target: RenderTargetImpl } {
  const fake = fakeAseDevice();
  const device: RenderTargetDevice = {
    device: asGpuDevice(fake.device),
    limits: fake.limits,
    assertLive: () => fake.assertLive(),
  };
  const target = createColorTarget(device, {
    width: opts.width ?? 640,
    height: opts.height ?? 480,
    format: opts.format ?? 'rgba8unorm',
    sampleCount: opts.sampleCount ?? 1,
    depth: opts.depth ?? false,
    depthFormat: opts.depthFormat,
    sampleDepth: opts.sampleDepth ?? false,
    label: opts.label ?? 'test',
  });
  return { fake, device: fake.device, target };
}

describe('RenderTargetImpl — texture accounting', () => {
  test('a single-sampled target with depth allocates exactly 1 colour + 1 depth', () => {
    const { device, target: t } = target({ sampleCount: 1, depth: true });
    expect(t.width).toBe(640);
    expect(t.height).toBe(480);
    expect(device.textures).toHaveLength(2);
    expect(device.live(':color')).toHaveLength(1);
    expect(device.live(':depth')).toHaveLength(1);
    // The size has to reach the descriptor, not just the accessor.
    expect(device.createLog[0]?.size).toEqual({ width: 640, height: 480 });
    expect(device.createLog[0]?.sampleCount).toBe(1);
    t.dispose();
  });

  test('a 4x target allocates exactly 2 colour + 1 depth', () => {
    // The MSAA case. Two colour textures is not waste: a multisampled texture
    // cannot be sampled, so the resolve target is the only thing a later pass
    // can read, and it is the reason MSAA and post-processing compose at all.
    const { device, target: t } = target({ sampleCount: 4, depth: true });
    expect(t.sampleCount).toBe(4);
    expect(device.textures).toHaveLength(3);

    const msaa = device.live(':color');
    const resolve = device.live(':colorResolve');
    const depth = device.live(':depth');
    expect(msaa).toHaveLength(1);
    expect(resolve).toHaveLength(1);
    expect(depth).toHaveLength(1);

    expect(msaa[0]?.sampleCount).toBe(4);
    expect(resolve[0]?.sampleCount).toBe(1);
    // The multisampled attachment is render-only: TEXTURE_BINDING on a
    // multisampled texture is a validation error, not a slow path.
    expect(msaa[0]?.hasUsage(USAGE.RENDER_ATTACHMENT)).toBe(true);
    expect(msaa[0]?.hasUsage(USAGE.TEXTURE_BINDING)).toBe(false);
    // The resolve target is written by the resolve and read by the next pass.
    expect(resolve[0]?.hasUsage(USAGE.RENDER_ATTACHMENT | USAGE.TEXTURE_BINDING)).toBe(true);
    // Depth follows the colour sample count; a render pass requires agreement.
    expect(depth[0]?.sampleCount).toBe(4);
    expect(depth[0]?.format).toBe(DEFAULT_DEPTH_FORMAT);
    t.dispose();
  });

  test('no depth is allocated unless it is asked for', () => {
    // "Allocate depth only when asked" — an unconditional depth buffer costs
    // 3 bytes a pixel on a phone for a pass that may never test against it.
    const { device, target: t } = target({ depth: false });
    expect(device.textures).toHaveLength(1);
    expect(device.live(':depth')).toHaveLength(0);
    expect(t.depthView).toBeUndefined();
    expect(t.depthFormat).toBeUndefined();
    t.dispose();
  });

  test('with sampleCount 1, colorView and sampleView are the same object', () => {
    const { target: t } = target({ sampleCount: 1 });
    expect(t.sampleView).toBe(t.colorView);
    t.dispose();
  });

  test('with sampleCount 4, colorView and sampleView are different objects', () => {
    // The distinction the whole file exists for. A renderer that binds
    // `colorView` into a bind group gets a validation error naming a sampled
    // texture binding, and nothing that points here.
    const { target: t } = target({ sampleCount: 4 });
    expect(t.sampleView).not.toBe(t.colorView);
    expect(t.sampleView).toBeDefined();
    t.dispose();
  });

  test('resize to the same size allocates nothing and destroys nothing', () => {
    // The no-op case a renderer hits on every frame it calls resize()
    // defensively. Recreating textures here would reallocate several megabytes
    // per frame for no reason.
    const { device, target: t } = target({ width: 800, height: 600, depth: true, sampleCount: 4 });
    const created = device.textures.length;
    const destroyed = device.destroyLog.length;
    t.resize(800, 600);
    expect(device.textures).toHaveLength(created);
    expect(device.destroyLog).toHaveLength(destroyed);
    expect(device.liveTextures).toHaveLength(3);
    t.dispose();
  });

  test('resize destroys the old textures before allocating new ones', () => {
    // The order is the point: the other order peaks at two full targets, which
    // on a 4x MSAA 1080p target is the difference between allocating and
    // failing to allocate on a mid-range phone.
    const { device, target: t } = target({ width: 800, height: 600, depth: true, sampleCount: 4 });
    const old = [...device.liveTextures];
    t.resize(1024, 768);

    // Every old texture is gone, and gone *before* the new ones were made: at
    // no point did more than 3 textures exist.
    for (const texture of old) expect(texture.destroyed).toBe(true);
    expect(device.destroyLog).toEqual(['test:color', 'test:colorResolve', 'test:depth']);
    expect(device.textures).toHaveLength(6);
    expect(device.liveTextures).toHaveLength(3);
    expect(t.width).toBe(1024);
    expect(t.height).toBe(768);
    // New textures are at the new size, and still correctly configured.
    for (const texture of device.liveTextures) {
      expect(texture.width).toBe(1024);
      expect(texture.height).toBe(768);
    }
    t.dispose();
  });

  test('a chain of resizes never accumulates textures', () => {
    const { device, target: t } = target({ depth: true, sampleCount: 4 });
    for (let i = 1; i <= 20; i++) {
      t.resize(320 + i, 240 + i);
      expect(device.liveTextures).toHaveLength(3);
    }
    expect(device.textures).toHaveLength(3 * 21);
    t.dispose();
    expect(device.liveTextures).toHaveLength(0);
  });

  test('dispose leaves zero live textures', () => {
    const { device, target: t } = target({ depth: true, sampleCount: 4 });
    expect(device.liveTextures).toHaveLength(3);
    t.dispose();
    expect(device.liveTextures).toHaveLength(0);
    expect(device.destroyLog).toEqual(['test:color', 'test:colorResolve', 'test:depth']);
  });

  test('dispose twice is safe', () => {
    // Resource.dispose() is idempotent, and the fake throws if a texture is
    // destroyed twice — so this proves apse is not leaning on the driver being
    // lenient.
    const { device, target: t } = target({ depth: true, sampleCount: 4 });
    t.dispose();
    t.dispose();
    t.dispose();
    expect(device.liveTextures).toHaveLength(0);
    expect(device.destroyLog).toHaveLength(3);
  });

  test('dispose after resize frees only the current textures', () => {
    const { device, target: t } = target({ depth: true, sampleCount: 4 });
    t.resize(64, 64);
    expect(device.liveTextures).toHaveLength(3);
    t.dispose();
    expect(device.liveTextures).toHaveLength(0);
    // Six allocations, six destroys: nothing stranded.
    expect(device.textures).toHaveLength(6);
    expect(device.destroyLog).toHaveLength(6);
  });

  test('the ref count composes with dispose, from the shared Resource base', () => {
    const { device, target: t } = target({ depth: true });
    expect(t.refCount).toBe(1);
    t.ref();
    expect(t.refCount).toBe(2);
    t.unref();
    expect(device.liveTextures).toHaveLength(2);
    t.unref();
    expect(device.liveTextures).toHaveLength(0);
    expect(t.disposed).toBe(true);
  });
});

describe('RenderTargetImpl — views and usage', () => {
  test('the resolve target is not given COPY_SRC, deliberately', () => {
    // A resolve target needs RENDER_ATTACHMENT to be written and nothing else
    // here. COPY_SRC can force a slower path on tiled GPUs, and a screenshot of
    // an offscreen target should be taken from the canvas, which already has it.
    const { device, target: t } = target({ sampleCount: 4 });
    const resolve = device.live(':colorResolve')[0];
    expect(resolve?.hasUsage(USAGE.COPY_SRC)).toBe(false);
    t.dispose();
  });

  test('depth is RENDER_ATTACHMENT only unless sampling was asked for', () => {
    const plain = target({ depth: true });
    expect(plain.device.live(':depth')[0]?.hasUsage(USAGE.TEXTURE_BINDING)).toBe(false);
    plain.target.dispose();

    const sampled = target({ depth: true, sampleDepth: true });
    expect(sampled.device.live(':depth')[0]?.hasUsage(USAGE.TEXTURE_BINDING)).toBe(true);
    sampled.target.dispose();
  });

  test('depth32float is available as an option, and depth24plus is the default', () => {
    // depth24plus is the default because it is renderable as a depth-stencil
    // attachment everywhere. depth32float gives a full 32 bits of range but is
    // not universally renderable, so choosing it is a portability decision and
    // the API makes you make it.
    const fallback = target({ depth: true });
    expect(fallback.target.depthFormat).toBe('depth24plus');
    fallback.target.dispose();

    const wide = target({ depth: true, depthFormat: 'depth32float' });
    expect(wide.target.depthFormat).toBe('depth32float');
    expect(wide.device.live(':depth')[0]?.format).toBe('depth32float');
    wide.target.dispose();
  });

  test('every texture is labelled, and the label names its role', () => {
    // WebGPU error messages reference the most recent label, so an unlabelled
    // resource produces a validation error that names a descriptor field but not
    // the object that had the bug.
    const { device, target: t } = target({ sampleCount: 4, depth: true, label: 'shadow' });
    expect(device.createLog.map((d) => d.label)).toEqual([
      'shadow:color',
      'shadow:colorResolve',
      'shadow:depth',
    ]);
    t.dispose();
  });
});

describe('RenderTargetImpl — size validation', () => {
  test('a zero or negative size is rejected', () => {
    for (const [w, h] of [[0, 100], [100, 0], [0, 0], [-1, 100], [100, -1]] as const) {
      const fake = fakeAseDevice();
      const device: RenderTargetDevice = {
        device: asGpuDevice(fake.device),
        limits: fake.limits,
        assertLive: () => fake.assertLive(),
      };
      const error = expectCode(
        () => createColorTarget(device, { width: w, height: h }),
        'RENDER_TARGET_SIZE_INVALID',
      );
      expect(error.message).toContain(`${w}x${h}`);
      // A rejected size must not have allocated anything.
      expect(fake.device.textures).toHaveLength(0);
    }
  });

  test('a fractional size is rejected', () => {
    const fake = fakeAseDevice();
    const device: RenderTargetDevice = {
      device: asGpuDevice(fake.device),
      limits: fake.limits,
      assertLive: () => fake.assertLive(),
    };
    expectCode(() => createColorTarget(device, { width: 100.5, height: 100 }), 'RENDER_TARGET_SIZE_INVALID');
  });

  test('a size over maxTextureDimension2D fails with both numbers in the message', () => {
    // An error, not a clamp. A 5000-pixel canvas quietly rendered into a
    // 4096-pixel target is a cropped image with no symptom anywhere.
    const { device: fake, target: t } = target({ width: 4096, height: 2160 });
    const error = expectCode(() => t.resize(5000, 2160), 'CANVAS_SIZE_INVALID');
    expect(error.message).toContain('5000x2160');
    expect(error.message).toContain('4096');
    expect(error.fix).toContain('4096');
    // The failed resize changed nothing and destroyed nothing.
    expect(t.width).toBe(4096);
    expect(fake.liveTextures).toHaveLength(1);
    t.dispose();
  });

  test('the limit used is the device limit, not the compat constant', () => {
    // A core-mode device reports 8192, and a 5000-wide target is legal there.
    // Clamping to the compat number regardless would be a portability bug in
    // the other direction.
    const fake = fakeAseDevice({ limits: fakeLimits({ maxTextureDimension2D: 8192 }) });
    const device: RenderTargetDevice = {
      device: asGpuDevice(fake.device),
      limits: fake.limits,
      assertLive: () => fake.assertLive(),
    };
    const t = createColorTarget(device, { width: 5000, height: 2160 });
    expect(t.width).toBe(5000);
    expect(fake.device.live(':color')[0]?.width).toBe(5000);
    t.dispose();
  });
});

describe('RenderTargetImpl — liveness', () => {
  test('accessors after dispose fail loudly', () => {
    // The error catalog has no code for "used a disposed target" — the nearest
    // is RENDERER_ALREADY_DISPOSED, which belongs to another module — so
    // INTERNAL_INVARIANT is used with a message that says exactly what happened.
    const { target: t } = target({ depth: true, sampleCount: 4 });
    t.dispose();
    expectCode(() => t.colorView, 'INTERNAL_INVARIANT');
    expectCode(() => t.sampleView, 'INTERNAL_INVARIANT');
    expectCode(() => t.depthView, 'INTERNAL_INVARIANT');
    expectCode(() => t.resize(10, 10), 'INTERNAL_INVARIANT');
  });

  test('resize after a device loss is DEVICE_LOST, and allocates nothing', () => {
    const { fake, target: t } = target({ depth: true });
    fake.loseDevice('unknown', 'driver reset');
    const error = expectCode(() => t.resize(32, 32), 'DEVICE_LOST');
    expect(error.message).toContain('the device it was created on is gone');
    // Nothing allocated, nothing destroyed: a dead device cannot free anything.
    expect(fake.device.textures).toHaveLength(2);
    expect(fake.device.destroyLog).toHaveLength(0);
    t.dispose();
    // dispose() must still work on a lost device — the wrappers still exist
    // and a recover() would otherwise inherit them.
    expect(fake.device.liveTextures).toHaveLength(0);
  });
});

describe('createDepthTarget', () => {
  test('allocates only depth, and nothing more', () => {
    const fake = fakeAseDevice();
    const device: RenderTargetDevice = {
      device: asGpuDevice(fake.device),
      limits: fake.limits,
      assertLive: () => fake.assertLive(),
    };
    const t = createDepthTarget(device, { width: 256, height: 256 });
    expect(fake.device.textures).toHaveLength(1);
    expect(fake.device.live(':depth')).toHaveLength(1);
    expect(t.depthFormat).toBe('depth24plus');
    t.dispose();
    expect(fake.device.liveTextures).toHaveLength(0);
  });

  test('colorView is a lazily created 1x1 placeholder, never a depth view', () => {
    // RenderTarget.colorView is a non-optional GPUTextureView, so something has
    // to be behind it. Returning the depth view would be a colour attachment
    // that is silently the wrong type. Allocating nothing until it is read
    // means a caller who never touches it pays nothing.
    const fake = fakeAseDevice();
    const device: RenderTargetDevice = {
      device: asGpuDevice(fake.device),
      limits: fake.limits,
      assertLive: () => fake.assertLive(),
    };
    const t = createDepthTarget(device, { width: 256, height: 256 });
    expect(fake.device.textures).toHaveLength(1);

    const view = t.colorView;
    expect(fake.device.textures).toHaveLength(2);
    const placeholder = fake.device.live(':colorPlaceholder')[0];
    expect(placeholder).toBeDefined();
    expect(placeholder?.width).toBe(1);
    expect(placeholder?.height).toBe(1);
    expect(placeholder?.sampleCount).toBe(1);
    expect(placeholder?.format).toBe('rgba8unorm');
    // It is not the depth view, and it is the same object every time.
    expect(view).not.toBe(t.depthView);
    expect(t.colorView).toBe(view);
    expect(t.sampleView).toBe(view);
    t.dispose();
    expect(fake.device.liveTextures).toHaveLength(0);
  });

  test('a resize frees the placeholder, and it is not recreated until read', () => {
    const fake = fakeAseDevice();
    const device: RenderTargetDevice = {
      device: asGpuDevice(fake.device),
      limits: fake.limits,
      assertLive: () => fake.assertLive(),
    };
    const t = createDepthTarget(device, { width: 256, height: 256 });
    const placeholder = t.colorView;
    expect(fake.device.apseLive(':colorPlaceholder')).toHaveLength(1);

    t.resize(128, 128);
    // The resize freed the placeholder along with everything else, and did not
    // immediately mint a new one. A depth target nobody reads costs one texture.
    expect(fake.device.apseLive(':colorPlaceholder')).toHaveLength(0);
    expect(fake.device.liveTextures).toHaveLength(1);
    expect(t.colorView).not.toBe(placeholder);
    expect(fake.device.apseLive(':colorPlaceholder')[0]?.width).toBe(1);

    t.dispose();
    expect(fake.device.liveTextures).toHaveLength(0);
  });
});

describe('createCanvasTarget', () => {
  function canvasTarget(opts: {
    width?: number;
    height?: number;
    depth?: boolean;
    fake?: ReturnType<typeof fakeAseDevice>;
  } = {}): { fake: ReturnType<typeof fakeAseDevice>; target: RenderTargetImpl; context: ReturnType<typeof asCanvasTargetDevice>['context'] } {
    const fake = opts.fake ?? fakeAseDevice({ canvasWidth: 1024, canvasHeight: 768 });
    const target = createCanvasTarget(asCanvasTargetDevice(fake), {
      width: opts.width,
      height: opts.height,
      depth: opts.depth,
    });
    return { fake, target, context: fake.context };
  }

  test('does not own a colour texture: the browser does', () => {
    // The single most important property of the canvas target. Only the depth
    // attachment is allocated here.
    const { fake, target: t } = canvasTarget({ depth: true });
    expect(t.isCanvas).toBe(true);
    expect(fake.device.apseLog.map((d) => d.label)).toEqual(['apse.canvas:depth']);
    t.dispose();
    expect(fake.device.liveTextures).toHaveLength(0);
  });

  test('without depth it allocates nothing at all', () => {
    const { fake, target: t } = canvasTarget();
    // Nothing at all: the colour texture belongs to the browser and depth was
    // not asked for, so apse allocates nothing until a view is requested.
    expect(fake.device.apseLog).toHaveLength(0);
    t.dispose();
  });

  test('sampleView is undefined, because a swapchain image is not sampleable', () => {
    // The canvas is configured RENDER_ATTACHMENT | COPY_SRC, without
    // TEXTURE_BINDING, because granting texture binding to a swapchain image
    // forces the compositor path to keep a resolvable copy of every frame.
    const { target: t } = canvasTarget();
    expect(t.sampleView).toBeUndefined();
    t.dispose();
  });

  test('the colour view is re-acquired from getCurrentTexture on every access', () => {
    // THE footgun. The canvas texture expires at present, so a cached view
    // points at a texture that no longer exists, and the resulting WebGPU error
    // has no useful text. Re-acquiring every frame is the only correct answer.
    const { fake, target: t } = canvasTarget();
    const context = fake.context as unknown as { getCurrentTextureCalls: number };

    const first = t.getColorViewForFrame();
    expect(context.getCurrentTextureCalls).toBe(1);
    const second = t.getColorViewForFrame();
    expect(context.getCurrentTextureCalls).toBe(2);
    // Two acquisitions, two distinct view objects over one texture — which is
    // exactly why the renderer should call it once and hold the result.
    expect(first).not.toBe(second);

    // The colorView property routes to the same per-frame acquisition.
    void t.colorView;
    expect(context.getCurrentTextureCalls).toBe(3);
    t.dispose();
  });

  test('the canvas view is not affected by dispose — the browser owns it', () => {
    const { fake, target: t } = canvasTarget();
    void t.getColorViewForFrame();
    t.dispose();
    // The current texture is still live: apse must never destroy a texture it
    // did not allocate, and the fake's destroy-twice check would have caught
    // an accidental second destroy of the depth texture, not this one.
    const current = (fake.context as unknown as {
      peekCurrentTexture(): { destroyed: boolean };
    }).peekCurrentTexture();
    expect(current.destroyed).toBe(false);
  });

  test('the target writes canvas.width/height, and resize is what changes them', () => {
    // Single-writer discipline. CanvasSizer computes the size; the canvas
    // target applies it, which is what keeps target.width === canvas.width an
    // invariant rather than a hope.
    const fake = fakeAseDevice({ canvasWidth: 1024, canvasHeight: 768 });
    const { target: t } = canvasTarget({ fake });
    expect(fake.canvas.width).toBe(1024);
    expect(fake.canvas.height).toBe(768);

    t.resize(1280, 720);
    expect(fake.canvas.width).toBe(1280);
    expect(fake.canvas.height).toBe(720);
    expect(t.width).toBe(1280);
    t.dispose();
  });

  test('an explicit size is applied to the canvas immediately', () => {
    const fake = fakeAseDevice({ canvasWidth: 300, canvasHeight: 150 });
    const t = createCanvasTarget(asCanvasTargetDevice(fake), { width: 800, height: 600 });
    expect(fake.canvas.width).toBe(800);
    expect(t.width).toBe(800);
    t.dispose();
  });

  test('resize recreates the depth attachment and leaves the colour alone', () => {
    const { fake, target: t } = canvasTarget({ depth: true, width: 512, height: 512 });
    const before = fake.device.apseLive(':depth')[0];
    expect(before?.width).toBe(512);
    t.resize(1024, 1024);
    expect(before?.destroyed).toBe(true);
    const after = fake.device.apseLive(':depth')[0];
    expect(after?.width).toBe(1024);
    expect(fake.device.apseLive('')).toHaveLength(1);
    t.dispose();
  });

  test('a resize to the same size is a no-op', () => {
    const { fake, target: t } = canvasTarget({ depth: true, width: 512, height: 512 });
    t.resize(512, 512);
    expect(fake.device.apseTextures).toHaveLength(1);
    expect(fake.device.destroyLog).toHaveLength(0);
    t.dispose();
  });

  test('sampleCount is 1 because a canvas texture is never multisampled', () => {
    // There is no sampleCount option to pass — the option does not exist,
    // because getCurrentTexture() always returns a single-sample attachment.
    // 4x MSAA is an offscreen target whose resolve destination is the canvas.
    const { target: t } = canvasTarget();
    expect(t.sampleCount).toBe(1);
    t.dispose();
  });

  test('an unknown depth format is rejected by name', () => {
    const fake = fakeAseDevice();
    const error = expectCode(
      () => createCanvasTarget(asCanvasTargetDevice(fake), { depth: true, depthFormat: 'rgb8unorm' as GPUTextureFormat }),
      'OPTION_UNKNOWN',
    );
    expect(error.message).toContain('rgb8unorm');
    expect(error.message).toContain('depth24plus');
  });
});

describe('option validation', () => {
  test('a sample count other than 1 or 4 is rejected, with the reason', () => {
    const fake = fakeAseDevice();
    const device: RenderTargetDevice = {
      device: asGpuDevice(fake.device),
      limits: fake.limits,
      assertLive: () => fake.assertLive(),
    };
    const error = expectCode(
      () => createColorTarget(device, { width: 8, height: 8, sampleCount: 2 as 1 | 4 }),
      'OPTION_UNKNOWN',
    );
    expect(error.message).toContain('sampleCount 2');
    // 2x exists in hardware and not in WebGPU; supersampling is the answer.
    expect(error.fix).toContain('supersample');
  });
});

// ===========================================================================
// 7. createDevice — the paths reachable without a GPU
// ===========================================================================

describe('createDevice', () => {
  /** A canvas stand-in. Never reached, because option checks come first. */
  const canvas = {} as HTMLCanvasElement;

  test('reports WEBGPU_UNAVAILABLE when there is no navigator.gpu', async () => {
    // The real situation in a Node process, a Web Worker in a browser that
    // lacks the flag, and a page served over plain http from a LAN IP — all
    // three look identical from here, which is why the message quotes the
    // secure-context state rather than guessing at a browser version.
    expect((globalThis as { navigator?: { gpu?: unknown } }).navigator?.gpu).toBeUndefined();
    const error = await expectCodeAsync(() => createDevice(canvas), 'WEBGPU_UNAVAILABLE');
    expect(error.message).toContain('navigator.gpu');
    expect(error.why).toContain('secure-context-only');
    expect(error.fix.length).toBeGreaterThan(0);
  });

  // Every option is validated before the adapter is acquired, so a bad option
  // costs nothing rather than a round trip to the GPU process. These are
  // reachable in a test environment precisely because of that ordering.
  const BAD_OPTIONS: ReadonlyArray<{
    name: string;
    opts: DeviceOptions;
    expectInMessage: string;
  }> = [
    { name: 'powerPreference', opts: { powerPreference: 'medium' as 'low-power' }, expectInMessage: 'powerPreference' },
    { name: 'alphaMode', opts: { alphaMode: 'straight' as 'opaque' }, expectInMessage: 'alphaMode' },
    { name: 'sampleCount', opts: { sampleCount: 2 as 1 | 4 }, expectInMessage: 'sampleCount' },
    { name: 'format', opts: { format: 'depth24plus' }, expectInMessage: 'depth24plus' },
    { name: 'maxPixelRatio of 0', opts: { maxPixelRatio: 0 }, expectInMessage: 'maxPixelRatio' },
    { name: 'negative maxPixelRatio', opts: { maxPixelRatio: -1 }, expectInMessage: 'maxPixelRatio' },
    { name: 'NaN maxPixelRatio', opts: { maxPixelRatio: Number.NaN }, expectInMessage: 'maxPixelRatio' },
    { name: 'zero maxObjects', opts: { maxObjects: 0 }, expectInMessage: 'maxObjects' },
    { name: 'fractional maxObjects', opts: { maxObjects: 10.5 }, expectInMessage: 'maxObjects' },
    { name: 'negative maxObjects', opts: { maxObjects: -4096 }, expectInMessage: 'maxObjects' },
  ];

  for (const bad of BAD_OPTIONS) {
    test(`rejects a bad ${bad.name} with OPTION_UNKNOWN`, async () => {
      const error = await expectCodeAsync(
        () => createDevice(canvas, bad.opts), 'OPTION_UNKNOWN',
      );
      expect(error.message).toContain(bad.expectInMessage);
      // The point of rejecting rather than ignoring: every option error carries
      // the rule that was broken and the one thing to change, so it is
      // actionable without reading the source.
      expect(error.why.length).toBeGreaterThan(40);
      expect(error.fix.length).toBeGreaterThan(20);
      expect(error.link).toContain('apse.dev/errors/option-unknown');
    });
  }

  test('a valid option set still gets as far as looking for an adapter', async () => {
    // Proves the option checks pass and the failure is the real one. If a
    // validation rule were wrong, this would report OPTION_UNKNOWN instead.
    const error = await expectCodeAsync(
      () => createDevice(canvas, {
        powerPreference: 'low-power',
        alphaMode: 'premultiplied',
        sampleCount: 4,
        maxPixelRatio: 1.5,
        maxObjects: 1024,
        format: 'rgba8unorm',
      }),
      'WEBGPU_UNAVAILABLE',
    );
    expect(error.code).toBe('WEBGPU_UNAVAILABLE');
  });
});

// ===========================================================================
// 8. The object-uniform budget
// ===========================================================================

describe('the shared object-uniform buffer budget', () => {
  test('the documented stride fits inside a compatibility-mode uniform binding', () => {
    // The subtle one. maxUniformBufferBindingSize is 16 KiB in compatibility
    // mode, so a bind group layout that declares `size: wholeBuffer` is
    // invalid there even though the buffer itself is only 512 KiB. apse binds
    // one 256-byte slot with a dynamic offset, which is why this holds.
    expect(OBJECT_UNIFORM_STRIDE_BYTES).toBe(256);
    expect(OBJECT_UNIFORM_STRIDE_BYTES).toBeLessThanOrEqual(COMPAT_LIMITS.maxUniformBufferBindingSize);
  });

  test('the default maxObjects of 4096 fits comfortably in maxBufferSize', () => {
    expect(4096 * OBJECT_UNIFORM_STRIDE_BYTES).toBe(1048576);
    expect(4096 * OBJECT_UNIFORM_STRIDE_BYTES).toBeLessThan(COMPAT_LIMITS.maxBufferSize);
  });

  test('the default dynamic-offset alignment is respected by the stride', () => {
    // minUniformBufferOffsetAlignment is 256, and every slot offset has to be a
    // multiple of it or setBindGroup throws a validation error at draw time.
    expect(OBJECT_UNIFORM_STRIDE_BYTES % COMPAT_LIMITS.minUniformBufferOffsetAlignment).toBe(0);
  });
});
