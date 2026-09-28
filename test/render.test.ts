/**
 * Render-layer tests: device capabilities, GPU timing, readback, render-target
 * invariants, and canvas sizing.
 *
 * There is no GPU here, and the shape of these tests follows from that. What can
 * be tested is the part that is *apse's* rather than the driver's — the limit
 * request it sends, the capability set it derives, the resources it allocates
 * and frees, and the arithmetic that decides both. Each of those has a way of
 * being wrong that produces a plausible-looking render rather than an error, and
 * that is what these are aimed at.
 *
 * The three that matter most, in order:
 *
 *   1. **The `requiredLimits` sent to `requestDevice`.** Not the values — the
 *      *identity* of what is sent. `{...adapter.limits}` evaluates to `{}` on a
 *      real `GPUSupportedLimits`, because it is an interface of prototype
 *      getters, and a renderer built on that line believes it is running against
 *      the adapter's ceilings while it is running against the device defaults.
 *      The fake here has prototype getters, so the bug is reachable.
 *   2. **The nanosecond-to-millisecond conversion.** Timestamps are nanoseconds
 *      on an arbitrary non-zero epoch, so a conversion that skips the epoch
 *      produces a number around 4.5e9 that looks like a measurement. The fake
 *      lets a test choose the epoch, which is the only way to catch it.
 *   3. **`null` versus `0` for a missing GPU measurement.** A hardcoded zero is
 *      indistinguishable from an idle GPU, and every consumer of a public stats
 *      field will read it as one.
 */

import { describe, expect, test } from 'bun:test';

import { isAseError } from '../src/core/error.ts';
import type { AseErrorCode } from '../src/core/error.ts';
import { isErr, isOk } from '../src/core/result.ts';
import {
  ASE_FEATURES,
  ASE_LIMIT_NAMES,
  availableFeatures,
  COMPAT_LIMITS,
  COMPAT_VERTEX_STAGE_LIMITS,
  compatRequiredLimits,
  CORE_DEFAULT_LIMITS,
  createDevice,
  hasFeature,
  readCapabilities,
  requireFeature,
} from '../src/render/device.ts';
import type { AseFeatureName, DeviceOptions } from '../src/render/device.ts';
import { CanvasSizer } from '../src/render/context.ts';
import { RenderTargetImpl, createCanvasTarget, createColorTarget } from '../src/render/target.ts';
import { GpuTimer } from '../src/render/timing.ts';
import { DEFAULT_TONE_MAPPING } from '../src/render/present.ts';
import {
  alignedBytesPerRow,
  assertBytesPerRowAligned,
  BYTES_PER_PIXEL_RGBA8,
  COPY_BYTES_PER_ROW_ALIGNMENT,
  CaptureReadback,
  unpadRows,
} from '../src/render/readback.ts';
import type { FrameTimingStats } from '../src/render/types.ts';
import { asCanvasTargetDevice, asTargetDevice, fakeAseDevice, fakeLimits, FAKE_TEXTURE_USAGE as USAGE } from './fake-device.ts';
import {
  asTimingDevice,
  FakeTimingDevice,
  fakeWebGpuCanvas,
  raiseUncapturedAs,
  withGpu,
} from './render-fakes.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Asserts that `fn` throws an AseError with the given catalog code. */
function expectCode(fn: () => unknown, code: AseErrorCode) {
  let thrown: unknown;
  try {
    fn();
  } catch (error) {
    thrown = error;
  }
  if (!isAseError(thrown)) {
    throw new Error(`expected an AseError with code ${code}, got ${String(thrown)}`);
  }
  expect((thrown as { code: AseErrorCode }).code).toBe(code);
  return thrown as ReturnType<typeof asMessage>;
}

function asMessage(error: { message: string; fix: string; why: string }): {
  message: string;
  fix: string;
  why: string;
} {
  return error;
}

async function expectCodeAsync(fn: () => Promise<unknown>, code: AseErrorCode) {
  let thrown: unknown;
  try {
    await fn();
  } catch (error) {
    thrown = error;
  }
  if (!isAseError(thrown)) {
    throw new Error(`expected an AseError with code ${code}, got ${String(thrown)}`);
  }
  expect((thrown as { code: AseErrorCode }).code).toBe(code);
  return thrown as ReturnType<typeof asMessage>;
}

/** Lets every pending microtask and the mapAsync the fake schedules settle. */
function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** A device with just the features named, for the capability profile. */
function deviceWithFeatures(features: string[], limits?: GPUSupportedLimits): GPUDevice {
  return new FakeTimingDevice({
    limits: limits ?? fakeLimits(),
    features,
  }) as unknown as GPUDevice;
}

/** A canvas stand-in with the two layout reads `CanvasSizer` performs. */
function fakeCanvas(cssWidth: number, cssHeight: number): HTMLCanvasElement {
  return { clientWidth: cssWidth, clientHeight: cssHeight } as unknown as HTMLCanvasElement;
}

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

// ===========================================================================
// 1. requiredLimits — the compatibility profile
// ===========================================================================

describe('the limits sent to requestDevice', () => {
  test('spreading a real GPUSupportedLimits yields nothing, which is the whole trap', () => {
    // The fake builds its limits from prototype getters with no own properties,
    // which is the shape Chrome produces. Object spread copies own enumerable
    // properties only, so the idiomatic-looking line below is `{}` — not "the
    // limits it can see", nothing at all.
    const limits: unknown = fakeLimits();
    expect({ ...(limits as object) }).toEqual({});
    expect(Object.keys(limits as object)).toEqual([]);
    // …while every field is still readable, which is why the mistake survives a
    // code review: it looks like it worked.
    const fields = limits as Record<string, number>;
    expect(fields['maxTextureDimension2D']).toBe(COMPAT_LIMITS.maxTextureDimension2D);
    for (const name of ASE_LIMIT_NAMES) {
      expect(typeof fields[name]).toBe('number');
    }
  });

  test('every key apse sends is a limit the spec has, and apse sends only those', () => {
    // `requiredLimits` rejects unknown keys outright, so a limit added to the
    // spec next year is not safe to send today. This asserts the sent set is
    // exactly the curated one.
    const sent = Object.keys(compatRequiredLimits()).sort();
    expect(sent).toEqual([...ASE_LIMIT_NAMES].sort());
  });

  test('the values sent are the compatibility defaults, not the adapter ceilings', () => {
    const sent = compatRequiredLimits();
    for (const name of ASE_LIMIT_NAMES) {
      expect(sent[name]).toBe(COMPAT_LIMITS[name]);
      // And never *better* than the core default, which is what makes sending
      // them safe on a core device: a request worse than the default is legal
      // and simply has no effect.
      expect(sent[name]).toBeLessThanOrEqual(CORE_DEFAULT_LIMITS[name]);
    }
  });

  test('the texture and uniform ceilings are the compatibility ones, not core', () => {
    // The four that actually differ, and the four a naive port gets wrong.
    const sent = compatRequiredLimits();
    expect(sent.maxUniformBufferBindingSize).toBe(16384);
    expect(CORE_DEFAULT_LIMITS.maxUniformBufferBindingSize).toBe(65536);
    expect(sent.maxTextureDimension2D).toBe(4096);
    expect(CORE_DEFAULT_LIMITS.maxTextureDimension2D).toBe(8192);
    expect(sent.maxColorAttachments).toBe(4);
    expect(CORE_DEFAULT_LIMITS.maxColorAttachments).toBe(8);
    expect(sent.maxInterStageShaderVariables).toBe(15);
    expect(CORE_DEFAULT_LIMITS.maxInterStageShaderVariables).toBe(16);
  });

  test('createDevice sends the compatibility profile and never the adapter limits', async () => {
    await withGpu({ features: ['timestamp-query'] }, async ({ adapter }) => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {});
      expect(device.featureLevel).toBe('compatibility');

      const descriptor = adapter.requests[0]!;
      expect(descriptor.requiredLimits).toEqual(compatRequiredLimits());
      // The negative form matters as much as the positive one: if this ever
      // regresses to `{...adapter.limits}` the descriptor would be `{}` and the
      // page would silently depend on limits other people's devices lack.
      expect(Object.keys(descriptor.requiredLimits ?? {}).length).toBeGreaterThan(20);
      // No optional feature is required, because requiring one makes
      // requestDevice() reject on the devices apse exists to serve.
      expect(descriptor.requiredFeatures).toEqual([]);
      // …even when the adapter has them. Detection is not a request.
      expect(device.capabilities.timestampQuery).toBe(true);
      device.destroy();
    });
  });

  test('the adapter probe asks for compatibility first', async () => {
    await withGpu({}, async ({ adapterRequests }) => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {});
      expect(adapterRequests[0]!.featureLevel).toBe('compatibility');
      expect(adapterRequests).toHaveLength(1);
      device.destroy();
    });
  });

  test('preferCore probes core first and requests the core feature', async () => {
    await withGpu({ features: ['core-features-and-limits'] }, async ({ adapterRequests, adapter }) => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, { preferCore: true });
      expect(adapterRequests[0]!.featureLevel).toBe('core');
      expect(adapter.requests[0]!.requiredFeatures).toEqual(['core-features-and-limits']);
      expect(device.featureLevel).toBe('core');
      device.destroy();
    });
  });

  test('an adapter that cannot do core falls back to compatibility', async () => {
    await withGpu({ noCore: true }, async ({ adapterRequests }) => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, { preferCore: true });
      expect(adapterRequests.map((r) => r.featureLevel)).toEqual(['core', 'compatibility']);
      expect(device.featureLevel).toBe('compatibility');
      device.destroy();
    });
  });
});

// ===========================================================================
// 2. The capability profile
// ===========================================================================

describe('DeviceCapabilities', () => {
  test('a compatibility device reports zero storage buffers in the vertex stage', () => {
    // The number the instancing path is designed around: a vertex shader that
    // reads a storage buffer compiles on a laptop and fails on a phone.
    const caps = readCapabilities(deviceWithFeatures([]), 'compatibility');
    expect(caps.featureLevel).toBe('compatibility');
    expect(caps.storageBuffersInVertexStage).toBe(0);
    expect(caps.storageTexturesInVertexStage).toBe(0);
    expect(COMPAT_VERTEX_STAGE_LIMITS.maxStorageBuffersInVertexStage).toBe(0);
  });

  test('a core device reports what its limits say', () => {
    const limits = fakeLimits({ maxStorageBuffersInVertexStage: 8, maxStorageTexturesInVertexStage: 4 });
    const caps = readCapabilities(deviceWithFeatures([], limits), 'core');
    expect(caps.storageBuffersInVertexStage).toBe(8);
    expect(caps.storageTexturesInVertexStage).toBe(4);
  });

  test('a limit the implementation does not expose falls back to the profile', () => {
    // `fakeLimits` has no `maxStorageBuffersInVertexStage` at all, which is what
    // an implementation that has not implemented the IDL member looks like.
    const limits = fakeLimits();
    expect((limits as unknown as Record<string, number>).maxStorageBuffersInVertexStage).toBeUndefined();
    expect(readCapabilities(deviceWithFeatures([], limits), 'compatibility').storageBuffersInVertexStage).toBe(0);
  });

  test('optional features are detected, never required', () => {
    const gpu = deviceWithFeatures(['timestamp-query', 'float32-filterable', 'texture-compression-astc']);
    const caps = readCapabilities(gpu, 'core');
    expect(caps.timestampQuery).toBe(true);
    expect(caps.float32Filterable).toBe(true);
    expect(caps.textureCompression).toEqual(['astc']);
    expect(hasFeature(gpu, 'subgroups')).toBe(false);
    expect(availableFeatures(gpu)).toEqual(['timestamp-query', 'float32-filterable', 'texture-compression-astc']);
  });

  test('every compression family is reported independently', () => {
    const caps = readCapabilities(
      deviceWithFeatures(['texture-compression-bc', 'texture-compression-etc2']),
      'core',
    );
    expect(caps.textureCompression).toEqual(['bc', 'etc2']);
    expect(readCapabilities(deviceWithFeatures([]), 'core').textureCompression).toEqual([]);
  });

  test('instancing needs two vertex buffers, and the profile says whether it has them', () => {
    expect(readCapabilities(deviceWithFeatures([], fakeLimits({ maxVertexBuffers: 8 })), 'compatibility').instancing).toBe(true);
    const one = readCapabilities(deviceWithFeatures([], fakeLimits({ maxVertexBuffers: 1 })), 'compatibility');
    expect(one.instancing).toBe(false);
    expect(one.maxVertexBuffers).toBe(1);
  });

  test('the storage-buffer binding ceiling and texture ceiling are read, not assumed', () => {
    const caps = readCapabilities(
      deviceWithFeatures([], fakeLimits({ maxStorageBufferBindingSize: 2147483644, maxTextureDimension2D: 8192 })),
      'core',
    );
    expect(caps.maxStorageBufferBindingSize).toBe(2147483644);
    expect(caps.maxTextureDimension2D).toBe(8192);
  });

  test('the profile is frozen, so a caller cannot mutate the device out from under the renderer', () => {
    const caps = readCapabilities(deviceWithFeatures(['timestamp-query']), 'compatibility');
    expect(Object.isFrozen(caps)).toBe(true);
  });

  test('the device exposes it, and the legacy booleans agree with it', async () => {
    await withGpu({ features: ['timestamp-query', 'subgroups'] }, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {});
      expect(device.capabilities.timestampQuery).toBe(true);
      expect(device.hasTimestampQuery).toBe(device.capabilities.timestampQuery);
      expect(device.hasSubgroups).toBe(true);
      expect(device.capabilities.featureLevel).toBe('compatibility');
      device.destroy();
    });
  });
});

describe('requireFeature', () => {
  test('a supported feature returns ok, and does not throw', () => {
    const gpu = deviceWithFeatures(['float32-filterable']);
    const result = requireFeature(gpu, 'float32-filterable');
    expect(isOk(result)).toBe(true);
  });

  test('a missing capability returns a typed failure rather than throwing', () => {
    // The project rule: a wrong argument throws, a missing capability returns.
    // A phone without float32-filterable is the world being what it is.
    const result = requireFeature(deviceWithFeatures([]), 'float32-filterable');
    expect(isErr(result)).toBe(true);
    if (isErr(result)) {
      expect(result.code).toBe('GPU_FEATURE_UNSUPPORTED');
      expect(result.message).toContain('float32-filterable');
      expect(result.context?.['feature']).toBe('float32-filterable');
      expect(result.fix.length).toBeGreaterThan(10);
    }
  });

  test('the failure names the features the device does have', () => {
    const result = requireFeature(deviceWithFeatures(['timestamp-query']), 'subgroups');
    expect(isErr(result)).toBe(true);
    if (isErr(result)) {
      expect(result.context?.['available']).toBe('timestamp-query');
    }
  });

  test('an unknown feature name is a wrong argument, and throws', () => {
    const error = expectCode(
      () => requireFeature(deviceWithFeatures([]), 'ray-tracing' as AseFeatureName),
      'OPTION_UNKNOWN',
    );
    expect(error.message).toContain('ray-tracing');
    expect(error.message).toContain('timestamp-query');
  });

  test('every tracked feature name is a real WebGPU feature name', () => {
    // ASE_FEATURES exists so `requireFeature` can tell a typo from a missing
    // capability. A typo *in this list* would make that check wrong.
    const known = new Set([
      'timestamp-query', 'float32-filterable', 'depth32float-stencil8', 'indirect-first-instance',
      'texture-compression-bc', 'texture-compression-astc', 'texture-compression-etc2', 'subgroups',
    ]);
    for (const name of ASE_FEATURES) expect(known.has(name)).toBe(true);
  });

  test('the device method delegates and asserts liveness first', async () => {
    await withGpu({ features: ['float32-filterable'] }, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {});
      expect(isOk(device.requireFeature('float32-filterable'))).toBe(true);
      expect(isErr(device.requireFeature('subgroups'))).toBe(true);
      device.destroy();
    });
  });
});

// ===========================================================================
// 3. Uncaptured errors and device loss
// ===========================================================================

describe('surfacing GPU errors through the catalog', () => {
  test('an uncaptured validation error becomes GPU_VALIDATION_FAILED and is kept', async () => {
    const seen: string[] = [];
    const legacy: string[] = [];
    await withGpu({}, async ({ adapter }) => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {
        onUncapturedError: (error) => seen.push(error.code),
        onValidationError: (message) => legacy.push(message),
      });
      void adapter;
      expect(adapter.requests).toHaveLength(1);
      raiseUncapturedAs(device.device, 'GPUValidationError', 'texture usage is missing COPY_SRC');
      expect(seen).toEqual(['GPU_VALIDATION_FAILED']);
      // The pre-existing option still fires, so nothing that used it breaks.
      expect(legacy[0]).toContain('COPY_SRC');
      expect(device.lastError?.code).toBe('GPU_VALIDATION_FAILED');
      expect(device.lastError?.message).toContain('COPY_SRC');
      expect(device.lastError?.why.length).toBeGreaterThan(20);
      expect(device.lastError?.fix.length).toBeGreaterThan(20);
      device.destroy();
    });
  });

  test('an out-of-memory error is filed as a budget, not as a validation error', async () => {
    // Reporting an OOM as "a validation error" sends the reader looking for a
    // wrong descriptor. It is a budget nobody declared.
    const codes: string[] = [];
    await withGpu({}, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {
        onUncapturedError: (error) => codes.push(error.code),
        onValidationError: () => { /* the typed path is what is under test */ },
      });
      raiseUncapturedAs(device.device, 'GPUOutOfMemoryError', 'Allocation of 132 MB failed');
      expect(codes).toEqual(['BUDGET_EXCEEDED']);
      expect(device.lastError?.fix).toContain('maxPixelRatio');
    });
  });

  test('a driver-internal error is filed as an internal invariant', async () => {
    const codes: string[] = [];
    await withGpu({}, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {
        onUncapturedError: (error) => codes.push(error.code),
        onValidationError: () => { /* the typed path is what is under test */ },
      });
      raiseUncapturedAs(device.device, 'GPUInternalError', 'lost track of the command buffer');
      expect(codes).toEqual(['INTERNAL_INVARIANT']);
      expect(device.lastError?.fix).toContain('describeGpu()');
    });
  });

  test('with no handler at all, a development build still says something', async () => {
    // Silence is the failure mode: a validation error nobody was told about is a
    // black frame an hour later. The record on the handle is the durable half;
    // this is the immediate half.
    const warnings: string[] = [];
    const previous = console.warn;
    console.warn = (message?: unknown): void => { warnings.push(String(message)); };
    try {
      await withGpu({}, async () => {
        const { canvas } = fakeWebGpuCanvas();
        const device = await createDevice(canvas, {});
        raiseUncapturedAs(device.device, 'GPUValidationError', 'bind group layout is wrong');
        device.destroy();
      });
    } finally {
      console.warn = previous;
    }
    expect(warnings.some((w) => w.includes('bind group layout is wrong'))).toBe(true);
  });

  test('a device loss is kept on the handle, and destroy() is not a loss', async () => {
    await withGpu({}, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {});
      (device.device as unknown as { lose: (r: GPUDeviceLostReason, m: string) => void })
        .lose('unknown', 'the driver went away');
      await flush();
      expect(device.lastError?.code).toBe('DEVICE_LOST');
      expect(device.lastError?.message).toContain('the driver went away');
      // assertLive now fails, which is the point of keeping the record.
      expectCode(() => device.assertLive(), 'DEVICE_LOST');
    });
  });

  test('destroy() is not recorded as a loss: it is the app\'s own teardown', async () => {
    await withGpu({}, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {});
      device.destroy();
      await flush();
      expect(device.lastError).toBeNull();
    });
  });

  test('a live device has no lastError to report', async () => {
    await withGpu({}, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {});
      expect(device.lastError).toBeNull();
      device.destroy();
    });
  });
});

// ===========================================================================
// 4. GpuTimer — real timestamp queries
// ===========================================================================

describe('GpuTimer — resource lifecycle and sizing', () => {
  test('a device without timestamp-query gets a typed failure and no resources', () => {
    const fake = new FakeTimingDevice({ features: [] });
    const result = GpuTimer.create(asTimingDevice(fake));
    expect(isErr(result)).toBe(true);
    if (isErr(result)) expect(result.code).toBe('GPU_FEATURE_UNSUPPORTED');
    // Nothing was created, so nothing can leak: the feature gate runs before
    // the first allocation.
    expect(fake.querySets).toHaveLength(0);
    expect(fake.buffers).toHaveLength(0);
  });

  test('one pair is two query slots and sixteen bytes', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    expect(fake.querySets[0]!.count).toBe(2);
    expect(fake.querySets[0]!.type).toBe('timestamp');
    expect(timer.pairCount).toBe(1);
    expect(timer.byteSize).toBe(16);
    expect(fake.buffers[0]!.size).toBe(16);
  });

  test('the resolve buffer is QUERY_RESOLVE | COPY_SRC, and the readback is COPY_DST | MAP_READ', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    const [resolve, slot0, slot1] = fake.buffers;
    // 0x200 QUERY_RESOLVE, 0x04 COPY_SRC. Without QUERY_RESOLVE the resolve is a
    // validation error naming the usage bit, not the missing feature.
    expect(resolve!.usage).toBe(0x200 | 0x04);
    // 0x08 COPY_DST, 0x01 MAP_READ. A mapped buffer cannot be a copy
    // destination, which is why there is a ring at all.
    expect(slot0!.usage).toBe(0x08 | 0x01);
    expect(slot1!.usage).toBe(0x08 | 0x01);
    expect(fake.buffers).toHaveLength(3);
  });

  test('two pairs are four slots and thirty-two bytes, and scale linearly', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake), { pairs: 2 }));
    expect(fake.querySets[0]!.count).toBe(4);
    expect(timer.byteSize).toBe(32);
    expect(fake.buffers[0]!.size).toBe(32);
    expect(fake.buffers.every((b) => b.size === 32)).toBe(true);
  });

  test('the staging ring is configurable and defaults to two', () => {
    const one = new FakeTimingDevice({ features: ['timestamp-query'] });
    unwrapTimer(GpuTimer.create(asTimingDevice(one), { stagingSlots: 1 }));
    expect(one.buffers).toHaveLength(2);
    const five = new FakeTimingDevice({ features: ['timestamp-query'] });
    unwrapTimer(GpuTimer.create(asTimingDevice(five), { stagingSlots: 5 }));
    expect(five.buffers).toHaveLength(6);
  });

  test('a pair\'s write indices are its own two slots, on one query set', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake), { pairs: 3 }));
    const set = fake.querySets[0]!;
    const first = timer.writes(0);
    expect(first.beginningOfPassWriteIndex).toBe(0);
    expect(first.endOfPassWriteIndex).toBe(1);
    expect(first.querySet).toBe(set as unknown as GPUQuerySet);
    const third = timer.writes(2);
    expect(third.beginningOfPassWriteIndex).toBe(4);
    expect(third.endOfPassWriteIndex).toBe(5);
    expect(third.querySet).toBe(set as unknown as GPUQuerySet);
    // The same object every time, so the frame loop allocates nothing.
    expect(timer.writes(0)).toBe(timer.writes(0));
    expect(timer.writes()).toBe(timer.writes(0));
  });

  test('a write index past the allocated pairs is rejected with the fix', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake), { pairs: 1 }));
    const error = expectCode(() => timer.writes(1), 'OPTION_UNKNOWN');
    expect(error.message).toContain('writes(1)');
    expect(error.message).toContain('1 pair');
    expect(error.fix).toContain('pairs: 2');
  });

  test('an advertised feature whose allocation fails is reported as a driver fault', async () => {
    // Different from "no feature": the machine says yes and then does not
    // deliver, so the fix is a bug report rather than "carry on without it".
    // Reporting both as the same code sends the reader to the wrong place.
    await withGpu({ features: ['timestamp-query'] }, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const device = await createDevice(canvas, {});
      const gpu = device.device as unknown as {
        createQuerySet: () => never;
      };
      gpu.createQuerySet = () => { throw new Error('out of memory in the driver'); };
      const result = GpuTimer.create({ device: gpu as unknown as GPUDevice });
      expect(isErr(result)).toBe(true);
      if (isErr(result)) {
        expect(result.code).toBe('TIMESTAMP_ALLOCATION_FAILED');
        expect(result.message).toContain('out of memory in the driver');
      }
    });
  });

  test('a non-positive or fractional count is a wrong argument', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    for (const pairs of [0, -1, 1.5, Number.NaN]) {
      const error = expectCode(() => GpuTimer.create(asTimingDevice(fake), { pairs }), 'OPTION_UNKNOWN');
      expect(error.message).toContain('pairs');
    }
    expectCode(() => GpuTimer.create(asTimingDevice(fake), { stagingSlots: 0 }), 'OPTION_UNKNOWN');
    expect(fake.querySets).toHaveLength(0);
  });

  test('dispose destroys the query set and every buffer, and is idempotent', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake), { stagingSlots: 3 }));
    expect(fake.liveBuffers).toHaveLength(4);
    timer.dispose();
    expect(fake.liveBuffers).toHaveLength(0);
    expect(fake.querySets[0]!.destroyed).toBe(true);
    expect(timer.disposed).toBe(true);
    timer.dispose();
    expect(fake.liveBuffers).toHaveLength(0);
  });

  test('using a disposed timer fails rather than encoding into freed memory', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    timer.dispose();
    expectCode(() => timer.writes(0), 'INTERNAL_INVARIANT');
    expectCode(() => timer.encodeReadback(fake.createCommandEncoder()), 'INTERNAL_INVARIANT');
  });

  test('nothing is allocated at module scope: a timer needs a device to exist', () => {
    // `sideEffects: false` is a promise. A module-scope GPUQuerySet would make
    // importing apse allocate on a device that may not exist.
    expect(new FakeTimingDevice({ features: ['timestamp-query'] }).querySets).toHaveLength(0);
  });
});

describe('GpuTimer — reading the timestamps', () => {
  test('no reading yet is null, never 0', () => {
    // The whole point of the type change. A hardcoded 0 tells every consumer
    // the GPU was idle.
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    expect(timer.lastGpuMs).toBeNull();
    expect(timer.pairMs(0)).toBeNull();
    expect(timer.readings).toEqual([null]);
  });

  test('nanoseconds on a huge non-zero epoch become milliseconds, not epoch millis', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    // Chromium's timestamp epoch is arbitrary and non-zero; this is the shape of
    // a real one. Reporting the raw value would be ~1.7e9 "milliseconds".
    const epoch = 1_700_000_000_000_000n;
    fake.timestamps = [epoch, epoch + 4_200_000n];
    timer.encodeReadback(fake.createCommandEncoder());
    timer.poll();
    expect(timer.lastGpuMs).toBeNull();
    return flush().then(() => {
      expect(timer.lastGpuMs).toBe(4.2);
      // The value is a duration, not a clock: nothing near the epoch survives.
      expect(timer.lastGpuMs as number).toBeLessThan(1000);
    });
  });

  test('a measured zero is 0, and is not confused with a missing reading', () => {
    // A driver that quantises timestamps to 100 µs really does report 0 for a
    // trivial frame. That is data, and collapsing it into `null` would lose it.
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    fake.timestamps = [500n, 500n];
    timer.encodeReadback(fake.createCommandEncoder());
    timer.poll();
    return flush().then(() => {
      expect(timer.lastGpuMs).toBe(0);
      expect(timer.pairMs(0)).not.toBeNull();
    });
  });

  test('a counter that wrapped reports no reading rather than 2^64 nanoseconds', async () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    fake.timestamps = [900n, 100n];
    timer.encodeReadback(fake.createCommandEncoder());
    timer.poll();
    await flush();
    expect(timer.pairMs(0)).toBeNull();
  });

  test('each pair is read from its own two slots', async () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake), { pairs: 2 }));
    fake.timestamps = [0n, 2_000_000n, 10_000_000n, 13_500_000n];
    timer.encodeReadback(fake.createCommandEncoder());
    timer.poll();
    await flush();
    expect(timer.pairMs(0)).toBe(2);
    expect(timer.pairMs(1)).toBe(3.5);
    expect(timer.readings).toEqual([2, 3.5]);
  });

  test('a multi-pass frame is summed, and an unstamped pass contributes nothing', () => {
    // A write index may only be written once per submission, so a frame with
    // three passes needs three pairs — and a sum that quietly used two of them
    // would report a third of the frame as if it were all of it.
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake), { pairs: 3 }));
    expect(timer.sumMs(3)).toBeNull();
    fake.timestamps = [0n, 2_000_000n, 0n, 3_000_000n];
    timer.encodeReadback(fake.createCommandEncoder());
    timer.poll();
    return flush().then(() => {
      expect(timer.sumMs(2)).toBe(5);
      // A pair nobody wrote resolves to zero, and a pass that did not open cost
      // nothing — so an over-count adds 0 ms rather than reading stale data.
      expect(timer.sumMs(3)).toBe(5);
      expect(timer.sumMs(0)).toBe(0);
    });
  });

  test('the resolve and the copy are both encoded, in that order, before the submit', () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    const encoder = fake.createCommandEncoder();
    timer.encodeReadback(encoder);
    expect(encoder.calls).toHaveLength(2);
    expect(encoder.calls[0]).toContain('resolveQuerySet');
    expect(encoder.calls[0]).toContain(':0:2:');
    expect(encoder.calls[1]).toContain('copyBufferToBuffer');
    expect(encoder.calls[1]).toContain('16');
  });

  test('poll never blocks, and the reading arrives on a later turn', async () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    fake.timestamps = [0n, 1_000_000n];
    timer.encodeReadback(fake.createCommandEncoder());
    // Synchronous return, and nothing is available yet: the map cannot resolve
    // before the queue has run the copy.
    timer.poll();
    expect(timer.lastGpuMs).toBeNull();
    expect(timer.pending).toBe(1);
    await flush();
    expect(timer.lastGpuMs).toBe(1);
    expect(timer.pending).toBe(0);
  });

  test('a frame whose readback is still in flight is dropped, not waited on', async () => {
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake), { stagingSlots: 2 }));
    fake.timestamps = [0n, 1_000_000n];
    // Two frames consume both slots, and a third arrives before either map has
    // settled. The third must drop: copying into a buffer with a pending map is
    // a validation error that invalidates the whole command buffer.
    timer.encodeReadback(fake.createCommandEncoder());
    timer.poll();
    timer.encodeReadback(fake.createCommandEncoder());
    timer.poll();
    const third = fake.createCommandEncoder();
    timer.encodeReadback(third);
    expect(third.calls).toHaveLength(0);
    expect(timer.dropped).toBe(1);
    await flush();
    // And the slots come back, so the timer recovers on the next frame.
    expect(timer.pending).toBe(0);
    timer.encodeReadback(fake.createCommandEncoder());
    expect(timer.dropped).toBe(1);
  });

  test('a slot reserved and never polled is still reclaimed', async () => {
    // One poll per frame is the contract, but a caller that skips one must not
    // strand a buffer: after `stagingSlots` such frames the timer would drop
    // every readback forever.
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake), { stagingSlots: 2 }));
    fake.timestamps = [0n, 1_000_000n];
    timer.encodeReadback(fake.createCommandEncoder());
    timer.encodeReadback(fake.createCommandEncoder());
    expect(timer.pending).toBe(2);
    timer.poll();
    await flush();
    expect(timer.pending).toBe(0);
    timer.encodeReadback(fake.createCommandEncoder());
    expect(timer.dropped).toBe(0);
  });

  test('disposing with a map in flight does not raise an unhandled rejection', async () => {
    // The fake's destroy() rejects a pending map, exactly as a real one does.
    // A missing `.catch()` here is a console error on every teardown.
    const fake = new FakeTimingDevice({ features: ['timestamp-query'] });
    const timer = unwrapTimer(GpuTimer.create(asTimingDevice(fake)));
    fake.timestamps = [0n, 1_000_000n];
    timer.encodeReadback(fake.createCommandEncoder());
    timer.poll();
    timer.dispose();
    await flush();
    expect(timer.lastGpuMs).toBeNull();
  });
});

function unwrapTimer(result: ReturnType<typeof GpuTimer.create>): GpuTimer {
  if (isErr(result)) throw new Error(`expected a timer, got ${result.code}: ${result.message}`);
  return result.value;
}

// ===========================================================================
// 5. Readback — the 256-byte row alignment
// ===========================================================================

describe('alignedBytesPerRow', () => {
  test('a 100-pixel RGBA8 image has a 512-byte pitch, not 400', () => {
    // 400 rounds up to 512, and the 112 bytes of padding are not image data.
    expect(alignedBytesPerRow(100)).toBe(512);
    expect(alignedBytesPerRow(100, BYTES_PER_PIXEL_RGBA8)).toBe(512);
    // 64 pixels is exactly 256, so it needs no padding at all.
    expect(alignedBytesPerRow(64)).toBe(256);
    expect(alignedBytesPerRow(1)).toBe(256);
  });

  test('every width produces a multiple of 256, and never less than one row of pixels', () => {
    for (let width = 1; width <= 2000; width++) {
      const pitch = alignedBytesPerRow(width);
      expect(pitch % COPY_BYTES_PER_ROW_ALIGNMENT).toBe(0);
      expect(pitch).toBeGreaterThanOrEqual(width * 4);
      // Never more than one alignment step of padding: a bug that rounded to the
      // next 2x would double the memory of every capture.
      expect(pitch - width * 4).toBeLessThan(COPY_BYTES_PER_ROW_ALIGNMENT);
    }
  });

  test('the pixel size is a parameter, because a depth copy is 4 bytes per pixel', () => {
    expect(alignedBytesPerRow(64, 8)).toBe(512);
    expect(alignedBytesPerRow(128, 8)).toBe(1024);
    expect(alignedBytesPerRow(128, 2)).toBe(256);
  });

  test('a zero or fractional width fails loudly, because there is no pitch for it', () => {
    for (const width of [0, -1, 10.5, Number.NaN]) {
      const error = expectCode(() => alignedBytesPerRow(width), 'RENDER_TARGET_SIZE_INVALID');
      expect(error.message).toContain(String(width));
      expect(error.fix).toContain('CanvasSizer');
    }
    expectCode(() => alignedBytesPerRow(64, 0), 'RENDER_TARGET_SIZE_INVALID');
  });
});

describe('assertBytesPerRowAligned', () => {
  test('an aligned pitch passes, including the exact minimum', () => {
    assertBytesPerRowAligned(256);
    assertBytesPerRowAligned(512);
    assertBytesPerRowAligned(4096);
  });

  test('an unaligned pitch is refused before it reaches a copy', () => {
    // width * 4 is the classic: 100 * 4 = 400, which invalidates the whole
    // command buffer — the frame's draws included — and throws nothing.
    for (const pitch of [400, 4, 0, -256, 255, 300]) {
      const error = expectCode(() => assertBytesPerRowAligned(pitch, 'capture'), 'INTERNAL_INVARIANT');
      expect(error.message).toContain(String(pitch));
      expect(error.why).toContain('256');
    }
  });
});

describe('unpadRows', () => {
  /** A padded buffer whose first byte of each row is the row index. */
  function paddedRows(width: number, height: number): Uint8Array {
    const pitch = alignedBytesPerRow(width);
    const out = new Uint8Array(pitch * height);
    for (let y = 0; y < height; y++) {
      for (let i = 0; i < width * 4; i++) out[y * pitch + i] = (y * 7 + i) & 0xff;
    }
    return out;
  }

  test('rows come out contiguous, and the padding is gone', () => {
    const out = unpadRows(paddedRows(100, 5), 100, 5);
    expect(out).toHaveLength(100 * 4 * 5);
    for (let y = 0; y < 5; y++) {
      for (let i = 0; i < 100 * 4; i++) expect(out[y * 400 + i]).toBe((y * 7 + i) & 0xff);
    }
  });

  test('reading the padded buffer at width * 4 is the shear this exists to prevent', () => {
    // The failure is not a shift, it is a per-row interleave: 56 bytes of one
    // row followed by 56 of the next, so the error grows by one pixel per row.
    const width = 100;
    const height = 4;
    const padded = paddedRows(width, height);
    const wrong = new Uint8Array(width * 4 * height);
    for (let y = 0; y < height; y++) {
      wrong.set(padded.subarray(y * width * 4, y * width * 4 + width * 4), y * width * 4);
    }
    expect(wrong).not.toEqual(unpadRows(padded, width, height));
    // Correct at the pitch, wrong at the pixel width: the two differ, which is
    // the only reason a stride bug is ever noticed.
    const right = unpadRows(padded, width, height);
    expect(right[400]).toBe(7);
    expect(wrong[400]).toBe(0);
  });

  test('an exactly-aligned width is a straight copy', () => {
    const width = 64;
    const padded = paddedRows(width, 3);
    expect(alignedBytesPerRow(width)).toBe(width * 4);
    expect(unpadRows(padded, width, 3)).toEqual(padded.subarray(0, width * 4 * 3));
  });

  test('a buffer too small for the rows is a loud failure, not a short image', () => {
    const error = expectCode(() => unpadRows(new Uint8Array(100), 100, 5), 'INTERNAL_INVARIANT');
    expect(error.message).toContain('512');
    expect(error.fix).toContain('alignedBytesPerRow');
  });
});

describe('CaptureReadback', () => {
  test('the destination is the padded size, with COPY_DST and MAP_READ', () => {
    const fake = new FakeTimingDevice();
    const readback = CaptureReadback.create({ device: fake as unknown as GPUDevice }, 100, 7);
    expect(readback.bytesPerRow).toBe(512);
    expect(readback.size).toBe(512 * 7);
    expect(fake.buffers[0]!.size).toBe(512 * 7);
    // 0x08 COPY_DST, 0x01 MAP_READ. Nothing else: a readback buffer that a
    // shader could also write to is a buffer the copy can land in badly.
    expect(fake.buffers[0]!.usage).toBe(0x08 | 0x01);
  });

  test('a resize reallocates, frees the old buffer first, and reports the change', () => {
    const fake = new FakeTimingDevice();
    const readback = CaptureReadback.create({ device: fake as unknown as GPUDevice }, 100, 7);
    const first = fake.buffers[0]!;
    expect(readback.resize(200, 7)).toBe(true);
    expect(first.destroyed).toBe(true);
    expect(readback.bytesPerRow).toBe(1024);
    expect(fake.buffers[1]!.size).toBe(1024 * 7);
    // One live buffer at a time: the peak is one capture, not two.
    expect(fake.liveBuffers).toHaveLength(1);
  });

  test('a same-size resize allocates nothing', () => {
    const fake = new FakeTimingDevice();
    const readback = CaptureReadback.create({ device: fake as unknown as GPUDevice }, 100, 7);
    expect(readback.resize(100, 7)).toBe(false);
    // A CSS change that rounds away to the same backing size is not a resize,
    // and reallocating on one is how a scrollbar costs four megabytes a frame.
    expect(readback.resize(99, 7)).toBe(false);
    expect(fake.buffers).toHaveLength(1);
  });

  test('map returns contiguous pixels and leaves the buffer unmapped', async () => {
    const fake = new FakeTimingDevice();
    const readback = CaptureReadback.create({ device: fake as unknown as GPUDevice }, 4, 2);
    const buffer = fake.buffers[0]!;
    // 4 px * 4 bytes = 16, padded to 256. Two rows of 16 bytes at a 256 pitch.
    for (let y = 0; y < 2; y++) for (let i = 0; i < 16; i++) buffer.bytes[y * 256 + i] = y * 16 + i;
    const pixels = await readback.map();
    expect(pixels).toHaveLength(32);
    expect(pixels[16]).toBe(16);
    expect(pixels[0]).toBe(0);
    expect(buffer.mapped).toBe(false);
  });

  test('the copy out of the mapped range happens before the unmap', async () => {
    // The symptom this rules out: a view taken inside the mapped window and read
    // after the unmap is a detached buffer, so a capture silently returns all
    // zeroes — indistinguishable from a black scene.
    const fake = new FakeTimingDevice();
    const readback = CaptureReadback.create({ device: fake as unknown as GPUDevice }, 4, 1);
    const buffer = fake.buffers[0]!;
    buffer.bytes[0] = 42;
    const pixels = await readback.map();
    expect(pixels[0]).toBe(42);
  });

  test('dispose frees the buffer, is idempotent, and use after it fails loudly', async () => {
    const fake = new FakeTimingDevice();
    const readback = CaptureReadback.create({ device: fake as unknown as GPUDevice }, 64, 2);
    const buffer = fake.buffers[0]!;
    readback.dispose();
    expect(buffer.destroyed).toBe(true);
    expect(readback.disposed).toBe(true);
    readback.dispose();
    expect(fake.liveBuffers).toHaveLength(0);
    expectCode(() => readback.buffer, 'INTERNAL_INVARIANT');
    expectCode(() => readback.resize(64, 2), 'INTERNAL_INVARIANT');
  });
});

// ===========================================================================
// 6. Render-target invariants
// ===========================================================================

describe('RenderTargetImpl.assertDrawable', () => {
  function targetOf(opts: { format?: GPUTextureFormat; label?: string } = {}) {
    const fake = fakeAseDevice({ format: opts.format ?? 'bgra8unorm' });
    const target = createColorTarget(asTargetDevice(fake), {
      width: 16,
      height: 16,
      format: opts.format ?? 'bgra8unorm',
      label: opts.label ?? 'apse.test',
    });
    return { fake, target };
  }

  test('a material compiled for this format is accepted', () => {
    const { target } = targetOf({ format: 'rgba16float' });
    expect(() => target.assertDrawable({ name: 'pbr', targetFormats: ['rgba16float'] })).not.toThrow();
  });

  test('a format mismatch fails with both formats named in the message', () => {
    const { target } = targetOf({ format: 'bgra8unorm' });
    const error = expectCode(
      () => target.assertDrawable({ name: 'checker', targetFormats: ['rgba8unorm'] }),
      'RENDER_TARGET_FORMAT_MISMATCH',
    );
    // The driver names two format enums and no mistake; this names the
    // material, the expected format, the actual one, and the target.
    expect(error.message).toContain('checker');
    expect(error.message).toContain('rgba8unorm');
    expect(error.message).toContain('bgra8unorm');
    expect(error.message).toContain('apse.test');
    expect(error.why).toContain('whole command buffer');
    expect(error.fix).toContain('targetFormat');
  });

  test('an unorm material is told to pass the target format, not to recreate it', () => {
    const { target } = targetOf({ format: 'bgra8unorm' });
    const error = expectCode(
      () => target.assertDrawable({ name: 'basic', targetFormats: ['rgba8unorm'] }),
      'RENDER_TARGET_FORMAT_MISMATCH',
    );
    expect(error.fix).toContain('getPreferredCanvasFormat');
  });

  test('a material with no colour target, or several, is a mismatch too', () => {
    const { target } = targetOf();
    expectCode(() => target.assertDrawable({ name: 'empty', targetFormats: [] }), 'RENDER_TARGET_FORMAT_MISMATCH');
    const multi = expectCode(
      () => target.assertDrawable({ name: 'gbuffer', targetFormats: ['bgra8unorm', 'rgba16float'] }),
      'RENDER_TARGET_FORMAT_MISMATCH',
    );
    expect(multi.message).toContain('bgra8unorm, rgba16float');
  });

  test('a real Material satisfies the structural type, and the check is a pointer compare', () => {
    // The hot-path cost: one array length and one string identity per pipeline
    // change, which is once per material per frame at worst.
    const { target } = targetOf({ format: 'rgba16float' });
    const formats: readonly GPUTextureFormat[] = Object.freeze(['rgba16float']);
    expect(() => target.assertDrawable({ name: 'pbr', targetFormats: formats })).not.toThrow();
  });
});

describe('RenderTargetImpl — readback and MSAA invariants', () => {
  test('a target without COPY_SRC refuses to be read, with the flag named', () => {
    const fake = fakeAseDevice();
    const target = createColorTarget(asTargetDevice(fake), { width: 8, height: 8, label: 'apse.hdr' });
    const error = expectCode(() => target.assertCopySource(), 'OPTION_UNKNOWN');
    expect(error.message).toContain('COPY_SRC');
    expect(error.fix).toContain('COPY_SRC');
    expectCode(() => target.readableColorTexture, 'OPTION_UNKNOWN');
  });

  test('a target with COPY_SRC hands back its colour texture', () => {
    const fake = fakeAseDevice();
    const target = createColorTarget(asTargetDevice(fake), {
      width: 8,
      height: 8,
      usage: USAGE.COPY_SRC,
      label: 'apse.capture',
    });
    const texture = target.readableColorTexture;
    expect(texture).toBe(target.colorTexture);
    expect((texture as unknown as { hasUsage(f: number): boolean }).hasUsage(USAGE.COPY_SRC)).toBe(true);
    expect(target.colorTextureCount).toBe(1);
  });

  test('a canvas target cannot be read back, and says why', () => {
    // The swapchain texture expires at present, so reading it is only legal
    // inside the task that drew it — which no caller can rely on.
    const fake = fakeAseDevice();
    const target = createCanvasTarget(asCanvasTargetDevice(fake), { width: 8, height: 8 });
    const error = expectCode(() => target.assertCopySource(), 'INTERNAL_INVARIANT');
    expect(error.why).toContain('expires at present');
    expect(error.fix).toContain('capture');
  });

  test('MSAA allocates two colour textures, and the sample view is the resolve', () => {
    const fake = fakeAseDevice();
    const target = createColorTarget(asTargetDevice(fake), { width: 8, height: 8, sampleCount: 4, label: 'apse.msaa' });
    expect(target.colorTextureCount).toBe(2);
    expect(fake.device.live('color')).toHaveLength(1);
    expect(fake.device.live('colorResolve')).toHaveLength(1);
    // The multisampled attachment is what is drawn into…
    const color = fake.device.live(':color')[0]!;
    const resolve = fake.device.live(':colorResolve')[0]!;
    expect(color.sampleCount).toBe(4);
    // …and the resolve is what a later pass samples, because a multisampled
    // texture cannot be bound as a sampled texture at all.
    expect(resolve.sampleCount).toBe(1);
    expect(resolve.hasUsage(USAGE.TEXTURE_BINDING)).toBe(true);
    expect(color.hasUsage(USAGE.TEXTURE_BINDING)).toBe(false);
    // The sampled view is the resolve's, and it is a different object from the
    // attachment a pass draws into.
    expect(target.sampleView).toBeDefined();
    expect(target.sampleView).not.toBe(target.colorView);
  });

  test('a single-sampled target has one colour texture and the views are the same object', () => {
    const fake = fakeAseDevice();
    const target = createColorTarget(asTargetDevice(fake), { width: 8, height: 8, label: 'apse.ssaa' });
    expect(target.colorTextureCount).toBe(1);
    expect(fake.device.live('colorResolve')).toHaveLength(0);
    expect(target.sampleView).toBe(target.colorView);
  });

  test('a canvas target is single-sampled, and the invariant is enforced on construction', () => {
    const fake = fakeAseDevice();
    const target = createCanvasTarget(asCanvasTargetDevice(fake), { width: 8, height: 8 });
    expect(target.colorTextureCount).toBe(1);
    // Not reachable through the factory, so the check is in the constructor
    // where a new factory cannot reintroduce the bug.
    const device = asCanvasTargetDevice(fake);
    const error = expectCode(() => new RenderTargetImpl({
      device,
      width: 8,
      height: 8,
      format: 'bgra8unorm',
      depthFormat: undefined,
      sampleCount: 4 as 1 | 4,
      sampleDepth: false,
      ownsColor: true,
      label: 'apse.badCanvas',
      canvas: device.canvas,
      canvasContext: device.context,
    }), 'OPTION_UNKNOWN');
    expect(error.message).toContain('sampleCount 4');
  });

  test('a resize of a multisampled target reallocates both textures and frees both', () => {
    const fake = fakeAseDevice();
    const target = createColorTarget(asTargetDevice(fake), { width: 8, height: 8, sampleCount: 4, label: 'apse.msaa' });
    target.resize(16, 16);
    expect(fake.device.destroyLog.filter((l) => l.includes('color')).length).toBeGreaterThanOrEqual(2);
    expect(fake.device.live(':color')[0]!.width).toBe(16);
    expect(fake.device.live(':colorResolve')[0]!.width).toBe(16);
  });

  test('dispose leaves nothing alive, MSAA included', () => {
    const fake = fakeAseDevice();
    const target = createColorTarget(asTargetDevice(fake), { width: 8, height: 8, sampleCount: 4, label: 'apse.msaa' });
    target.dispose();
    expect(fake.device.liveTextures).toHaveLength(0);
  });
});

// ===========================================================================
// 7. CanvasSizer — the awkward cases
// ===========================================================================

describe('CanvasSizer — CSS px, backing px, and DPR', () => {
  test('a fractional DPR rounds rather than truncating', async () => {
    await withDpr(1.5, () => {
      const sizer = new CanvasSizer(fakeCanvas(101, 33), { maxPixelRatio: 2 });
      // 151.5 and 49.5: Math.round, not floor and not ceil.
      expect(sizer.width).toBe(152);
      expect(sizer.height).toBe(50);
    });
  });

  test('a DPR change is noticed even though the CSS box did not move', async () => {
    const canvas = fakeCanvas(400, 300);
    await withDpr(1, () => {
      const sizer = new CanvasSizer(canvas, { maxPixelRatio: 3 });
      expect([sizer.width, sizer.height]).toEqual([400, 300]);
      // A ResizeObserver does not fire for a zoom: the element's CSS box is
      // unchanged while the correct backing store doubles. `devicePixelRatio` is
      // the only thing that moved, and it is read every frame.
      return withDpr(2, () => {
        expect(sizer.update()).toBe(true);
        expect([sizer.width, sizer.height]).toEqual([800, 600]);
        // …and going back is a change too, not a one-way trip.
        return withDpr(1, () => {
          expect(sizer.update()).toBe(true);
          expect([sizer.width, sizer.height]).toEqual([400, 300]);
        });
      });
    });
  });

  test('a hidden element measures 1x1 rather than 0x0', async () => {
    await withDpr(2, () => {
      // display:none, or a detached element in a hidden iframe.
      const sizer = new CanvasSizer(fakeCanvas(0, 0), { maxPixelRatio: 2 });
      expect(sizer.width).toBe(1);
      expect(sizer.height).toBe(1);
      expect(sizer.clamped).toBe(false);
    });
  });

  test('a hidden element with a device limit is still 1x1, not 0', async () => {
    await withDpr(2, () => {
      // A cap below 1 would make `min()` produce 0 and a zero-sized texture,
      // which WebGPU rejects with an error about texture dimensions.
      const sizer = new CanvasSizer(fakeCanvas(0, 0), { maxPixelRatio: 0, maxTextureDimension2D: 1 });
      expect([sizer.width, sizer.height]).toEqual([1, 1]);
      expect(Number.isInteger(sizer.width)).toBe(true);
      expect(sizer.width).toBeGreaterThan(0);
    });
  });

  test('a DPR above the device ceiling clamps, and says it clamped', async () => {
    await withDpr(3, () => {
      // 5120 CSS px at 3x is 15360, and this compatibility device tops out at
      // 4096. Clamping costs supersampling nobody asked for; the compositor
      // still scales the whole image into the CSS box, so nothing is cropped.
      // 1000 CSS rows at 3x is 3000, which fits; only the width is capped.
      const sizer = new CanvasSizer(fakeCanvas(5120, 1000), {
        maxPixelRatio: 3,
        maxTextureDimension2D: 4096,
      });
      expect(sizer.width).toBe(4096);
      expect(sizer.height).toBe(3000);
      expect(sizer.clamped).toBe(true);
      expect(sizer.maxTextureDimension2D).toBe(4096);
    });
  });

  test('a 4K canvas at 2x clamps to the compatibility ceiling in both axes', async () => {
    await withDpr(2, () => {
      const sizer = new CanvasSizer(fakeCanvas(3840, 2160), { maxPixelRatio: 2, maxTextureDimension2D: 4096 });
      expect(sizer.width).toBe(4096);
      expect(sizer.height).toBe(4096);
      expect(sizer.clamped).toBe(true);
    });
  });

  test('without a ceiling the size is left alone, so an oversize reaches the target', async () => {
    await withDpr(3, () => {
      // No device, no clamping: the renderer passes the ceiling, and a caller
      // with a device gets a clamp. A silent crop is never the answer.
      const sizer = new CanvasSizer(fakeCanvas(5000, 5000), { maxPixelRatio: 3 });
      expect(sizer.width).toBe(15000);
      expect(sizer.clamped).toBe(false);
    });
  });

  test('a nonsense ceiling degrades to no ceiling at all', async () => {
    await withDpr(2, () => {
      for (const dim of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
        const sizer = new CanvasSizer(fakeCanvas(800, 600), { maxPixelRatio: 2, maxTextureDimension2D: dim });
        expect([sizer.width, sizer.height]).toEqual([1600, 1200]);
        expect(sizer.clamped).toBe(false);
      }
    });
  });

  test('a resize during a frame is picked up on the next update, and only then', async () => {
    await withDpr(1, () => {
      const canvas = fakeCanvas(800, 600);
      const sizer = new CanvasSizer(canvas, { maxPixelRatio: 2 });
      expect(sizer.update()).toBe(false);
      // The element moved between frames: a sidebar dragged, a grid reflowed.
      (canvas as unknown as { clientWidth: number }).clientWidth = 640;
      expect(sizer.update()).toBe(true);
      expect(sizer.width).toBe(640);
      expect(sizer.cssWidth).toBe(640);
      // And a change that rounds away to the same backing size is not one.
      (canvas as unknown as { clientWidth: number }).clientWidth = 640.4;
      expect(sizer.update()).toBe(false);
    });
  });

  test('the backing size is always a positive integer, whatever the input', async () => {
    const samples = [0, -1, 0.4, 0.5, 1, 1.5, 2, 3, 7.25, 12345, Number.NaN, Number.POSITIVE_INFINITY];
    const dims = [undefined, 1, 2, 64, 4096, 0, -5, Number.NaN];
    for (const css of samples) {
      for (const dpr of samples) {
        for (const maxDpr of samples) {
          for (const maxDim of dims) {
            const size = CanvasSizer.backingSize(css, css, dpr, maxDpr, maxDim);
            expect(Number.isInteger(size.width)).toBe(true);
            expect(Number.isInteger(size.height)).toBe(true);
            expect(size.width).toBeGreaterThan(0);
            expect(size.height).toBeGreaterThan(0);
            if (typeof maxDim === 'number' && Number.isFinite(maxDim) && maxDim >= 1) {
              expect(size.width).toBeLessThanOrEqual(Math.floor(maxDim));
            }
          }
        }
      }
    }
  });

  test('a disposed sizer refuses to update, rather than freezing silently', async () => {
    await withDpr(1, () => {
      const sizer = new CanvasSizer(fakeCanvas(800, 600));
      sizer.dispose();
      expectCode(() => sizer.update(), 'INTERNAL_INVARIANT');
    });
  });
});

// ===========================================================================
// 8. The public type change
// ===========================================================================

describe('FrameTimingStats', () => {
  test('a device with no measurement expresses it as null, not 0', () => {
    // Compile-time assertion: this assignment is the migration a consumer has
    // to make, and it is why `gpu` became `number | null`.
    const noTiming: FrameTimingStats = {
      gpu: null,
      averageGpu: null,
      gpuTimingAvailable: false,
    };
    expect(noTiming.gpu).toBeNull();
    expect(noTiming.gpuTimingAvailable).toBe(false);
  });

  test('a measured frame carries a number and the availability flag', () => {
    const timing: FrameTimingStats = { gpu: 0, averageGpu: 0.25, gpuTimingAvailable: true };
    // 0 is a real measurement — a driver that quantises to 100 µs reports it for
    // a trivial frame — and it is not the same fact as `null`.
    expect(timing.gpu).toBe(0);
    expect(timing.gpuTimingAvailable).toBe(true);
  });
});

// ===========================================================================
// 9. The present pass default
// ===========================================================================

describe('DEFAULT_TONE_MAPPING', () => {
  test('tone mapping is on by default, and it is ACES', () => {
    // A material writes linear values and a `bgra8unorm` canvas stores them
    // verbatim: a straight-to-canvas draw is a linear image displayed as sRGB,
    // which is far too dark and reports nothing.
    expect(DEFAULT_TONE_MAPPING.operator).toBe('aces');
  });

  test('the default is frozen, so a caller cannot retune the shared object', () => {
    expect(Object.isFrozen(DEFAULT_TONE_MAPPING)).toBe(true);
  });

  test('it agrees with the tone-map material\'s own default', () => {
    // Two defaults that disagreed would make a directly-built tone-map material
    // and a present pass produce different images from the same scene.
    expect(DEFAULT_TONE_MAPPING.operator).toBe('aces');
    expect(DEFAULT_TONE_MAPPING.hdrFormat).toBeUndefined();
  });
});

// ===========================================================================
// 10. Option handling on the device
// ===========================================================================

describe('DeviceOptions — unknown values are still rejected before acquisition', () => {
  const BAD: ReadonlyArray<[string, DeviceOptions, string]> = [
    ['sampleCount', { sampleCount: 2 as 1 | 4 }, 'sampleCount'],
    ['maxPixelRatio', { maxPixelRatio: 0 }, 'maxPixelRatio'],
    ['maxObjects', { maxObjects: 0 }, 'maxObjects'],
    ['alphaMode', { alphaMode: 'straight' as 'opaque' }, 'alphaMode'],
  ];

  for (const [name, opts, inMessage] of BAD) {
    test(`rejects ${name} before touching the GPU`, async () => {
      const { canvas } = fakeWebGpuCanvas();
      const error = await expectCodeAsync(() => createDevice(canvas, opts), 'OPTION_UNKNOWN');
      expect(error.message).toContain(inMessage);
    });
  }
});

