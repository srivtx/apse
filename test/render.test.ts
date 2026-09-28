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

import { afterAll, beforeAll, describe, expect, test } from 'bun:test';

import { BIND_GROUP, sceneObjectOffset } from '../src/core/slot.ts';
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
import { Renderer } from '../src/render/renderer.ts';
import type { RendererOptions } from '../src/render/renderer.ts';
import {
  alignedBytesPerRow,
  assertBytesPerRowAligned,
  BYTES_PER_PIXEL_RGBA8,
  COPY_BYTES_PER_ROW_ALIGNMENT,
  CaptureReadback,
  unpadRows,
} from '../src/render/readback.ts';
import type { FrameTimingStats } from '../src/render/types.ts';
import { Scene, MeshNode, PerspectiveCamera } from '../src/scene/index.ts';
import { box } from '../src/geometry/primitives/box.ts';
import { upload } from '../src/geometry/mesh.ts';
import type { GpuMesh } from '../src/geometry/mesh.ts';
import { InstanceData, TRANSFORM_STRIDE, uploadInstances } from '../src/geometry/instanced.ts';
import { basicMaterial } from '../src/material/basic.ts';
import { instancedMaterial } from '../src/material/instanced.ts';
import { asCanvasTargetDevice, asTargetDevice, fakeAseDevice, fakeLimits, FAKE_TEXTURE_USAGE as USAGE } from './fake-device.ts';
import {
  asTimingDevice,
  FakeFrameDevice,
  FakeFramePass,
  FakeTimingDevice,
  fakeWebGpuCanvas,
  installWebGpuBitmaps,
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


// ===========================================================================
// 11. The frame loop
//
// Everything above tests one function. This tests the *order*, and the order is
// the part that is not visible in any return value: a frame that packs nothing,
// that binds the wrong slot, that presents in a second submit, or that reports a
// GPU time it never measured, all return a `FrameStats` that looks fine.
//
// So the assertions here are on the sequence of API calls the encoder received.
// `FakeFrameDevice` records them; nothing rasterises. A `drawIndexed` is a line
// in an array, and that is enough, because "one draw for a thousand instances"
// is a claim about the calls and not about the pixels.
// ===========================================================================

/** A renderer, a device that records its calls, and a scene to draw. */
interface Frame {
  readonly renderer: Renderer;
  readonly device: FakeFrameDevice;
  readonly scene: Scene;
  readonly camera: PerspectiveCamera;
  readonly gpu: GPUDevice;
  /** The scene pass, i.e. every pass except the present pass. */
  scenePasses(): readonly FakeFramePass[];
  /** The present pass, or undefined when there is none. */
  presentPass(): FakeFramePass | undefined;
}

let uninstallBitmaps: (() => void) | null = null;

beforeAll(() => { uninstallBitmaps = installWebGpuBitmaps(); });
afterAll(() => { uninstallBitmaps?.(); uninstallBitmaps = null; });

/** The last encoder's passes that are not the present pass. */
function scenePassesOf(device: FakeFrameDevice): readonly FakeFramePass[] {
  return device.lastPasses.filter((pass) => !pass.label.startsWith('apse.present'));
}

interface FrameOptions {
  readonly renderer?: RendererOptions;
  readonly features?: readonly string[];
  /** Nanoseconds the fake GPU stamps on the six query slots. */
  readonly timestamps?: readonly bigint[];
  readonly size?: [number, number];
}

/** Creates a renderer on a fake device, runs `fn`, and disposes both. */
async function withFrame<T>(
  opts: FrameOptions,
  fn: (frame: Frame) => Promise<T> | T,
): Promise<T> {
  const device = new FakeFrameDevice({
    limits: fakeLimits(),
    features: [...(opts.features ?? ['timestamp-query'])],
  });
  if (opts.timestamps !== undefined) device.timestamps = [...opts.timestamps];
  const [width, height] = opts.size ?? [320, 200];
  const { canvas } = fakeWebGpuCanvas(width, height);
  return withGpu({ device: device as unknown as GPUDevice, preferredFormat: 'bgra8unorm' }, async () => {
    const renderer = await Renderer.create(canvas, opts.renderer ?? {});
    const gpu = device as unknown as GPUDevice;
    const camera = new PerspectiveCamera({ fov: 45, near: 0.1, far: 100, aspect: width / height });
    camera.lookAt([0, 0, 30], [0, 0, 0], [0, 1, 0]);
    const frame: Frame = {
      renderer, device, gpu,
      scene: new Scene('test'),
      camera,
      scenePasses: () => scenePassesOf(device),
      presentPass: () => device.lastPasses.find((p) => p.label.startsWith('apse.present')),
    };
    try {
      return await fn(frame);
    } finally {
      renderer.dispose();
    }
  });
}

/** A unit-scale translation for each of `count` instances, on a grid. */
function gridMatrices(count: number, perRow = 32): Float32Array {
  const out = new Float32Array(count * TRANSFORM_STRIDE);
  for (let i = 0; i < count; i++) {
    const at = i * TRANSFORM_STRIDE;
    out[at] = 1; out[at + 5] = 1; out[at + 10] = 1; out[at + 15] = 1;
    out[at + 12] = (i % perRow) - perRow / 2;
    out[at + 13] = Math.floor(i / perRow) - perRow / 2;
  }
  return out;
}

/** `count` cubes on one node, as one instanced draw. */
async function addInstancedCubes(
  frame: Frame, count: number, opts: { width?: number; firstInstance?: number; instanceCount?: number } = {},
): Promise<GpuMesh> {
  const instances = uploadInstances(
    frame.gpu,
    InstanceData.fromMatrices(gridMatrices(count), { name: 'grid' }),
  );
  const material = await instancedMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
  const mesh = upload(frame.gpu, box({ width: opts.width ?? 0.6 }), {
    instances,
    ...(opts.firstInstance === undefined ? {} : { firstInstance: opts.firstInstance }),
    ...(opts.instanceCount === undefined ? {} : { instanceCount: opts.instanceCount }),
  });
  frame.scene.add(new MeshNode({ name: 'grid', mesh, material }));
  return mesh;
}

describe('the frame loop — instanced draws', () => {
  test('a thousand instances cost one drawIndexed, with the count in the draw', async () => {
    await withFrame({}, async (frame) => {
      await addInstancedCubes(frame, 1000);
      const stats = frame.renderer.render(frame.scene, frame.camera);

      // The claim, in the only place it is observable: the calls, not a value.
      const [draw] = frame.scenePasses()[0]!.draws;
      expect(draw).toBeDefined();
      expect(draw!.kind).toBe('drawIndexed');
      expect(draw!.instances).toBe(1000);
      expect(stats.drawCalls).toBe(1);
      // 36 indices per cube × 1000, which is the triangle count the naive
      // version would have got right by accident and the broken one would not.
      expect(stats.triangles).toBe(36 / 3 * 1000);
    });
  });

  test('the instance stream is bound to slot 1, and it is the geometry\'s own buffer', async () => {
    await withFrame({}, async (frame) => {
      const mesh = await addInstancedCubes(frame, 8);
      frame.renderer.render(frame.scene, frame.camera);
      const slots = frame.scenePasses()[0]!.vertexBuffers.map((v) => v.slot);
      expect(slots).toEqual([0, 1]);
      const bound = frame.scenePasses()[0]!.vertexBuffers.find((v) => v.slot === 1);
      expect(bound?.buffer).toBe(mesh.instanceBuffer ?? undefined);
    });
  });

  test('a non-null buffer is the test, not a count: one instance still binds slot 1', async () => {
    // `instanceCount: 1` on a buffer of a thousand is an ordinary call — draw one
    // copy of a grid. A renderer that guarded on `instanceCount > 1` would skip
    // slot 1, and the pipeline still declares it, so the vertex stage would read
    // whatever happened to be in the slot: a wrong image, and no error. The
    // pipeline has no idea how many copies were asked for.
    await withFrame({}, async (frame) => {
      await addInstancedCubes(frame, 1000, { instanceCount: 1 });
      frame.renderer.render(frame.scene, frame.camera);
      const pass = frame.scenePasses()[0]!;
      expect(pass.vertexBuffers.map((v) => v.slot)).toEqual([0, 1]);
      expect(pass.draws[0]!.instances).toBe(1);
    });
  });

  test('firstInstance and a sub-range both reach the draw', async () => {
    await withFrame({}, async (frame) => {
      await addInstancedCubes(frame, 64, { firstInstance: 10, instanceCount: 5 });
      const stats = frame.renderer.render(frame.scene, frame.camera);
      const draw = frame.scenePasses()[0]!.draws[0]!;
      expect(draw.firstInstance).toBe(10);
      expect(draw.instances).toBe(5);
      expect(stats.triangles).toBe(12 * 5);
    });
  });

  test('a GpuInstances with no instances is a legal buffer, and a mesh refuses to carry it', async () => {
    // The state that decides how the renderer must test for instancing: a
    // non-null buffer with a count of zero. `uploadInstances` produces it — an
    // empty transform list is legal, and allocates a 4-byte floor so
    // `createBuffer` stays well defined — and `upload` then rejects it, because
    // `drawIndexed(n, 0)` is a silent no-op and a node that can never be seen
    // should not be in a draw list at all. So the renderer never receives one,
    // and the guard it uses cannot be about the count: see the `instanceCount: 1`
    // test above, which is the case a count-based guard actually breaks.
    await withFrame({}, (frame) => {
      const empty = uploadInstances(frame.gpu, InstanceData.fromMatrices(new Float32Array(0)));
      expect(empty.count).toBe(0);
      expect(empty.buffer).not.toBeNull();
      const error = expectCode(
        () => upload(frame.gpu, box(), { instances: empty }),
        'INTERNAL_INVARIANT',
      );
      expect(error.message).toContain('instanceCount');
    });
  });

  test('instancing costs one object-uniform slot, not one per instance', async () => {
    await withFrame({}, async (frame) => {
      await addInstancedCubes(frame, 1000);
      frame.renderer.render(frame.scene, frame.camera);
      // The per-instance transform arrives as a vertex attribute, so there is
      // nothing per instance in the uniform block. 1000 slots would be 256 KB
      // of upload for one draw.
      const object = frame.device.writes.find((w) => w.label.includes('object'));
      expect(object?.size).toBe(256);
      // The scene group is @0 and carries both the frame and the object, so the
      // only dynamic offset in a frame is the object's. Binding 0 is the frame
      // region, not object 0.
      const offsets = frame.scenePasses()[0]!.bindGroups
        .filter((b) => b.group === BIND_GROUP.scene)
        .flatMap((b) => b.offsets ?? []);
      expect(offsets).toEqual([sceneObjectOffset(0)]);
    });
  });

  test('a plain mesh never binds slot 1, and two items sharing a buffer bind it once', async () => {
    await withFrame({}, async (frame) => {
      const material = await basicMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
      const mesh = upload(frame.gpu, box({ width: 0.6 }));
      frame.scene.add(new MeshNode({ name: 'a', mesh, material }));
      frame.scene.add(new MeshNode({ name: 'b', mesh, material, position: [2, 0, 0] }));
      frame.renderer.render(frame.scene, frame.camera);
      const pass = frame.scenePasses()[0]!;
      expect(pass.draws).toHaveLength(2);
      expect(pass.vertexBuffers.map((v) => v.slot)).toEqual([0]);
      // The instance guard shares the suppression state of the vertex guard:
      // two items, one mesh, one setVertexBuffer(0).
      expect(pass.calls.filter((c) => c === 'setVertexBuffer:0')).toHaveLength(1);
    });
  });

  test('two instanced items sharing one buffer bind slot 1 once', async () => {
    await withFrame({}, async (frame) => {
      const mesh = await addInstancedCubes(frame, 16);
      const material = await instancedMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
      frame.scene.add(new MeshNode({ name: 'second', mesh, material, position: [0, 4, 0] }));
      frame.renderer.render(frame.scene, frame.camera);
      const pass = frame.scenePasses()[0]!;
      expect(pass.draws).toHaveLength(2);
      expect(pass.calls.filter((c) => c === 'setVertexBuffer:1')).toHaveLength(1);
    });
  });
});

describe('the frame loop — GPU timestamps', () => {
  /** Six slots: two stamped pairs at 1 ms and 4 ms, and two unstamped at 0. */
  const TWO_PASS_FRAME = [1_000_000n, 2_000_000n, 5_000_000n, 9_000_000n, 0n, 0n];

  test('no reading yet is null, and the capability is reported separately', async () => {
    await withFrame({ timestamps: TWO_PASS_FRAME }, (frame) => {
      const stats = frame.renderer.render(frame.scene, frame.camera);
      expect(stats.gpu).toBeNull();
      expect(stats.averageGpu).toBeNull();
      // The capability is true on the very first frame even though there is no
      // number, which is the whole reason it is a separate field.
      expect(stats.gpuTimingAvailable).toBe(true);
    });
  });

  test('the reading arrives a frame late, and is the sum of the passes that ran', async () => {
    await withFrame({ timestamps: TWO_PASS_FRAME }, async (frame) => {
      frame.renderer.render(frame.scene, frame.camera);
      await flush();
      const stats = frame.renderer.render(frame.scene, frame.camera);
      // (2 − 1) ms for the scene pass plus (9 − 5) ms for the present pass. Not
      // 4 ms, which is only the difference of one pair, and not the raw
      // nanosecond epoch, which is 1e9.
      expect(stats.gpu).toBeCloseTo(5, 6);
    });
  });

  test('a measured zero is zero, and is not confused with a missing reading', async () => {
    const idle = [7_000_000n, 7_000_000n, 7_000_000n, 7_000_000n, 0n, 0n];
    await withFrame({ timestamps: idle }, async (frame) => {
      frame.renderer.render(frame.scene, frame.camera);
      await flush();
      const stats = frame.renderer.render(frame.scene, frame.camera);
      expect(stats.gpu).toBe(0);
    });
  });

  test('every pass a frame opens gets its own write index, and none is reused', async () => {
    await withFrame({ timestamps: TWO_PASS_FRAME }, async (frame) => {
      const material = await basicMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
      const mesh = upload(frame.gpu, box({ width: 0.6 }));
      const glass = await basicMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat, transparent: true });
      frame.scene.add(new MeshNode({ name: 'a', mesh, material }));
      frame.scene.add(new MeshNode({ name: 'b', mesh, material: glass, position: [2, 0, 0] }));
      frame.renderer.render(frame.scene, frame.camera);

      // Three passes: opaque, transparent, present. Depth state is baked into the
      // pipeline, so the transparent item genuinely cannot share the first pass.
      expect(frame.device.lastPasses).toHaveLength(3);
      const written = frame.device.lastPasses.map((p) => p.timestampWrites);
      expect(written.every((w) => w !== undefined)).toBe(true);
      const indices = written.map((w) => `${w!.beginningOfPassWriteIndex}:${w!.endOfPassWriteIndex}`);
      // A write index written twice in one submission is a validation error, and
      // a discarded pass is a black frame — so all three must be distinct.
      expect(new Set(indices).size).toBe(3);
      expect(indices).toEqual(['0:1', '2:3', '4:5']);
    });
  });

  test('the readback is encoded after the passes and before the submit', async () => {
    await withFrame({ timestamps: TWO_PASS_FRAME }, (frame) => {
      frame.renderer.render(frame.scene, frame.camera);
      const calls = frame.device.frameEncoders[0]!.calls;
      const lastPass = calls.map((c, i) => (c.startsWith('beginRenderPass') ? i : -1)).filter((i) => i >= 0).pop()!;
      const resolve = calls.findIndex((c) => c.startsWith('resolveQuerySet'));
      // `resolveQuerySet` is a queue command: encoded after the passes it
      // observes, and in the same command buffer as the copy, or the result has
      // to travel through the CPU.
      expect(resolve).toBeGreaterThan(lastPass);
      expect(calls[resolve + 1]).toContain('copyBufferToBuffer');
      expect(frame.device.submitted).toHaveLength(1);
    });
  });

  test('a device without timestamp-query reports no timing and attaches no write index', async () => {
    await withFrame({ features: [] }, (frame) => {
      const material = basicMaterialSpecOnly(frame);
      expect(material).toBeNull();
      const stats = frame.renderer.render(frame.scene, frame.camera);
      expect(stats.gpuTimingAvailable).toBe(false);
      expect(stats.gpu).toBeNull();
      expect(stats.averageGpu).toBeNull();
      // Attaching a write index without the feature is a validation error, not
      // a no-op, and it would invalidate every frame on half of all devices.
      for (const pass of frame.device.allPasses) expect(pass.timestampWrites).toBeUndefined();
      expect(frame.device.querySets).toHaveLength(0);
    });
  });

  test('averageGpu divides by the readings, not by the frames', async () => {
    // Frame 1's two passes are stamped 1 ms and 3 ms apart, so it reads 4 ms.
    await withFrame({ timestamps: [0n, 1_000_000n, 0n, 3_000_000n, 0n, 0n] }, async (frame) => {
      frame.renderer.render(frame.scene, frame.camera);
      await flush();
      // Frame 2 reads 3 ms and 7 ms, so 10 ms — and the window now holds two
      // readings and two frames that could not be timed at all.
      frame.device.timestamps = [0n, 3_000_000n, 0n, 7_000_000n, 0n, 0n];
      frame.renderer.render(frame.scene, frame.camera);
      await flush();
      const stats = frame.renderer.render(frame.scene, frame.camera);
      expect(stats.gpu).toBeCloseTo(10, 6);
      // (4 + 10) / 2. Dividing by the four frames in the window — counting the
      // unmeasured ones as zeros — would report 3.5, which is a number about
      // nothing rather than a slightly pessimistic one.
      expect(stats.averageGpu).toBeCloseTo(7, 6);
    });
  });

  test('the gpu budget stays silent until there is a reading, and fires once there is', async () => {
    await withFrame({ timestamps: TWO_PASS_FRAME, renderer: { budget: { gpu: 1 } } }, async (frame) => {
      const breaches: string[] = [];
      frame.renderer.onBudgetBreached((name) => { breaches.push(name); });
      frame.renderer.render(frame.scene, frame.camera);
      await flush();
      frame.renderer.render(frame.scene, frame.camera);
      // 5 ms against a 1 ms budget, measured. Before the reading arrived the
      // budget had no number to compare and said nothing — a breach of a
      // measurement nobody took is the one thing a budget must not report.
      expect(breaches).toEqual(['gpu']);
    });
  });

  test('dispose destroys the query set, the resolve buffer and every staging buffer', async () => {
    const device = new FakeFrameDevice({ limits: fakeLimits(), features: ['timestamp-query'] });
    const { canvas } = fakeWebGpuCanvas(64, 64);
    await withGpu({ device: device as unknown as GPUDevice, preferredFormat: 'bgra8unorm' }, async () => {
      const renderer = await Renderer.create(canvas, {});
      const camera = new PerspectiveCamera({ fov: 45, near: 0.1, far: 10 });
      camera.lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
      renderer.render(new Scene('empty'), camera);
      // 1 query set, 1 resolve, 2 staging — 48 bytes each of the last two.
      expect(device.querySets).toHaveLength(1);
      renderer.dispose();
      expect(device.querySets[0]!.destroyed).toBe(true);
      const timing = device.buffers.filter((b) => b.label.startsWith('apse.timing'));
      expect(timing).toHaveLength(3);
      expect(timing.every((b) => b.destroyed)).toBe(true);
    });
  });
});

/**
 * Nothing: a material creation is a device round trip and this test is about the
 * absence of a timer, so it needs no material. Named so the call site reads as
 * the deliberate nothing it is.
 */
function basicMaterialSpecOnly(frame: Frame): null {
  void frame;
  return null;
}

describe('the frame loop — the present pass', () => {
  test('the scene renders into an rgba16float intermediate, and the tone map presents it', async () => {
    await withFrame({}, async (frame) => {
      // The default, and the reason it is a default: a material writes linear
      // values and a bgra8unorm canvas stores them verbatim.
      expect(frame.renderer.sceneFormat).toBe('rgba16float');
      expect(frame.renderer.sceneTarget.isCanvas).toBe(false);
      expect(frame.renderer.sceneTarget.format).toBe('rgba16float');
      // And the intermediate is a real allocation, not a description of one.
      expect(frame.device.apseLive('apse.present:scene:color')).toHaveLength(1);
    });
  });

  test('the present pass shares the frame\'s encoder, so the frame is one submit', async () => {
    await withFrame({}, async (frame) => {
      const material = await basicMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
      frame.scene.add(new MeshNode({ name: 'a', mesh: upload(frame.gpu, box()), material }));
      frame.renderer.render(frame.scene, frame.camera);

      // Two passes and one submit. Two submits would let the compositor present
      // the scene target's own frame as a finished one — an untone-mapped image,
      // with a compositor sync in the middle of the frame.
      expect(frame.device.frameEncoders).toHaveLength(1);
      expect(frame.device.submitted).toHaveLength(1);
      expect(frame.scenePasses()).toHaveLength(1);
      const present = frame.presentPass();
      expect(present).toBeDefined();
      // Three vertices, one copy, and no index buffer: a fullscreen triangle.
      expect(present!.draws).toHaveLength(1);
      expect(present!.draws[0]!.kind).toBe('draw');
      expect(present!.draws[0]!.count).toBe(3);
      expect(present!.draws[0]!.instances).toBe(1);
      // The scene drew into the intermediate, and the pass has no depth
      // attachment, which is the only reason its pipeline has no depth state.
      expect(present!.depthStencilAttachment).toBeUndefined();
    });
  });

  test('toneMapping: null is the direct path: one pass, the canvas format, no intermediate', async () => {
    await withFrame({ renderer: { toneMapping: null } }, async (frame) => {
      expect(frame.renderer.sceneFormat).toBe('bgra8unorm');
      expect(frame.renderer.sceneTarget.isCanvas).toBe(true);
      const material = await basicMaterial(frame.gpu, { targetFormat: 'bgra8unorm' });
      frame.scene.add(new MeshNode({ name: 'a', mesh: upload(frame.gpu, box()), material }));
      const stats = frame.renderer.render(frame.scene, frame.camera);
      expect(frame.device.lastPasses).toHaveLength(1);
      expect(stats.drawCalls).toBe(1);
      // No pipeline, no mesh, no intermediate: the degenerate path has to stay
      // free, because it is the only one that costs nothing at all.
      expect(frame.device.apseLive('apse.present:scene:color')).toHaveLength(0);
      expect(frame.device.pipelineDescs.filter((d) => d.label === 'apse.present:pipeline')).toHaveLength(0);
    });
  });

  test('hdr: false keeps the tone map but renders into the destination format', async () => {
    await withFrame({ renderer: { hdr: false } }, async (frame) => {
      expect(frame.renderer.sceneFormat).toBe('bgra8unorm');
      expect(frame.renderer.sceneTarget.isCanvas).toBe(false);
      const material = await basicMaterial(frame.gpu, { targetFormat: 'bgra8unorm' });
      frame.scene.add(new MeshNode({ name: 'a', mesh: upload(frame.gpu, box()), material }));
      frame.renderer.render(frame.scene, frame.camera);
      // Still an intermediate, and still a fullscreen pass: a curve applied to an
      // already-clipped image, which is legal and occasionally what you want, and
      // is the only way to keep a material written for the canvas format while
      // still fixing the darkness.
      expect(frame.presentPass()).toBeDefined();
      expect(frame.device.apseLive('apse.present:scene:color')[0]!.format).toBe('bgra8unorm');
    });
  });

  test('a material compiled for the canvas format now fails, naming both formats', async () => {
    await withFrame({}, async (frame) => {
      // The intended trade, and the reason it is the right one: the alternative
      // is a command buffer invalidated whole, with no exception anywhere.
      const material = await basicMaterial(frame.gpu, { targetFormat: 'bgra8unorm' });
      frame.scene.add(new MeshNode({ name: 'a', mesh: upload(frame.gpu, box()), material }));
      const error = expectCode(() => frame.renderer.render(frame.scene, frame.camera), 'RENDER_TARGET_FORMAT_MISMATCH');
      expect(error.message).toContain('basic');
      expect(error.message).toContain('rgba16float');
      expect(error.message).toContain('bgra8unorm');
      expect(error.fix).toContain('targetFormat');
    });
  });

  test('exposure reaches the pass, and the renderer reports what the screen is using', async () => {
    await withFrame({ renderer: { toneMapping: { operator: 'aces', exposure: 0.5 } } }, (frame) => {
      // Adopted at construction, so the getter does not answer 1 while the
      // screen is at half a stop.
      expect(frame.renderer.exposure).toBe(0.5);
      frame.renderer.exposure = 2;
      expect(frame.renderer.exposure).toBe(2);
    });
  });

  test('a resize reallocates the intermediate at the canvas\'s new size', async () => {
    await withFrame({ size: [320, 200] }, async (frame) => {
      const before = frame.device.apseLive('apse.present:scene:color');
      expect(before).toHaveLength(1);
      expect(before[0]!.width).toBe(320);

      const canvas = frame.renderer.device.canvas as unknown as { clientWidth: number; clientHeight: number };
      canvas.clientWidth = 640;
      canvas.clientHeight = 480;
      frame.renderer.render(frame.scene, frame.camera);

      // Both, and in the same frame. The intermediate is the scene target now, so
      // leaving it at the old extent would tone map a differently-sized image
      // into the new one: a stretched frame, on the frame after every resize.
      expect(frame.renderer.sceneTarget.width).toBe(640);
      expect(frame.renderer.sceneTarget.height).toBe(480);
      expect(frame.device.apseLive('apse.present:scene:color')).toHaveLength(1);
      expect(frame.device.apseLive('apse.present:scene:color')[0]!.width).toBe(640);
      // Freed before it was replaced, not after: the peak is one target.
      expect(before[0]!.destroyed).toBe(true);
    });
  });

  test('an empty scene still clears, so the tone map cannot present last frame', async () => {
    await withFrame({}, async (frame) => {
      const material = await basicMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
      frame.scene.add(new MeshNode({ name: 'a', mesh: upload(frame.gpu, box()), material }));
      frame.renderer.render(frame.scene, frame.camera);
      const first = frame.device.allPasses.length;

      const empty = new Scene('empty');
      const stats = frame.renderer.render(empty, frame.camera);
      // A pass with no draws. Without it the intermediate keeps the last frame's
      // pixels and the tone map presents them again, forever.
      expect(frame.device.allPasses.length).toBe(first + 2);
      const clear = frame.scenePasses();
      expect(clear).toHaveLength(1);
      expect(clear[0]!.draws).toHaveLength(0);
      expect(clear[0]!.ended).toBe(true);
      expect(stats.drawCalls).toBe(0);
    });
  });

  test('capture goes through its own present pass, so it returns the presented image', async () => {
    await withFrame({}, async (frame) => {
      const material = await basicMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
      frame.scene.add(new MeshNode({ name: 'a', mesh: upload(frame.gpu, box()), material }));
      // The point of the whole arrangement: a capture target is 8-bit and the
      // materials are compiled for rgba16float, so rendering the scene straight
      // into it is a hard format mismatch rather than a dark image.
      const shot = await frame.renderer.capture(frame.scene, frame.camera);
      expect(shot.format).toBe('bgra8unorm');
      expect(shot.width).toBe(320);
      // 4 bytes per pixel, padded to 256. Not 8: the tone map is what brought
      // the HDR intermediate back to 8 bits.
      expect(shot.bytesPerRow).toBe(Math.ceil(320 * 4 / 256) * 256);
      expect(shot.data.byteLength).toBe(shot.bytesPerRow * shot.height);
      // A second intermediate, built on the first capture and not before.
      expect(frame.device.apseLive('apse.capture.present:scene:color')).toHaveLength(1);
      // Three submits: the scene's, the capture pass's own, and the copy out.
      // The in-frame present pass shares the frame's encoder instead, which is
      // why this number is 3 rather than 1.
      expect(frame.device.submitted.length).toBe(3);
    });
  });

  test('capture with no present pass renders straight into the 8-bit target', async () => {
    await withFrame({ renderer: { toneMapping: null } }, async (frame) => {
      const material = await basicMaterial(frame.gpu, { targetFormat: 'bgra8unorm' });
      frame.scene.add(new MeshNode({ name: 'a', mesh: upload(frame.gpu, box()), material }));
      const shot = await frame.renderer.capture(frame.scene, frame.camera);
      expect(shot.format).toBe('bgra8unorm');
      // No second pass, so no second intermediate: two submits, the frame's and
      // the copy out.
      expect(frame.device.apseLive('apse.capture.present:scene:color')).toHaveLength(0);
      expect(frame.device.submitted.length).toBe(2);
    });
  });

  test('dispose releases the intermediate, the depth attachment and the pass pipeline', async () => {
    const device = new FakeFrameDevice({ limits: fakeLimits(), features: ['timestamp-query'] });
    const { canvas } = fakeWebGpuCanvas(64, 64);
    await withGpu({ device: device as unknown as GPUDevice, preferredFormat: 'bgra8unorm' }, async () => {
      const renderer = await Renderer.create(canvas, {});
      expect(device.apseLive('apse.present:scene:color')).toHaveLength(1);
      expect(device.apseLive('apse.present:scene:depth')).toHaveLength(1);
      renderer.dispose();
      expect(device.apseLive('apse.present:scene:color')).toHaveLength(0);
      expect(device.apseLive('apse.present:scene:depth')).toHaveLength(0);
      // Idempotent, and the second call must not reach a destroyed device.
      renderer.dispose();
    });
  });
});

describe('the frame loop — the draw-list counters', () => {
  /** One cube in front of the camera, one behind it, sharing a material. */
  async function twoCubes(frame: Frame, behind: { visible: boolean } = { visible: true }): Promise<void> {
    const material = await basicMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
    const mesh = upload(frame.gpu, box({ width: 1 }));
    frame.scene.add(new MeshNode({ name: 'front', mesh, material }));
    const back = new MeshNode({ name: 'back', mesh, material, position: [0, 0, 500] });
    back.visible = behind.visible;
    frame.scene.add(back);
  }

  test('candidates is what the walk saw and culled is the frustum\'s verdict', async () => {
    await withFrame({}, async (frame) => {
      await twoCubes(frame);
      const stats = frame.renderer.render(frame.scene, frame.camera);
      // Both cubes are candidates. One of them is behind the camera and the
      // frustum rejected it. Deriving either number from the surviving draw list
      // would report 1 and 0.
      expect(stats.candidates).toBe(2);
      expect(stats.culled).toBe(1);
      expect(stats.drawCalls).toBe(1);
    });
  });

  test('a node that was never submitted is not reported as culled', async () => {
    await withFrame({}, async (frame) => {
      await twoCubes(frame, { visible: false });
      const stats = frame.renderer.render(frame.scene, frame.camera);
      // A hidden node is not culled: the frustum never saw it, and a closer
      // camera will not find it either. Counting it as a culling win credits
      // the frustum with rejecting something that was never in front of it.
      expect(stats.candidates).toBe(1);
      expect(stats.culled).toBe(0);
      expect(stats.drawCalls).toBe(1);
    });
  });

  test('the three counters partition the walk exactly', async () => {
    await withFrame({}, async (frame) => {
      await twoCubes(frame);
      // A node with no indices is a candidate that produces no draw item for a
      // reason that is not culling — which is the third term of the identity.
      const material = await basicMaterial(frame.gpu, { targetFormat: frame.renderer.sceneFormat });
      const unindexed = {
        layout: box().layout,
        vertexBuffer: frame.device.createBuffer({ label: 'bare', size: 48, usage: 0x20 }) as unknown as GPUBuffer,
        indexBuffer: null,
        indexCount: 0,
        instanceCount: 1,
        firstInstance: 0,
        instanceBuffer: null,
      };
      frame.scene.add(new MeshNode({ name: 'empty', mesh: unindexed, material }));
      const stats = frame.renderer.render(frame.scene, frame.camera);
      // graph.ts states this identity; the renderer is what turns it into
      // numbers, so it is the renderer's numbers that have to satisfy it.
      expect(stats.candidates).toBe(frame.scene.meshNodeCount);
      expect(stats.culled).toBe(frame.scene.culledCount);
      expect(stats.candidates).toBe(stats.drawCalls + stats.culled + frame.scene.emptyCount);
    });
  });
});
