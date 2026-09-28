/**
 * Test doubles for the parts of WebGPU that `test/fake-device.ts` does not model.
 *
 * The base fake covers textures, views, error scopes, and device loss, because
 * those are what the render-target layer touches. Three subsystems need more
 * than that, and all three are in this file:
 *
 *   - **`GPUQuerySet` and `createBuffer`**, for the timestamp timer. The fake
 *     lets a test *choose* the nanosecond values the GPU "wrote", which is the
 *     only way to test the three hazards that make GPU timing go wrong: an
 *     arbitrary non-zero epoch, a subtraction that must not cross frames, and a
 *     wrapped counter.
 *   - **`navigator.gpu`**, for `createDevice`. Without it the only reachable
 *     path is `WEBGPU_UNAVAILABLE`, so every property of a real device — the
 *     limits it sends, the capabilities it derives, the uncaptured-error path —
 *     would be untested.
 *   - **A mappable buffer**, for readback. It reproduces the one behaviour that
 *     matters: `getMappedRange()` returns a range that `unmap()` detaches, so a
 *     test that copies inside the mapped window and one that copies outside get
 *     different answers. That difference *is* the bug the readback helper
 *     exists to prevent.
 *
 * `test/fake-device-ext.ts` models the draw plan for the instancing tests and
 * `test/fake-device.ts` models textures; this file is the third thing, and it
 * is named for the subsystem rather than for "extension" so the three do not
 * collide.
 */

import { FakeGPUDevice, fakeLimits } from './fake-device.ts';
import type { FeatureLevel } from '../src/render/device.ts';

// ---------------------------------------------------------------------------
// Buffers and query sets
// ---------------------------------------------------------------------------

export interface FakeBufferDescriptor {
  label?: string;
  size: number;
  usage: number;
}

export class FakeBuffer {
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  destroyed = false;
  /** Bytes written by a `copyBufferToBuffer`, or by a test. */
  readonly bytes: Uint8Array;
  #mapped = false;
  /** Resolvers of the pending `mapAsync` calls, oldest first. */
  #pending: (() => void)[] = [];

  constructor(desc: FakeBufferDescriptor) {
    this.label = desc.label ?? '';
    this.size = desc.size;
    this.usage = desc.usage;
    this.bytes = new Uint8Array(desc.size);
  }

  /**
   * A `mapAsync` that resolves on the next microtask.
   *
   * The real one resolves after the queue has caught up, which is what makes it
   * a promise rather than a callback. A test can therefore observe "the reading
   * has not arrived yet" and then "it has", which is the whole point of the
   * timer never blocking the frame.
   */
  mapAsync(_mode?: number): Promise<void> {
    if (this.#mapped) return Promise.reject(new Error('FakeBuffer: already mapped'));
    return new Promise<void>((resolve) => {
      this.#pending.push(() => {
        this.#mapped = true;
        resolve();
      });
      queueMicrotask(() => {
        const next = this.#pending.shift();
        if (next !== undefined) next();
      });
    });
  }

  getMappedRange(offset = 0, size?: number): ArrayBuffer {
    if (!this.#mapped) {
      throw new Error(`FakeBuffer "${this.label}".getMappedRange() before the map resolved — this is the detached-range bug`);
    }
    // A copy, so that the detachment on unmap() is the fake's job and not an
    // accident of sharing one ArrayBuffer. The test asserts on the values.
    return this.bytes.slice(offset, size === undefined ? this.size : offset + size).buffer;
  }

  unmap(): void {
    this.#mapped = false;
  }

  destroy(): void {
    if (this.destroyed) throw new Error(`FakeBuffer "${this.label}".destroy() called twice`);
    this.destroyed = true;
    // A real destroy rejects a pending map; the rejection is what makes a
    // missing `.catch()` on mapAsync visible.
    const pending = this.#pending.splice(0);
    for (const resolve of pending) resolve();
  }

  get mapped(): boolean { return this.#mapped; }
  get mapPending(): number { return this.#pending.length; }
}

export class FakeQuerySet {
  readonly label: string;
  readonly count: number;
  readonly type: string;
  destroyed = false;

  constructor(desc: GPUQuerySetDescriptor) {
    this.label = desc.label ?? '';
    this.count = desc.count;
    this.type = desc.type;
  }

  destroy(): void { this.destroyed = true; }
}

/**
 * `GPUCommandEncoder`, structurally.
 *
 * `GpuTimer.encodeReadback` takes the real interface, so a test needs a real
 * interface; casting once here beats casting at every call site.
 */
export type FakeEncoderAsGpu = GPUCommandEncoder & FakeEncoder;

export class FakeEncoder {
  /** Every call, in order, so a test can assert the exact sequence. */
  readonly calls: string[] = [];
  /** What `resolveQuerySet` writes, as nanoseconds, per query index. */
  timestamps: bigint[] = [];

  resolveQuerySet(
    querySet: FakeQuerySet,
    firstQuery: number,
    queryCount: number,
    destination: FakeBuffer,
    destinationOffset: number,
  ): void {
    this.calls.push(`resolveQuerySet:${querySet.count}:${firstQuery}:${queryCount}:${destination.label}:${destinationOffset}`);
    const view = new BigUint64Array(destination.bytes.buffer, destinationOffset, queryCount);
    for (let i = 0; i < queryCount; i++) {
      view[i] = this.timestamps[firstQuery + i] ?? 0n;
    }
  }

  copyBufferToBuffer(
    source: FakeBuffer,
    sourceOffset: number,
    destination: FakeBuffer,
    destinationOffset: number,
    size: number,
  ): void {
    this.calls.push(`copyBufferToBuffer:${source.label}:${destination.label}:${size}`);
    destination.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), destinationOffset);
  }

  finish(): unknown { return { calls: this.calls }; }
}

/** A {@link FakeGPUDevice} with buffers, query sets, and an encoder. */
export class FakeTimingDevice extends FakeGPUDevice {
  readonly buffers: FakeBuffer[] = [];
  readonly querySets: FakeQuerySet[] = [];
  readonly encoders: FakeEncoder[] = [];
  /** Nanoseconds the "GPU" stamps on each query index. Defaults to 0. */
  timestamps: bigint[] = [];
  readonly submitted: unknown[] = [];
  /** The device's own `uncapturederror` listener, so a test can dispatch to it. */
  #uncaptured: ((event: unknown) => void) | null = null;

  createBuffer(desc: FakeBufferDescriptor): FakeBuffer {
    const buffer = new FakeBuffer(desc);
    this.buffers.push(buffer);
    return buffer;
  }

  createQuerySet(desc: GPUQuerySetDescriptor): FakeQuerySet {
    const set = new FakeQuerySet(desc);
    this.querySets.push(set);
    return set;
  }

  createCommandEncoder(_desc?: GPUCommandEncoderDescriptor): FakeEncoderAsGpu {
    const encoder = new FakeEncoder();
    encoder.timestamps = this.timestamps;
    this.encoders.push(encoder);
    return encoder as FakeEncoderAsGpu;
  }

  /** Live buffers, so a leak assertion has something to count. */
  get liveBuffers(): FakeBuffer[] { return this.buffers.filter((b) => !b.destroyed); }

  override addEventListener(type: string, listener: (event: never) => void): void {
    // The base fake keeps the listener private, and the point of this subclass
    // is to dispatch errors of all three GPUError classes at it.
    if (type === 'uncapturederror') this.#uncaptured = listener as (event: unknown) => void;
    super.addEventListener(type, listener);
  }

  /**
   * Raises an uncaptured error of a chosen class.
   *
   * The base fake hardcodes `GPUValidationError`, and the other two classes are
   * the whole reason apse classifies rather than assumes: an out-of-memory error
   * filed as a validation error sends the reader looking for a wrong descriptor
   * when the real problem is an allocation nobody budgeted for.
   */
  raiseUncapturedAs(className: string, message: string): void {
    if (this.#uncaptured === null) throw new Error('FakeTimingDevice: no uncapturederror listener');
    this.#uncaptured({ error: { message, constructor: { name: className } } });
  }

  get queue(): { submit: (buffers: unknown[]) => void; writeBuffer: () => void } {
    return {
      submit: (buffers: unknown[]) => { this.submitted.push(...buffers); },
      writeBuffer: () => { /* not used by the layer under test */ },
    };
  }
}

/** The device as `GpuTimer` sees it: one field. */
export function asTimingDevice(fake: FakeTimingDevice): { readonly device: GPUDevice } {
  return { device: fake as unknown as GPUDevice };
}

/**
 * Raises an uncaptured error of `className` through a device's own listener.
 *
 * The listener was registered by `attach()`, so this reaches exactly the code
 * path the browser reaches.
 */
export function raiseUncapturedAs(device: GPUDevice, className: string, message: string): void {
  (device as unknown as FakeTimingDevice).raiseUncapturedAs(className, message);
}

// ---------------------------------------------------------------------------
// navigator.gpu
// ---------------------------------------------------------------------------

export interface FakeGpuOptions {
  /** Features the adapter advertises. */
  readonly features?: string[];
  /** Limits the adapter advertises. Defaults to the compatibility defaults. */
  readonly limits?: GPUSupportedLimits;
  readonly preferredFormat?: GPUTextureFormat;
  /** Make `requestAdapter` return null, to reach ADAPTER_UNAVAILABLE. */
  readonly noAdapter?: boolean;
  /** Make a 'core' probe return null, to reach the compatibility fallback. */
  readonly noCore?: boolean;
  /** Reject `requestDevice`, to reach DEVICE_REQUEST_FAILED. */
  readonly rejectDevice?: Error;
  /** Features the *device* is created with. Defaults to the adapter's. */
  readonly deviceFeatures?: string[];
}

export class FakeAdapter {
  readonly features: ReadonlySet<string>;
  readonly limits: GPUSupportedLimits;
  readonly info: { vendor: string; architecture: string; device: string; description: string };
  /** The descriptor `requestDevice` was last called with, verbatim. */
  readonly requests: GPUDeviceDescriptor[] = [];
  #opts: FakeGpuOptions;

  constructor(opts: FakeGpuOptions = {}) {
    this.#opts = opts;
    this.features = new Set(opts.features ?? []);
    this.limits = opts.limits ?? fakeLimits();
    this.info = { vendor: 'fake', architecture: 'none', device: 'fake', description: 'FakeAdapter' };
  }

  async requestDevice(desc: GPUDeviceDescriptor = {}): Promise<GPUDevice> {
    this.requests.push(desc);
    if (this.#opts.rejectDevice !== undefined) throw this.#opts.rejectDevice;
    return new FakeTimingDevice({
      limits: copyPrototypeLimits(this.limits),
      features: this.#opts.deviceFeatures ?? [...this.features],
    }) as unknown as GPUDevice;
  }
}

/**
 * Rebuilds a limits object with the same prototype-getter shape.
 *
 * The shape is the point: `{...limits}` on the result is `{}`, so a device that
 * reported plain own properties would hide the `requiredLimits` bug that this
 * file's tests exist to catch.
 */
function copyPrototypeLimits(limits: GPUSupportedLimits): GPUSupportedLimits {
  const source = Object.getPrototypeOf(limits) as Record<string, number>;
  const out: Record<string, number> = {};
  for (const key of Object.getOwnPropertyNames(source)) {
    Object.defineProperty(out, key, { get: () => source[key] as number, enumerable: true, configurable: true });
  }
  return Object.create(out) as GPUSupportedLimits;
}

export interface FakeGpuHandle {
  gpu: GPU;
  adapter: FakeAdapter;
  /** Every `requestAdapter` call, in order, so the probe order is testable. */
  readonly adapterRequests: GPURequestAdapterOptions[];
}

/**
 * Installs a fake `navigator.gpu` for the duration of `fn`, and restores
 * whatever was there afterwards.
 *
 * Restoring matters: `createDevice` keeps a module-level `WeakMap` of configured
 * canvases, and a test that leaked a `navigator` would make every later test in
 * the file see a GPU that is not there.
 */
export async function withGpu<T>(opts: FakeGpuOptions, fn: (handle: FakeGpuHandle) => Promise<T> | T): Promise<T> {
  const adapter = new FakeAdapter(opts);
  const adapterRequests: GPURequestAdapterOptions[] = [];
  const gpu = {
    async requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null> {
      adapterRequests.push(options ?? {});
      if (opts.noAdapter === true) return null;
      if (options?.featureLevel === 'core' &&
          (opts.noCore === true || !adapter.features.has('core-features-and-limits'))) {
        // An adapter that cannot do core returns null for a core request, which
        // is what the compatibility fallback path exists for.
        return null;
      }
      return adapter as unknown as GPUAdapter;
    },
    getPreferredCanvasFormat(): GPUTextureFormat {
      return opts.preferredFormat ?? 'bgra8unorm';
    },
  } as unknown as GPU;

  const globals = globalThis as { navigator?: unknown };
  const hadNavigator = 'navigator' in globals;
  const previous = globals.navigator;
  globals.navigator = { gpu };
  try {
    return await fn({ gpu, adapter, adapterRequests });
  } finally {
    if (hadNavigator) globals.navigator = previous;
    else delete globals.navigator;
  }
}

/** A canvas whose `getContext('webgpu')` returns a recording fake context. */
export function fakeWebGpuCanvas(width = 300, height = 200): {
  canvas: HTMLCanvasElement;
  configurations: GPUCanvasConfiguration[];
} {
  const configurations: GPUCanvasConfiguration[] = [];
  const texture = { label: 'canvas.current', createView: () => ({}) };
  const context = {
    configure(d: GPUCanvasConfiguration): void { configurations.push(d); },
    unconfigure(): void { /* no-op */ },
    getCurrentTexture(): unknown { return texture; },
  };
  const canvas = {
    width,
    height,
    clientWidth: width,
    clientHeight: height,
    getContext(kind: string): unknown { return kind === 'webgpu' ? context : null; },
  } as unknown as HTMLCanvasElement;
  return { canvas, configurations };
}

/** The feature levels a fake adapter can pretend to be. */
export const COMPAT: FeatureLevel = 'compatibility';
export const CORE: FeatureLevel = 'core';
