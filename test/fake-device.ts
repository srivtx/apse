/**
 * A fake GPU device, for testing the render-target layer without a GPU.
 *
 * There is no GPU in the test environment, and there should not be one: the
 * point of these tests is the *bookkeeping* — that an MSAA target allocates two
 * colour textures, that a same-size resize allocates nothing, that dispose
 * leaves nothing alive — and none of that is the driver's business. It is
 * apse's, and a fake that records every `createTexture` and every `destroy`
 * proves it exactly.
 *
 * The fake mirrors WebGPU in the specific ways that make bugs possible:
 *
 *   - **`limits` is an object of prototype getters with no own properties**,
 *     exactly like a real `GPUSupportedLimits`. This is not incidental. It is
 *     the shape that makes `{...adapter.limits}` evaluate to `{}`, and a fake
 *     built with plain own properties would let that bug pass a test suite.
 *   - **Error scopes are real**: `pushErrorScope`/`popErrorScope` record and
 *     replay, so `withErrorScope` is genuinely exercised rather than mocked.
 *   - **`getCurrentTexture()` is tracked, not cached**, and a new texture
 *     appears after a canvas resize, because that is how the real one behaves
 *     and it is what the canvas-target tests need to observe.
 */

import { COMPAT_LIMITS, copyLimits } from '../src/render/device.ts';
import type { AseLimits, FeatureLevel } from '../src/render/device.ts';
import type { CanvasTargetDevice, RenderTargetDevice } from '../src/render/target.ts';

// ---------------------------------------------------------------------------
// Limits as prototype getters
// ---------------------------------------------------------------------------

/**
 * Builds a `GPUSupportedLimits`-shaped object whose members are **prototype
 * getters**, with no own properties at all.
 *
 * This is the shape Chrome and every other conformant implementation actually
 * produces, and it is the reason this file exists. `{ ...limits }` copies own
 * enumerable properties, so on this object the spread is `{}` — not "the
 * limits it can see", nothing. A renderer that wrote
 *
 *     const requiredLimits = { ...adapter.limits };
 *
 * would send an empty record, believe it had mirrored the adapter, and then
 * overflow a 16 KiB uniform block on a compatibility-mode device. A fake built
 * with plain own properties would hide that completely.
 */
export function fakeLimits(overrides: Partial<Record<string, number>> = {}): GPUSupportedLimits {
  const values: Record<string, number> = { ...COMPAT_LIMITS, ...overrides };
  const prototype: Record<string, number> = {};
  for (const [key, value] of Object.entries(values)) {
    // `enumerable: true` matters: real IDL attributes are enumerable, so
    // `for...in` and `Object.assign` see them. It is *own-ness* that they lack,
    // and own-ness is the only thing spread looks at.
    Object.defineProperty(prototype, key, { get: () => value, enumerable: true, configurable: true });
  }
  return Object.create(prototype) as GPUSupportedLimits;
}

// ---------------------------------------------------------------------------
// Textures and views
// ---------------------------------------------------------------------------

/** The label the fake gives the browser-owned canvas texture. */
const CANVAS_TEXTURE_LABEL = 'canvas.current';

export class FakeTexture {
  /** The device that created it, so `destroy()` can be logged centrally. */
  readonly owner: FakeGPUDevice | null;
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly format: GPUTextureFormat;
  readonly sampleCount: number;
  readonly usage: number;
  destroyed = false;
  viewsCreated = 0;

  constructor(desc: GPUTextureDescriptor, owner: FakeGPUDevice | null = null) {
    this.owner = owner;
    this.label = desc.label ?? '';
    const size = desc.size as { width: number; height?: number };
    this.width = size.width;
    this.height = size.height ?? 1;
    this.format = desc.format;
    this.sampleCount = desc.sampleCount ?? 1;
    this.usage = desc.usage ?? 0;
  }

  createView(desc?: GPUTextureViewDescriptor): GPUTextureView {
    this.viewsCreated++;
    // A real view of a destroyed texture is invalid and silently does nothing.
    // Reproduce the invalidity so a test that uses a stale view fails loudly
    // rather than appearing to work.
    const texture = this;
    return {
      label: desc?.label ?? this.label,
      get destroyed() { return texture.destroyed; },
    } as unknown as GPUTextureView;
  }

  destroy(): void {
    if (this.destroyed) {
      throw new Error(`FakeTexture "${this.label}".destroy() called twice — the real API allows it, so apse must not be relying on it.`);
    }
    this.destroyed = true;
    this.owner?.destroyLog.push(this.label);
  }

  /** True when the usage flags include every bit in `flags`. */
  hasUsage(flags: number): boolean {
    return (this.usage & flags) === flags;
  }
}

// ---------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------

export interface FakeDeviceOptions {
  limits?: GPUSupportedLimits;
  features?: Iterable<string>;
  /** Errors to hand back from `popErrorScope`, oldest first. */
  errorScopes?: (GPUError | null)[];
}

export class FakeGPUDevice {
  readonly limits: GPUSupportedLimits;
  readonly features: ReadonlySet<string>;
  readonly lost: Promise<{ reason: GPUDeviceLostReason; message: string }>;

  /** Every texture handed out, in creation order, destroyed or not. */
  readonly textures: FakeTexture[] = [];
  /** `createTexture` descriptors, in order. The clearest test assertion. */
  readonly createLog: GPUTextureDescriptor[] = [];
  /** Labels of every `destroy()` call, in order. */
  readonly destroyLog: string[] = [];
  /**
   * Set to skip recording the browser-owned canvas current texture.
   *
   * The fake fabricates one so `getCurrentTexture()` has something to hand out,
   * and it must not show up in assertions about what *apse* allocated.
   */
  recordCanvasTexture = true;

  destroyed = false;
  /** Error scopes pushed but not yet popped. */
  readonly scopeStack: GPUErrorFilter[] = [];
  /** Everything popErrorScope has returned, in order. */
  readonly poppedErrors: (GPUError | null)[] = [];

  #queuedErrors: (GPUError | null)[];
  #resolveLost: ((info: { reason: GPUDeviceLostReason; message: string }) => void) | null = null;
  #uncaptured: ((event: GPUUncapturedErrorEvent) => void) | null = null;

  constructor(opts: FakeDeviceOptions = {}) {
    this.limits = opts.limits ?? fakeLimits();
    this.features = new Set(opts.features ?? []);
    this.#queuedErrors = [...(opts.errorScopes ?? [])];
    this.lost = new Promise((resolve) => { this.#resolveLost = resolve; });
  }

  /** Textures created and not yet destroyed. The core assertion of this file. */
  get liveTextures(): FakeTexture[] {
    return this.textures.filter((t) => !t.destroyed);
  }

  /** Live textures whose label ends with the given suffix, in creation order. */
  live(labelSuffix: string): FakeTexture[] {
    return this.liveTextures.filter((t) => t.label.endsWith(labelSuffix));
  }

  /** Returns the concrete fake, so tests can inspect it. See asGpuDevice(). */
  /** Only the entries apse allocated, excluding the browser-owned canvas one. */
  get apseTextures(): FakeTexture[] {
    return this.textures.filter((t) => t.label !== CANVAS_TEXTURE_LABEL);
  }

  /** `createTexture` descriptors apse caused, in order. */
  get apseLog(): GPUTextureDescriptor[] {
    return this.createLog.filter((d) => d.label !== CANVAS_TEXTURE_LABEL);
  }

  /** Live apse textures whose label ends with the given suffix. */
  apseLive(labelSuffix: string): FakeTexture[] {
    return this.apseTextures.filter((t) => !t.destroyed && t.label.endsWith(labelSuffix));
  }

  createTexture(desc: GPUTextureDescriptor): FakeTexture {
    if (this.destroyed) {
      throw new Error('FakeGPUDevice: createTexture() after destroy()');
    }
    const texture = new FakeTexture(desc, this);
    if (desc.label === CANVAS_TEXTURE_LABEL || this.recordCanvasTexture) {
      this.textures.push(texture);
      this.createLog.push(desc);
    }
    return texture;
  }

  pushErrorScope(filter: GPUErrorFilter): void {
    this.scopeStack.push(filter);
  }

  popErrorScope(): Promise<GPUError | null> {
    const filter = this.scopeStack.pop();
    if (filter === undefined) {
      throw new Error('FakeGPUDevice: popErrorScope() with no matching pushErrorScope()');
    }
    const next = this.#queuedErrors.shift() ?? null;
    this.poppedErrors.push(next);
    return Promise.resolve(next);
  }

  addEventListener(type: string, listener: (event: never) => void): void {
    if (type === 'uncapturederror') {
      this.#uncaptured = listener as (event: GPUUncapturedErrorEvent) => void;
    }
  }

  removeEventListener(): void {}

  destroy(): void {
    this.destroyed = true;
    // Mirrors the real contract: destroy() resolves `lost` with reason
    // "destroyed", and destroying twice is explicitly allowed.
    this.#resolveLost?.({ reason: 'destroyed', message: 'Device was destroyed' });
  }

  /** Simulate a driver-level loss. */
  lose(reason: GPUDeviceLostReason = 'unknown', message = 'simulated loss'): void {
    this.#resolveLost?.({ reason, message });
  }

  /** Raise an uncaptured error, as the browser would. */
  raiseUncaptured(message: string): void {
    if (this.#uncaptured === null) throw new Error('no uncapturederror listener');
    const error = { message, constructor: { name: 'GPUValidationError' } };
    this.#uncaptured({ error } as unknown as GPUUncapturedErrorEvent);
  }
}

// ---------------------------------------------------------------------------
// A fake AseDevice
// ---------------------------------------------------------------------------

/** `GPUTextureUsage` values the fake needs, inlined so the test has no globals. */
const USAGE = {
  COPY_SRC: 0x01,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;

export { USAGE as FAKE_TEXTURE_USAGE };

export interface FakeAseDeviceOptions {
  limits?: GPUSupportedLimits;
  features?: Iterable<string>;
  format?: GPUTextureFormat;
  canvasWidth?: number;
  canvasHeight?: number;
  sampleCount?: 1 | 4;
  maxPixelRatio?: number;
  maxObjects?: number;
  featureLevel?: FeatureLevel;
}

export interface FakeCanvasContext {
  /** Number of `getCurrentTexture()` calls. */
  getCurrentTextureCalls: number;
  /** The texture the next `getCurrentTexture()` will return. */
  /** The FakeTexture the next getCurrentTexture() will hand out, creating it. */
  peekCurrentTexture(): FakeTexture;
  configure(desc: GPUCanvasConfiguration): void;
  unconfigure(): void;
  getCurrentTexture(): GPUTexture;
  getConfiguration(): GPUCanvasConfiguration | null;
  /** Simulate a canvas resize: the current texture is replaced, as in a browser. */
  resize(width: number, height: number): void;
}

export interface FakeAseDevice {
  device: FakeGPUDevice;
  limits: AseLimits;
  format: GPUTextureFormat;
  featureLevel: FeatureLevel;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  adapterInfo: { vendor: string; architecture: string; device: string; description: string };
  hasTimestampQuery: boolean;
  hasSubgroups: boolean;
  maxObjects: number;
  sampleCount: 1 | 4;
  maxPixelRatio: number;
  /** True once `loseDevice()` has been called. */
  lost(): boolean;
  loseDevice(reason?: GPUDeviceLostReason, message?: string): void;
  assertLive(): void;
}

/**
 * An `AseDevice` made of a fake device, a fake canvas, and a fake context.
 *
 * `assertLive()` throws the same `AseError` code the real one does, so the
 * target's liveness checks are genuinely exercised — a stub that returned
 * silently would let a use-after-loss regression through.
 */
export function fakeAseDevice(opts: FakeAseDeviceOptions = {}): FakeAseDevice {
  const device = new FakeGPUDevice({
    limits: opts.limits ?? fakeLimits(),
    features: opts.features,
  });
  const format = opts.format ?? 'bgra8unorm';
  let lostFlag = false;

  const canvas = {
    width: opts.canvasWidth ?? 800,
    height: opts.canvasHeight ?? 600,
  } as unknown as HTMLCanvasElement;

  const makeCurrentTexture = (): FakeTexture => device.createTexture({
    label: CANVAS_TEXTURE_LABEL,
    size: { width: canvas.width, height: canvas.height },
    format,
    sampleCount: 1,
    // The real canvas texture is created by the browser with whatever usage the
    // configuration asked for. The target must never destroy it, so this fake
    // texture is tracked separately from the ones apse allocated.
    usage: USAGE.RENDER_ATTACHMENT | USAGE.COPY_SRC,
  });

  // The texture the browser owns, created lazily on the first
  // `getCurrentTexture()`. apse must never destroy it, which is why it is a
  // distinct field from anything the target allocates — and lazy so that a test
  // about an offscreen target sees only apse's own allocations.
  let current: FakeTexture | null = null;
  const currentTexture = (): FakeTexture => current ??= makeCurrentTexture();

  const context: FakeCanvasContext = {
    getCurrentTextureCalls: 0,
    configure() { /* no-op */ },
    unconfigure() { /* no-op */ },
    getConfiguration() { return null; },
    getCurrentTexture(): GPUTexture {
      context.getCurrentTextureCalls++;
      // The current texture survives until the next resize; a real one expires
      // at present, but for the purpose of "was a new texture handed out" a
      // resize is the only observable change, and that is what matters here.
      return currentTexture() as unknown as GPUTexture;
    },
    peekCurrentTexture(): FakeTexture {
      return currentTexture();
    },
    resize(width: number, height: number): void {
      canvas.width = width;
      canvas.height = height;
      // The old current texture is expired by the resize. Not destroyed — the
      // browser owns it and apse must never call destroy() on it.
      current = makeCurrentTexture();
    },
  };

  const ase: FakeAseDevice = {
    device,
    // Copy through the real copyLimits() so the fake reports the limits its
    // device actually has — a core-mode device reports 8192, and a target that
    // clamped to the compat number regardless would be a portability bug.
    limits: copyLimits(device.limits),
    format,
    featureLevel: opts.featureLevel ?? 'compatibility',
    canvas,
    context: context as unknown as GPUCanvasContext,
    adapterInfo: { vendor: 'fake', architecture: 'fake', device: 'fake', description: 'FakeGPUDevice' },
    hasTimestampQuery: device.features.has('timestamp-query'),
    hasSubgroups: device.features.has('subgroups'),
    maxObjects: opts.maxObjects ?? 4096,
    sampleCount: opts.sampleCount ?? 1,
    maxPixelRatio: opts.maxPixelRatio ?? 2,
    lost: () => lostFlag,
    loseDevice(reason: GPUDeviceLostReason = 'unknown', message = 'simulated loss'): void {
      lostFlag = true;
      device.lose(reason, message);
    },
    assertLive(): void {
      if (!lostFlag) return;
      // Throws the same code the real assertLive() does, so the target's
      // liveness checks are genuinely exercised and a test can assert on
      // `error.code` rather than on a message string.
      throw new FakeDeviceLostError('simulated device loss');
    },
  };
  return ase;
}

/** The error `FakeAseDevice.assertLive()` throws. Matches the real code. */
export class FakeDeviceLostError extends Error {
  readonly code = 'DEVICE_LOST';
  constructor(message: string) {
    super(message);
    this.name = 'AseError';
  }
}

/** Type-narrowing helper: treat a fake device as a real one where required. */
export function asGpuDevice(device: FakeGPUDevice): GPUDevice {
  return device as unknown as GPUDevice;
}

/** Type-narrowing helper for the fake limits object. */
export function asAseLimits(limits: GPUSupportedLimits): AseLimits {
  return limits as AseLimits;
}

/**
 * The fake as the minimal device a render target accepts.
 *
 * `RenderTargetImpl` deliberately takes a three-field structural type rather
 * than the whole `AseDevice`, which is what makes this possible: the target's
 * texture accounting can be tested with no canvas, no context and no adapter.
 */
export function asTargetDevice(fake: FakeAseDevice): RenderTargetDevice {
  return {
    device: asGpuDevice(fake.device),
    limits: fake.limits,
    assertLive: () => fake.assertLive(),
  };
}

/** The fake as a canvas target's device, context included. */
export function asCanvasTargetDevice(fake: FakeAseDevice): CanvasTargetDevice {
  return {
    ...asTargetDevice(fake),
    canvas: fake.canvas,
    context: fake.context,
    format: fake.format,
  };
}
