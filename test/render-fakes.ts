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
  /**
   * Hand the buffer over already mapped.
   *
   * `mappedAtCreation` is the only way to write a buffer without a staging copy,
   * and `GpuMesh` uses it for every vertex and index upload. The fake honours
   * it, because the alternative — a `getMappedRange()` that throws — would make
   * every geometry test a test of the fake.
   */
  mappedAtCreation?: boolean;
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
    this.#mapped = desc.mappedAtCreation === true;
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

  get queue(): GPUQueue {
    return {
      submit: (buffers: unknown[]) => { this.submitted.push(...buffers); },
      writeBuffer: () => { /* not used by the layer under test */ },
    } as unknown as GPUQueue;
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
// A frame: render passes, pipelines, and a queue that records writes
//
// The frame loop is the one layer with no fake anywhere in the repo, because
// everything it does is an API call on a device — and the properties worth
// testing are all *about* those calls. Whether N instances cost one
// `drawIndexed` or N is not visible in a return value; it is visible in the
// sequence of calls the encoder received, and nowhere else. Same for whether the
// present pass shares the frame's encoder, whether the frame uniform was
// actually marked dirty, and whether a timestamp write index is attached to a
// pass that has already used it.
//
// So this models the calls, not the GPU. Nothing rasterises; `drawIndexed` is a
// line in an array.
// ---------------------------------------------------------------------------

/** One `draw` or `drawIndexed`, with the arguments apse passed. */
export interface RecordedDraw {
  readonly kind: 'draw' | 'drawIndexed';
  /** `vertexCount` for `draw`, `indexCount` for `drawIndexed`. */
  readonly count: number;
  /** The second argument either way: how many copies of the mesh. */
  readonly instances: number;
  /** `firstVertex` or `firstIndex`. */
  readonly first: number;
  /** `baseVertex` for `drawIndexed`, `0` for `draw`. */
  readonly base: number;
  /** `firstInstance`, the same for both. */
  readonly firstInstance: number;
}

/** A `GPURenderPassEncoder`, recorded. */
export class FakeFramePass {
  readonly label: string;
  readonly timestampWrites: GPURenderPassTimestampWrites | undefined;
  readonly depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined;
  readonly colorFormats: readonly GPUTextureFormat[] = [];
  readonly calls: string[] = [];
  readonly pipelines: GPURenderPipeline[] = [];
  /** `[group, bindGroup, dynamic offsets]`, in call order. */
  readonly bindGroups: { group: number; bindGroup: GPUBindGroup; offsets: number[] | null }[] = [];
  /** `[slot, buffer]`, in call order. Slot 1 is the per-instance stream. */
  readonly vertexBuffers: { slot: number; buffer: GPUBuffer }[] = [];
  readonly indexBuffers: { buffer: GPUBuffer; format: GPUIndexFormat }[] = [];
  readonly draws: RecordedDraw[] = [];
  ended = false;

  constructor(desc: GPURenderPassDescriptor) {
    this.label = desc.label ?? '';
    this.timestampWrites = desc.timestampWrites;
    this.depthStencilAttachment = desc.depthStencilAttachment;
    this.calls.push('beginRenderPass');
  }

  setPipeline(pipeline: GPURenderPipeline): void {
    this.calls.push('setPipeline');
    this.pipelines.push(pipeline);
  }

  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: Iterable<number>): void {
    this.calls.push(`setBindGroup:${index}`);
    this.bindGroups.push({
      group: index,
      bindGroup,
      offsets: dynamicOffsets === undefined ? null : [...dynamicOffsets],
    });
  }

  setVertexBuffer(slot: number, buffer: GPUBuffer, _offset?: number): void {
    this.calls.push(`setVertexBuffer:${slot}`);
    this.vertexBuffers.push({ slot, buffer });
  }

  setIndexBuffer(buffer: GPUBuffer, format: GPUIndexFormat, _offset?: number): void {
    this.calls.push(`setIndexBuffer:${format}`);
    this.indexBuffers.push({ buffer, format });
  }

  draw(vertexCount: number, instanceCount: number, firstVertex: number, firstInstance: number): void {
    this.calls.push('draw');
    this.draws.push({ kind: 'draw', count: vertexCount, instances: instanceCount, first: firstVertex, base: 0, firstInstance });
  }

  drawIndexed(
    indexCount: number,
    instanceCount: number,
    firstIndex: number,
    baseVertex: number,
    firstInstance: number,
  ): void {
    this.calls.push('drawIndexed');
    this.draws.push({ kind: 'drawIndexed', count: indexCount, instances: instanceCount, first: firstIndex, base: baseVertex, firstInstance });
  }

  end(): void {
    this.ended = true;
    this.calls.push('end');
  }
}

/** A `GPUCommandEncoder` that can open a pass. */
export class FakeFrameEncoder extends FakeEncoder {
  readonly passes: FakeFramePass[] = [];

  beginRenderPass(desc: GPURenderPassDescriptor): GPURenderPassEncoder {
    const pass = new FakeFramePass(desc);
    this.passes.push(pass);
    this.calls.push(`beginRenderPass:${pass.label}`);
    return pass as unknown as GPURenderPassEncoder;
  }

  copyTextureToBuffer(
    source: { texture: GPUTexture },
    destination: { buffer: GPUBuffer; bytesPerRow: number },
    copySize: { width: number; height: number },
  ): void {
    const from = (source.texture as unknown as { label: string }).label ?? '';
    this.calls.push(`copyTextureToBuffer:${from}:${copySize.width}x${copySize.height}`);
    const buffer = destination.buffer as unknown as FakeBuffer;
    buffer.bytes.fill(0, 0, destination.bytesPerRow * copySize.height);
  }
}

/** One `queue.writeBuffer`, with the bytes copied out rather than retained. */
export interface RecordedFrameWrite {
  readonly label: string;
  readonly bufferOffset: number;
  readonly size: number;
  readonly bytes: Uint8Array;
}

/**
 * A {@link FakeTimingDevice} that can encode a whole frame.
 *
 * The addition is a render pass, a pipeline, and a queue — nothing else. The
 * shader module answers with whatever diagnostics the test asks for, because the
 * fake must never compile WGSL: a second WGSL front end would only prove that it
 * agrees with itself.
 */
export class FakeFrameDevice extends FakeTimingDevice {
  readonly modules: { label: string; code: string }[] = [];
  readonly pipelineDescs: GPURenderPipelineDescriptor[] = [];
  readonly bindGroupLayouts: GPUBindGroupLayoutDescriptor[] = [];
  readonly pipelineLayouts: GPUPipelineLayoutDescriptor[] = [];
  readonly bindGroups: GPUBindGroupDescriptor[] = [];
  readonly samplers: GPUSamplerDescriptor[] = [];
  readonly writes: RecordedFrameWrite[] = [];
  /** Every encoder handed out, so a test can see one submit and not two. */
  readonly frameEncoders: FakeFrameEncoder[] = [];

  override createCommandEncoder(desc?: GPUCommandEncoderDescriptor): FakeEncoderAsGpu {
    const encoder = new FakeFrameEncoder();
    encoder.timestamps = this.timestamps;
    this.frameEncoders.push(encoder);
    this.encoders.push(encoder);
    void desc;
    return encoder as unknown as FakeEncoderAsGpu;
  }

  createShaderModule(desc: GPUShaderModuleDescriptor): GPUShaderModule {
    this.modules.push({ label: desc.label ?? '', code: desc.code });
    return {
      label: desc.label,
      getCompilationInfo: () => Promise.resolve({ messages: [] } as unknown as GPUCompilationInfo),
    } as unknown as GPUShaderModule;
  }

  createRenderPipelineAsync(desc: GPURenderPipelineDescriptor): Promise<GPURenderPipeline> {
    this.pipelineDescs.push(desc);
    return Promise.resolve({ label: desc.label } as unknown as GPURenderPipeline);
  }

  createBindGroupLayout(desc: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
    this.bindGroupLayouts.push(desc);
    return { label: desc.label } as unknown as GPUBindGroupLayout;
  }

  createPipelineLayout(desc: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
    this.pipelineLayouts.push(desc);
    return { label: desc.label } as unknown as GPUPipelineLayout;
  }

  createBindGroup(desc: GPUBindGroupDescriptor): GPUBindGroup {
    this.bindGroups.push(desc);
    return { label: desc.label } as unknown as GPUBindGroup;
  }

  createSampler(desc: GPUSamplerDescriptor): GPUSampler {
    this.samplers.push(desc);
    return { label: desc.label } as unknown as GPUSampler;
  }

  override get queue(): GPUQueue {
    const device = this;
    return {
      writeBuffer(
        buffer: GPUBuffer,
        bufferOffset: number,
        data: BufferSource,
        dataOffset?: number,
        size?: number,
      ): void {
        // Copied, because apse hands over a live CPU mirror and writes to it
        // again next frame: a retained view would make every recorded frame read
        // the last one's values.
        const from = dataOffset ?? 0;
        const bytes = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(from, from + (size ?? Infinity))
          : new Uint8Array(data).slice(from, from + (size ?? Infinity));
        device.writes.push({
          label: (buffer as unknown as { label: string }).label ?? '',
          bufferOffset,
          size: bytes.byteLength,
          bytes,
        });
      },
      submit(buffers: Iterable<unknown>): void {
        for (const buffer of buffers) device.submitted.push(buffer);
      },
    } as unknown as GPUQueue;
  }

  /** The passes of the most recent encoder, in order. */
  get lastPasses(): readonly FakeFramePass[] {
    return this.frameEncoders[this.frameEncoders.length - 1]?.passes ?? [];
  }

  /** Every pass opened by any encoder, flattened. */
  get allPasses(): readonly FakeFramePass[] {
    return this.frameEncoders.flatMap((encoder) => encoder.passes);
  }

  /** Every draw of the most recent encoder, across all of its passes. */
  get lastDraws(): readonly RecordedDraw[] {
    return this.lastPasses.flatMap((pass) => pass.draws);
  }
}

/**
 * The WebGPU bit-flag globals, which bun does not have.
 *
 * **`src` reads `GPUBufferUsage`, `GPUShaderStage` and `GPUTextureUsage` as
 * globals** wherever the value is needed to build a descriptor, so a headless
 * test that constructs a material, a mesh or a renderer has to install them —
 * that is the only reason the frame loop had no test at all. The values are the
 * spec's, written out rather than imported, so a wrong bit here shows up as a
 * wrong `usage` in a recorded descriptor instead of being masked by a constant
 * that happens to agree.
 */
export function installWebGpuBitmaps(): () => void {
  const globals = globalThis as Record<string, unknown>;
  const previous = new Map<string, unknown>();
  const set = (name: string, value: unknown): void => {
    previous.set(name, globals[name]);
    globals[name] = value;
  };
  set('GPUBufferUsage', {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
  });
  set('GPUTextureUsage', {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  });
  set('GPUShaderStage', { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 });
  set('GPUMapMode', { READ: 0x0001, WRITE: 0x0002 });
  return () => {
    for (const [name, value] of previous) {
      if (value === undefined) delete globals[name];
      else globals[name] = value;
    }
  };
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
  /**
   * Hand this device back from `requestDevice` instead of a fresh
   * {@link FakeTimingDevice}.
   *
   * For a test that needs a fake with more surface than the timer does — a render
   * pass, a pipeline, a queue that records writes — while still going through
   * the real `createDevice`. Going through the real one is the point: the canvas
   * configuration, the 1x1 validity probe, and the limits copy are three things
   * only `createDevice` does, and a test that built an `AseDevice` by hand would
   * bypass all three.
   */
  readonly device?: GPUDevice;
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
    if (this.#opts.device !== undefined) return this.#opts.device;
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
