/**
 * GPU timing, by timestamp query.
 *
 * # Why this is not a `performance.now()` around a submit
 *
 * Because that measures the CPU. `queue.submit()` returns after the commands
 * have been *recorded*, not after the GPU has run them, so a frame that takes
 * 0.1 ms of CPU and 14 ms of GPU reports 0.1. A renderer that reports CPU time
 * as "GPU time" is worse than one that reports nothing, because the number looks
 * real: a user profiling a stuttering scene sees a comfortable 0.4 ms and
 * concludes the problem is elsewhere. It is the single most common way a WebGPU
 * renderer's own instrumentation lies.
 *
 * # Why `gpu` is `null` and not `0`
 *
 * `timestamp-query` is an *optional* feature, present on roughly half of all
 * devices, and apse never requires it — requiring it would make `requestDevice`
 * reject on exactly the phones the library exists to run on. So on a device
 * without it there is no measurement. The honest report is `null`, and that is
 * why `FrameStats.gpu` is `number | null`: **a missing measurement and a
 * measured zero are different facts, and a type that cannot express the
 * difference guarantees one of them gets reported as the other.**
 *
 * # The three hazards, all of which are silent
 *
 * 1. **The epoch is arbitrary and non-zero.** Timestamps are nanoseconds on a
 *    monotonic clock whose origin is unspecified — on Chromium, a
 *    `QuerySet`-scaled value derived from `base::TimeTicks` and deliberately
 *    offset so that the answer is not a wall-clock timestamp. Reporting the raw
 *    value as milliseconds produces a number around 4.5e9, which is
 *    indistinguishable from a real measurement to any code that just prints it.
 *    Only the *difference* between the two ends of a pair is a duration, and
 *    only within one frame. Nothing here ever subtracts across frames.
 * 2. **`resolveQuerySet` and `copyBufferToBuffer` must be encoded, not
 *    awaited.** The resolve is a command on the queue, so it happens after the
 *    pass that wrote the timestamps, in submission order, whatever the JS does.
 *    The map then takes a promise round trip, and awaiting it inside `render()`
 *    would stall the frame for a whole frame time.
 * 3. **A mapped buffer cannot be a copy destination.** `copyBufferToBuffer` into
 *    a buffer with a pending or active map is a validation error that invalidates
 *    the command buffer — so the readback uses a small ring, and a frame whose
 *    slot is still in flight is *dropped*, counted in {@link GpuTimer.dropped},
 *    and never blocks.
 *
 * # What is not here, and why
 *
 * A `resize()`. Every resource this owns is a fixed 8 bytes per query
 * regardless of how large the framebuffer is, so a resize cannot change their
 * size, and a reallocation path that provably cannot do anything useful is one
 * more way to strand an in-flight map. Framebuffer-dependent readback lives in
 * `readback.ts`, where the size genuinely depends on the resolution.
 */

import { fail } from '../core/error.ts';
import { Resource } from '../core/resource.ts';
import { err, ok } from '../core/result.ts';
import type { Result } from '../core/result.ts';
import { BUFFER_USAGE, MAP_MODE, TIMESTAMP_BYTES, requireFeature } from './device.ts';
import type { GpuFeatureFailure } from './device.ts';

/**
 * The two ways {@link GpuTimer.create} can decline, which are different problems
 * with different fixes: the device has no `timestamp-query` (a fact about the
 * machine, so carry on with `stats.gpu === null`), or it advertises the feature
 * and then refuses the allocation (a driver fault, worth reporting).
 */
export type GpuTimerFailure = GpuFeatureFailure | 'TIMESTAMP_ALLOCATION_FAILED';

/**
 * One `(beginning, end)` pair is two query slots, and each resolved slot is a
 * `uint64`. Stated once because the buffer size is derived from it and a
 * hand-written 16 there is a bug waiting for a second pair.
 */
const SLOTS_PER_PAIR = 2;

/**
 * Default readback slots.
 *
 * Two, not one: with one, every frame's copy would land on the buffer the
 * previous frame's `mapAsync` is still using, so the timer would report
 * nothing at all. With two, one can be mapping while the other is written, and
 * the copy is dropped only if a frame ever takes longer than two readbacks.
 */
const DEFAULT_STAGING_SLOTS = 2;

/** The structural minimum: an `AseDevice` satisfies it, a three-field fake does too. */
export interface TimingDevice {
  readonly device: GPUDevice;
}

export interface GpuTimerOptions {
  /**
   * `(beginning, end)` pairs to allocate. Default 1.
   *
   * One pair is normally enough, because a pair's two ends can be attached to
   * *different* passes: put the beginning on the first pass of the frame and the
   * end on the last, and the measurement covers the whole frame including the
   * present. More than one is for per-pass attribution, which is worth a second
   * pipeline slot when a frame is over budget and you need to know which pass.
   */
  readonly pairs?: number;
  /**
   * MAP_READ buffers to cycle through. Default 2. Must be at least 1.
   *
   * See {@link DEFAULT_STAGING_SLOTS} for why the default is not 1. Larger costs
   * `pairs * 2 * 8` bytes each and buys tolerance for a slow readback.
   */
  readonly stagingSlots?: number;
  /** Prefix for every label this creates. Default 'apse.timing'. */
  readonly label?: string;
}

/** Per-slot state. Values are named so the transitions are readable. */
const FREE = 0;
/** `mapAsync` called, promise not yet settled. */
const MAPPING = 1;
/** Mapped, or mapped-then-unmapped, and awaiting its result. */
const IN_FLIGHT = 2;

export class GpuTimer extends Resource {
  readonly #querySet: GPUQuerySet;
  readonly #resolve: GPUBuffer;
  readonly #staging: GPUBuffer[];
  /** One entry per staging buffer: FREE, MAPPING, or IN_FLIGHT. */
  readonly #state: Uint8Array;
  /** Pre-built pass descriptors, so a frame allocates nothing here. */
  readonly #writes: GPURenderPassTimestampWrites[];
  /** Milliseconds per pair, or `null` for a pair with no reading yet. */
  readonly #ms: (number | null)[];
  readonly #pairCount: number;
  readonly #byteSize: number;
  readonly #label: string;
  #dropped = 0;

  /**
   * Creates a timer, or explains why this device cannot have one.
   *
   * Returns rather than throws, because "this GPU cannot do that" is a fact
   * about the machine, not a mistake by the caller — the same rule as
   * {@link requireFeature}, which this goes through. A caller that has *already*
   * checked `device.capabilities.timestampQuery` and wants the throwing form has
   * a proof that the check passed, so the `Result` is unwrapped by them rather
   * than hidden here.
   */
  static create(
    device: TimingDevice,
    opts: GpuTimerOptions = {},
  ): Result<GpuTimer, GpuTimerFailure> {
    const gate = requireFeature(device.device, 'timestamp-query');
    if (!gate.ok) return gate;

    const pairs = positiveInteger(opts.pairs ?? 1, 'pairs', 'GpuTimer.create');
    const slots = positiveInteger(opts.stagingSlots ?? DEFAULT_STAGING_SLOTS, 'stagingSlots', 'GpuTimer.create');
    const label = opts.label ?? 'apse.timing';
    const queryCount = pairs * SLOTS_PER_PAIR;
    const byteSize = queryCount * TIMESTAMP_BYTES;
    const gpu = device.device;

    // Declared outside the try so the failure path can free what it built. There
    // is no `dispose` on a half-built object to lean on, and a query set plus
    // two buffers stranded per failed creation is a leak in a code path that
    // runs on every device that advertises the feature and then refuses it.
    const staging: GPUBuffer[] = [];
    let querySet: GPUQuerySet | null = null;
    let resolve: GPUBuffer | null = null;
    try {
      querySet = gpu.createQuerySet({
        label: `${label}:querySet`,
        type: 'timestamp',
        count: queryCount,
      });
      // QUERY_RESOLVE is what makes a query set readable at all; COPY_SRC is what
      // lets the result be copied onwards. Without the first the resolve is a
      // validation error, and the message names the usage bit rather than the
      // feature that is actually missing.
      resolve = gpu.createBuffer({
        label: `${label}:resolve`,
        size: byteSize,
        usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
      });
      const state = new Uint8Array(slots);
      const writes: GPURenderPassTimestampWrites[] = [];
      for (let i = 0; i < pairs; i++) {
        writes.push({
          beginningOfPassWriteIndex: i * SLOTS_PER_PAIR,
          endOfPassWriteIndex: i * SLOTS_PER_PAIR + 1,
          querySet,
        });
      }
      for (let i = 0; i < slots; i++) {
        staging.push(gpu.createBuffer({
          label: `${label}:readback${i}`,
          size: byteSize,
          // COPY_DST because the resolved result is copied in; MAP_READ because
          // the CPU reads it. A readback destination needs both.
          usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
        }));
      }
      return ok(new GpuTimer(querySet, resolve, staging, state, writes, pairs, byteSize, label));
    } catch (error) {
      for (const buffer of staging) buffer.destroy();
      resolve?.destroy();
      querySet?.destroy();
      return err('TIMESTAMP_ALLOCATION_FAILED',
        `Could not allocate ${pairs} timestamp quer${pairs === 1 ? 'y' : 'ies'} on this device: ${error instanceof Error ? error.message : String(error)}`,
        'timestamp-query is advertised but the allocation failed, which is a driver fault rather than a caller error. Check the adapter description with describeGpu() and report it with the browser version. Carry on with stats.gpu = null.',
        { feature: 'timestamp-query', required: String(pairs), available: 'allocation failed' });
    }
  }

  private constructor(
    querySet: GPUQuerySet,
    resolve: GPUBuffer,
    staging: GPUBuffer[],
    state: Uint8Array,
    writes: GPURenderPassTimestampWrites[],
    pairs: number,
    byteSize: number,
    label: string,
  ) {
    super();
    this.#querySet = querySet;
    this.#resolve = resolve;
    this.#staging = staging;
    this.#state = state;
    this.#writes = writes;
    this.#pairCount = pairs;
    this.#byteSize = byteSize;
    this.#label = label;
    this.#ms = new Array<number | null>(pairs).fill(null);
  }

  /** `(beginning, end)` pairs allocated. */
  get pairCount(): number { return this.#pairCount; }

  /** Bytes in the resolve buffer, and in each staging buffer. */
  get byteSize(): number { return this.#byteSize; }

  /**
   * The `timestampWrites` value for one pass.
   *
   * `pair` 0 by default, and the same object may be attached to two different
   * passes in a frame — the beginning to the first, the end to the last — which
   * is how one pair measures a whole frame that spans several render passes.
   * The object is reused, so read it into a variable if you need it after the
   * next call.
   */
  writes(pair: number = 0): GPURenderPassTimestampWrites {
    this.assertLive('GpuTimer');
    const found = this.#writes[pair];
    if (found === undefined) {
      fail('OPTION_UNKNOWN',
        `GpuTimer "${this.#label}".writes(${pair}) but only ${this.#pairCount} pair${this.#pairCount === 1 ? ' is' : 's are'} allocated.`, {
        why: 'A write index past the end of the query set is a validation error at pass-begin, and the message names an index rather than the pass that asked for it.',
        fix: `Allocate the pairs you need: GpuTimer.create(device, { pairs: ${pair + 1} }). One pair is enough to time a whole frame; see GpuTimerOptions.pairs.`,
      });
    }
    return found;
  }

  /**
   * Resolves this frame's queries and copies them into a free readback buffer.
   *
   * **Call it after every pass that writes a timestamp has ended, and before
   * `queue.submit()`.** Two ordering rules, both load-bearing:
   *
   *   - `resolveQuerySet` is itself a queue command, so it observes the
   *     timestamps written by the passes submitted before it and not after.
   *   - the copy has to be encoded in the same command buffer, or the result has
   *     to travel through the CPU, which is the thing being avoided.
   *
   * Frames whose readback slot is still in flight are dropped and counted in
   * {@link dropped} rather than waited on. A dropped frame is a missing sample,
   * and the ring makes it rare; blocking would make it a stutter every time.
   */
  encodeReadback(encoder: GPUCommandEncoder): void {
    this.assertLive('GpuTimer');
    const slot = this.#freeSlot();
    if (slot < 0) {
      this.#dropped++;
      return;
    }
    // Reserved before the resolve is encoded, so a second encodeReadback in the
    // same frame cannot take the same buffer.
    this.#state[slot] = IN_FLIGHT;
    encoder.resolveQuerySet(this.#querySet, 0, this.#pairCount * SLOTS_PER_PAIR, this.#resolve, 0);
    encoder.copyBufferToBuffer(this.#resolve, 0, this.#staging[slot]!, 0, this.#byteSize);
  }

  /**
   * Starts the readback of whatever the last {@link encodeReadback} reserved.
   *
   * **Call it after `queue.submit()`,** because the map cannot resolve until the
   * copy has actually run on the queue. Never `await` this: the promise is
   * deliberately unhandled, the reading lands in {@link pairMs} when it arrives,
   * and the frame does not wait. `mapAsync` is fenced by the queue, so this
   * cannot observe a value the GPU has not written.
   */
  poll(): void {
    this.assertLive('GpuTimer');
    // Every reserved slot, not just one: a slot that was reserved and never
    // polled would otherwise stay non-FREE forever, and after `stagingSlots`
    // frames the timer would drop every readback forever. Cheap to be wrong in
    // this direction: two buffers, two maps, no allocation.
    for (let slot = 0; slot < this.#staging.length; slot++) {
      if (this.#state[slot] !== IN_FLIGHT) continue;
      this.#state[slot] = MAPPING;
      const buffer = this.#staging[slot]!;
      // `.catch` is not optional: `dispose()` while this is pending rejects the
      // map, and an unhandled rejection here would surface as a console error
      // on a perfectly normal teardown.
      void buffer.mapAsync(MAP_MODE.READ).then(() => {
        if (this.disposed) return;
        this.#read(slot, buffer);
      }).catch(() => {
        this.#state[slot] = FREE;
      });
    }
  }

  /**
   * Milliseconds for one pair, or `null` when no reading has arrived yet.
   *
   * `null` means "no data". A reading of exactly `0` means the GPU spent less
   * than the driver's timestamp quantisation — Chromium rounds to 100 µs, so a
   * trivial frame genuinely reads 0 — and that is a measurement, not a gap.
   * Collapsing the two is the mistake this whole module exists to avoid.
   */
  pairMs(pair: number = 0): number | null {
    return this.#ms[pair] ?? null;
  }

  /**
   * The whole-frame GPU time in milliseconds, or `null` with no measurement.
   *
   * This is the value that belongs in `FrameStats.gpu`. It is `null` — not 0 —
   * when the device has no `timestamp-query`, when the timer has not been
   * created, and when no reading has arrived since it was.
   */
  get lastGpuMs(): number | null {
    return this.pairMs(0);
  }

  /** Every pair's latest reading, in allocation order. */
  get readings(): readonly (number | null)[] {
    return this.#ms;
  }

  /**
   * The first `count` pairs added together, or `null` before any have arrived.
   *
   * For a frame made of more than one pass. A `(beginning, end)` pair may only
   * be written **once** per submission — reusing a write index is a validation
   * error — so a frame that opens an opaque pass, a transparent pass and a
   * present pass cannot be covered by one pair. It takes one pair per pass, and
   * `count` is how many the frame opened.
   *
   * A count larger than the number of passes stamped is harmless rather than
   * stale: `resolveQuerySet` writes the whole destination range every frame and
   * a query nobody wrote resolves to zero, so an absent pass contributes 0 ms —
   * which is what a pass that did not open actually cost.
   *
   * Readings are one to two frames late, which is why this is a trend and not a
   * per-frame verdict.
   */
  sumMs(count: number): number | null {
    let total = 0;
    for (let i = 0; i < count; i++) {
      const value = this.#ms[i];
      if (value === undefined || value === null) return null;
      total += value;
    }
    return total;
  }

  /** Readbacks in flight. Non-zero means a reading is on its way. */
  get pending(): number {
    let n = 0;
    for (let i = 0; i < this.#state.length; i++) if (this.#state[i] !== FREE) n++;
    return n;
  }

  /** Frames whose readback was skipped because every slot was busy. */
  get dropped(): number { return this.#dropped; }

  protected override onDispose(): void {
    // Destroying a buffer with a pending map is legal and rejects that map, which
    // the catch in poll() absorbs. Nothing here is awaited: dispose() is called
    // from teardown, and teardown does not block.
    this.#querySet.destroy();
    this.#resolve.destroy();
    for (const buffer of this.#staging) buffer.destroy();
  }

  #freeSlot(): number {
    for (let i = 0; i < this.#state.length; i++) if (this.#state[i] === FREE) return i;
    return -1;
  }

  /**
   * Reads one resolved buffer.
   *
   * `getMappedRange()` is detached by `unmap()` and a `BigUint64Array` over it is
   * a *view*, so the deltas are copied out here, inside the mapped window, and
   * nothing view-like escapes. Holding the view and reading it after the unmap
   * yields a detached buffer: every value zero, which would be reported as a
   * measured 0 ms GPU time for every frame forever.
   */
  #read(slot: number, buffer: GPUBuffer): void {
    const raw = new BigUint64Array(buffer.getMappedRange(0, this.#byteSize));
    for (let pair = 0; pair < this.#pairCount; pair++) {
      const begin = raw[pair * SLOTS_PER_PAIR]!;
      const end = raw[pair * SLOTS_PER_PAIR + 1]!;
      // `end < begin` is not a fast frame, it is a counter that wrapped or a
      // device that went away mid-flight. Subtracting it anyway wraps to
      // 2^64 nanoseconds, so the pair reports no reading at all instead.
      this.#ms[pair] = end < begin ? null : Number(end - begin) / NS_PER_MS;
    }
    buffer.unmap();
    this.#state[slot] = FREE;
  }
}

/** Nanoseconds per millisecond. Written out so the conversion is greppable. */
const NS_PER_MS = 1e6;

function positiveInteger(value: number, option: string, where: string): number {
  if (Number.isInteger(value) && value > 0) return value;
  fail('OPTION_UNKNOWN',
    `${where} was given ${option}: ${value}, which is not a positive integer.`, {
    why: `\`${option}\` sizes a fixed-count allocation — a query set slot count or a ring length. A non-integer or non-positive value produces either a validation error at createQuerySet or a ring that can never find a free slot, and the error message from the driver names a count rather than the option.`,
    fix: `Pass a positive integer. ${option === 'pairs' ? '1 measures a whole frame; 2 attributes it to the scene pass and the present pass.' : '2 is the minimum that reads every frame without dropping one.'}`,
    detail: { kind: 'numeric', field: option, value, min: 1 },
  });
}
