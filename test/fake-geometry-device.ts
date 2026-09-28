/**
 * A recording GPU device, for proving that a *draw plan* is right without a GPU.
 *
 * `test/fake-device.ts` is built around textures, because that is what the
 * render-target layer needed. Instancing and batching are a different problem:
 * the risky part is not any GPU call, it is the **plan** — which buffers get
 * bound to which slots, with which stride, and how many instances the draw
 * covers. Every one of those is a number, and a number can be recorded and
 * asserted on without a driver.
 *
 * So this records:
 *
 *   - every `createBuffer` descriptor, so usage flags and sizes are checkable —
 *     in particular that nothing is ever created with `STORAGE`, which is the
 *     limit the compatibility profile zeroes and the reason instance data is a
 *     vertex buffer at all;
 *   - every `writeBuffer`, with the **byte** offset and the **bytes** written,
 *     so a partial re-upload can be proven to be partial and a packed instance
 *     record can be read back exactly as the GPU would read it;
 *   - the `getMappedRange` / `unmap` pair, so the bytes uploaded at creation are
 *     recoverable — the trap being that the mapped range is *detached* by
 *     `unmap`, so this copies out of it before unmapping, which is exactly the
 *     mistake the note in `AGENTS.md` is about;
 *   - every encoder call, in order, so "one `drawIndexed` for N instances" is an
 *     assertion rather than a claim.
 *
 * The encoder is a recorder, not an implementation: it accepts the calls the
 * renderer makes and remembers them. A test that wants to know what a draw *is*
 * asks it here rather than re-deriving the answer from the same source the bug
 * is in.
 */

const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

/** The spec's own bit values, installed on `globalThis` the way a browser has them. */
function installUsageGlobals(): void {
  (globalThis as unknown as { GPUBufferUsage: unknown }).GPUBufferUsage = GPU_BUFFER_USAGE;
}

export { GPU_BUFFER_USAGE as FAKE_BUFFER_USAGE };

/** One recorded `writeBuffer`, with the payload as raw bytes. */
export interface RecordedWrite {
  readonly label: string;
  /** Byte offset in the destination buffer. */
  readonly offset: number;
  /** Number of elements of the source typed array, as WebGPU counts them. */
  readonly elements: number;
  /** The bytes that were handed over, copied. */
  readonly bytes: Uint8Array;
  /** The same bytes as floats. `bytes.length / 4` long. */
  readonly floats: Float32Array;
}

export class RecordingBuffer {
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  destroyed = false;
  /** Every byte ever written to this buffer, so a read-back sees the latest. */
  readonly contents: Uint8Array;
  #mapping = false;

  constructor(desc: GPUBufferDescriptor) {
    this.label = desc.label ?? '';
    this.size = desc.size;
    this.usage = desc.usage;
    this.contents = new Uint8Array(desc.size);
    if (desc.mappedAtCreation === true) this.#mapping = true;
  }

  getMappedRange(offset?: number, size?: number): ArrayBuffer {
    if (!this.#mapping) {
      throw new Error(`RecordingBuffer "${this.label}".getMappedRange() — the buffer is not mapped`);
    }
    const byteOffset = offset ?? 0;
    const length = size === undefined ? this.contents.byteLength - byteOffset : size;
    if (byteOffset === 0 && length === this.contents.byteLength) {
      // The buffer's **own** storage, not a copy of it. A real mapped range is a
      // view onto the mapped memory: `new Float32Array(range).set(data)` writes
      // into the buffer. Handing back a copy here silently discards every byte
      // uploaded at `mappedAtCreation` — the upload "succeeds", the recorder
      // writes nothing, and every read-back sees zeroes. The cast is sound: the
      // buffer came from `new Uint8Array(size)`, so its ArrayBuffer is exclusive.
      return this.contents.buffer as ArrayBuffer;
    }
    // A partial range cannot be an `ArrayBuffer` view, because the platform type
    // is an `ArrayBuffer` and there is no such thing as a sub-array one. No apse
    // upload path asks for one, so this is here to fail loudly rather than
    // quietly wrong.
    throw new Error(
      `RecordingBuffer "${this.label}".getMappedRange(${byteOffset}, ${length}) — the fake only models a full-buffer mapping`,
    );
  }

  unmap(): void {
    if (!this.#mapping) {
      throw new Error(`RecordingBuffer "${this.label}".unmap() without a matching getMappedRange()`);
    }
    // A real mapped range is *detached* by unmap, so nothing may hold a view of
    // it afterwards. The bytes went into `contents` before the mapping flag was
    // cleared, which is the only order that reads back correctly.
    this.#mapping = false;
  }

  destroy(): void {
    this.destroyed = true;
  }
}

/** One recorded `draw` / `drawIndexed`. */
export interface RecordedDraw {
  readonly kind: 'draw' | 'drawIndexed';
  /** `indexCount` for a `drawIndexed`, `vertexCount` for a `draw`. */
  readonly vertexCount: number;
  readonly instanceCount: number;
  readonly firstVertex: number;
  readonly firstInstance: number;
  /** `drawIndexed` only. Index **elements**, not bytes. */
  readonly firstIndex?: number;
  /** `drawIndexed` only. */
  readonly baseVertex?: number;
}

/** One recorded `setVertexBuffer`. */
export interface RecordedVertexBinding {
  readonly slot: number;
  readonly label: string;
  readonly offset: number;
  readonly size: number;
}

export interface RecordedIndexBinding {
  readonly label: string;
  readonly format: GPUIndexFormat;
  readonly offset: number;
  readonly size: number;
}

/** The command stream a test encoded, in order. */
export class RecordingEncoder {
  readonly vertexBindings: RecordedVertexBinding[] = [];
  readonly indexBindings: RecordedIndexBinding[] = [];
  readonly draws: RecordedDraw[] = [];
  /** Every method name called, in order. Enough to assert the *sequence*. */
  readonly calls: string[] = [];

  setVertexBuffer(slot: number, buffer: GPUBuffer, offset = 0, size?: number): void {
    const b = buffer as unknown as RecordingBuffer;
    this.vertexBindings.push({ slot, label: b.label, offset, size: size ?? (b.size - offset) });
    this.calls.push(`setVertexBuffer(${slot})`);
  }

  setIndexBuffer(buffer: GPUBuffer, format: GPUIndexFormat, offset = 0, size?: number): void {
    const b = buffer as unknown as RecordingBuffer;
    this.indexBindings.push({ label: b.label, format, offset, size: size ?? (b.size - offset) });
    this.calls.push('setIndexBuffer');
  }

  draw(vertexCount: number, instanceCount: number, firstVertex: number, firstInstance: number): void {
    this.draws.push({ kind: 'draw', vertexCount, instanceCount, firstVertex, firstInstance });
    this.calls.push('draw');
  }

  drawIndexed(
    indexCount: number,
    instanceCount: number,
    firstIndex: number,
    baseVertex: number,
    firstInstance: number,
  ): void {
    this.draws.push({
      kind: 'drawIndexed',
      vertexCount: indexCount,
      instanceCount,
      firstVertex: baseVertex,
      firstInstance,
      firstIndex,
      baseVertex,
    });
    this.calls.push('drawIndexed');
  }

  finish(): Record<string, never> {
    return {} as Record<string, never>;
  }
}

export interface RecordingDevice {
  readonly device: GPUDevice;
  readonly buffers: RecordingBuffer[];
  readonly writes: RecordedWrite[];
  readonly encoders: RecordingEncoder[];
  /** `createBuffer` descriptors, in order. */
  readonly descriptors: GPUBufferDescriptor[];
  /** The buffer with this label. Throws, naming what does exist. */
  buffer(label: string): RecordingBuffer;
  /** A fresh encoder, also recorded on the device. */
  encoder(): RecordingEncoder;
  /** The bytes currently in `buffer`, decoded as floats, copied out. */
  readFloats(buffer: GPUBuffer): Float32Array;
}

/**
 * A `GPUDevice` stub with a real byte-level buffer model.
 *
 * `maxBufferSize` defaults to 256 MiB — the compatibility profile's
 * `maxBufferSize` is 256 MiB, so this is the size apse asks for in production
 * and the size a test should be measuring against.
 */
export function recordingDevice(maxBufferSize = 256 * 1024 * 1024): RecordingDevice {
  installUsageGlobals();

  const buffers: RecordingBuffer[] = [];
  const writes: RecordedWrite[] = [];
  const encoders: RecordingEncoder[] = [];
  const descriptors: GPUBufferDescriptor[] = [];

  const device = {
    limits: { maxBufferSize },
    createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
      descriptors.push(desc);
      const b = new RecordingBuffer(desc);
      buffers.push(b);
      return b as unknown as GPUBuffer;
    },
    queue: {
      writeBuffer(
        target: GPUBuffer,
        bufferOffset: number,
        data: ArrayBufferView<ArrayBuffer>,
        dataOffset = 0,
        size?: number,
      ): void {
        const b = target as unknown as RecordingBuffer;
        const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const elemSize = (data as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
        const first = dataOffset * elemSize;
        const length = (size ?? data.byteLength / elemSize - dataOffset) * elemSize;
        // Copied, because a real queue copies. A test that held a reference to
        // the source array would otherwise see a later upload mutate it.
        const bytes = new Uint8Array(length);
        bytes.set(view.subarray(first, first + length));
        b.contents.set(bytes, bufferOffset);
        writes.push({
          label: b.label,
          offset: bufferOffset,
          elements: size ?? length / elemSize,
          bytes,
          floats: new Float32Array(bytes.buffer, 0, length >> 2),
        });
      },
    },
  } as unknown as GPUDevice;

  return {
    device,
    buffers,
    writes,
    encoders,
    descriptors,
    buffer(label: string): RecordingBuffer {
      const found = buffers.find((b) => b.label === label);
      if (found === undefined) {
        throw new Error(`recordingDevice: no buffer labelled "${label}". Have: ${buffers.map((b) => b.label).join(', ')}`);
      }
      return found;
    },
    encoder(): RecordingEncoder {
      const enc = new RecordingEncoder();
      encoders.push(enc);
      return enc;
    },
    readFloats(buffer: GPUBuffer): Float32Array {
      const b = buffer as unknown as RecordingBuffer;
      // A copy, because a Float32Array over a Uint8Array's buffer is a view and
      // the caller is going to keep it past the next write.
      const count = b.contents.byteLength >> 2;
      const out = new Float32Array(count);
      out.set(new Float32Array(b.contents.buffer.slice(0, count * 4)));
      return out;
    },
  };
}
