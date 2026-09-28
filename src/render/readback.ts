/**
 * Reading a texture back to the CPU.
 *
 * # The 256-byte row alignment is the whole file
 *
 * `copyTextureToBuffer` requires `bytesPerRow` to be a multiple of 256. That is
 * not a performance hint, it is a validation rule, and it exists because the copy
 * is implemented as a linear DMA whose row pitch has to be a power of two on
 * every tiled GPU apse targets. Break it and the *entire command buffer* is
 * invalidated — not just the copy — so a screenshot with an unpadded stride
 * takes the frame's draw calls down with it, and the driver's error names
 * `bytesPerRow` rather than the readback.
 *
 * The consequence bites twice, and the second one is the trap:
 *
 *   1. The buffer must be `bytesPerRow * height`, not `width * 4 * height`, and
 *      the copy descriptor's `bytesPerRow` must be the padded value.
 *   2. **The result must be read at the padded stride.** A 100-pixel-wide RGBA
 *      image has a `bytesPerRow` of 256 and rows of 400 bytes of pixel data, so
 *      reading it as 400-byte rows interleaves the last 56 bytes of one row with
 *      the first 56 of the next. The image is sheared by exactly one pixel per
 *      row: a diagonal band, not a shift, and not obviously a readback bug.
 *
 * {@link alignedBytesPerRow} is the only place the padding is computed, and
 * {@link unpadRows} is the only place a padded buffer is turned back into
 * contiguous pixels.
 *
 * # Why there is no `preserveDrawingBuffer` equivalent to fall back on
 *
 * A WebGPU canvas swapchain image is transient. `getCurrentTexture()` returns a
 * texture that is valid for the frame being encoded and is expired by the
 * present, and sampling the canvas afterwards — by `drawImage`, or by copying
 * out of the swapchain — yields black. There is no flag that changes this. So a
 * capture reads from an *offscreen* target that was created with `COPY_SRC`, and
 * that is what {@link CaptureReadback} is for: it owns the destination buffer,
 * knows the size, and is the reason the size can be wrong in exactly one place
 * instead of in every caller.
 */

import { fail } from '../core/error.ts';
import { Resource } from '../core/resource.ts';
import { alignUp } from '../core/uniform.ts';
import { BUFFER_USAGE, MAP_MODE } from './device.ts';

/**
 * The row pitch alignment `copyTextureToBuffer` requires, in bytes.
 *
 * 256 is the spec's constant, not a tuning knob. The only way to get a legal
 * `bytesPerRow` is to round *up* to a multiple of it, which means every image
 * with a width that is not a multiple of 64 pixels carries padding.
 */
export const COPY_BYTES_PER_ROW_ALIGNMENT = 256;

/** Bytes per pixel for the formats apse reads back. `rgba8unorm` and friends. */
export const BYTES_PER_PIXEL_RGBA8 = 4;

/**
 * The row pitch a copy of a `width`-wide image must declare.
 *
 * `bytesPerPixel` is a parameter rather than a constant because the same rule
 * applies to `depth32float` readback and to a packed `r16float` view, and a
 * function that only knows about RGBA8 gets a stride wrong the first time anyone
 * copies a depth texture.
 */
export function alignedBytesPerRow(width: number, bytesPerPixel: number = BYTES_PER_PIXEL_RGBA8): number {
  if (!Number.isInteger(width) || width <= 0) {
    fail('RENDER_TARGET_SIZE_INVALID',
      `A readback was asked for ${width} pixels per row.`, {
      why: 'The row pitch is derived from the width, so a non-positive or fractional width has no pitch. The usual source is a canvas measured at 0x0 — display:none, or before the first layout pass.',
      fix: 'Pass the backing-store width the target actually has, and make sure a layout pass has run. CanvasSizer already floors a zero measurement to 1.',
      detail: { kind: 'numeric', field: 'width', value: width, min: 1 },
    });
  }
  if (!Number.isInteger(bytesPerPixel) || bytesPerPixel <= 0) {
    fail('RENDER_TARGET_SIZE_INVALID',
      `A readback was asked for ${bytesPerPixel} bytes per pixel.`, {
      why: 'The pitch is a multiple of the pixel size, so a zero or fractional pixel size makes the multiplication meaningless — and an unaligned pitch is a validation error on the whole command buffer.',
      fix: 'Pass a positive integer: 4 for rgba8unorm and bgra8unorm, 8 for a two-channel 32-bit format, 2 for r16float.',
      detail: { kind: 'numeric', field: 'bytesPerPixel', value: bytesPerPixel, min: 1 },
    });
  }
  return alignUp(width * bytesPerPixel, COPY_BYTES_PER_ROW_ALIGNMENT);
}

/**
 * Rejects a pitch the copy would refuse.
 *
 * Exists to be called on the way *in* to a copy, not just on the way out of the
 * helper: the failure mode is an invalidated command buffer that takes the
 * frame's draws with it, which is a far worse symptom than a thrown error, and
 * the driver's own error arrives as an uncaptured error attributed to nothing.
 */
export function assertBytesPerRowAligned(bytesPerRow: number, what: string = 'A readback'): void {
  if (Number.isInteger(bytesPerRow) && bytesPerRow > 0 &&
      bytesPerRow % COPY_BYTES_PER_ROW_ALIGNMENT === 0) {
    return;
  }
  fail('INTERNAL_INVARIANT',
    `${what} was given bytesPerRow ${bytesPerRow}, which is not a positive multiple of ${COPY_BYTES_PER_ROW_ALIGNMENT}.`, {
    why: 'WebGPU requires the row pitch of a texture-to-buffer copy to be a multiple of 256. Passing an unaligned value is a validation error that invalidates the whole command buffer, so it also discards the frame\'s draw calls and the copy — and it surfaces as an uncaptured error naming a descriptor field, with no stack.',
    fix: `Use alignedBytesPerRow(width, bytesPerPixel) for the pitch, and read the result at that same pitch. For a ${COPY_BYTES_PER_ROW_ALIGNMENT / BYTES_PER_PIXEL_RGBA8}-pixel-wide RGBA8 image the pitch is already aligned; narrower images are padded and the padding is not part of the image.`,
    detail: { kind: 'numeric', field: 'bytesPerRow', value: bytesPerRow, min: COPY_BYTES_PER_ROW_ALIGNMENT },
  });
}

/**
 * Strips the row padding, producing `width * bytesPerPixel * height` bytes.
 *
 * What an image consumer actually wants: the copy is stored padded because the
 * DMA needs it, and everything downstream — a PNG encoder, a `putImageData`, a
 * pixel diff — needs contiguous rows. Doing the unpad here means the
 * three-channel-vs-four and the last-row special cases are handled once.
 */
export function unpadRows(
  padded: Uint8Array,
  width: number,
  height: number,
  bytesPerPixel: number = BYTES_PER_PIXEL_RGBA8,
): Uint8Array {
  const bytesPerRow = alignedBytesPerRow(width, bytesPerPixel);
  const rowBytes = width * bytesPerPixel;
  if (padded.length < bytesPerRow * height) {
    fail('INTERNAL_INVARIANT',
      `A readback buffer of ${padded.length} bytes was unpacked as ${width}x${height} at ${bytesPerRow} bytes per row, which needs ${bytesPerRow * height}.`, {
      why: 'The copy writes bytesPerRow * height bytes, so a short buffer means the copy never ran, the target was smaller than the row count, or the buffer was sized at width * 4 instead of at the padded pitch — the exact mistake this module exists to make impossible.',
      fix: 'Size the destination at alignedBytesPerRow(width) * height, and pass that same value as the copy\'s bytesPerRow.',
      detail: { kind: 'numeric', field: 'buffer', value: padded.length, min: bytesPerRow * height },
    });
  }
  const out = new Uint8Array(rowBytes * height);
  if (bytesPerRow === rowBytes) {
    out.set(padded.subarray(0, out.length));
    return out;
  }
  for (let y = 0; y < height; y++) {
    out.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + rowBytes), y * rowBytes);
  }
  return out;
}

/**
 * The MAP_READ destination for one texture copy.
 *
 * Owns the size, so a resize cannot desynchronise it from the pitch the copy
 * declared — the failure being a buffer too small for the copy, which
 * invalidates the command buffer, or a pitch that no longer matches the target.
 * Both are per-frame at most (a capture is not a hot path) and both are silent.
 */
export class CaptureReadback extends Resource {
  readonly #gpu: GPUDevice;
  readonly #bytesPerPixel: number;
  readonly #label: string;
  #buffer: GPUBuffer | null;
  #width: number;
  #height: number;

  private constructor(gpu: GPUDevice, buffer: GPUBuffer, width: number, height: number, bytesPerPixel: number, label: string) {
    super();
    this.#gpu = gpu;
    this.#buffer = buffer;
    this.#width = width;
    this.#height = height;
    this.#bytesPerPixel = bytesPerPixel;
    this.#label = label;
  }

  static create(
    device: { readonly device: GPUDevice },
    width: number,
    height: number,
    bytesPerPixel: number = BYTES_PER_PIXEL_RGBA8,
    label: string = 'apse:readback',
  ): CaptureReadback {
    const bytesPerRow = alignedBytesPerRow(width, bytesPerPixel);
    const buffer = device.device.createBuffer({
      label,
      size: bytesPerRow * height,
      // MAP_READ to read it, COPY_DST to be the destination of a copy. Nothing
      // else: a readback buffer that could also be written by the GPU is a
      // buffer a shader can scribble on, and on a tiled GPU granting STORAGE
      // changes how the copy lands.
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    });
    return new CaptureReadback(device.device, buffer, width, height, bytesPerPixel, label);
  }

  get width(): number { return this.#width; }
  get height(): number { return this.#height; }
  get bytesPerPixel(): number { return this.#bytesPerPixel; }

  /**
   * The pitch to declare on the copy, and the pitch to read the result at.
   *
   * One property on purpose: the copy and the read must agree, and the way they
   * are made to disagree is by reading `width * bytesPerPixel` in one place and
   * this in the other.
   */
  get bytesPerRow(): number {
    return alignedBytesPerRow(this.#width, this.#bytesPerPixel);
  }

  /** Bytes the destination buffer holds. */
  get size(): number {
    return this.bytesPerRow * this.#height;
  }

  /** The buffer, for `copyTextureToBuffer`. Valid until `resize` or `dispose`. */
  get buffer(): GPUBuffer {
    this.assertLive('CaptureReadback');
    return this.#buffer as GPUBuffer;
  }

  /**
   * Reallocates at a new size, or does nothing when the size is unchanged.
   *
   * **Free before allocating**, the same order as `RenderTargetImpl.resize` and
   * for the same reason: two full-size readback buffers alive at once is a
   * needless peak on exactly the devices least able to afford it, and a buffer
   * still mapped cannot be destroyed usefully.
   */
  resize(width: number, height: number): boolean {
    this.assertLive('CaptureReadback');
    const nextRow = alignedBytesPerRow(width, this.#bytesPerPixel);
    if (nextRow === this.bytesPerRow && height === this.#height) return false;
    const previous = this.#buffer;
    this.#buffer = null;
    previous?.destroy();
    this.#width = width;
    this.#height = height;
    this.#buffer = this.#gpu.createBuffer({
      label: this.#label,
      size: nextRow * height,
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    });
    return true;
  }

  /**
   * Maps the buffer and returns `width * bytesPerPixel * height` bytes.
   *
   * **Await this; it is a real GPU round trip.** It is the price of getting
   * pixels out of WebGPU, and there is no cheaper path — hence `capture()` being
   * an async call rather than a synchronous one that lies about its cost.
   *
   * The copy out of the mapped range happens *before* `unmap()`, because
   * `unmap()` detaches it: a `Uint8Array` view taken inside the mapped window
   * and read after the unmap is a detached buffer, which reads as all zeroes. A
   * capture that returns a zero-filled image is the classic symptom, and it is
   * indistinguishable from "the scene really is black" unless you check the
   * stride.
   */
  async map(): Promise<Uint8Array> {
    this.assertLive('CaptureReadback');
    const buffer = this.buffer;
    await buffer.mapAsync(MAP_MODE.READ);
    try {
      return unpadRows(new Uint8Array(buffer.getMappedRange(0, this.size)).slice(), this.#width, this.#height, this.#bytesPerPixel);
    } finally {
      buffer.unmap();
    }
  }

  protected override onDispose(): void {
    this.#buffer?.destroy();
    this.#buffer = null;
  }
}
