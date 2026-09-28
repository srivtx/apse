/**
 * The boundary between the material layer and the render layer.
 *
 * `src/material` knows how to compile WGSL and create pipelines.
 * `src/render` knows how to run a frame. Neither imports the other.
 *
 * This file is the whole of that contract, which is what lets both sides be
 * developed, tested, and reasoned about in isolation. If you find yourself
 * wanting to add something to this file, first check whether it is really a
 * contract or just shared convenience.
 */

import type { VertexLayout } from '../geometry/layout.ts';
import type { TextureSlotSpec } from '../material/texture-slot.ts';
import type { SlotType } from '../core/slot.ts';
import type { BlendSpec, DepthSpec, PrimitiveTopology } from './pipeline-state.ts';

// ---------------------------------------------------------------------------
// Draw state
// ---------------------------------------------------------------------------

/** Where a draw item sits relative to other items, after sorting. */
export type DrawPhase = 'opaque' | 'transparent';

/** One sorted, drawable thing produced by the scene traversal. */
export interface DrawItem {
  /** Stable per-frame index, written into `obj.objectId`. */
  readonly objectId: number;
  readonly phase: DrawPhase;
  /** Explicit sort key. Higher draws later within a phase. */
  readonly order: number;
  /** View-space depth used to sort within the default order. */
  readonly depth: number;
  /** Byte offset into the object uniform buffer for this item's transform. */
  readonly objectOffset: number;
  /**
   * The node's world matrix, as a **live reference** to the node's own
   * `Float32Array(16)`.
   *
   * A reference, not a copy: the transform pass rewrites the array's contents
   * in place and the array identity never changes, so holding the reference
   * costs nothing and stays correct as the node moves. Copying 64 bytes per
   * object per frame would be a real cost at a few thousand objects, and copying
   * it once at collection time would silently freeze every transform.
   *
   * This is the only place a draw item and the scene graph meet, so it is also
   * the only place that has to know how to get a transform. Everything
   * downstream reads `model`.
   */
  readonly model: Float32Array;
  /**
   * The node's {@link worldVersion} at collection time.
   *
   * Half of the renderer's dirty key for an object. The other half is the
   * identity of `model` itself, which is unique per node and never replaced —
   * the version alone is not enough, because it comes from a global clock, so
   * two nodes written in the same traversal pass carry the same number with
   * entirely different matrices.
   */
  readonly worldVersion: number;
  readonly material: Drawable;
  readonly geometry: DrawableGeometry;
  /** 1 for a plain mesh, n for instanced. */
  readonly instanceCount: number;
  /** First instance index, for instanced draws. */
  readonly firstInstance: number;
  /** False when the node was hidden or fully clipped; the item is skipped. */
  visible: boolean;
}

/** The GPU-side geometry a draw item needs. Supplied by `src/geometry`. */
export interface DrawableGeometry {
  readonly layout: VertexLayout;
  readonly vertexBuffer: GPUBuffer;
  readonly indexBuffer: GPUBuffer | null;
  readonly indexCount: number;
  readonly instanceCount: number;
  readonly firstInstance: number;
  /**
   * Per-instance transforms, bound to vertex slot 1 with `stepMode: 'instance'`.
   *
   * `null` for a non-instanced mesh, and the renderer skips the slot entirely
   * rather than binding null — a null binding on slot 1 when the pipeline
   * declares no instance attributes is a validation error, not a no-op.
   *
   * A vertex buffer, not a storage buffer, and that is not a preference:
   * compatibility mode sets `maxStorageBuffersInVertexStage` to 0, so a vertex
   * shader reading a storage buffer compiles on a laptop and fails on a phone.
   */
  readonly instanceBuffer: GPUBuffer | null;
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/**
 * What a material must expose to be drawable.
 *
 * Implemented by `Material` in `src/material/material.ts`. The renderer
 * touches nothing else on a material — no pipeline rebuilds, no per-frame
 * string lookups, no `isXxx` brand checks in the hot path.
 */
export interface Drawable {
  readonly name: string;
  /** Vertex layout this material's pipeline was built for. */
  readonly layout: VertexLayout;

  // --- GPU objects, created once at material construction ---
  readonly renderPipeline: GPURenderPipeline;
  /** @group(0) — frame uniforms. */
  readonly frameBindGroup: GPUBindGroup;
  /** @group(1) — object uniforms, bound with a dynamic offset. */
  readonly objectBindGroup: GPUBindGroup;
  /** @group(2) — material slots. Null when the material declares no slots. */
  readonly materialBindGroup: GPUBindGroup | null;
  /** @group(3) — textures and the shared sampler. Null when untextured. */
  readonly textureBindGroup: GPUBindGroup | null;

  // --- State the renderer needs when sorting or opening a pass ---
  readonly phase: DrawPhase;
  readonly depth: DepthSpec;
  readonly blend: BlendSpec | null;
  readonly topology: PrimitiveTopology;

  // --- Per-frame mutation ---
  /** Copy the frame uniform into place. Called once per frame, before drawing. */
  writeFrameUniform(bytes: ArrayBuffer): void;
  /** Write one material slot. Validates and packs in one place. */
  setSlot(name: string, value: number | ArrayLike<number>): void;
  /** Bump a slot without a value copy, for a whole-material change. */
  markSlotsDirty(): void;
  /** True when the material block has pending writes to flush. */
  readonly slotsDirty: boolean;
  /** Flush pending slot writes to the GPU. Called once per frame, lazily. */
  flushSlots(): void;
  /**
   * Colour formats this material's pipeline was compiled for.
   *
   * A WebGPU pipeline bakes its attachment formats in at creation, so a
   * material is permanently bound to the format it was built for. The renderer
   * compares this against the target's format before `setPipeline`, because
   * Dawn's own report for a mismatch names two format enums rather than the
   * mistake, and invalidates the whole command buffer with no exception.
   */
  readonly targetFormats: readonly GPUTextureFormat[];
  /** True when the material declares at least one texture. */
  readonly textured: boolean;
  /** Texture slot names, for tooling and validation. */
  readonly textureNames: readonly string[];
  /** Resolved slot types, for tooling and validation. */
  readonly slotTypes: Readonly<Record<string, SlotType>>;
  /** The full generated WGSL. Exposed for debugging and for the docs site. */
  readonly wgsl: string;
}

// ---------------------------------------------------------------------------
// Render target
// ---------------------------------------------------------------------------

/** A colour attachment, plus optional depth. */
export interface RenderTarget {
  readonly width: number;
  readonly height: number;
  readonly format: GPUTextureFormat;
  /** May be undefined for colour-only passes. */
  readonly depthFormat: GPUTextureFormat | undefined;
  /** The colour texture view, as a render pass attachment. */
  readonly colorView: GPUTextureView;
  /** The depth texture view, as a render pass depth-stencil attachment. */
  readonly depthView: GPUTextureView | undefined;
  /** Present: whether this target is the canvas backbuffer. */
  readonly isCanvas: boolean;
  /** Resolves to this view when sampled by a later pass. */
  readonly sampleView: GPUTextureView | undefined;
  /** MSAA sample count: 1 or 4. */
  readonly sampleCount: number;
  /** Recreate at a new size. Called on canvas resize. */
  resize(width: number, height: number): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Frame description
// ---------------------------------------------------------------------------

/**
 * The GPU-timing half of `FrameStats`, defined here rather than in
 * `renderer.ts` so that the *absence* of a measurement is a type decision made
 * in the layer that owns the absence.
 *
 * `gpu` and `averageGpu` are `number | null`, and that is a **breaking change**
 * from the `number` they were. It is the honest type, and the reason is
 * structural rather than stylistic:
 *
 *   - `timestamp-query` is an optional feature on roughly half of all devices,
 *     and apse never requires it — requiring it would make `requestDevice()`
 *     reject on exactly the phones the library exists to serve.
 *   - A hardcoded `0` reads, to any consumer, as "the GPU was idle". `stats.gpu
 *     === 0` is a completely ordinary line of instrumentation code, and it
 *     concludes that the GPU is not the problem — which is the one conclusion a
 *     renderer must never help someone reach.
 *   - `null` and `0` are different facts: no measurement, versus a measurement of
 *     nothing. A driver that quantises timestamps to 100 µs really does report
 *     0 for a trivial frame, and that 0 is data.
 *
 * So a caller migrates by handling the null:
 *
 *     stats.gpu === null ? 'no GPU timing on this device' : `${stats.gpu.toFixed(2)} ms`
 *
 * `renderer.ts` declares `FrameStats extends FrameTimingStats` so there is one
 * definition of the shape rather than two that can drift.
 */
export interface FrameTimingStats {
  /**
   * GPU time for the frame, in milliseconds, or `null` for no measurement.
   *
   * Measured by timestamp query and therefore 1–2 frames late: the value comes
   * from a buffer the GPU writes after the frame is submitted, and the frame
   * loop never waits for it. Treat it as a trend, not as a per-frame verdict.
   */
  readonly gpu: number | null;
  /** Mean of the last `sampleSize` readings, or `null` while none have arrived. */
  readonly averageGpu: number | null;
  /**
   * True when the device has `timestamp-query` **and** a timer is running.
   *
   * A capability flag rather than a derived value, so a caller can say "this
   * device cannot report GPU time" without inferring it from a `null` that might
   * also mean "not read back yet". The `gpu` budget in `RenderBudget` is checked
   * only when this is true, because a budget that cannot be measured must not
   * report a breach of a number it never had.
   */
  readonly gpuTimingAvailable: boolean;
}

/** Everything that changes once per frame. Written into the frame uniform. */
export interface FrameState {
  readonly view: Float32Array;
  readonly proj: Float32Array;
  readonly viewProj: Float32Array;
  readonly invView: Float32Array;
  readonly invProj: Float32Array;
  readonly invViewProj: Float32Array;
  readonly camPos: readonly number[];
  readonly time: number;
  readonly delta: number;
  readonly elapsed: number;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly exposure: number;
  readonly alpha: number;
}

// ---------------------------------------------------------------------------
// Re-exports that keep both sides aligned
// ---------------------------------------------------------------------------

export type { TextureSlotSpec };
export type { BlendSpec, DepthSpec, PrimitiveTopology } from './pipeline-state.ts';
