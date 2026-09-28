/**
 * The present pass: where a rendered frame becomes a visible one.
 *
 * # The step this adds to the frame order
 *
 * `renderer.ts` owns the frame order and this module is step 6b, inserted
 * between it and the submit:
 *
 * ```txt
 *   6.  encode   — one command encoder, one render pass, N draws
 *   6b. present  — one more pass on the same encoder: the fullscreen tone map
 *   7.  submit   — one queue.submit
 *   8.  stats
 * ```
 *
 * It is 6b rather than 7 because it is not a phase: it always runs, it has no
 * relationship to the opaque/transparent split, and it must see the finished
 * image. It is *inside* step 6 rather than after it because it should share the
 * encoder — see {@link PresentPass.render}.
 *
 * # One mechanism, two uses: the offscreen intermediate
 *
 * **A canvas texture is never multisampled, and it is never sampleable.** Both
 * facts force the same shape, which is why there is one code path and not two:
 *
 *   - `getCurrentTexture()` always returns a single-sample attachment, so 4x MSAA
 *     has to be built by hand: an offscreen 4x target whose resolve destination
 *     is the thing you want to end up with.
 *   - The swapchain is configured `RENDER_ATTACHMENT | COPY_SRC` and no
 *     `TEXTURE_BINDING` — granting texture binding to a swapchain image forces
 *     the compositor path to keep a resolvable copy of every frame — so the
 *     presented image **cannot be read back by a shader**. So a tone map cannot
 *     read the canvas either.
 *
 * Therefore: anything that needs to read the frame, or that needs MSAA, renders
 * into an offscreen target and then gets copied to the presentation target by a
 * fullscreen pass. `hdr: true` and `sampleCount: 4` are the same mechanism with a
 * different intermediate format.
 *
 * There is exactly one case that does **not** need an intermediate: the plain
 * straight-to-canvas draw with no tone map and no MSAA. Then the scene is already
 * in the right place, no intermediate is allocated, and {@link PresentPass.render}
 * does nothing at all. That case is free, and it is the one that has to stay
 * free — see the note on the default in the report that accompanied this file.
 *
 * **What that free case costs you.** A material writes *linear* values. A canvas
 * target in `bgra8unorm` — which is what `getPreferredCanvasFormat()` returns on
 * every desktop — stores them verbatim, with no transfer function applied
 * anywhere. So the direct path produces a linear image displayed as if it were
 * sRGB: far too dark, and a lit surface looks unlit. Nothing errors. That is
 * precisely the bug the tone map material exists to fix, so the direct path is a
 * deliberate opt-out rather than a neutral default: it is right when the target
 * format is `*-srgb` (the hardware encodes), or when a caller wants the raw
 * linear values for their own reasons.
 *
 * # The depth attachment nobody expects
 *
 * The intermediate always carries depth, even though this pass does not read it.
 * The scene's materials declare `depthStencil` (apse materials do by default), and
 * **a render pass must agree with its pipelines about whether a depth attachment
 * exists.** A pass with no depth attachment and a pipeline that has one is a
 * validation error whose consequence is that the *entire pass is discarded*,
 * including the clear — a uniformly black frame and no error anywhere. This is
 * the most expensive kind of bug there is, and it is why `renderer.ts` puts depth
 * on its readback target even though nothing samples it.
 *
 * The mirror image applies to this pass: the pipeline here declares **no**
 * `depthStencil` at all, because the fullscreen triangle needs neither testing
 * nor writing, and a pipeline that has one cannot be used in a pass that has
 * none. Which is why {@link PresentPass} compiles its own pipeline rather than
 * drawing with the material's — see that method for the details.
 *
 * # What this owns, and when it frees it
 *
 * Every texture the intermediates need, on every path, including across
 * `resize()`. A resize leak is the classic bug in this layer: it is invisible
 * until the tab has been open for an hour, and the symptom is an OOM on exactly
 * the devices least able to survive one. So resize destroys before it allocates,
 * never the other way round, and `dispose()` is idempotent.
 */

import { fail } from '../core/error.ts';
import { Resource, ResourceScope } from '../core/resource.ts';
import { BIND_GROUP, FRAME_BLOCK, SCENE_FRAME_BYTES } from '../core/slot.ts';
import { upload } from '../geometry/mesh.ts';
import type { GpuMesh } from '../geometry/mesh.ts';
import { FRAGMENT_ENTRY, VERTEX_ENTRY, Material, deviceCache } from '../material/material.ts';
import {
  EXPOSURE_FRAME_FIELD,
  FULLSCREEN_LAYOUT,
  HDR_TARGET_FORMATS,
  fullscreenMesh,
  tonemapMaterial,
} from '../material/tonemap.ts';
import type { TonemapOptions } from '../material/tonemap.ts';
import { TEXTURE_USAGE, withErrorScope } from './device.ts';
import type { AseDevice } from './device.ts';
import { createColorTarget } from './target.ts';
import type { RenderTargetImpl, RenderTargetDevice } from './target.ts';
import type { RenderTarget } from './types.ts';

// ---------------------------------------------------------------------------
// Scratch
//
// `render()` runs sixty times a second and must not allocate: every pass
// descriptor, every dynamic-offset list, and every string it needs is built once
// at module scope and mutated. WebGPU converts an IDL dictionary to a native
// struct synchronously inside the call, so handing the same object to
// `beginRenderPass` on every frame is exactly as safe as building a new one — and
// it is the difference between a frame that allocates nothing and one that hands
// the collector three short-lived objects per frame.
//
// Two PresentPass instances alternating will overwrite each other's scratch. That
// is fine: every field is written before it is read.
// ---------------------------------------------------------------------------

/** Byte offset 0 into the object block. A fullscreen pass has one "object". */
/**
 * The dynamic offset the fullscreen triangle binds.
 *
 * **Not 0.** Byte 0 of the shared scene buffer is the frame region; object 0 now
 * begins at `SCENE_FRAME_BYTES`. Binding 0 here would satisfy the binding with
 * the camera and read `camPos` as the first float of a world matrix — a valid
 * in-range read, a successful draw, and a fullscreen triangle transformed to
 * nothing.
 */
const OBJECT_OFFSETS: readonly number[] = Object.freeze([SCENE_FRAME_BYTES]);

/**
 * Float index of `frame.exposure` in the frame block's CPU mirror.
 *
 * Resolved from the generated `FRAME_BLOCK` rather than written as a literal,
 * because a hand-computed offset in a uniform block is exactly the class of bug
 * that produces a plausible-looking wrong image rather than an error.
 */
const EXPOSURE_INDEX = ((): number => {
  const field = FRAME_BLOCK.fields.find((f) => f.name === EXPOSURE_FRAME_FIELD);
  if (field === undefined) {
    fail('INTERNAL_INVARIANT',
      `The frame block has no "${EXPOSURE_FRAME_FIELD}" field.`, {
      why: 'apse reserves the name in FRAME_FIELDS for exactly this pass, and the generated struct is the only authority on where it sits.',
      fix: 'Restore the field in FRAME_FIELDS in src/core/slot.ts.',
    });
  }
  return field.offset >> 2;
})();

const ENCODER_DESC: GPUCommandEncoderDescriptor = { label: '' };

/**
 * The single colour attachment, mutated per frame.
 *
 * Its own object rather than `PASS_DESC.colorAttachments[0]`, because the
 * descriptor's `colorAttachments` is an `Iterable` in the IDL — there is no index
 * to hold onto, and one shared object is one fewer allocation.
 */
const ATTACHMENT: GPURenderPassColorAttachment = {
  // Replaced every frame. A canvas target's view expires at present, so this has
  // to be re-acquired rather than cached — see RenderTargetImpl.
  view: undefined as unknown as GPUTextureView,
  resolveTarget: undefined,
  // Unused while loadOp is 'load', but present so switching to 'clear' is a
  // one-field change rather than a type error.
  clearValue: { r: 0, g: 0, b: 0, a: 0 },
  // 'load', not 'clear'. The scene's colour is in this very attachment in the
  // direct path, and clearing it would throw away the frame this pass exists to
  // deliver. Note the spelling: WebGPU has no `loadOp: 'store'` — the two enums
  // are `GPULoadOp = 'load' | 'clear'` and `GPUStoreOp = 'store' | 'discard'`,
  // so "load and keep" is `loadOp: 'load'` plus `storeOp: 'store'`.
  loadOp: 'load',
  storeOp: 'store',
};

const PASS_DESC: GPURenderPassDescriptor = {
  label: '',
  colorAttachments: [ATTACHMENT],
  // Absent on purpose. See the module comment and #compile: the pipeline has no
  // depthStencil, and a pass with a depth attachment and a pipeline without one
  // is as much a validation error as the reverse.
  depthStencilAttachment: undefined,
};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface PresentOptions {
  /** Where the finished image ends up. Usually the canvas target. */
  readonly target: RenderTarget;
  /**
   * The tone map, or `null` for no present pass at all.
   *
   * `null` is only legal when no intermediate is needed — that is, when `hdr` is
   * false *and* `sampleCount` is 1. Anything else would produce a frame that is
   * never delivered, which is rejected rather than accepted quietly.
   */
  readonly toneMapping?: TonemapOptions | null;
  /** Render the scene into an intermediate instead of straight to the target. */
  readonly hdr?: boolean;
  readonly sampleCount?: 1 | 4;
  /** Prefix for every label this pass creates. Default 'apse.present'. */
  readonly label?: string;
}

/** MSAA levels the intermediate can be created at. Matches `RenderTargetImpl`. */
const SAMPLE_COUNTS: readonly (1 | 4)[] = [1, 4];

/**
 * The tone map a renderer should use when the caller does not choose.
 *
 * **ON, and `aces`, and that is a deliberate default rather than a neutral
 * one.** A material writes linear values; a canvas in `bgra8unorm` — which is
 * what `getPreferredCanvasFormat()` returns on every desktop — stores them
 * verbatim with no transfer function applied. A straight-to-canvas draw with no
 * present pass therefore produces a linear image displayed as if it were sRGB:
 * far too dark, with a lit surface reading as unlit and nothing anywhere
 * reporting an error. A library whose out-of-the-box output is darker than
 * everybody else's loses the argument before it starts.
 *
 * `aces` rather than `none` because `operator: 'none'` still applies the sRGB
 * encode — that is the part which fixes the darkness — so both fix the bug, and
 * the curve is what makes an unbounded highlight read as a highlight instead of
 * a flat white blob. It matches `tonemapMaterial`'s own default
 * (`tonemapMaterialSpec`), so a material built directly and a present pass built
 * here agree. Set `toneMapping: null` to turn the pass off entirely.
 */
export const DEFAULT_TONE_MAPPING: TonemapOptions = Object.freeze({ operator: 'aces' });

// ---------------------------------------------------------------------------
// PresentPass
// ---------------------------------------------------------------------------

export class PresentPass extends Resource {
  /**
   * The target the scene should be rendered INTO.
   *
   * Differs from `opts.target` whenever `hdr` or `sampleCount: 4` is on: it is
   * then the offscreen intermediate, and the *sampled* image is its
   * `sampleView` — with MSAA that is the resolve texture, not the multisampled
   * attachment, because binding a multisampled texture as a sampled texture is a
   * validation error rather than a blurry result.
   *
   * Typed as the `RenderTarget` interface rather than `RenderTargetImpl` because
   * that is the whole contract this class needs, and a renderer's `target`
   * parameter should be the interface too. Note the direction: a
   * `RenderTargetImpl` satisfies `RenderTarget`, not the other way round, so a
   * caller passing `sceneTarget` into a function that demands `RenderTargetImpl`
   * has a type error — which is the correct outcome, because that function is
   * about to need methods the interface does not have.
   */
  readonly sceneTarget: RenderTarget;

  /**
   * The tone map material, or `null` in the direct-to-target path.
   *
   * Exposed so a caller can retune the image per frame with no pipeline rebuild:
   * `setSlot('operator', OPERATOR_IDS.reinhard)` and `setSlot('gamma', 0.2)` go
   * here, and exposure goes through {@link PresentPass.setExposure} because it
   * lives in the frame block rather than the material block. {@link
   * PresentPass.render} flushes any pending slot writes before it encodes.
   */
  readonly material: Material | null;

  readonly #gpu: GPUDevice;
  readonly #target: RenderTarget;
  /** The intermediate, or `null` in the direct path where we do not own one. */
  readonly #owned: RenderTargetImpl | null;
  readonly #mesh: GpuMesh | null;
  /** Pipelines by `sourceFormat:destinationFormat:sampleCount`. */
  readonly #pipelines = new Map<string, GPURenderPipeline>();
  readonly #destinationSamples: 1 | 4;
  readonly #label: string;
  #scope: ResourceScope;
  /** What `frame.exposure` should read. See {@link PresentPass.setExposure}. */
  #exposure = 1;

  /**
   * Builds a present pass. There is no public constructor.
   *
   * **Why this is a factory and not `new`:** a present pass compiles at least one
   * shader pipeline, and pipeline compilation is asynchronous on purpose —
   * `createRenderPipeline` blocks the calling thread for two to five seconds on a
   * cold cache, which is the single most common cause of a slow first frame in
   * WebGPU. A synchronous constructor could only either block or hand back an
   * object that cannot draw yet, and an object that cannot draw yet is a
   * half-built renderer: the first frame renders nothing, with no error. The same
   * reasoning, and the same shape, as `Material.create` and `Renderer.create`.
   */
  private constructor(init: {
    gpu: GPUDevice;
    target: RenderTarget;
    owned: RenderTargetImpl | null;
    mesh: GpuMesh | null;
    material: Material | null;
    destinationSamples: 1 | 4;
    label: string;
    scope: ResourceScope;
  }) {
    super();
    this.#gpu = init.gpu;
    this.#target = init.target;
    this.#owned = init.owned;
    this.#mesh = init.mesh;
    this.material = init.material;
    this.sceneTarget = init.owned ?? init.target;
    this.#destinationSamples = init.destinationSamples;
    this.#label = init.label;
    // The same scope `create()` filled: it is what makes a half-built pass
    // release everything it had already allocated, and it is what `dispose()`
    // empties. Two scopes would mean one of them owns nothing.
    this.#scope = init.scope;
  }

  /**
   * Creates a present pass and compiles everything it will ever need.
   *
   * `device` is typed as the same structural minimum `createColorTarget` takes, so
   * an `AseDevice` is accepted and a three-field fake is enough to test the
   * texture accounting without a GPU. Nothing here needs the canvas, the context,
   * or the adapter: the destination format comes from `opts.target`, which knows
   * it already.
   */
  static async create(
    device: AseDevice | RenderTargetDevice,
    opts: PresentOptions,
  ): Promise<PresentPass> {
    const gpu = device.device;
    const target = opts.target;
    const label = opts.label ?? 'apse.present';
    const samples = opts.sampleCount ?? 1;
    const tone = opts.toneMapping ?? null;
    const hdr = opts.hdr === true;

    assertSampleCount(samples, label);
    const destinationSamples = assertDestinationSamples(target, label);

    // One rule, three ways of tripping it: if the scene renders anywhere other
    // than the destination, something has to move the image across, and the only
    // thing that can is a tone map. Saying "no tone map" in that state describes
    // a frame that is never delivered, so it is refused with the specific reason
    // rather than accepted and left to render nothing.
    const needsIntermediate = hdr || samples > 1 || tone !== null;
    if (needsIntermediate && tone === null) {
      const what = hdr
        ? 'hdr: true renders the scene into an offscreen intermediate'
        : `sampleCount: ${samples} needs an offscreen multisampled attachment, because a canvas texture can never be multisampled`;
      fail('OPTION_UNKNOWN',
        `PresentPass "${label}" was asked for ${what}, and toneMapping: null, so nothing would put the image on "${label}"'s target.`, {
          why: 'The intermediate exists so that a later pass can read the frame. With no later pass, the image is rendered into a texture nobody samples and the target keeps whatever it had — a black or garbage frame with no validation error anywhere.',
          fix: 'Pass a tone map: `toneMapping: { operator: "none" }` is a straight copy with no curve and is the cheapest thing that will move the image across.',
        });
    }

    const scope = new ResourceScope();
    try {
      if (!needsIntermediate) {
        // The degenerate path, and the cheapest one in the library: no
        // intermediate, no material, no pipeline, no mesh. `render()` is a no-op
        // because the scene has already drawn the right pixels into the right
        // place.
        return new PresentPass({
          gpu, target, owned: null, mesh: null, material: null, destinationSamples, label, scope,
        });
      }

      // Non-null from here on: the guard above fails for every case where it is
      // not, and `needsIntermediate` cannot be true with a null tone map past it.
      const toneOptions = tone as TonemapOptions;

      const scene = createColorTarget(device, {
        width: target.width,
        height: target.height,
        format: intermediateFormat(hdr, toneOptions, target, label),
        sampleCount: samples,
        // Always. See the module comment: the scene's pipelines have
        // depthStencil, and a pass without a depth attachment is discarded whole.
        depth: true,
        // Sampleable — but only when it is single-sampled. `createColorTarget`
        // ORs this onto the colour attachment, and a multisampled texture with
        // TEXTURE_BINDING is a validation error rather than a wasted flag, so with
        // 4x the resolve texture (which the target allocates with
        // TEXTURE_BINDING of its own accord) is what the tone map samples.
        usage: samples > 1 ? 0 : TEXTURE_USAGE.TEXTURE_BINDING,
        label: `${label}:scene`,
      });
      scope.own(scene);

      const source = scene.sampleView;
      if (source === undefined) {
        fail('INTERNAL_INVARIANT',
          `PresentPass "${label}" created an offscreen target with no sampleable view.`, {
          why: 'A colour target that owns its texture always has a sample view. Reaching this means the intermediate was built as something other than a colour target, and the present pass would sample nothing.',
          fix: 'This is a bug in apse. Please report it with the PresentOptions you passed.',
        });
      }

      const material = await tonemapMaterial(gpu, {
        ...toneOptions,
        // The destination decides the sRGB question, and it must be the format
        // actually drawn into — a mismatch here is a too-dark image with no
        // error, which is the failure this whole module exists to prevent.
        targetFormat: target.format,
      });
      scope.own(material);
      material.setTexture('texture', source);

      const mesh = upload(gpu, fullscreenMesh(), { name: `${label}:fullscreen` });
      scope.own(mesh);

      const pass = new PresentPass({
        gpu, target, owned: scene, mesh, material, destinationSamples, label, scope,
      });

      // Built here rather than lazily in render(), because a pipeline is a
      // compiled program and compiling one on the frame the user resizes the
      // window is a multi-hundred-millisecond stall. The cache is keyed by
      // everything a pipeline bakes in: the format it samples, the format it
      // writes, and the sample count.
      const pipeline = await pass.#compile(destinationSamples);
      pass.#pipelines.set(pipeline.key, pipeline.pipeline);
      pass.#exposure = toneOptions.exposure ?? 1;

      // Warm the bind groups now. `setTexture` above invalidated the texture
      // bind group, and rebuilding it inside render() would mean a GPU object
      // allocated on the first frame after every resize.
      void material.textureBindGroup;
      void material.materialBindGroup;
      void material.objectBindGroup;
      void material.frameBindGroup;

      return pass;
    } catch (error) {
      scope.dispose();
      throw error;
    }
  }

  /**
   * Sets the exposure the tone map applies, in stops-free linear units.
   *
   * Exposure is **not** a material slot. `core/slot.ts` reserves the name in the
   * generated frame block — with the comment "tone-map exposure, applied by the
   * present pass", which is this pass — and `RESERVED_SLOT_NAMES` refuses a
   * material slot that shadows it. So it goes in `frame.exposure`, which is also
   * where it belongs: one exposure for the whole frame, not one per material.
   *
   * That has a consequence worth knowing about. The renderer writes
   * `exposure: 1` on every frame's upload, so a value written before
   * `renderer.render()` is overwritten before the frame is drawn. This pass
   * therefore re-asserts the value on every `render()`, and only when it differs
   * from what the frame block currently holds — which is a 4-byte compare in the
   * common case and one 432-byte `writeBuffer` on the frame it changes.
   */
  setExposure(value: number): void {
    this.assertLive('PresentPass');
    if (!Number.isFinite(value) || value <= 0) {
      fail('SLOT_VALUE_NOT_FINITE',
        `PresentPass "${this.#label}" was given an exposure of ${String(value)}.`, {
        why: 'Exposure multiplies every pixel of the scene before the curve. Zero produces a black frame, a negative value clamps to black through the operator, and NaN propagates through every fragment and blanks the pass.',
        fix: 'Pass a positive, finite number. 1 is the identity; 0.5 darkens by one stop.',
        detail: { kind: 'numeric', field: 'exposure', value, min: 0 },
      });
    }
    this.#exposure = value;
  }

  /** The exposure the next frame will use. */
  get exposure(): number {
    return this.#exposure;
  }

  /**
   * Runs the tone map into `opts.target`. Call after the scene render.
   *
   * Pass an encoder to record into a command buffer someone else will submit —
   * that is one `queue.submit` per frame instead of two, and it is what the
   * renderer should do. Called with no argument it opens and submits its own,
   * which is correct and costs a submit.
   *
   * `stamp` is a `timestampWrites` value for this pass, or nothing. The
   * renderer needs one because a `(beginning, end)` write index may only be
   * written once per submission, so the present pass cannot share a pair with
   * the scene passes that precede it — a frame that opens an opaque pass, a
   * transparent pass and this one takes three pairs. It is a parameter rather
   * than something the pass owns because the query set is the renderer's, and
   * the two are disposed independently.
   *
   * A no-op in the direct-to-target path, where the scene already drew the final
   * image. It is a no-op *by design* and not by accident: that path exists to cost
   * nothing.
   */
  render(encoder?: GPUCommandEncoder, stamp?: GPURenderPassTimestampWrites): void {
    this.assertLive('PresentPass');
    const scene = this.#owned;
    const mesh = this.#mesh;
    const material = this.material;
    if (scene === null || mesh === null || material === null) return;

    // Exposure is re-asserted only when it is actually wrong. Reading the CPU
    // mirror directly rather than through `UniformBlock.get` is what keeps this
    // allocation-free; `get` builds a fresh array on every call.
    const frame = material.frameUniforms;
    if (frame.block.f32[EXPOSURE_INDEX] !== this.#exposure) {
      frame.set('exposure', this.#exposure);
      frame.flush();
    }

    // Through the same accessor the renderer uses, and read once into a local:
    // for a canvas target this re-acquires from getCurrentTexture(), which is
    // valid only for the frame being encoded. Reading it inside a loop would
    // mint a view per iteration and hand later code a view of a dead texture.
    const target = this.#target;
    ATTACHMENT.view = target.colorView;
    ATTACHMENT.resolveTarget = this.#destinationSamples > 1 ? target.sampleView : undefined;
    // 'load', not 'clear': in the direct path the scene's colour is in this very
    // attachment, and clearing it would throw away the frame this pass exists to
    // deliver. The fullscreen triangle covers every pixel regardless, so nothing
    // of the previous contents survives into the output.
    ATTACHMENT.loadOp = 'load';
    ATTACHMENT.storeOp = 'store';
    PASS_DESC.label = `${this.#label}:pass`;
    // Assigned unconditionally, including to `undefined`. The descriptor is
    // module scratch, so a pass that did not get a stamp has to clear the last
    // one's — otherwise a timestamp is attributed to whichever pass the previous
    // caller happened to be timing.
    PASS_DESC.timestampWrites = stamp;

    let enc: GPUCommandEncoder;
    let own: GPUCommandEncoder | null = null;
    if (encoder === undefined) {
      ENCODER_DESC.label = `${this.#label}:present`;
      own = this.#gpu.createCommandEncoder(ENCODER_DESC);
      enc = own;
    } else {
      enc = encoder;
    }

    // Slot writes are queue-ordered against submits, so flushing before the
    // submit is the same as flushing after the encoding — and flushing before is
    // also correct when the caller owns the encoder and submits later. A clean
    // block costs nothing: this is the renderer's own lazy pattern.
    if (material.slotsDirty) material.flushSlots();

    const pass = enc.beginRenderPass(PASS_DESC);
    // No depthStencilAttachment: see the module comment. The pipeline agrees,
    // which is the only reason this pass is not discarded whole.
    pass.setPipeline(this.#pipeline());
    // All four groups. The scaffold declares the frame and object bindings on
    // every material whether or not the body reads them, so an unsatisfied
    // pipeline layout is a validation error — not a warning. The frame uniform
    // this binds is the renderer's own, written this frame; the tone map does
    // not read it (it takes its UV from the source texture's dimensions, so it
    // cannot be reading a stale one), but the binding has to be satisfied.
    // Frame and object are one group and one buffer now, so this is one call
    // where the draw loop used to make two.
    pass.setBindGroup(BIND_GROUP.scene, material.sceneBindGroup, OBJECT_OFFSETS);
    pass.setBindGroup(BIND_GROUP.material, material.materialBindGroup);
    pass.setBindGroup(BIND_GROUP.texture, material.textureBindGroup);
    pass.setVertexBuffer(0, mesh.vertexBuffer);
    // Three vertices, not indexed. `cull: 'none'`, so a reversed winding could
    // not delete the screen — but the mesh is correct anyway, and a test asserts
    // it, because a fullscreen triangle with the wrong third vertex leaves a
    // diagonal seam across the whole image.
    pass.draw(mesh.indexCount, 1, 0, 0);
    pass.end();

    if (own !== null) this.#gpu.queue.submit([own.finish()]);
  }

  /**
   * Resizes the intermediates. Call when the canvas resizes.
   *
   * A no-op in the direct-to-target path: the presentation target's size is the
   * canvas's business, and resizing it here would fight `CanvasSizer`.
   */
  resize(width: number, height: number): void {
    this.assertLive('PresentPass');
    const scene = this.#owned;
    if (scene === null) return;
    // RenderTargetImpl destroys before it allocates, so the peak is one target
    // rather than two, and there is no window in which a field still points at a
    // destroyed texture.
    scene.resize(width, height);
    // The views were recreated with the textures, so the material is now bound to
    // a dead one. Rebinding marks the texture bind group invalid, and the read
    // here rebuilds it — outside render(), which is the point.
    const view = scene.sampleView;
    if (view === undefined) {
      fail('INTERNAL_INVARIANT',
        `PresentPass "${this.#label}" lost its sampleable view across a resize.`, {
          why: 'A colour target that owns its texture always has a sample view. If it is gone, the intermediate is disposed and every subsequent frame would sample a destroyed texture — which in WebGPU does nothing at all, silently.',
          fix: 'This is a bug in apse. Please report it with the PresentOptions you passed.',
      });
    }
    this.material?.setTexture('texture', view);
    void this.material?.textureBindGroup;
  }

  /**
   * Releases the intermediate, the fullscreen mesh, and the material. Idempotent.
   *
   * The material's uniform buffer and the mesh's vertex buffer go with it. The
   * pipeline does not: it lives in the per-device pipeline cache and is shared
   * with any other present pass built from the same generated WGSL, so
   * destroying it here would break those. That is `Material`'s contract, not a
   * special case here.
   */
  protected override onDispose(): void {
    this.#scope.dispose();
    this.#pipelines.clear();
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /**
   * The pipeline for this pass's current formats. Present in the cache because
   * `create()` put it there; a miss is an apse bug, not a runtime condition, so it
   * is named rather than compiled on demand — `render()` must not allocate, and
   * "compile the missing one" would be an allocation on exactly the frame that
   * cannot afford one.
   */
  #pipeline(): GPURenderPipeline {
    const key = pipelineKey(this.#owned?.format ?? this.#target.format, this.#target.format, this.#destinationSamples);
    const hit = this.#pipelines.get(key);
    if (hit !== undefined) return hit;
    fail('INTERNAL_INVARIANT',
      `PresentPass "${this.#label}" has no pipeline for ${key}.`, {
      why: 'Pipelines are compiled in create() and cached by (source format, destination format, sample count), so every key this pass can ask for was populated before the first frame.',
      fix: 'This is a bug in apse. Please report it with the PresentOptions you passed.',
    });
  }

  /**
   * Compiles the fullscreen pipeline for this pass's destination format.
   *
   * **Not the material's `renderPipeline`,** and the reason is specific: a WGSL
   * pipeline created through `Material` always declares a `depthStencil` state,
   * because the scaffold bakes one in from the spec. A pipeline with depth state
   * may only be used in a pass that has a matching depth attachment, and this
   * pass deliberately has none — a fullscreen triangle has nothing to test and
   * nothing to write, and attaching depth to it would make the present cost a
   * full-screen depth resolve for nothing. The failure mode of getting this
   * backwards is the worst one in this area: the pass is discarded *whole*,
   * including its clear, and nothing anywhere reports an error.
   *
   * So this compiles the same generated WGSL, through the same pipeline layout
   * (so the bind groups the material owns are exactly the ones this expects), with
   * the destination format it will actually be drawn into, and with no
   * `depthStencil` key at all.
   *
   * The cost is that the material also owns a pipeline this class never draws
   * with. That is the price of building the tone map through `Material.create` at
   * all, and it is one extra compile at startup rather than per frame. A
   * `depthStencil: null` option on `MaterialSpec` would remove it; that belongs to
   * the scaffold, not to this file.
   */
  async #compile(sampleCount: 1 | 4): Promise<{ key: string; pipeline: GPURenderPipeline }> {
    const material = this.material as Material;
    const key = pipelineKey(this.#owned?.format ?? this.#target.format, this.#target.format, sampleCount);
    const cache = deviceCache(this.#gpu);
    const module = this.#gpu.createShaderModule({
      label: `${this.#label}:shader`,
      code: material.wgsl,
    });
    const pipeline = await withErrorScope(
      this.#gpu,
      'validation',
      () => this.#gpu.createRenderPipelineAsync({
        label: `${this.#label}:pipeline`,
        layout: cache.pipelineLayout(material.generated.resolved),
        vertex: {
          module,
          entryPoint: VERTEX_ENTRY,
          buffers: [FULLSCREEN_LAYOUT.gpuLayout()],
        },
        fragment: {
          module,
          entryPoint: FRAGMENT_ENTRY,
          targets: [{ format: this.#target.format }],
        },
        primitive: {
          topology: 'triangle-list',
          // The material spec says 'none' too. It is stated in both places
          // because the two pipelines are built by different code and only one of
          // them reads the spec.
          cullMode: 'none',
          frontFace: 'ccw',
        },
        multisample: { count: sampleCount },
        // depthStencil is intentionally absent. See the method comment.
      }),
      `PresentPass "${this.#label}" fullscreen pipeline`,
    );
    return { key, pipeline };
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * The pipeline cache key: everything a `GPURenderPipeline` bakes in at creation.
 *
 * All three fields, in the API's own order. A cache that silently ignored the
 * sample count would be wrong the first time a present pass is built for two
 * destinations with different sample counts, and would be wrong *quietly* — the
 * second pass would draw with the first one's pipeline and the command buffer
 * would be invalidated with no exception.
 */
function pipelineKey(
  sourceFormat: GPUTextureFormat,
  destinationFormat: GPUTextureFormat,
  sampleCount: number,
): string {
  return `${sourceFormat}:${destinationFormat}:${sampleCount}`;
}

/**
 * The intermediate's colour format.
 *
 * `hdr` picks the HDR format; without it, the intermediate is in the
 * destination's own format, so the present pass is a straight same-format copy
 * plus a curve. That combination is legal and occasionally wanted (a debug view,
 * or MSAA on a device that cannot render `rgba16float` multisampled), but it
 * costs a full-screen pass to move an already-clipped image, and it cannot
 * recover a highlight that was already discarded. Say so rather than let it be a
 * silent performance cliff.
 */
function intermediateFormat(
  hdr: boolean,
  tone: TonemapOptions,
  target: RenderTarget,
  label: string,
): GPUTextureFormat {
  if (!hdr) return target.format;
  const format = tone.hdrFormat ?? 'rgba16float';
  if (!HDR_TARGET_FORMATS.includes(format)) {
    fail('OPTION_UNKNOWN',
      `PresentPass "${label}" was given hdrFormat "${format}", which is not one of: ${HDR_TARGET_FORMATS.join(', ')}.`, {
      why: 'The HDR intermediate has to be renderable and filterable: it is rendered into and then sampled. rgba16float is both. rgba32float is renderable on the core profile but is the format most often refused in compatibility mode, where 32-bit float render targets and blending are restricted.',
      fix: `Use one of: ${HDR_TARGET_FORMATS.join(', ')}, or omit it for the rgba16float default. Check \`device.featureLevel\` — on 'compatibility' prefer rgba16float unconditionally.`,
    });
  }
  return format;
}

function assertSampleCount(sampleCount: number, label: string): void {
  if (SAMPLE_COUNTS.includes(sampleCount as 1 | 4)) return;
  fail('OPTION_UNKNOWN',
    `PresentPass "${label}" was given sampleCount ${sampleCount}, which is not one of: ${SAMPLE_COUNTS.join(', ')}.`, {
    why: 'The sample count is baked into the render pipeline as well as the attachment, and apse builds exactly two shapes: single-sampled, and 4x with a resolve target. 2x exists in the hardware and not in WebGPU.',
    fix: 'Use 1, or 4. For more than 4x, render at a higher internal resolution and downsample in a post pass, which is what 8x MSAA amounts to anyway.',
  });
}

/**
 * The destination's sample count, which has to be 1 or 4.
 *
 * A multisampled *destination* is supported — the resolve destination is set and
 * the pipeline is built for it — because it is the same two lines `renderer.ts`
 * already writes, and it makes a "present into an offscreen target for a later
 * pass" path work. A canvas can never be one.
 */
function assertDestinationSamples(target: RenderTarget, label: string): 1 | 4 {
  const count = target.sampleCount;
  if (SAMPLE_COUNTS.includes(count as 1 | 4)) return count as 1 | 4;
  fail('RENDER_TARGET_FORMAT_MISMATCH',
    `PresentPass "${label}" was given a target whose sampleCount is ${count}.`, {
    why: 'A render pipeline bakes its sample count in at creation, and this pass builds its pipeline once. Presenting into an attachment with a sample count it was not built for invalidates the whole command buffer with no exception.',
    fix: `Present into a target created with \`createColorTarget(device, { sampleCount: 1 | 4 })\`. A canvas target is always 1.`,
  });
}
