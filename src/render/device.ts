/**
 * Device acquisition.
 *
 * This is the bottom of the stack: the adapter, the device, and the canvas
 * configuration everything else is built on. Nothing above this file is
 * allowed to call `navigator.gpu` — a renderer that re-acquires its own device
 * is a renderer with two devices, and that is a bug you find in production.
 *
 * Four verified facts drive the whole design. They are not opinions.
 *
 * 1. **Compatibility mode is the default target.** WebGPU is ~87% of global
 *    traffic and is *not* Baseline: Firefox on Linux is Nightly-only, Firefox
 *    on Android sits behind a flag, Firefox on Intel Macs is unsupported, and
 *    Safari on iOS below A12 is unsupported. The dominant mobile cohort is
 *    GLES 3.1-class hardware. So we ask for `featureLevel: 'compatibility'`
 *    first and only reach for `'core'` when the adapter says it can.
 *
 * 2. **We never spread `adapter.limits` into `requiredLimits`.** That is the
 *    classic WebGPU bug. It reads like "checking what is available" but it
 *    silently makes the shipped page depend on limits that other people's
 *    devices do not have. Separately, `{...adapter.limits}` copies *nothing*:
 *    `GPUSupportedLimits` is an interface of prototype getters, not own
 *    properties, so the spread is `{}`. Both halves of that are load-bearing
 *    here, and both are tested.
 *
 * 3. **A bad descriptor does not surface as a JS exception.** `requestDevice`
 *    *rejects* with an `OperationError` for limits and a `TypeError` for
 *    features — and separately, any object it does hand back can be *invalid*,
 *    and invalidity is contagious: every buffer, texture and pipeline derived
 *    from it is invalid too, silently, forever. See {@link probeDeviceValidity}
 *    for how the second half is caught.
 *
 * 4. **Device loss is routine.** It happens on driver resets, on tab
 *    backgrounding, on sleep/wake, and especially often on macOS. It is not a
 *    crash and it is not exceptional; it is a normal event with a callback.
 *
 * Everything here is labelled. WebGPU error messages reference the most recent
 * `label`, so an unlabelled resource produces a validation error that names a
 * descriptor field but not the object that had the bug.
 */

import { AseError, fail } from '../core/error.ts';
import type { AseErrorCode } from '../core/error.ts';
import { err, ok } from '../core/result.ts';
import type { Result } from '../core/result.ts';

// ---------------------------------------------------------------------------
// Feature level
// ---------------------------------------------------------------------------

/**
 * Which set of limits and validation rules the device was created under.
 *
 * `'compatibility'` is the target. `'core'` means the adapter advertised
 * `core-features-and-limits` and we asked for it — either because `preferCore`
 * was set, or because the implementation ignored our compatibility request.
 */
export type FeatureLevel = 'core' | 'compatibility';

/** The feature that lifts compatibility mode's extra validation rules. */
const CORE_FEATURE: GPUFeatureName = 'core-features-and-limits';

/**
 * The texture usage bits, resolved from the platform when it has them.
 *
 * These three are constants fixed by the spec, and reading them through the
 * `GPUTextureUsage` global makes every call site untestable outside a browser
 * — which is most of a renderer. The fallbacks are the spec values, so a headless
 * run and a real one build the same descriptors.
 */
export const TEXTURE_USAGE = Object.freeze({
  COPY_SRC: typeof GPUTextureUsage === 'undefined' ? 0x01 : GPUTextureUsage.COPY_SRC,
  COPY_DST: typeof GPUTextureUsage === 'undefined' ? 0x02 : GPUTextureUsage.COPY_DST,
  TEXTURE_BINDING: typeof GPUTextureUsage === 'undefined' ? 0x04 : GPUTextureUsage.TEXTURE_BINDING,
  RENDER_ATTACHMENT: typeof GPUTextureUsage === 'undefined' ? 0x10 : GPUTextureUsage.RENDER_ATTACHMENT,
});

/**
 * `GPUBufferUsage` and `GPUMapMode`, resolved the same way and for the same
 * reason as {@link TEXTURE_USAGE}: the spec values are constants, and reading
 * them through the globals makes every call site untestable outside a browser.
 * `QUERY_RESOLVE` in particular is the bit that makes a timestamp query set
 * readable at all, and it has no `TEXTURE_USAGE` equivalent.
 */
export const BUFFER_USAGE = Object.freeze({
  MAP_READ: typeof GPUBufferUsage === 'undefined' ? 0x01 : GPUBufferUsage.MAP_READ,
  COPY_SRC: typeof GPUBufferUsage === 'undefined' ? 0x04 : GPUBufferUsage.COPY_SRC,
  COPY_DST: typeof GPUBufferUsage === 'undefined' ? 0x08 : GPUBufferUsage.COPY_DST,
  QUERY_RESOLVE: typeof GPUBufferUsage === 'undefined' ? 0x200 : GPUBufferUsage.QUERY_RESOLVE,
});

export const MAP_MODE = Object.freeze({
  READ: typeof GPUMapMode === 'undefined' ? 0x01 : GPUMapMode.READ,
});

/** Bytes in one resolved timestamp. A query result is a `uint64`. */
export const TIMESTAMP_BYTES = 8;

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/**
 * The limits apse reads, and the only ones it will ever request.
 *
 * Every entry needs a reason to exist. Anything not listed here is
 * deliberately not copied, not requested, and not part of the public surface:
 * a limit apse does not read is a limit apse has no correct fallback for.
 *
 * Two limits in the full spec table are absent on purpose.
 * `maxBindGroupsPlusVertexBuffers` and `maxImmediateSize` are recent additions,
 * and `requiredLimits` rejects *unknown keys* outright — so a limit added to
 * the spec next year is not safe to send today. apse reads neither.
 */
export type AseLimitName =
  | 'maxTextureDimension2D'
  | 'maxBufferSize'
  | 'maxUniformBufferBindingSize'
  | 'maxStorageBufferBindingSize'
  | 'minUniformBufferOffsetAlignment'
  | 'maxBindGroups'
  | 'maxBindingsPerBindGroup'
  | 'maxDynamicUniformBuffersPerPipelineLayout'
  | 'maxSampledTexturesPerShaderStage'
  | 'maxSamplersPerShaderStage'
  | 'maxStorageBuffersPerShaderStage'
  | 'maxStorageTexturesPerShaderStage'
  | 'maxUniformBuffersPerShaderStage'
  | 'maxVertexBuffers'
  | 'maxVertexAttributes'
  | 'maxVertexBufferArrayStride'
  | 'maxInterStageShaderVariables'
  | 'maxColorAttachments'
  | 'maxColorAttachmentBytesPerSample'
  | 'maxComputeWorkgroupStorageSize'
  | 'maxComputeInvocationsPerWorkgroup'
  | 'maxComputeWorkgroupSizeX'
  | 'maxComputeWorkgroupSizeY'
  | 'maxComputeWorkgroupSizeZ'
  | 'maxComputeWorkgroupsPerDimension';

/** The copied limits: a plain frozen object with no prototype surprises. */
export type AseLimits = Required<Pick<GPUSupportedLimits, AseLimitName>>;

/**
 * Which direction "better" runs for each limit.
 *
 * This is not decoration. For `maximum` limits, higher is better, so the
 * compatibility value is a *ceiling* on what we may request. For `alignment`
 * limits, lower is better, so it is a *floor* — `minUniformBufferOffsetAlignment`
 * of 256 means a dynamic offset of 0 is fine and 128 is not. Code, and tests,
 * that compare limits with `Math.max` get alignment limits exactly backwards.
 */
export const LIMIT_CLASSES: Readonly<Record<AseLimitName, 'maximum' | 'alignment'>> = Object.freeze({
  maxTextureDimension2D: 'maximum',
  maxBufferSize: 'maximum',
  maxUniformBufferBindingSize: 'maximum',
  maxStorageBufferBindingSize: 'maximum',
  minUniformBufferOffsetAlignment: 'alignment',
  maxBindGroups: 'maximum',
  maxBindingsPerBindGroup: 'maximum',
  maxDynamicUniformBuffersPerPipelineLayout: 'maximum',
  maxSampledTexturesPerShaderStage: 'maximum',
  maxSamplersPerShaderStage: 'maximum',
  maxStorageBuffersPerShaderStage: 'maximum',
  maxStorageTexturesPerShaderStage: 'maximum',
  maxUniformBuffersPerShaderStage: 'maximum',
  maxVertexBuffers: 'maximum',
  maxVertexAttributes: 'maximum',
  maxVertexBufferArrayStride: 'maximum',
  maxInterStageShaderVariables: 'maximum',
  maxColorAttachments: 'maximum',
  maxColorAttachmentBytesPerSample: 'maximum',
  maxComputeWorkgroupStorageSize: 'maximum',
  maxComputeInvocationsPerWorkgroup: 'maximum',
  maxComputeWorkgroupSizeX: 'maximum',
  maxComputeWorkgroupSizeY: 'maximum',
  maxComputeWorkgroupSizeZ: 'maximum',
  maxComputeWorkgroupsPerDimension: 'maximum',
});

/**
 * The WebGPU compatibility-mode default limits — the ceiling of what apse will
 * ever ask a device for.
 *
 * Source: WebGPU Editor's Draft (23 September 2026) §3.6.2 "Limits", the
 * "Compatibility Mode Default" column. They exist because compatibility mode
 * targets GLES 3.1-class hardware, and the entire point of the mode is that a
 * WebGPU page runs on a phone a WebGL page runs on. The differences that
 * actually bite:
 *
 *   maxTextureDimension2D ............... 4096    (core 8192)
 *   maxUniformBufferBindingSize ........ 16 KiB  (core 64 KiB)   <- the classic one
 *   maxColorAttachments ................. 4       (core 8)
 *   maxInterStageShaderVariables ....... 15      (core 16)       <- one fewer varying
 *   maxComputeInvocationsPerWorkgroup .. 128     (core 256)
 *   maxComputeWorkgroupSizeX / Y ........ 128     (core 256)
 *
 * The remaining values match the core defaults. They are listed anyway so this
 * table is a complete, checkable statement of apse's floor rather than a list of
 * exceptions to somebody else's list.
 */
export const COMPAT_LIMITS: Readonly<AseLimits> = Object.freeze({
  maxTextureDimension2D: 4096,
  maxBufferSize: 268435456,
  maxUniformBufferBindingSize: 16384,
  maxStorageBufferBindingSize: 134217728,
  minUniformBufferOffsetAlignment: 256,
  maxBindGroups: 4,
  maxBindingsPerBindGroup: 1000,
  maxDynamicUniformBuffersPerPipelineLayout: 8,
  maxSampledTexturesPerShaderStage: 16,
  maxSamplersPerShaderStage: 16,
  maxStorageBuffersPerShaderStage: 8,
  maxStorageTexturesPerShaderStage: 4,
  maxUniformBuffersPerShaderStage: 12,
  maxVertexBuffers: 8,
  maxVertexAttributes: 16,
  maxVertexBufferArrayStride: 2048,
  maxInterStageShaderVariables: 15,
  maxColorAttachments: 4,
  maxColorAttachmentBytesPerSample: 32,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 128,
  maxComputeWorkgroupSizeX: 128,
  maxComputeWorkgroupSizeY: 128,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
});

/**
 * The core-profile default limits, for the same limit set.
 *
 * Present so the difference is legible rather than folklore, and so a test can
 * assert the property that makes {@link compatRequiredLimits} safe: no
 * compatibility value is ever *better* than the core default, so asking for the
 * compatibility value on a core device is always legal and simply yields the
 * better of the two.
 *
 * Note `maxBufferSize` and `maxStorageBufferBindingSize` are identical on both
 * profiles. Compatibility mode's savings are in uniform buffer size, texture
 * dimension, and compute shape — not in how much memory you may hold.
 */
export const CORE_DEFAULT_LIMITS: Readonly<AseLimits> = Object.freeze({
  maxTextureDimension2D: 8192,
  maxBufferSize: 268435456,
  maxUniformBufferBindingSize: 65536,
  maxStorageBufferBindingSize: 134217728,
  minUniformBufferOffsetAlignment: 256,
  maxBindGroups: 4,
  maxBindingsPerBindGroup: 1000,
  maxDynamicUniformBuffersPerPipelineLayout: 8,
  maxSampledTexturesPerShaderStage: 16,
  maxSamplersPerShaderStage: 16,
  maxStorageBuffersPerShaderStage: 8,
  maxStorageTexturesPerShaderStage: 4,
  maxUniformBuffersPerShaderStage: 12,
  maxVertexBuffers: 8,
  maxVertexAttributes: 16,
  maxVertexBufferArrayStride: 2048,
  maxInterStageShaderVariables: 16,
  maxColorAttachments: 8,
  maxColorAttachmentBytesPerSample: 32,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
});

/**
 * Per-stage limits, deliberately *not* part of {@link AseLimits}.
 *
 * The spec normalises these at device creation: `maxStorageBuffersPerShaderStage`
 * is raised to `max(itself, inVertexStage, inFragmentStage)`, and the reverse
 * normalisation happens when `core-features-and-limits` is on. In compatibility
 * mode the vertex stage gets **zero** storage buffers and zero storage textures:
 * there is no read-write buffer in a vertex shader on the hardware that
 * compatibility mode targets.
 *
 * They are read from the *device* at runtime, never requested, and the IDL
 * marks them as not-yet-universally-implemented, so consumers must
 * feature-detect. They are declared here because they are the limits most
 * likely to turn a shader that compiles on a laptop into one that fails to
 * compile on a phone.
 */
export const COMPAT_VERTEX_STAGE_LIMITS = Object.freeze({
  maxStorageBuffersInVertexStage: 0,
  maxStorageTexturesInVertexStage: 0,
  maxStorageBuffersInFragmentStage: 8,
  maxStorageTexturesInFragmentStage: 4,
});

/**
 * Copies the limits apse depends on, field by field.
 *
 * This function exists because of a specific, verified trap: `GPUSupportedLimits`
 * is an interface of **prototype getters**, not own properties. Object spread
 * copies own enumerable properties only, so the idiomatic-looking
 *
 *     const requiredLimits = { ...adapter.limits };
 *
 * produces `{}` — not "all the limits", not "the limits it can see", nothing.
 * A renderer built on that line runs against the *device defaults* while its
 * author believes it is running against the adapter's ceilings, and then
 * overflows a 16 KiB uniform block on a phone.
 *
 * The copy is explicit, one property read per line, because the point is that a
 * reader can see exactly which limits cross the boundary. It is frozen and
 * plain: no prototype getters, no `__brand` nominal-typing marker, and no way
 * for a consumer to mutate a shared table.
 */
export function copyLimits(limits: GPUSupportedLimits): AseLimits {
  return Object.freeze({
    maxTextureDimension2D: limits.maxTextureDimension2D,
    maxBufferSize: limits.maxBufferSize,
    maxUniformBufferBindingSize: limits.maxUniformBufferBindingSize,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
    maxBindGroups: limits.maxBindGroups,
    maxBindingsPerBindGroup: limits.maxBindingsPerBindGroup,
    maxDynamicUniformBuffersPerPipelineLayout: limits.maxDynamicUniformBuffersPerPipelineLayout,
    maxSampledTexturesPerShaderStage: limits.maxSampledTexturesPerShaderStage,
    maxSamplersPerShaderStage: limits.maxSamplersPerShaderStage,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    maxStorageTexturesPerShaderStage: limits.maxStorageTexturesPerShaderStage,
    maxUniformBuffersPerShaderStage: limits.maxUniformBuffersPerShaderStage,
    maxVertexBuffers: limits.maxVertexBuffers,
    maxVertexAttributes: limits.maxVertexAttributes,
    maxVertexBufferArrayStride: limits.maxVertexBufferArrayStride,
    maxInterStageShaderVariables: limits.maxInterStageShaderVariables,
    maxColorAttachments: limits.maxColorAttachments,
    maxColorAttachmentBytesPerSample: limits.maxColorAttachmentBytesPerSample,
    maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
    maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
  } satisfies AseLimits);
}

/** Every limit apse depends on, in declaration order. */
export const ASE_LIMIT_NAMES: readonly AseLimitName[] = Object.freeze(
  Object.keys(COMPAT_LIMITS) as AseLimitName[],
);

/**
 * The `requiredLimits` apse sends: the compatibility defaults, verbatim.
 *
 * This is the whole trick, and it is worth stating plainly. Per the spec, a
 * device is created with the defaults for its feature level, and each entry in
 * `requiredLimits` raises a limit to `max(requested, default)`. Requesting a
 * value *worse* than the default is legal and simply has no effect. So sending
 * the compatibility defaults means:
 *
 *   - on a compatibility device you get exactly the compatibility defaults,
 *   - on a core device you get the core defaults, which are never worse,
 *   - and neither case requires a single optional feature, so device creation
 *     can never fail because a phone lacks something apse only wanted.
 *
 * The alternative — mirroring the adapter's limits upward — is what makes a
 * page work on the developer's machine and produce a validation error on
 * everyone else's. Note also that `requiredLimits` rejects unknown keys
 * outright, which is the second reason this list is curated rather than
 * derived: a limit added to the spec next year is not safe to send today.
 *
 * The return type is `Record<AseLimitName, number>`, not `GPUSupportedLimits`:
 * the record passed to `requestDevice` is a sparse dictionary, and typing it as
 * the interface — with its `__brand` marker and 34 members — would suggest apse
 * has an opinion about the 24 limits it deliberately does not send.
 */
export type AseRequiredLimits = Record<AseLimitName, number>;

export function compatRequiredLimits(): AseRequiredLimits {
  return {
    maxTextureDimension2D: COMPAT_LIMITS.maxTextureDimension2D,
    maxBufferSize: COMPAT_LIMITS.maxBufferSize,
    maxUniformBufferBindingSize: COMPAT_LIMITS.maxUniformBufferBindingSize,
    maxStorageBufferBindingSize: COMPAT_LIMITS.maxStorageBufferBindingSize,
    minUniformBufferOffsetAlignment: COMPAT_LIMITS.minUniformBufferOffsetAlignment,
    maxBindGroups: COMPAT_LIMITS.maxBindGroups,
    maxBindingsPerBindGroup: COMPAT_LIMITS.maxBindingsPerBindGroup,
    maxDynamicUniformBuffersPerPipelineLayout: COMPAT_LIMITS.maxDynamicUniformBuffersPerPipelineLayout,
    maxSampledTexturesPerShaderStage: COMPAT_LIMITS.maxSampledTexturesPerShaderStage,
    maxSamplersPerShaderStage: COMPAT_LIMITS.maxSamplersPerShaderStage,
    maxStorageBuffersPerShaderStage: COMPAT_LIMITS.maxStorageBuffersPerShaderStage,
    maxStorageTexturesPerShaderStage: COMPAT_LIMITS.maxStorageTexturesPerShaderStage,
    maxUniformBuffersPerShaderStage: COMPAT_LIMITS.maxUniformBuffersPerShaderStage,
    maxVertexBuffers: COMPAT_LIMITS.maxVertexBuffers,
    maxVertexAttributes: COMPAT_LIMITS.maxVertexAttributes,
    maxVertexBufferArrayStride: COMPAT_LIMITS.maxVertexBufferArrayStride,
    maxInterStageShaderVariables: COMPAT_LIMITS.maxInterStageShaderVariables,
    maxColorAttachments: COMPAT_LIMITS.maxColorAttachments,
    maxColorAttachmentBytesPerSample: COMPAT_LIMITS.maxColorAttachmentBytesPerSample,
    maxComputeWorkgroupStorageSize: COMPAT_LIMITS.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: COMPAT_LIMITS.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: COMPAT_LIMITS.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: COMPAT_LIMITS.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: COMPAT_LIMITS.maxComputeWorkgroupSizeZ,
    maxComputeWorkgroupsPerDimension: COMPAT_LIMITS.maxComputeWorkgroupsPerDimension,
  };
}

/** `{ maxUniformBufferBindingSize=16384, ... }` for a device-creation error. */
export function formatLimitRequest(limits: Partial<Record<string, number>>): string {
  return ASE_LIMIT_NAMES.map((k) => `${k}=${limits[k] as number}`).join(', ');
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/**
 * The optional device features apse knows how to read.
 *
 * **None of them are ever required at device creation**, deliberately: requiring
 * one means `requestDevice()` rejects on exactly the devices that lack it, which
 * is the population apse exists to serve. They are *detected* and reported
 * through {@link DeviceCapabilities}, and a caller that needs one asks for it
 * with {@link requireFeature} — which returns a failure rather than throwing,
 * because a phone without `float32-filterable` is a fact, not a bug.
 */
export const ASE_FEATURES = [
  'timestamp-query',
  'float32-filterable',
  'depth32float-stencil8',
  'indirect-first-instance',
  'texture-compression-bc',
  'texture-compression-astc',
  'texture-compression-etc2',
  'subgroups',
] as const;

export type AseFeatureName = (typeof ASE_FEATURES)[number];

/** The `Result` failure code {@link requireFeature} returns. */
export type GpuFeatureFailure = 'GPU_FEATURE_UNSUPPORTED';

/**
 * Checks for an optional feature, as a value rather than an exception.
 *
 * The project rule is that a wrong argument throws and a missing capability
 * returns, and this is the canonical missing capability in the whole library: a
 * phone that does not implement `float32-filterable` is the world being what it
 * is. So this hands back `ok(undefined)` or an `Err` carrying the catalog's
 * capability detail, and the caller branches.
 *
 *     const r = requireFeature(gpu, 'timestamp-query');
 *     if (isErr(r)) { stats.gpu = null; } else { timer = GpuTimer.create(gpu); }
 *
 * A name outside {@link ASE_FEATURES} is a *wrong argument* rather than a
 * missing capability, and throws `OPTION_UNKNOWN` — the same split apse applies
 * everywhere else.
 */
export function requireFeature(
  device: GPUDevice,
  name: AseFeatureName,
): Result<void, GpuFeatureFailure> {
  if (!ASE_FEATURES.includes(name)) {
    fail('OPTION_UNKNOWN',
      `requireFeature() was given the feature "${name}", which is not one of: ${ASE_FEATURES.join(', ')}.`, {
      why: 'apse only reports on features it has a fallback for. An unrecognised name would resolve to "not supported" and be indistinguishable from a device that genuinely lacks it, so a typo becomes a silent feature downgrade.',
      fix: `Use one of: ${ASE_FEATURES.join(', ')}. A feature apse does not know about has no correct behaviour, which is the same rule that keeps requiredLimits curated.`,
    });
  }
  if (hasFeature(device, name)) return ok(undefined);
  return err('GPU_FEATURE_UNSUPPORTED',
    `This device does not implement the optional feature "${name}".`,
    `Carry on without it, or pick a path that needs it only when ${JSON.stringify(name)} is present — AseDevice.capabilities has the answer for every optional feature apse tracks.`,
    { feature: name, required: name, available: availableFeatures(device).join(', ') || 'none' });
}

/** True when the device was created with `name`. */
export function hasFeature(device: GPUDevice, name: AseFeatureName): boolean {
  const features = device.features as { has(name: GPUFeatureName): boolean };
  return typeof features?.has === 'function' && features.has(name);
}

/** Every optional feature this device has, sorted, for diagnostics. */
export function availableFeatures(device: GPUDevice): string[] {
  return ASE_FEATURES.filter((name) => hasFeature(device, name));
}

/** Texture-compression families, as the feature names they arrive under. */
export type TextureCompressionFamily = 'bc' | 'astc' | 'etc2';

const COMPRESSION_FEATURE: Readonly<Record<TextureCompressionFamily, AseFeatureName>> = Object.freeze({
  bc: 'texture-compression-bc',
  astc: 'texture-compression-astc',
  etc2: 'texture-compression-etc2',
});

/**
 * Every capability the renderer branches on, resolved once at device creation.
 *
 * Structured rather than a bag of booleans on the device, because the renderer's
 * decisions are all of the form "which of these two paths may I take", and a
 * `true` with no explanation is a trap. Each field is a fact about *this*
 * device, read from the device rather than assumed from the feature level —
 * except where a value is genuinely per-level, which is said in the field's own
 * comment.
 */
export interface DeviceCapabilities {
  /** Which set of limits and validation rules the device was created under. */
  readonly featureLevel: FeatureLevel;
  /**
   * `timestamp-query`. `false` on roughly half of all devices, so anything
   * that reports a GPU time must handle its absence with a *null*, never a 0.
   */
  readonly timestampQuery: boolean;
  /**
   * Storage buffers a vertex shader may read.
   *
   * **Zero in compatibility mode**, and that is the single most portable-bug in
   * WebGPU: a vertex shader that reads a storage buffer compiles on a laptop and
   * fails to compile on a phone. apse's instancing path binds per-instance
   * transforms as a *vertex buffer* with `stepMode: 'instance'` precisely so
   * this number never has to be greater than zero. See
   * {@link COMPAT_VERTEX_STAGE_LIMITS}.
   */
  readonly storageBuffersInVertexStage: number;
  /** Storage textures a vertex shader may read. Zero in compatibility mode. */
  readonly storageTexturesInVertexStage: number;
  /**
   * Which compressed texture formats can be sampled here.
   *
   * `bc` is desktop, `astc` is modern mobile, `etc2` is the GLES 3.1 baseline
   * that compatibility mode targets. An empty list means every texture in the
   * app has to be `rgba8unorm`, which is a real portability constraint and not
   * something to discover on a device you do not have.
   */
  readonly textureCompression: readonly TextureCompressionFamily[];
  /** `rgba32float` usable as a sampled texture with a linear sampler. */
  readonly float32Filterable: boolean;
  /**
   * Largest single storage-buffer binding. 128 MiB on both profiles, so this is
   * here because the renderer *asks* what it needs rather than assuming.
   */
  readonly maxStorageBufferBindingSize: number;
  /** Vertex buffer slots available. 8 on both profiles. */
  readonly maxVertexBuffers: number;
  /**
   * Whether per-instance vertex buffers are possible: slot 0 for the mesh and
   * slot 1 for the instances, so at least 2.
   */
  readonly instancing: boolean;
  /** The 2D texture extent ceiling. 4096 in compatibility mode, 8192 in core. */
  readonly maxTextureDimension2D: number;
}

/**
 * Resolves {@link DeviceCapabilities} from a live device.
 *
 * The two per-stage vertex limits are read defensively and fall back by feature
 * level, because the IDL marks them as not yet universally implemented: an
 * implementation that omits the property entirely would otherwise read as
 * `undefined`, and `undefined >= 1` is false, so the fallback is the safe
 * direction — a capability claimed absent is a path not taken, while one claimed
 * present is a shader that fails to compile.
 */
export function readCapabilities(device: GPUDevice, featureLevel: FeatureLevel): DeviceCapabilities {
  const limits = device.limits as unknown as Partial<Record<string, number>>;
  const vertexStorage = numberOr(
    limits.maxStorageBuffersInVertexStage,
    featureLevel === 'compatibility'
      ? COMPAT_VERTEX_STAGE_LIMITS.maxStorageBuffersInVertexStage
      : (limits.maxStorageBuffersPerShaderStage ?? 0),
  );
  const vertexTextures = numberOr(
    limits.maxStorageTexturesInVertexStage,
    featureLevel === 'compatibility'
      ? COMPAT_VERTEX_STAGE_LIMITS.maxStorageTexturesInVertexStage
      : (limits.maxStorageTexturesPerShaderStage ?? 0),
  );
  const maxVertexBuffers = numberOr(limits.maxVertexBuffers, 0);
  return Object.freeze({
    featureLevel,
    timestampQuery: hasFeature(device, 'timestamp-query'),
    storageBuffersInVertexStage: vertexStorage,
    storageTexturesInVertexStage: vertexTextures,
    textureCompression: (Object.keys(COMPRESSION_FEATURE) as TextureCompressionFamily[])
      .filter((family) => hasFeature(device, COMPRESSION_FEATURE[family])),
    float32Filterable: hasFeature(device, 'float32-filterable'),
    maxStorageBufferBindingSize: numberOr(limits.maxStorageBufferBindingSize, 0),
    maxVertexBuffers,
    // Two slots: mesh in 0, instances in 1. See the slot-1 comment in types.ts.
    instancing: maxVertexBuffers >= 2,
    maxTextureDimension2D: numberOr(limits.maxTextureDimension2D, COMPAT_LIMITS.maxTextureDimension2D),
  });
}

function numberOr(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

// ---------------------------------------------------------------------------
// Options and the device handle
// ---------------------------------------------------------------------------

export interface DeviceOptions {
  /** Defaults to 'high-performance'. */
  powerPreference?: 'high-performance' | 'low-power';
  /**
   * Try 'core' before 'compatibility'. Default false.
   *
   * Reverses the adapter probe order *and*, when the resulting adapter
   * advertises `core-features-and-limits`, requests that feature so the device
   * gets core limits. On an adapter that cannot do core this is silently a
   * no-op — check `device.featureLevel` rather than assuming.
   */
  preferCore?: boolean;
  /** Allow a software adapter. Default false. */
  fallbackAdapter?: boolean;
  /**
   * Max objects in the shared dynamic-offset uniform buffer. Default 4096.
   *
   * apse binds per-object transforms from one buffer using dynamic offsets, so
   * the buffer is `maxObjects * OBJECT_UNIFORM_STRIDE_BYTES` and is bounded by
   * `maxBufferSize`. Each individual binding is bounded by
   * `maxUniformBufferBindingSize` (16 KiB in compatibility mode), which is why
   * the bind group layout must declare a `size` of one object slot rather than
   * of the whole buffer.
   */
  maxObjects?: number;
  /** MSAA sample count: 1 or 4. Default 1. */
  sampleCount?: 1 | 4;
  /** Pixel ratio cap. Default 2. */
  maxPixelRatio?: number;
  /**
   * Target texture format. Defaults to `getPreferredCanvasFormat()`.
   *
   * Always ask the platform. It is `bgra8unorm` on desktop and `rgba8unorm` on
   * Android, and guessing wrong costs a full-frame copy on the way to the
   * compositor on both.
   */
  format?: GPUTextureFormat;
  /** Alpha mode for the canvas context. Default 'opaque'. */
  alphaMode?: 'opaque' | 'premultiplied';
  /** Receive GPU validation errors. Development aid. */
  onValidationError?: (message: string) => void;
  /**
   * Receive every uncaptured WebGPU error, already classified into an
   * {@link AseError} from the catalog.
   *
   * The difference from `onValidationError`: this one carries a `code`, a `why`,
   * a `fix`, and the raw driver text as structured detail, so a bug report does
   * not need a human to classify the message. An out-of-memory condition becomes
   * `BUDGET_EXCEEDED` and a driver-internal error becomes `INTERNAL_INVARIANT`,
   * because those are what they are — a validation error naming two format enums
   * is not the only thing that can go wrong on a device, and treating an OOM as
   * a "validation error" sends the reader looking in the wrong place.
   */
  onUncapturedError?: (error: AseError) => void;
  /**
   * Called when the device is lost.
   *
   * This fires for `reason === 'destroyed'` too, because that is what
   * {@link AseDevice.destroy} does. Ignore that reason unless you want to
   * report the app's own teardown as a GPU failure.
   */
  onDeviceLost?: (reason: GPUDeviceLostReason, message: string) => void;
}

export interface AseDevice {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly canvas: HTMLCanvasElement;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;
  readonly featureLevel: FeatureLevel;
  /** The limits apse depends on, copied field by field. */
  readonly limits: AseLimits;
  readonly adapterInfo: { vendor: string; architecture: string; device: string; description: string };
  /**
   * Every capability the renderer branches on, resolved at creation.
   *
   * Branch on this rather than on the adapter: a capability is a property of the
   * *device*, and the device is what a pipeline is compiled against.
   */
  readonly capabilities: DeviceCapabilities;
  /** Kept for callers written against the first device API. See `capabilities`. */
  readonly hasTimestampQuery: boolean;
  readonly hasSubgroups: boolean;
  /**
   * The most recent uncaptured error or device loss, as a catalog-typed error.
   *
   * Non-null means the driver told this device something was wrong. It is kept
   * rather than only reported, because the usual failure is nobody listening: a
   * console warning scrolls away and the frame is still black an hour later.
   */
  readonly lastError: AseError | null;
  /**
   * Checks an optional feature without throwing on its absence.
   *
   * `requireFeature('float32-filterable')` returns `ok(undefined)` or an `Err`
   * naming the feature. See the free function for the reasoning.
   */
  requireFeature(name: AseFeatureName): Result<void, GpuFeatureFailure>;
  /** Live objects in the shared object-uniform buffer. */
  readonly maxObjects: number;
  /** MSAA level this device was configured with. Pipelines must match it. */
  readonly sampleCount: 1 | 4;
  /** Pixel ratio cap, so the sizer and the device cannot disagree. */
  readonly maxPixelRatio: number;
  /**
   * Dispose + re-acquire. Rejects on failure.
   *
   * Only legal after a loss; on a live device it throws `INTERNAL_INVARIANT`,
   * because a device that has not been lost does not need replacing and
   * replacing it anyway hides a bug in the caller. The returned value is a
   * *new* device — the old one stays dead on purpose so a stale reference fails
   * loudly instead of drawing nothing — so reassign your reference. Every GPU
   * resource built on the old device is gone and must be rebuilt: buffers and
   * textures do not survive a device loss.
   */
  recover(): Promise<AseDevice>;
  destroy(): void;
  /**
   * Throws `DEVICE_LOST` if this device is no longer usable.
   *
   * Added to the handle deliberately: `device.lost` is a promise that does not
   * settle while the device lives, so it cannot be used as a liveness flag.
   * Anything that touches the GPU should call this first, and every accessor in
   * `target.ts` already does.
   */
  assertLive(): void;
}

/**
 * Bytes reserved per object in the shared object-uniform buffer.
 *
 * 4x4 model matrix (64 B) + 3x3 normal matrix packed as three vec4 (48 B) +
 * tint, material index and padding (16 B) = 128 B of data, **rounded up to
 * 256** because of `minUniformBufferOffsetAlignment`.
 *
 * That rounding is not tidiness, it is a hard constraint. Object N lives at byte
 * offset `N * OBJECT_UNIFORM_STRIDE_BYTES`, and every dynamic offset passed to
 * `setBindGroup` must be a multiple of `minUniformBufferOffsetAlignment` — 256
 * on every profile. A 128-byte stride puts object 1 at offset 128, which is
 * invalid, and the failure surfaces as a validation error at the first draw of
 * the second object rather than at buffer creation. Halve the stride and you
 * trade a compile-time constant for a per-draw error.
 *
 * Exported so the renderer can assert that the layout it generates matches the
 * budget `maxObjects` was validated against, instead of quietly overflowing it.
 */
export const OBJECT_UNIFORM_STRIDE_BYTES = 256;

/** Formats apse accepts as a render-target colour format. */
const RENDERABLE_COLOR_FORMATS: readonly GPUTextureFormat[] = [
  'bgra8unorm',
  'rgba8unorm',
  'bgra8unorm-srgb',
  'rgba8unorm-srgb',
  'rgba16float',
];

/**
 * The three `GPUError` classes, and what each one actually is.
 *
 * A tuple rather than an object per row because there are four fields and
 * nothing here is read by name outside this file.
 */
const UNCAPTURED_ERRORS: Readonly<Record<string, {
  code: AseErrorCode; label: string; why: string; fix: string;
}>> = Object.freeze({
  GPUValidationError: {
    code: 'GPU_VALIDATION_FAILED',
    label: 'Uncaptured WebGPU validation error',
    why: 'Raised on the device timeline rather than through an error scope, so nothing on the call stack raised it. The object it names is invalid, and so is everything derived from it.',
    fix: 'Read the raw driver text above: it names the object and the field. Nearly all of these are a descriptor apse built, and the label apse gave it is in the message.',
  },
  GPUOutOfMemoryError: {
    code: 'BUDGET_EXCEEDED',
    label: 'The GPU ran out of memory',
    why: 'The device could not satisfy an allocation. On the hardware apse targets that is nearly always a framebuffer: a 4x rgba16float target at 4K is 132 MB of colour plus a resolve.',
    fix: 'Lower maxPixelRatio, drop sampleCount to 1, or render at a lower internal resolution. maxBufferSize and maxTextureDimension2D in AseDevice.limits decide it.',
  },
  GPUInternalError: {
    code: 'INTERNAL_INVARIANT',
    label: 'The GPU driver raised an internal error',
    why: 'The driver lost track of its own state. No descriptor apse built is wrong, and no amount of fixing apse code addresses it.',
    fix: 'Report it with the adapter description from describeGpu(). A driver update usually fixes it; device.recover() clears it meanwhile.',
  },
});

// ---------------------------------------------------------------------------
// Development error scopes
// ---------------------------------------------------------------------------

function detectDevelopment(): boolean {
  // `process` does not exist in a browser bundle unless the bundler injected
  // it, so the guard is load-bearing rather than defensive.
  const p = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  const env = p?.env?.['NODE_ENV'];
  return env === 'development' || env === 'test';
}

let developmentMode: boolean = detectDevelopment();

/** True when apse wraps its own risky calls in error scopes. */
export function isDevelopmentMode(): boolean {
  return developmentMode;
}

/**
 * Force the error-scope behaviour on or off, overriding the `NODE_ENV` probe.
 *
 * Error scopes are a promise round trip per wrapped call. They are worth it in
 * a debug build and are not worth it in a shipped one, so a "GPU validation"
 * devtools toggle can call this with `true`.
 */
export function setDevelopmentMode(value: boolean): void {
  developmentMode = value;
}

/**
 * Runs `fn` inside a WebGPU error scope and turns a raised error into a typed
 * failure that quotes the raw driver text.
 *
 * WebGPU reports most mistakes on the device timeline through `GPUError` rather
 * than by throwing, and the *contagious invalidity* rule means the object you
 * got back is silently dead. This helper is the only way to convert that into a
 * normal exception, so use it around anything whose descriptor you did not
 * write by hand: `createRenderPipeline`, `createBindGroup`, `createTexture`
 * with a computed size, `queue.writeBuffer` with a computed offset.
 *
 *     const pipeline = await withErrorScope(
 *       device, 'validation',
 *       () => device.createRenderPipelineAsync(desc),
 *       'material "checker" pipeline',
 *     );
 *
 * The scope is always popped, including when `fn` throws, so a failing body
 * cannot leave the device's scope stack unbalanced and mis-attribute the next
 * unrelated error to this call.
 */
export async function withErrorScope<T>(
  device: GPUDevice,
  scope: GPUErrorFilter,
  fn: () => T | Promise<T>,
  operation: string = 'A WebGPU operation',
): Promise<T> {
  device.pushErrorScope(scope);
  let result: T;
  try {
    result = await fn();
  } catch (thrown) {
    // Pop on the way out. A rejection from popErrorScope here means the device
    // is gone, which the caller is about to discover anyway — it must not
    // replace the real error.
    await device.popErrorScope().catch(() => null);
    throw thrown;
  }
  // popErrorScope rejects only on a lost device; treat that as "no error"
  // rather than inventing a validation failure for work that already failed.
  const error = await device.popErrorScope().catch(() => null);
  if (error !== null && error !== undefined) {
    const raw = error.message;
    fail('GPU_VALIDATION_FAILED',
      `${operation} raised a WebGPU "${scope}" error. Driver said: ${raw}`, {
      why: 'WebGPU validates descriptors on the device timeline and reports failures through an error scope. The object returned by the call is invalid, and so is everything derived from it — silently, with no exception anywhere in your code.',
      fix: `Read the raw driver text above: it names the exact descriptor field that was wrong. \`${raw}\``,
      detail: { kind: 'gpu-validation', scope, raw },
    });
  }
  return result;
}

// ---------------------------------------------------------------------------
// The implementation
// ---------------------------------------------------------------------------

/**
 * Canvases this module has configured, so a second renderer on the same element
 * is a typed error rather than two devices fighting over one context. A
 * WeakMap, so holding a canvas here never keeps it alive.
 */
const CONFIGURED = new WeakMap<HTMLCanvasElement, AseDeviceImpl>();

class AseDeviceImpl implements AseDevice {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly canvas: HTMLCanvasElement;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;
  readonly featureLevel: FeatureLevel;
  readonly limits: AseLimits;
  readonly adapterInfo: { vendor: string; architecture: string; device: string; description: string };
  readonly capabilities: DeviceCapabilities;
  readonly hasTimestampQuery: boolean;
  readonly hasSubgroups: boolean;
  lastError: AseError | null = null;
  readonly maxObjects: number;
  readonly sampleCount: 1 | 4;
  readonly maxPixelRatio: number;

  readonly #opts: DeviceOptions;
  #lostMessage = 'the device was destroyed or lost before this call';
  #lost = false;

  constructor(init: {
    adapter: GPUAdapter;
    device: GPUDevice;
    canvas: HTMLCanvasElement;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
    featureLevel: FeatureLevel;
    alphaMode: GPUCanvasAlphaMode;
    maxObjects: number;
    sampleCount: 1 | 4;
    maxPixelRatio: number;
    opts: DeviceOptions;
  }) {
    this.adapter = init.adapter;
    this.device = init.device;
    this.canvas = init.canvas;
    this.context = init.context;
    this.format = init.format;
    this.featureLevel = init.featureLevel;
    this.maxObjects = init.maxObjects;
    this.sampleCount = init.sampleCount;
    this.maxPixelRatio = init.maxPixelRatio;
    this.#opts = init.opts;
    this.limits = copyLimits(init.device.limits);

    // `device.adapterInfo` is the newer spelling; `adapter.info` is what
    // shipped first. Read defensively: adapter identity is exactly the kind of
    // thing that lands late in a compatibility-mode rollout.
    const info = (init.device as { adapterInfo?: GPUAdapterInfo }).adapterInfo ?? init.adapter.info;
    this.adapterInfo = {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    };

    // Features are *detected*, never required. `timestamp-query` is around
    // 44% of devices and its values are quantised to 100 us, so it is useless
    // as a gate; `subgroups` is Chromium-only. Requesting either would make
    // device creation fail outright on exactly the devices that lack them.
    this.capabilities = readCapabilities(init.device, init.featureLevel);
    this.hasTimestampQuery = this.capabilities.timestampQuery;
    this.hasSubgroups = hasFeature(init.device, 'subgroups');
  }

  requireFeature(name: AseFeatureName): Result<void, GpuFeatureFailure> {
    this.assertLive();
    return requireFeature(this.device, name);
  }

  /**
   * Wires the two things that must be wired for the device to be usable.
   *
   * Both handlers swallow exceptions thrown by user callbacks: an exception
   * raised inside a `lost.then` becomes an unhandled rejection whose stack
   * points nowhere near the caller's bug, which is worse than a missed
   * notification.
   */
  attach(): void {
    void this.device.lost
      .then((info) => {
        this.#lost = true;
        this.#lostMessage = info.message === ''
          ? `the device was lost (reason: ${info.reason})`
          : `${info.message} (reason: ${info.reason})`;
        // A destroyed device is an expected event, not a fault, but the canvas
        // is free either way once the device is gone.
        CONFIGURED.delete(this.canvas);
        // Recorded before the callback so a handler that inspects `lastError`
        // sees the loss, and so a silent device (no handler at all) still leaves
        // a diagnosable trace on the handle.
        if (info.reason !== 'destroyed') this.lastError = this.#lostError();
        const cb = this.#opts.onDeviceLost;
        if (cb === undefined) return;
        try {
          cb(info.reason, this.#lostMessage);
        } catch {
          /* a broken callback must not escalate into an unhandled rejection */
        }
      })
      .catch(() => {
        this.#lost = true;
        this.#lostMessage = 'the device was lost and the reason was not reported';
        this.lastError = this.#lostError();
      });

    const onUncaptured = (event: GPUUncapturedErrorEvent): void => {
      const error = this.#classifyUncaptured(event.error);
      this.lastError = error;
      const cb = this.#opts.onUncapturedError;
      if (cb !== undefined) {
        try {
          cb(error);
        } catch {
          /* ditto */
        }
      }
      const legacy = this.#opts.onValidationError;
      if (legacy === undefined) {
        if (isDevelopmentMode()) console.warn(`apse: uncaptured WebGPU error — ${error.message}`);
        return;
      }
      try {
        legacy(error.message);
      } catch {
        /* ditto */
      }
    };
    this.device.addEventListener('uncapturederror', onUncaptured);
  }

  #lostError(): AseError {
    return new AseError('DEVICE_LOST', `The GPU device was lost: ${this.#lostMessage}.`, {
      why: 'A device is a lease on a driver context, not an object. After a loss every buffer, texture, pipeline and bind group made from it is invalid, and touching one does not throw — it silently does nothing.',
      fix: 'Handle onDeviceLost, then call device.recover() and rebuild every GPU resource. Reassign the returned handle. Ignore reason "destroyed": that is your own teardown.',
      detail: { kind: 'lifecycle', resource: 'GPUDevice', state: 'destroyed' },
    });
  }

  /**
   * Turns a `GPUError` into a catalog error, by the class the driver used.
   *
   * A table rather than three branches, for two reasons. The rule is that the
   * three classes are three different problems: a validation error is a
   * descriptor that is wrong, an out-of-memory error is a budget nobody
   * declared, and an internal error is the driver saying it lost track of its
   * own state. Filing all three as "a validation error" sends the reader to the
   * wrong place, and an OOM investigated as a descriptor is an afternoon
   * wasted. And one record compresses better than three near-copies, which
   * matters when the strings are the bulk of the code.
   *
   * An unrecognised class is treated as a validation error: it is the only one
   * of the three whose fix is "read what the driver said".
   */
  #classifyUncaptured(error: GPUError): AseError {
    const raw = error.message;
    const row = UNCAPTURED_ERRORS[error.constructor.name] ?? UNCAPTURED_ERRORS.GPUValidationError;
    return new AseError(row.code, `${row.label}: ${raw}`, {
      why: row.why,
      fix: row.fix,
      detail: { kind: 'gpu-validation', scope: 'uncaptured', raw },
    });
  }

  assertLive(): void {
    if (!this.#lost) return;
    fail('DEVICE_LOST',
      `This apse device can no longer be used: ${this.#lostMessage}.`, {
      why: 'A GPUDevice is a lease on a driver context, not an object. When it is lost — driver reset, tab backgrounded, machine sleep, or an explicit destroy() — every buffer, texture, pipeline and bind group created from it is invalid, and the browser raises no exception when you touch one.',
      fix: 'Handle `onDeviceLost`, then call `device.recover()` and rebuild every GPU resource. Reassign the returned handle: the old one stays dead on purpose, so a stale reference fails loudly instead of drawing nothing. Check `reason` and ignore "destroyed", which is your own teardown.',
      detail: { kind: 'lifecycle', resource: 'GPUDevice', state: 'destroyed' },
    });
  }

  async recover(): Promise<AseDevice> {
    if (!this.#lost) {
      fail('INTERNAL_INVARIANT',
        'recover() was called on a device that has not been lost.', {
        why: 'A device that is still alive does not need replacing. Reaching for recover() on a live device means the caller is trying to paper over a bug — usually a validation error that was swallowed, or a device-lost callback firing for a reason other than a loss.',
        fix: 'Only call recover() from your onDeviceLost handler, and only for reason !== "destroyed". If you are hitting this without a loss, set onValidationError and look for the real validation failure first.',
      });
    }
    // Release the canvas registration before re-acquiring: the same element and
    // the same context object are about to be reused by a new device, and the
    // registry exists to stop two *live* devices sharing one context.
    if (CONFIGURED.get(this.canvas) === this) CONFIGURED.delete(this.canvas);
    return createDevice(this.canvas, this.#opts);
  }

  destroy(): void {
    if (this.#lost) return;
    this.#lost = true;
    this.#lostMessage = 'destroy() was called on it';
    if (CONFIGURED.get(this.canvas) === this) CONFIGURED.delete(this.canvas);
    try {
      this.context.unconfigure();
    } catch {
      /* unconfigure on a lost device is a no-op at best */
    }
    // Fires `lost` with reason "destroyed", which reaches onDeviceLost. That
    // is deliberate: one teardown path, one notification, and the reason tells
    // the listener this was on purpose.
    this.device.destroy();
  }
}

// ---------------------------------------------------------------------------
// Acquisition
// ---------------------------------------------------------------------------

function assertOneOf<T extends string | number>(value: T, known: readonly T[], option: string): void {
  if (known.includes(value)) return;
  fail('OPTION_UNKNOWN',
    `Option ${option} was given the value ${JSON.stringify(value)}, which is not one of: ${known.join(', ')}.`, {
    why: 'apse rejects options it does not recognise instead of ignoring them, so a typo or a value from a different version surfaces at the call site rather than as mysterious behaviour three layers up.',
    fix: `Use one of: ${known.join(', ')}. If you need a value that is not in this list, it needs a fallback path in apse, not a silent coercion.`,
  });
}

function gpuOrThrow(): GPU {
  const nav = (globalThis as { navigator?: Navigator }).navigator;
  if (nav === undefined || nav.gpu === undefined) {
    const secure = (globalThis as { isSecureContext?: boolean }).isSecureContext === true;
    const protocol = (globalThis as { location?: { protocol?: string } }).location?.protocol;
    const insecure = secure
      ? ''
      : ` and this is not a secure context (isSecureContext: false, protocol: ${protocol ?? 'unknown'})`;
    fail('WEBGPU_UNAVAILABLE',
      `\`navigator.gpu\` is undefined${insecure}, so this context exposes no WebGPU implementation.`, {
      why: 'WebGPU is a secure-context-only API. A page served over plain http from a non-localhost origin has no `navigator.gpu` at all, which is indistinguishable from an old browser. Workers have `navigator.gpu` but no `document`, so a canvas target will not work there.',
      fix: secure
        ? 'Confirm the browser has WebGPU: Chrome/Edge 113+, Safari 26+, or Firefox 141+. Firefox on Linux is Nightly-only and on Intel Macs is unsupported, which is why apse targets compatibility mode.'
        : 'Serve over https://, or over http://localhost. This is almost always the cause when identical code works on a laptop and not in a container or on a LAN IP.',
    });
  }
  return nav.gpu;
}

/**
 * Asks for an adapter, compatibility first.
 *
 * The spec says a single `requestAdapter({ featureLevel: 'compatibility' })`
 * is the right call: an implementation that cannot enforce the stricter
 * compatibility rules ignores the request and hands back a core adapter, so in
 * the common case there is nothing to fall back *to*. The second attempt exists
 * for the remaining case — an implementation that treats an unrecognised
 * `featureLevel` as a failure rather than ignoring it — and costs nothing when
 * it is not needed.
 */
async function acquireAdapter(
  gpu: GPU,
  opts: DeviceOptions,
): Promise<{ adapter: GPUAdapter; requested: FeatureLevel }> {
  const order: FeatureLevel[] = opts.preferCore === true
    ? ['core', 'compatibility']
    : ['compatibility', 'core'];
  const powerPreference = opts.powerPreference ?? 'high-performance';

  const tried: string[] = [];
  for (const level of order) {
    tried.push(level);
    const adapter = await gpu.requestAdapter({ featureLevel: level, powerPreference });
    if (adapter !== null) return { adapter, requested: level };

    if (opts.fallbackAdapter === true) {
      tried.push(`${level} + software`);
      const software = await gpu.requestAdapter({
        featureLevel: level,
        forceFallbackAdapter: true,
        powerPreference,
      });
      if (software !== null) return { adapter: software, requested: level };
    }
  }

  fail('ADAPTER_UNAVAILABLE',
    `\`requestAdapter()\` returned null for ${tried.join(' and ')} with powerPreference "${powerPreference}".`, {
    why: 'WebGPU is present and reachable, but no adapter could be created. The usual causes are a blocklisted driver, a headless run with no GPU, a page in a background tab, or a machine whose only adapter the content process is not allowed to use.',
    fix: 'Pass `fallbackAdapter: true` to run on a software adapter — slow, but it renders, and it proves the rest of your code is correct. In headless Chrome add `--enable-unsafe-swiftshader`. If it is one specific machine, `chrome://gpu` names the reason that adapter was rejected.',
  });
}

/**
 * Proves a freshly-created device is actually valid.
 *
 * **There is no way to put an error scope around `requestDevice` itself**, and
 * that is a real limitation of the API rather than an oversight. `pushErrorScope`
 * is a method on `GPUDevice`, and the device is what the call is trying to
 * produce. Browsers do report a bad `GPUDeviceDescriptor` by rejecting the
 * promise, so the rejection path covers the spec-mandated cases; what is left is
 * the "invalid object" half of the rule, and the only way to see that is to do
 * a real operation and read the device's error scope afterwards.
 *
 * So we allocate a one-pixel texture, destroy it, and read the scope. On a valid
 * device that is free and silent. On an invalid one it raises a validation error
 * naming the object that is invalid, which is the whole point: without this
 * probe the application gets a device-shaped object that silently accepts
 * every call and produces an empty screen.
 */
async function probeDeviceValidity(device: GPUDevice): Promise<void> {
  device.pushErrorScope('validation');
  let probe: GPUTexture | undefined;
  try {
    probe = device.createTexture({
      label: 'apse.device.probe',
      size: { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING,
    });
    probe.destroy();
  } catch {
    await device.popErrorScope().catch(() => null);
    throw new AseProbeFailure('the device refused to create even a 1x1 texture');
  }
  const error = await device.popErrorScope().catch(() => null);
  if (error !== null && error !== undefined) {
    throw new AseProbeFailure(error.message);
  }
}

/** Internal carrier so {@link probeDeviceValidity} can hand raw text upward. */
class AseProbeFailure extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AseProbeFailure';
  }
}

function requestDeviceOrFail(
  adapter: GPUAdapter,
  requiredFeatures: readonly GPUFeatureName[],
): Promise<GPUDevice> {
  const requiredLimits = compatRequiredLimits();
  return adapter.requestDevice({ label: 'apse.device', requiredFeatures, requiredLimits })
    .catch((thrown: unknown) => {
      fail('DEVICE_REQUEST_FAILED',
        `requestDevice() rejected: ${thrown instanceof Error ? `${thrown.name}: ${thrown.message}` : String(thrown)}`, {
        why: 'A requestDevice() rejection means the adapter could not satisfy the descriptor at all: an unknown feature name is a TypeError, and a limit that is unknown or better than the adapter supports is an OperationError. A *validation* failure, by contrast, does not reject — it resolves with an invalid device and reports through the probe that follows.',
        fix: `Requested features: [${requiredFeatures.join(', ') || 'none'}]. Requested limits: ${formatLimitRequest(requiredLimits)}. apse requests only the compatibility defaults, so a rejection here almost always means a device that is already consumed — a second requestDevice() on the same adapter, usually from a stray recover().`,
        cause: thrown,
      });
    });
}

/**
 * Creates a device on `canvas` and configures the canvas for it.
 *
 * **This is a factory, not a constructor, on purpose.** Adapter and device
 * acquisition are both asynchronous: `requestAdapter()` consults the GPU
 * process and `requestDevice()` can synchronise with the driver. A class
 * constructor cannot await, so the only ways to hide that are to hand back a
 * half-built object or to make the caller await a second call. Both produce the
 * same failure — a renderer that exists before its device does, throwing
 * somewhere unrelated on the first frame. An async factory makes that ordering
 * impossible to get wrong.
 *
 * Rejects with a typed `AseError`:
 *   - `WEBGPU_UNAVAILABLE`        — no `navigator.gpu` here
 *   - `ADAPTER_UNAVAILABLE`       — no adapter, not even a software one
 *   - `DEVICE_REQUEST_FAILED`     — descriptor rejected; the message quotes the driver
 *   - `CANVAS_CONTEXT_ALREADY_TAKEN` — another live apse device owns this canvas
 *   - `CANVAS_CONTEXT_INVALID`    — the canvas already has a different context type
 *   - `BUDGET_EXCEEDED`           — `maxObjects` cannot fit in a buffer
 */
export async function createDevice(
  canvas: HTMLCanvasElement,
  opts: DeviceOptions = {},
): Promise<AseDevice> {
  // --- options, validated before anything is acquired ----------------------
  // A bad option should cost nothing, not an adapter round trip.
  if (opts.powerPreference !== undefined) {
    assertOneOf(opts.powerPreference, ['high-performance', 'low-power'] as const, 'powerPreference');
  }
  if (opts.alphaMode !== undefined) {
    assertOneOf(opts.alphaMode, ['opaque', 'premultiplied'] as const, 'alphaMode');
  }
  if (opts.sampleCount !== undefined) {
    assertOneOf(opts.sampleCount, [1, 4] as const, 'sampleCount');
  }
  if (opts.format !== undefined) {
    assertOneOf(opts.format, RENDERABLE_COLOR_FORMATS, 'format');
  }
  const maxPixelRatio = opts.maxPixelRatio ?? 2;
  if (!Number.isFinite(maxPixelRatio) || maxPixelRatio <= 0) {
    fail('OPTION_UNKNOWN',
      `Option maxPixelRatio was given the value ${JSON.stringify(opts.maxPixelRatio)}, which is not a positive finite number.`, {
      why: 'The pixel ratio cap multiplies the CSS size into the backing-store size. A zero or non-finite cap produces a zero-sized canvas, which fails texture creation much later with an error that does not mention the option.',
      fix: 'Pass a number greater than 0: 2 is a good default, 1.5 for a large canvas on a 3x display, 1 to ignore DPR entirely.',
    });
  }
  const sampleCount = opts.sampleCount ?? 1;
  const maxObjects = opts.maxObjects ?? 4096;
  if (!Number.isInteger(maxObjects) || maxObjects <= 0) {
    fail('OPTION_UNKNOWN',
      `Option maxObjects was given the value ${JSON.stringify(opts.maxObjects)}, which is not a positive integer.`, {
      why: '`maxObjects` sizes the shared object-uniform buffer, and each object occupies a whole number of 256-byte-aligned slots. A fractional or non-positive count would produce a buffer size that is not a multiple of the dynamic-offset alignment.',
      fix: 'Pass a positive integer, e.g. 4096. The renderer raises this if a frame legitimately needs more.',
    });
  }

  const gpu = gpuOrThrow();

  // --- adapter ------------------------------------------------------------
  const { adapter, requested } = await acquireAdapter(gpu, opts);

  // An adapter obtained from a 'core' request cannot do compatibility whatever
  // its feature list says. Otherwise, 'core' is opt-in via preferCore, and only
  // when the adapter actually advertises the feature.
  const canGoCore = adapter.features.has(CORE_FEATURE);
  const featureLevel: FeatureLevel =
    requested === 'core' || (opts.preferCore === true && canGoCore) ? 'core' : 'compatibility';
  const requiredFeatures: GPUFeatureName[] = featureLevel === 'core' ? [CORE_FEATURE] : [];

  // --- device -------------------------------------------------------------
  const device = await requestDeviceOrFail(adapter, requiredFeatures);
  const limits = copyLimits(device.limits);

  // The object-uniform buffer is allocated by the renderer, not here, so this
  // is a budget check on the option rather than an allocation.
  const objectBufferBytes = maxObjects * OBJECT_UNIFORM_STRIDE_BYTES;
  if (objectBufferBytes > limits.maxBufferSize) {
    const ceiling = Math.floor(limits.maxBufferSize / OBJECT_UNIFORM_STRIDE_BYTES);
    fail('BUDGET_EXCEEDED',
      `maxObjects: ${maxObjects} needs a ${objectBufferBytes}-byte object-uniform buffer, over the ${limits.maxBufferSize}-byte maxBufferSize limit.`, {
      why: 'apse keeps one uniform buffer for every object\'s transform and binds a different 128-byte slot of it with a dynamic offset. The whole buffer must fit inside maxBufferSize.',
      fix: `Lower maxObjects — ${ceiling} is the ceiling on this device. Or batch the scene: cull harder, or spread the draw over more frames. Do not raise OBJECT_UNIFORM_STRIDE_BYTES to fit a bigger count; that grows the buffer too.`,
      detail: { kind: 'numeric', field: 'maxObjects', value: maxObjects, min: 1, max: ceiling },
    });
  }
  if (OBJECT_UNIFORM_STRIDE_BYTES > limits.maxUniformBufferBindingSize) {
    fail('BUDGET_EXCEEDED',
      `One object slot is ${OBJECT_UNIFORM_STRIDE_BYTES} bytes but maxUniformBufferBindingSize is only ${limits.maxUniformBufferBindingSize}.`, {
      why: 'A uniform binding cannot exceed maxUniformBufferBindingSize, which is 16 KiB in compatibility mode. The bind group layout must declare a `size` of one object slot, not of the whole buffer — if it declares the whole buffer, this is the error you will get.',
      fix: 'Declare the object bind group layout entry with `size: OBJECT_UNIFORM_STRIDE_BYTES` and bind the shared buffer with a dynamic offset per draw. See DeviceOptions.maxObjects.',
    });
  }

  // --- is the device real? ------------------------------------------------
  try {
    await probeDeviceValidity(device);
  } catch (thrown) {
    const raw = thrown instanceof AseProbeFailure ? thrown.message
      : thrown instanceof Error ? thrown.message : String(thrown);
    device.destroy();
    fail('DEVICE_REQUEST_FAILED',
      `The adapter returned a device that is not valid. Driver said: ${raw}`, {
      why: 'WebGPU objects can be invalid from creation, and invalidity is contagious: every buffer, texture and pipeline made from this device would be invalid too, and every call on them would succeed and do nothing. There is no exception anywhere in the chain — the only symptom is an empty screen. apse probes the device with a 1x1 texture allocation to find out at the boundary instead of three frames later.',
      fix: `Read the raw driver text: it names the device or descriptor that is invalid. This is a browser or driver fault, not an apse one — retry, or report it with the browser version. Requested limits: ${formatLimitRequest(compatRequiredLimits())}.`,
      detail: { kind: 'gpu-validation', scope: 'validation', raw },
      cause: thrown,
    });
  }

  // --- canvas -------------------------------------------------------------
  const prior = CONFIGURED.get(canvas);
  if (prior !== undefined) {
    prior.assertLive();
    fail('CANVAS_CONTEXT_ALREADY_TAKEN',
      'Another live apse device is already configured on this canvas.', {
      why: 'A canvas has one WebGPU context and one configuration at a time. Two devices configuring it means the second silently replaces the first, and the first goes on drawing to a device the compositor is no longer reading from.',
      fix: 'Dispose the previous device first, or give each renderer its own canvas element. If the previous device was lost, call recover() on it rather than creating a second one.',
    });
  }

  const context = canvas.getContext('webgpu');
  if (context === null) {
    // We cannot always tell *what* is holding the canvas. getContext() returns
    // null for a context type that has already been handed out, and the DOM
    // offers no way to ask which one. So this message says exactly that rather
    // than inventing a cause.
    fail('CANVAS_CONTEXT_INVALID',
      'canvas.getContext("webgpu") returned null, so this element will not give out a WebGPU context.', {
      why: 'A canvas element has exactly one context type for its whole lifetime. If anything already called getContext("2d") on it — or "webgl", or "bitmaprenderer" — this returns null, and it will keep returning null forever. There is no API to ask which type was taken, so apse cannot tell you here; a canvas that still draws in 2D almost certainly has a 2D context.',
      fix: 'Use a fresh <canvas> element. If another library owns this element, give it its own canvas and give apse another. Note that the context is cached and shared but the texture is not: see getCurrentTexture in src/render/target.ts.',
    });
  }

  const format = opts.format ?? gpu.getPreferredCanvasFormat();
  const alphaMode = opts.alphaMode ?? 'opaque';

  const config: GPUCanvasConfiguration = {
    device,
    format,
    alphaMode,
    // RENDER_ATTACHMENT is the only usage that means anything for presenting.
    // COPY_SRC is here so the canvas texture can be copied into a buffer later,
    // which is the whole of a "save frame as PNG" feature. It is not free: on
    // some drivers it forces a resolve into a separate surface, so a library
    // that never reads the canvas back should drop it. apse keeps it because
    // capturing a frame is a feature almost every app eventually wants, and
    // retrofitting a usage flag means re-plumbing the swapchain for everyone.
    usage: TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.COPY_SRC,
  };

  if (isDevelopmentMode()) {
    await withErrorScope(device, 'validation',
      () => context.configure(config),
      `canvas.configure({ format: "${format}", alphaMode: "${alphaMode}" })`);
  } else {
    context.configure(config);
  }

  const impl = new AseDeviceImpl({
    adapter,
    device,
    canvas,
    context,
    format,
    featureLevel,
    alphaMode,
    maxObjects,
    sampleCount,
    maxPixelRatio,
    opts,
  });
  impl.attach();
  CONFIGURED.set(canvas, impl);
  return impl;
}
