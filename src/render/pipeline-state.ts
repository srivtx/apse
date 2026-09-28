/**
 * Fixed-function pipeline state.
 *
 * These are the knobs WebGPU bakes into a pipeline rather than exposing as
 * settable state. Two materials with different values here are two different
 * pipelines, which is why they are part of the pipeline cache key and why they
 * are declared once, up front, rather than changed per frame.
 */

/** The primitive assembled from vertices. */
export type PrimitiveTopology =
  | 'point-list'
  | 'line-list'
  | 'line-strip'
  | 'triangle-list'
  | 'triangle-strip';

export const PRIMITIVE_TOPOLOGIES: readonly PrimitiveTopology[] = [
  'point-list',
  'line-list',
  'line-strip',
  'triangle-list',
  'triangle-strip',
];

export type ComparisonFunction =
  | 'never' | 'less' | 'equal' | 'less-equal'
  | 'greater' | 'not-equal' | 'greater-equal' | 'always';

export const COMPARISON_FUNCTIONS: readonly ComparisonFunction[] = [
  'never', 'less', 'equal', 'less-equal', 'greater', 'not-equal', 'greater-equal', 'always',
];

export type FrontFace = 'ccw' | 'cw';

export type CullMode = 'none' | 'front' | 'back';

export type BlendFactor =
  | 'zero' | 'one'
  | 'src' | 'one-minus-src'
  | 'src-alpha' | 'one-minus-src-alpha'
  | 'dst' | 'one-minus-dst'
  | 'dst-alpha' | 'one-minus-dst-alpha'
  | 'src-alpha-saturated'
  | 'constant' | 'one-minus-constant'
  | 'src1' | 'one-minus-src1';

export const BLEND_FACTORS: readonly BlendFactor[] = [
  'zero', 'one', 'src', 'one-minus-src', 'src-alpha', 'one-minus-src-alpha',
  'dst', 'one-minus-dst', 'dst-alpha', 'one-minus-dst-alpha',
  'src-alpha-saturated', 'constant', 'one-minus-constant', 'src1', 'one-minus-src1',
];

/**
 * Depth test and write.
 *
 * `write: true, compare: 'less'` is the default for opaque geometry: test
 * against what is already there, and keep the nearer fragment.
 * `write: false` is the default for transparent geometry, because a transparent
 * surface must not hide the opaque surface behind it.
 */
export interface DepthSpec {
  readonly write: boolean;
  readonly compare: ComparisonFunction;
}

export const DEFAULT_DEPTH: DepthSpec = Object.freeze({ write: true, compare: 'less' });

export const READ_ONLY_DEPTH: DepthSpec = Object.freeze({ write: false, compare: 'less' });

/** Colour blending. `null` means the write channel replaces the target. */
export interface BlendSpec {
  readonly color: { readonly srcFactor: BlendFactor; readonly dstFactor: BlendFactor; readonly operation: 'add' | 'subtract' | 'reverse-subtract' };
  readonly alpha: { readonly srcFactor: BlendFactor; readonly dstFactor: BlendFactor; readonly operation: 'add' | 'subtract' | 'reverse-subtract' };
}

export const ALPHA_BLEND: BlendSpec = Object.freeze({
  color: { srcFactor: 'src-alpha' as const, dstFactor: 'one-minus-src-alpha' as const, operation: 'add' as const },
  alpha: { srcFactor: 'one' as const, dstFactor: 'one-minus-src-alpha' as const, operation: 'add' as const },
});

export const ADDITIVE_BLEND: BlendSpec = Object.freeze({
  color: { srcFactor: 'src-alpha' as const, dstFactor: 'one' as const, operation: 'add' as const },
  alpha: { srcFactor: 'zero' as const, dstFactor: 'one' as const, operation: 'add' as const },
});

/** Which attachment(s) a material writes, and in what format. */
export interface TargetSpec {
  readonly format: GPUTextureFormat;
  /** Defaults to {@link BlendSpec} for the target, or replace when null. */
  readonly blend?: BlendSpec | null;
  /** Set on writeMask to restrict which channels this material may write. */
  readonly writeMask?: GPUColorWriteFlags;
}

/** Fixed-function state, resolved and normalised. Materials produce this. */
export interface ResolvedPipelineState {
  readonly topology: PrimitiveTopology;
  readonly cullMode: CullMode;
  readonly frontFace: FrontFace;
  readonly depth: DepthSpec;
  readonly targets: readonly ResolvedTarget[];
  readonly sampleCount: number;
}

export interface ResolvedTarget {
  readonly format: GPUTextureFormat;
  readonly blend: BlendSpec | null;
  readonly writeMask: GPUColorWriteFlags;
}

/** Normalises a user-supplied depth spec, applying the phase default. */
export function resolveDepth(depth: Partial<DepthSpec> | undefined, phase: 'opaque' | 'transparent'): DepthSpec {
  const fallback = phase === 'transparent' ? READ_ONLY_DEPTH : DEFAULT_DEPTH;
  if (depth === undefined) return fallback;
  return { write: depth.write ?? fallback.write, compare: depth.compare ?? fallback.compare };
}

export function resolveBlend(blend: BlendSpec | null | undefined): BlendSpec | null {
  return blend ?? null;
}

export const COLOR_WRITE_ALL: GPUColorWriteFlags = GPUColorWrite.ALL as GPUColorWriteFlags;
