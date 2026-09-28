/**
 * Texture slot declarations.
 *
 * A texture slot is a *named declaration* in a material, not a texture
 * assignment. The declaration is what lets apse generate the WGSL binding, the
 * bind group layout entry, and the sampler pairing — so a shader body can
 * write `texture(albedo, uv)` without ever touching a `@group` or a `@binding`.
 */

import { BIND_GROUP } from '../core/slot.ts';

export type TextureKind =
  | '2d'
  | 'cube'
  | '2d-array'
  | '3d'
  | 'depth-2d'
  | 'external';

export const TEXTURE_KINDS: readonly TextureKind[] = [
  '2d', 'cube', '2d-array', '3d', 'depth-2d', 'external',
];

export type TextureSampleType = 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint';

export interface TextureSlotSpec {
  /** Defaults to `'2d'`. */
  readonly kind?: TextureKind;
  /** Shader-side sample type. Defaults to `'float'`. */
  readonly sampleType?: TextureSampleType;
  /** Sampling is filtered and mipmapped. Defaults to true. */
  readonly mipmapFilter?: boolean;
  /** Address mode. Defaults to `'repeat'` for 2d, `'clamp-to-edge'` otherwise. */
  readonly addressMode?: 'repeat' | 'clamp-to-edge' | 'mirror-repeat';
  /** Comparison sampler, for depth textures. Defaults to false. */
  readonly compare?: boolean;
  /**
   * WGSL name of the binding variable. Defaults to the slot name.
   * Set this when the slot name is not a valid WGSL identifier.
   */
  readonly name?: string;
  /** Human label for tooling. */
  readonly label?: string;
}

export interface ResolvedTextureSlot {
  readonly slotName: string;
  readonly varName: string;
  readonly kind: TextureKind;
  readonly sampleType: TextureSampleType;
  readonly mipmapFilter: boolean;
  readonly addressMode: 'repeat' | 'clamp-to-edge' | 'mirror-repeat';
  readonly compare: boolean;
  readonly label: string;
  /** `@binding(n)` within the texture bind group. Assigned by the material, in declaration order. */
  readonly bindingIndex: number;
}

/** `sampler` for a filtering slot, `sampler_comparison` for a depth slot. */
export function samplerTypeName(sampleType: TextureSampleType, compare: boolean): string {
  if (compare) return 'sampler_comparison';
  return sampleType === 'float' || sampleType === 'unfilterable-float' ? 'sampler' : 'sampler';
}

export function textureViewDimension(kind: TextureKind): GPUTextureViewDimension {
  switch (kind) {
    case '2d':
    case 'depth-2d':
      return '2d';
    case 'cube':
      return 'cube';
    case '2d-array':
      return '2d-array';
    case '3d':
      return '3d';
    case 'external':
      return '2d';
  }
}

export function wgslTextureType(sampleType: TextureSampleType): string {
  switch (sampleType) {
    case 'float': return 'texture_2d<f32>';
    case 'unfilterable-float': return 'texture_2d<f32>';
    case 'depth': return 'texture_depth_2d';
    case 'sint': return 'texture_2d<i32>';
    case 'uint': return 'texture_2d<u32>';
  }
}

/** WGSL declaration for a texture binding, respecting its view dimension. */
export function wgslTextureDecl(name: string, slot: ResolvedTextureSlot): string {
  const dim = textureViewDimension(slot.kind);
  // The group comes from BIND_GROUP, never a literal. A hardcoded 3 survived a
  // renumbering of the other groups and emitted textures into a group no bind
  // group layout was built for, while samplers went to the real one -- a
  // validation error at draw, from a file nobody opens when the groups change.
  if (slot.sampleType === 'depth') {
    return `@group(${BIND_GROUP.texture}) @binding(${slot.bindingIndex}) var ${name} : texture_depth_${dim === '2d' ? '2d' : '2d_array'};`;
  }
  const scalar = slot.sampleType === 'sint' ? 'i32' : slot.sampleType === 'uint' ? 'u32' : 'f32';
  return `@group(${BIND_GROUP.texture}) @binding(${slot.bindingIndex}) var ${name} : texture_${dim}<${scalar}>;`;
}
