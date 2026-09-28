/**
 * Uniform slot types and the reserved block definitions.
 *
 * A "slot" is a named, typed field of a uniform block. Declaring slots in a
 * material is how a shader gets custom data without writing a single WGSL
 * binding: apse computes the struct, the offsets, and the writer.
 */

import type { UniformBlockSpec } from './uniform.ts';
import { buildUniformBlock } from './uniform.ts';

export type SlotType =
  | 'f32' | 'i32' | 'u32'
  | 'vec2f' | 'vec3f' | 'vec4f'
  | 'vec2u' | 'vec3u' | 'vec4u'
  | 'mat3x3f' | 'mat4x4f';

/** A user-declared uniform field. */
export interface SlotSpec {
  readonly type: SlotType;
  /**
   * Value written once at material creation. Component count must match
   * `type`. Omit to get a zeroed field of the right type.
   */
  readonly default?: number | ArrayLike<number>;
}

/** Shorthand: `slots: { roughness: 'f32' }` or `slots: { color: { type: 'vec3f', default: [1,0,0] } }`. */
export type SlotDefs = Readonly<Record<string, SlotType | SlotSpec>>;

/** Normalises the shorthand into a flat `name -> SlotType` map. */
export function resolveSlotTypes(slots: SlotDefs | undefined): Record<string, SlotType> {
  const out: Record<string, SlotType> = {};
  if (slots === undefined) return out;
  for (const [name, spec] of Object.entries(slots)) {
    out[name] = typeof spec === 'string' ? spec : spec.type;
  }
  return out;
}

/** Collects the declared defaults into a `field -> value` map for first-write. */
export function collectSlotDefaults(slots: SlotDefs | undefined): Map<string, number | ArrayLike<number>> {
  const out = new Map<string, number | ArrayLike<number>>();
  if (slots === undefined) return out;
  for (const [name, spec] of Object.entries(slots)) {
    if (typeof spec === 'object' && spec.default !== undefined) out.set(name, spec.default);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Reserved blocks
//
// These three groups are always present and always at the same binding, so a
// shader body can rely on them without declaring anything. Group indices are
// chosen by change frequency: frame changes once per frame, obj once per
// object, mat once per material, textures last.
// ---------------------------------------------------------------------------

/** @group(0) @binding(0) — camera and frame state. */
export const FRAME_FIELDS = {
  view:        'mat4x4f',
  proj:        'mat4x4f',
  viewProj:    'mat4x4f',
  invView:     'mat4x4f',
  invProj:     'mat4x4f',
  invViewProj: 'mat4x4f',
  camPos:      'vec3f',
  /** Sub-frame time in seconds since start. */
  time:        'f32',
  /** Frame delta in seconds, clamped to 100ms to survive tab stalls. */
  delta:       'f32',
  /** Total elapsed seconds. */
  elapsed:     'f32',
  /** 1 / width, 1 / height — for texel-centred UVs and screen-space effects. */
  resolution:  'vec2f',
  /** Backbuffer width, height in pixels. */
  viewport:    'vec2u',
  /** Tone-map exposure, applied by the present pass. */
  exposure:    'f32',
  /** Global alpha, 1 by default. */
  alpha:       'f32',
} as const satisfies Record<string, SlotType>;

export const FRAME_BLOCK: UniformBlockSpec = buildUniformBlock('Frame', FRAME_FIELDS, { maxBindingSize: 65536 });

/** @group(1) @binding(0) — per-object transform. Written with a dynamic offset. */
export const OBJECT_FIELDS = {
  /** World transform, column-major. */
  model:        'mat4x4f',
  /** Inverse-transpose of the upper 3x3, for normals under non-uniform scale. */
  normalMatrix: 'mat3x3f',
  /** Index into the per-frame draw list. Stable for the frame. */
  objectId:     'u32',
  /** 0 for the mesh, 1..n for instance slots. */
  instanceId:   'u32',
  /** 1.0 for a normal object; 0.0 when the owning node is hidden. */
  visibility:   'f32',
} as const satisfies Record<string, SlotType>;

export const OBJECT_BLOCK: UniformBlockSpec = buildUniformBlock('ObjectData', OBJECT_FIELDS, {
  dynamic: true,
  maxBindingSize: 65536,
});

/** @group(2) @binding(0) — per-material slots. */
export const MATERIAL_BLOCK: UniformBlockSpec = buildUniformBlock('MaterialData', {}, { maxBindingSize: 65536 });

/** Reserved slot names apse owns inside the material block. */
export const RESERVED_SLOT_NAMES: ReadonlySet<string> = new Set<string>([
  'model', 'normalMatrix', 'objectId', 'instanceId', 'visibility',
  'view', 'proj', 'viewProj', 'invView', 'invProj', 'invViewProj',
  'camPos', 'time', 'delta', 'elapsed', 'resolution', 'viewport', 'exposure', 'alpha',
]);

/** Bind group indices. Fixed so generated WGSL and pipeline layouts cannot drift. */
export const BIND_GROUP = {
  frame: 0,
  object: 1,
  material: 2,
  texture: 3,
} as const;
