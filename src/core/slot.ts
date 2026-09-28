/**
 * Uniform slot types and the reserved block definitions.
 *
 * A "slot" is a named, typed field of a uniform block. Declaring slots in a
 * material is how a shader gets custom data without writing a single WGSL
 * binding: apse computes the struct, the offsets, and the writer.
 */

import type { UniformBlockSpec } from './uniform.ts';
import { alignUp, buildUniformBlock } from './uniform.ts';

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
// chosen by change frequency: the scene group changes once per object, mat once
// per material, textures last.
// ---------------------------------------------------------------------------

/** Frame fields. Bound at `@group(0) @binding(1)` — see {@link SCENE_BLOCK}. */
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

/** Per-object transform. Bound at `@group(0) @binding(0)`, by dynamic offset. */
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

/** @group(1) @binding(0) — per-material slots. */
export const MATERIAL_BLOCK: UniformBlockSpec = buildUniformBlock('MaterialData', {}, { maxBindingSize: 65536 });

/** Reserved slot names apse owns inside the material block. */
export const RESERVED_SLOT_NAMES: ReadonlySet<string> = new Set<string>([
  'model', 'normalMatrix', 'objectId', 'instanceId', 'visibility',
  'view', 'proj', 'viewProj', 'invView', 'invProj', 'invViewProj',
  'camPos', 'time', 'delta', 'elapsed', 'resolution', 'viewport', 'exposure', 'alpha',
]);

/**
 * Bind group indices. Fixed so generated WGSL and pipeline layouts cannot drift.
 *
 * Three groups, not four. The frame and the object used to be `@group(0)` and
 * `@group(1)`, which meant a draw had to bind both: `setBindGroup` is the most
 * expensive call in the WebGPU API, and a recorded census at 1000 boxes found 921
 * of them per frame, half of them saying the same thing. One scene group holding
 * both regions of one buffer is the whole point — see {@link SCENE_BLOCK}.
 */
export const BIND_GROUP = {
  /** Frame and object state, one buffer, bound once per draw. */
  scene: 0,
  material: 1,
  texture: 2,
} as const;

// ---------------------------------------------------------------------------
// The merged scene buffer
//
//     byte 0                       the frame: FRAME_BLOCK, the struct every
//                                   material already reads
//     byte SCENE_FRAME_BYTES       object 0: OBJECT_BLOCK
//     byte SCENE_FRAME_BYTES + 256 object 1
//     ...                          one stride each
//
// **The frame region is rounded up to a whole stride, and that is load-bearing.**
// `FRAME_BLOCK` is 432 bytes — six mat4s is 384, and the scalars after them are
// another 48 — and 432 is not a multiple of 256. The obvious layout, "frame at 0,
// object *i* at `(i + 1) * stride`", therefore puts object 0 at byte 256, inside
// `invProj`, and every pack of object 0 would overwrite `camPos`, `time`, `delta`,
// `elapsed`, `resolution`, `viewport`, `exposure` and `alpha` with the first 176
// bytes of its own model matrix. Nothing validates: the writes are in range, the
// draws succeed, and the scene renders with a zeroed camera. So the object base
// is *computed* from the built frame block, and `test/slot.test.ts` asserts the
// two regions do not overlap rather than asserting a constant.
// ---------------------------------------------------------------------------

/**
 * Byte stride between consecutive objects, and between a dynamic offset and the
 * next one it could legally take.
 *
 * 256 is `minUniformBufferOffsetAlignment`'s guaranteed value: the limit may be
 * *larger* on some devices, never smaller, and a stride below it would be legal
 * nowhere. `OBJECT_BLOCK.stride` is built to the same number, and a test asserts
 * the two agree.
 */
export const SCENE_UNIFORM_STRIDE = 256;

/** @group(0) @binding(0) — the object, the one binding a draw moves. */
export const SCENE_OBJECT_BINDING = 0;

/**
 * @group(0) @binding(1) — the frame, the one binding a pass never moves.
 *
 * The object and the frame are two regions of one buffer, so a WGSL module needs
 * two declarations, so they need two bindings: "Two different resource variables
 * in a shader must not have the same group and binding values" (WGSL, resource
 * interface). A dynamic offset of 0 for the frame would be legal — 0 is a
 * multiple of 256 — but it would have to share the *object's* binding, and a
 * binding a draw rewrites is a binding that moves the frame with it. Two entries,
 * one dynamic, is the only shape that gives a draw one bind and a pass one
 * constant.
 */
export const SCENE_FRAME_BINDING = 1;

/** Bytes reserved for the frame, rounded up to a whole object stride. */
export const SCENE_FRAME_BYTES = alignUp(FRAME_BLOCK.size, SCENE_UNIFORM_STRIDE);

/** One region's placement inside the shared scene buffer. */
export interface SceneRegion {
  readonly name: 'frame' | 'object';
  /** Binding index within the scene group. */
  readonly binding: number;
  /**
   * Whether a draw supplies an offset for this region. Only the object does: it
   * is what changes per draw, and a static binding is one less number to write
   * per draw.
   */
  readonly hasDynamicOffset: boolean;
  /** Byte offset of the region from the start of the buffer. */
  readonly byteOffset: number;
  /**
   * Bytes the region occupies. The object region is a whole stride rather than
   * `OBJECT_BLOCK.size` because the stride is the alignment a dynamic offset is
   * measured in, so a stride-sized range is the alignment-rounded view of the
   * struct; it also means the last object in a full buffer ends exactly at the
   * end of the buffer.
   */
  readonly byteLength: number;
  /** The struct this region holds. Its `size` is the layout's `minBindingSize`. */
  readonly block: UniformBlockSpec;
}

/** The combined buffer: the frame region, then one stride per object. */
export interface SceneBlockSpec {
  readonly structName: 'SceneData';
  /** Bytes between consecutive objects. */
  readonly stride: number;
  readonly frame: SceneRegion;
  readonly object: SceneRegion;
  /**
   * The smallest legal buffer: the frame region plus one object stride. Zero
   * object slots is not a thing — a draw with no object still has to bind the
   * object region at some legal offset.
   */
  readonly minimumByteLength: number;
}

export const SCENE_BLOCK: SceneBlockSpec = {
  structName: 'SceneData',
  stride: SCENE_UNIFORM_STRIDE,
  frame: {
    name: 'frame',
    binding: SCENE_FRAME_BINDING,
    hasDynamicOffset: false,
    byteOffset: 0,
    byteLength: SCENE_FRAME_BYTES,
    block: FRAME_BLOCK,
  },
  object: {
    name: 'object',
    binding: SCENE_OBJECT_BINDING,
    hasDynamicOffset: true,
    byteOffset: SCENE_FRAME_BYTES,
    byteLength: SCENE_UNIFORM_STRIDE,
    block: OBJECT_BLOCK,
  },
  minimumByteLength: SCENE_FRAME_BYTES + SCENE_UNIFORM_STRIDE,
};

/**
 * The dynamic offset a draw binds to select object `index`.
 *
 * The one place this arithmetic exists. The renderer's draw list computes it once
 * per item, the uniform packer writes at the matching byte, and the test suite
 * reads it to check the regions do not collide — three consumers of one function,
 * which is the only reason the two copies can be compared at all.
 */
export function sceneObjectOffset(index: number): number {
  return SCENE_BLOCK.object.byteOffset + index * SCENE_BLOCK.stride;
}

/** Bytes a scene buffer holding `objects` object slots needs. */
export function sceneByteLength(objects: number): number {
  return SCENE_BLOCK.object.byteOffset + Math.max(1, objects) * SCENE_BLOCK.stride;
}
