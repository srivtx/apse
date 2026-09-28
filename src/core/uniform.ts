/**
 * The uniform type table and block builder.
 *
 * This module is the single source of truth for how a uniform field maps to
 * (a) a WGSL struct declaration, (b) a byte offset, and (c) a CPU write.
 * The material scaffold, the bind group layout, and the per-frame writer all
 * read from here, which is what makes it impossible for the generated WGSL and
 * the JS packing to disagree.
 *
 * Layout follows the WGSL uniform address space rules, which are stricter than
 * the storage address space: `vec3<f32>` is 16-byte aligned but only 12 bytes
 * wide, and any member following it starts at the next 16-byte boundary.
 */

import { fail } from './error.ts';
import type { SlotType } from './slot.ts';

export interface UniformTypeInfo {
  /** Declaration text used in the generated WGSL struct. */
  readonly wgsl: string;
  /** Scalar component count: f32=1, vec3f=3, mat4x4f=16. */
  readonly components: number;
  /** Byte alignment in the uniform address space. */
  readonly align: number;
  /** Byte width, excluding trailing padding. */
  readonly size: number;
  /** How the JS writer must interpret the source values. */
  readonly kind: 'float' | 'int' | 'uint';
  /** Declared type for error messages, e.g. "3 floats". */
  readonly describe: string;
}

const t = (
  wgsl: string,
  components: number,
  align: number,
  size: number,
  kind: UniformTypeInfo['kind'],
): UniformTypeInfo => ({
  wgsl,
  components,
  align,
  size,
  kind,
  describe: components === 1 ? '1 number' : `${components} numbers`,
});

/** Every uniform slot type apse understands. */
export const UNIFORM_TYPES: Readonly<Record<SlotType, UniformTypeInfo>> = Object.freeze({
  f32:      t('f32', 1, 4, 4, 'float'),
  i32:      t('i32', 1, 4, 4, 'int'),
  u32:      t('u32', 1, 4, 4, 'uint'),

  vec2f:    t('vec2<f32>', 2, 8, 8, 'float'),
  vec3f:    t('vec3<f32>', 3, 16, 12, 'float'),
  vec4f:    t('vec4<f32>', 4, 16, 16, 'float'),

  vec2u:    t('vec2<u32>', 2, 8, 8, 'uint'),
  vec3u:    t('vec3<u32>', 3, 16, 12, 'uint'),
  vec4u:    t('vec4<u32>', 4, 16, 16, 'uint'),

  // Three columns of vec3, each 16-byte aligned: 3 * 16 = 48.
  mat3x3f:  t('mat3x3<f32>', 12, 16, 48, 'float'),
  // Four columns of vec4: 4 * 16 = 64.
  mat4x4f:  t('mat4x4<f32>', 16, 16, 64, 'float'),
});

export const UNIFORM_TYPE_NAMES = Object.keys(UNIFORM_TYPES) as SlotType[];

export function uniformType(type: SlotType, field: string): UniformTypeInfo {
  const info = UNIFORM_TYPES[type];
  if (info === undefined) {
    fail('SLOT_TYPE_UNKNOWN', `Unknown uniform type "${type}" for field "${field}".`, {
      why: `The type "${type}" is not in apse's uniform type table, so it cannot be sized, aligned, or packed.`,
      fix: `Use one of: ${UNIFORM_TYPE_NAMES.join(', ')}. To add a type, add an entry to UNIFORM_TYPES in src/core/uniform.ts.`,
    });
  }
  return info;
}

/** Rounds `n` up to the next multiple of `align`. `align` must be a power of two. */
export function alignUp(n: number, align: number): number {
  return (n + align - 1) & ~(align - 1);
}

/** One resolved field inside a uniform block. */
export interface UniformField {
  readonly name: string;
  readonly type: SlotType;
  /** Byte offset from the start of the struct. */
  readonly offset: number;
  /** Byte width excluding padding. */
  readonly size: number;
  readonly components: number;
  readonly kind: UniformTypeInfo['kind'];
}

export interface UniformBlockSpec {
  readonly structName: string;
  readonly fields: readonly UniformField[];
  /** Byte size of the struct, padded up to its own alignment. */
  readonly size: number;
  /**
   * Byte stride between consecutive instances in a GPU buffer. This is
   * `size` rounded up to `minUniformBufferOffsetAlignment` (256 by default),
   * which is what dynamic offsets require. Zero for a non-dynamic block.
   */
  readonly stride: number;
  /** Generated WGSL `struct` declaration, including the trailing brace. */
  readonly wgsl: string;
}

/**
 * Builds a uniform block description from a field map.
 *
 * `maxBindingSize` is the device limit to check against; pass the device's
 * `maxUniformBufferBindingSize`. `dynamic` requests a 256-aligned stride for
 * use as a dynamic-offset buffer.
 */
export function buildUniformBlock(
  structName: string,
  fields: Readonly<Record<string, SlotType>>,
  opts: { dynamic?: boolean; maxBindingSize?: number; offsetAlignment?: number } = {},
): UniformBlockSpec {
  const resolved: UniformField[] = [];
  let offset = 0;
  let structAlign = 4;
  const lines: string[] = [];

  for (const [name, type] of Object.entries(fields)) {
    const info = uniformType(type, name);
    offset = alignUp(offset, info.align);
    resolved.push({
      name,
      type,
      offset,
      size: info.size,
      components: info.components,
      kind: info.kind,
    });
    lines.push(`  ${name} : ${info.wgsl},`);
    offset += info.size;
    if (info.align > structAlign) structAlign = info.align;
  }

  const size = alignUp(offset, structAlign);

  if (opts.maxBindingSize !== undefined && size > opts.maxBindingSize) {
    fail('UNIFORM_BLOCK_OVERFLOW',
      `Uniform block "${structName}" needs ${size} bytes but the device allows at most ${opts.maxBindingSize}.`, {
        why: 'WGSL uniform buffers are limited by the device `maxUniformBufferBindingSize` (64 KiB on the core profile, 16 KiB on compatibility).',
        fix: 'Move large or rarely-read fields into a storage buffer, or move them behind a separate bind group.',
      });
  }

  const stride = opts.dynamic ? alignUp(size, opts.offsetAlignment ?? 256) : 0;

  const wgsl = lines.length === 0
    ? `struct ${structName} {\n};`
    : `struct ${structName} {\n${lines.join('\n')}\n};`;

  return { structName, fields: resolved, size, stride, wgsl };
}

/**
 * A live instance of a uniform block: the backing storage plus typed views.
 *
 * All writes go through {@link set}, which validates component count and
 * finiteness once, in one place, rather than at six scattered call sites.
 */
export class UniformBlock {
  /** Raw backing storage, for `queue.writeBuffer`. */
  readonly data: ArrayBuffer;
  readonly f32: Float32Array;
  readonly u32: Uint32Array;
  readonly i32: Int32Array;
  readonly spec: UniformBlockSpec;

  constructor(spec: UniformBlockSpec) {
    this.spec = spec;
    this.data = new ArrayBuffer(spec.size);
    this.f32 = new Float32Array(this.data);
    this.u32 = new Uint32Array(this.data);
    this.i32 = new Int32Array(this.data);
  }

  /** A `subarray` view of just this instance's bytes, for a multi-instance buffer. */
  bytesAt(byteOffset: number): ArrayBuffer {
    return this.data.slice(byteOffset, byteOffset + this.spec.size);
  }

  private field(name: string): UniformField {
    const f = this.spec.fields.find((x) => x.name === name);
    if (f === undefined) {
      fail('INTERNAL_INVARIANT',
        `Uniform block "${this.spec.structName}" has no field "${name}".`, {
          why: 'apse writes uniform fields by name; writing an unknown name means the field table and the writer disagree.',
          fix: 'This is a bug in apse. Please report it with the material that triggered it.',
        });
      throw new Error('unreachable');
    }
    return f;
  }

  /**
   * Writes one field. `value` may be a number, a number-like array, or any
   * typed array of matching length. The value is range-checked so a NaN never
   * reaches the GPU and blanks the frame.
   */
  set(name: string, value: number | ArrayLike<number> | Float32Array | Uint32Array | Int32Array): void {
    const f = this.field(name);
    const start = f.offset >> 2;

    if (typeof value === 'number') {
      assertFinite(name, value);
      if (f.kind === 'float') this.f32[start] = value;
      else if (f.kind === 'uint') this.u32[start] = value >>> 0;
      else this.i32[start] = value | 0;
      return;
    }

    const len = value.length;
    if (len !== f.components) {
      fail('SLOT_VALUE_WRONG_LENGTH',
        `Field "${name}" is ${f.type} and needs ${f.components} numbers, but you passed ${len}.`, {
          why: 'Uniform fields are packed at a fixed byte offset. A wrong-length value would overwrite the neighbouring field.',
          fix: f.components === 1
            ? `Pass a single number for "${name}", not an array.`
            : `Pass exactly ${f.components} numbers for "${name}".`,
        });
    }

    for (let i = 0; i < len; i++) {
      const v = value[i] as number;
      assertFinite(name, v);
      if (f.kind === 'float') this.f32[start + i] = v;
      else if (f.kind === 'uint') this.u32[start + i] = v >>> 0;
      else this.i32[start + i] = v | 0;
    }
  }

  /** Reads one field back out, as a plain array. Useful in tests and tooling. */
  get(name: string): number[] {
    const f = this.field(name);
    const start = f.offset >> 2;
    const src = f.kind === 'float' ? this.f32 : f.kind === 'uint' ? this.u32 : this.i32;
    return Array.from(src.subarray(start, start + f.components));
  }

  /**
   * Sub-range view for writing directly into a larger shared buffer.
   * `byteOffset` must be a multiple of 4.
   */
  static viewFor(buffer: ArrayBuffer, spec: UniformBlockSpec, byteOffset: number): UniformBlock {
    const block = Object.create(UniformBlock.prototype) as {
      data: ArrayBuffer; f32: Float32Array; u32: Uint32Array;
      i32: Int32Array; spec: UniformBlockSpec;
    };
    block.spec = spec;
    block.data = buffer;
    block.f32 = new Float32Array(buffer, byteOffset, spec.size >> 2);
    block.u32 = new Uint32Array(buffer, byteOffset, spec.size >> 2);
    block.i32 = new Int32Array(buffer, byteOffset, spec.size >> 2);
    return block as UniformBlock;
  }
}

function assertFinite(field: string, value: number): void {
  if (Number.isFinite(value)) return;
  fail('SLOT_VALUE_NOT_FINITE',
    `Field "${field}" was set to ${String(value)}.`, {
      why: 'NaN and Infinity silently propagate through every fragment that reads the uniform, blanking the whole draw rather than raising an error.',
      fix: 'Guard the computation that produced it — usually a division by zero or an uninitialised value upstream.',
    });
}
