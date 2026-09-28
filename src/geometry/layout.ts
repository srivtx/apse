/**
 * Vertex layout — one description, three consumers.
 *
 * The same `VertexLayout` instance produces:
 *   1. the WGSL `struct VertexIn` declaration, injected into every material,
 *   2. the `GPUVertexBufferLayout` handed to the pipeline,
 *   3. the interleaved typed array the mesh uploads.
 *
 * Because all three come from here, a shader can never disagree with the
 * buffer it reads from. This is the reason apse materials take a layout
 * constant rather than a hand-written struct in the shader.
 */

import { fail } from '../core/error.ts';

/** Every vertex format WebGPU accepts, minus the half-float ones (see note). */
export type VertexFormat =
  | 'uint8x2' | 'uint8x4' | 'sint8x2' | 'sint8x4' | 'unorm8x2' | 'unorm8x4'
  | 'snorm8x2' | 'snorm8x4' | 'unorm8x4-bgra'
  | 'uint16x2' | 'uint16x4' | 'sint16x2' | 'sint16x4'
  | 'unorm16x2' | 'unorm16x4' | 'snorm16x2' | 'snorm16x4'
  | 'float32' | 'float32x2' | 'float32x3' | 'float32x4'
  | 'uint32' | 'uint32x2' | 'uint32x3' | 'uint32x4'
  | 'sint32' | 'sint32x2' | 'sint32x3' | 'sint32x4'
  | 'unorm10-10-10-2';

export interface VertexFormatInfo {
  /** WGSL type this format reads as. */
  readonly wgsl: string;
  readonly components: number;
  /** Bytes per vertex for this attribute. */
  readonly byteSize: number;
  /** Typed array used to pack it on the CPU. */
  readonly array: Float32ArrayConstructor | Uint32ArrayConstructor | Int32ArrayConstructor
    | Uint16ArrayConstructor | Int16ArrayConstructor | Uint8ArrayConstructor | Int8ArrayConstructor;
  /** Human-readable component list, for error messages. */
  readonly describe: string;
}

const f32 = Float32Array;
const u32 = Uint32Array;
const i32 = Int32Array;
const u16 = Uint16Array;
const i16 = Int16Array;
const u8 = Uint8Array;
const i8 = Int8Array;

const v = (
  wgsl: string,
  components: number,
  array: VertexFormatInfo['array'],
  unit: number,
): VertexFormatInfo => ({
  wgsl,
  components,
  array,
  byteSize: components * unit,
  describe: components === 1 ? wgsl : `${components} × ${wgsl}`,
});

/**
 * The format table.
 *
 * Note on `float16x2` / `float16x4`: WebGPU supports them, but packing them
 * requires an f16 encoder that no JS engine exposes natively. They are omitted
 * deliberately rather than offered with a slow path. Use `unorm16x2` for
 * compact half-precision data, or `float32x2` when you need actual precision.
 */
export const VERTEX_FORMATS: Readonly<Record<VertexFormat, VertexFormatInfo>> = Object.freeze({
  uint8x2:          v('vec2<u32>', 2, u8, 1),
  uint8x4:          v('vec4<u32>', 4, u8, 1),
  sint8x2:          v('vec2<i32>', 2, i8, 1),
  sint8x4:          v('vec4<i32>', 4, i8, 1),
  unorm8x2:         v('vec2<f32>', 2, u8, 1),
  unorm8x4:         v('vec4<f32>', 4, u8, 1),
  snorm8x2:         v('vec2<f32>', 2, i8, 1),
  snorm8x4:         v('vec4<f32>', 4, i8, 1),
  'unorm8x4-bgra':  v('vec4<f32>', 4, u8, 1),

  uint16x2:         v('vec2<u32>', 2, u16, 2),
  uint16x4:         v('vec4<u32>', 4, u16, 2),
  sint16x2:         v('vec2<i32>', 2, i16, 2),
  sint16x4:         v('vec4<i32>', 4, i16, 2),
  unorm16x2:        v('vec2<f32>', 2, u16, 2),
  unorm16x4:        v('vec4<f32>', 4, u16, 2),
  snorm16x2:        v('vec2<f32>', 2, i16, 2),
  snorm16x4:        v('vec4<f32>', 4, i16, 2),

  float32:          v('f32', 1, f32, 4),
  float32x2:        v('vec2<f32>', 2, f32, 4),
  float32x3:        v('vec3<f32>', 3, f32, 4),
  float32x4:        v('vec4<f32>', 4, f32, 4),

  uint32:           v('u32', 1, u32, 4),
  uint32x2:         v('vec2<u32>', 2, u32, 4),
  uint32x3:         v('vec3<u32>', 3, u32, 4),
  uint32x4:         v('vec4<u32>', 4, u32, 4),

  sint32:           v('i32', 1, i32, 4),
  sint32x2:         v('vec2<i32>', 2, i32, 4),
  sint32x3:         v('vec3<i32>', 3, i32, 4),
  sint32x4:         v('vec4<i32>', 4, i32, 4),

  'unorm10-10-10-2': v('vec4<f32>', 4, u16, 2),
});

export const VERTEX_FORMAT_NAMES = Object.keys(VERTEX_FORMATS) as VertexFormat[];

export function vertexFormat(format: string, attribute: string): VertexFormatInfo {
  const info = (VERTEX_FORMATS as Record<string, VertexFormatInfo | undefined>)[format];
  if (info === undefined) {
    fail('ATTRIBUTE_FORMAT_UNKNOWN',
      `Attribute "${attribute}" declares format "${format}", which apse does not know.`, {
        why: 'apse generates the WGSL struct, the GPU buffer layout, and the CPU packing from this format. An unknown format has no size, so it cannot be placed.',
        fix: `Use one of: ${VERTEX_FORMAT_NAMES.join(', ')}. To add a format, add an entry to VERTEX_FORMATS in src/geometry/layout.ts.`,
      });
  }
  return info;
}

/** `position: 'float32x3'` — the common case. */
export type AttributeDefs = Readonly<Record<string, VertexFormat>>;

/**
 * `maxVertexAttributes`: the number of `@location`s a vertex stage input may
 * declare, counted across *every* vertex buffer the pipeline binds. The vertex
 * half and the per-instance half draw on the same budget, which is the whole
 * reason this is one constant and not two limits.
 */
export const MAX_VERTEX_LOCATIONS = 16;

export interface ResolvedAttribute {
  readonly name: string;
  readonly format: VertexFormat;
  readonly info: VertexFormatInfo;
  /** Byte offset within the vertex. */
  readonly offset: number;
  /** `@location(n)` in the generated WGSL struct. */
  readonly location: number;
}

/**
 * An immutable, cached vertex layout.
 *
 * Build with {@link layout}. The returned object is shared by every mesh and
 * material that uses the same attributes, so construction cost is irrelevant
 * and identity comparison is a valid cache key.
 */
export class VertexLayout {
  readonly attributes: readonly ResolvedAttribute[];
  /**
   * Per-instance attributes, from the second vertex buffer. Empty when the
   * layout describes a non-instanced draw. Locations continue from the vertex
   * attributes, because the vertex stage reads both halves through one struct.
   */
  readonly instanceAttributes: readonly ResolvedAttribute[];
  /** Bytes per vertex. Must be a multiple of 4. */
  readonly stride: number;
  /** Bytes per instance in the second buffer, or 0 when there are none. */
  readonly instanceStride: number;
  /** Number of vertex attributes, i.e. how many locations the first buffer uses. */
  readonly attributeCount: number;
  /** Number of per-instance attributes, i.e. how many locations the second uses. */
  readonly instanceAttributeCount: number;

  readonly #byName: Map<string, ResolvedAttribute>;
  readonly #instanceByName: Map<string, ResolvedAttribute>;
  readonly #wgslCache: Map<string, string>;
  readonly #key: string;
  #gpuLayouts: readonly GPUVertexBufferLayout[] | null;

  constructor(attributes: AttributeDefs, maxStride = 2048, instanceAttributes?: AttributeDefs) {
    const resolved: ResolvedAttribute[] = [];
    const byName = new Map<string, ResolvedAttribute>();
    let offset = 0;
    let location = 0;

    for (const [name, format] of Object.entries(attributes)) {
      const info = vertexFormat(format, name);
      // WebGPU requires every vertex offset to be 4-byte aligned.
      offset = (offset + 3) & ~3;
      const attr: ResolvedAttribute = { name, format, info, offset, location: location++ };
      resolved.push(attr);
      byName.set(name, attr);
      offset += info.byteSize;
    }

    if (resolved.length === 0) {
      fail('MESH_NO_POSITION',
        'A vertex layout must declare at least one attribute.', {
        why: 'A pipeline with an empty vertex state cannot be created, and there is nothing to draw.',
        fix: 'Start from `STANDARD_LAYOUT` (position/normal/uv) or declare at least `position: "float32x3"`.',
      });
    }

    if (offset > maxStride) {
      fail('ATTRIBUTE_LAYOUT_OVERFLOW',
        `Vertex stride is ${offset} bytes, over the ${maxStride}-byte device limit.`, {
        why: 'All vertex attributes must live in a single buffer whose stride is bounded by `maxVertexBufferArrayStride` (2048 on the core profile).',
        fix: 'Pack an attribute smaller. `unorm8x4` costs 4 bytes where `float32x4` costs 16 — a position/normal/uv/tangent mesh drops from 56 to 28 bytes per vertex.',
      });
    }

    if (location > MAX_VERTEX_LOCATIONS) {
      fail('VARYING_LOCATION_OVERFLOW',
        `Layout declares ${location} attributes but the device allows ${MAX_VERTEX_LOCATIONS}.`, {
        why: 'Each attribute consumes one vertex `@location`, and WebGPU caps vertex attributes at 16.',
        fix: 'Pack attributes into wider formats — two `unorm8x2` attributes become one `unorm8x4`.',
      });
    }

    // The per-instance half. Resolved by the same rules, into the same
    // location sequence, because the vertex stage receives both halves as one
    // struct: `@location(0..2)` from the mesh buffer, `@location(3..7)` from the
    // instance buffer. A storage buffer per instance is the alternative and it
    // does not exist on the compatibility profile (`maxStorageBuffersInVertexStage`
    // is 0), which compiles on a desktop and fails on a phone.
    const instance: ResolvedAttribute[] = [];
    const instanceByName = new Map<string, ResolvedAttribute>();
    let instanceOffset = 0;

    for (const [name, format] of Object.entries(instanceAttributes ?? {})) {
      if (byName.has(name) || instanceByName.has(name)) {
        fail('ATTRIBUTE_MISSING',
          `Layout declares "${name}" twice — once as a vertex attribute and once as a per-instance attribute (or twice in the same map).`, {
          why: `Both halves are emitted into one WGSL struct, so two fields of the same name is a duplicate declaration that the shader compiler rejects. The vertex half declares: ${resolved.map((a) => a.name).join(', ')}.`,
          fix: 'Rename one of them. The per-instance names apse ships are prefixed `instanceTransform0..3` and `instanceColor` precisely so they cannot collide with a mesh attribute.',
        });
      }
      const info = vertexFormat(format, name);
      instanceOffset = (instanceOffset + 3) & ~3;
      const attr: ResolvedAttribute = { name, format, info, offset: instanceOffset, location: location++ };
      instance.push(attr);
      instanceByName.set(name, attr);
      instanceOffset += info.byteSize;
    }

    if (instanceOffset > maxStride) {
      fail('ATTRIBUTE_LAYOUT_OVERFLOW',
        `Instance stride is ${instanceOffset} bytes, over the ${maxStride}-byte device limit.`, {
        why: 'The per-instance buffer is a vertex buffer like any other, so its stride is bounded by the same `maxVertexBufferArrayStride` limit.',
        fix: 'Drop an instance attribute. A `mat4x4f` costs 64 bytes; four `float32x4`s cost the same, and a `vec3` translation plus a packed normal costs a third of that.',
      });
    }

    // Only reachable when the vertex half was within budget on its own, so the
    // message can name the split: the budget is shared and that is the surprise.
    if (instance.length > 0 && location > MAX_VERTEX_LOCATIONS) {
      fail('VARYING_LOCATION_OVERFLOW',
        `Layout declares ${resolved.length} vertex attributes and ${instance.length} per-instance attributes — ${location} locations in total, and the device allows ${MAX_VERTEX_LOCATIONS}.`, {
        why: `Vertex and per-instance attributes are counted against the same \`maxVertexAttributes\` limit of ${MAX_VERTEX_LOCATIONS}, because both are declared in the vertex stage's input struct.`,
        fix: 'Drop a per-instance attribute first: a `mat4x4f` transform costs four of them, and a translation-only instance costs one. Then pack the vertex attributes — two `unorm8x2` attributes become one `unorm8x4`.',
      });
    }

    this.attributes = Object.freeze(resolved);
    this.instanceAttributes = Object.freeze(instance);
    this.stride = offset;
    this.instanceStride = instanceOffset;
    this.attributeCount = resolved.length;
    this.instanceAttributeCount = instance.length;
    this.#byName = byName;
    this.#instanceByName = instanceByName;
    this.#wgslCache = new Map();
    this.#gpuLayouts = null;

    // Canonical key. Declaration order is significant because it determines
    // the byte layout, so this is not order-independent. The instance half is
    // tagged so a layout and the same layout plus instancing are two different
    // keys — they compile to different programs and cannot share a pipeline.
    const vertexKey = resolved.map((a) => `${a.name}:${a.format}`).join('|');
    this.#key = instance.length === 0
      ? vertexKey
      : `${vertexKey}|inst:${instance.map((a) => `${a.name}:${a.format}`).join('|')}`;
  }


  /** The canonical identity of this layout. Stable and cheap to compare. */
  get key(): string {
    return this.#key;
  }

  attribute(name: string): ResolvedAttribute | undefined {
    return this.#byName.get(name);
  }

  /** The per-instance attribute `name`, or undefined. */
  instanceAttribute(name: string): ResolvedAttribute | undefined {
    return this.#instanceByName.get(name);
  }

  has(name: string): boolean {
    return this.#byName.has(name);
  }

  /** True when this layout binds a per-instance buffer. */
  get instanced(): boolean {
    return this.instanceAttributeCount > 0;
  }

  byteLength(count: number): number {
    return this.stride * count;
  }

  /** Bytes the per-instance buffer occupies for `count` instances. */
  instanceByteLength(count: number): number {
    return this.instanceStride * count;
  }

  /**
   * The generated WGSL struct. `structName` is usually `VertexIn`.
   * Cached, so calling this per material is free.
   *
   * With per-instance attributes the struct carries **both** halves: the vertex
   * attributes at `@location(0..n-1)` and the instance attributes at
   * `@location(n..)`, because that is exactly how the vertex stage receives
   * them — one input struct, two `GPUVertexBufferLayout` slots. A body reads
   * both through `in.`.
   */
  wgslStruct(structName = 'VertexIn'): string {
    const cached = this.#wgslCache.get(structName);
    if (cached !== undefined) return cached;
    const lines = this.attributes.map(
      (a) => `  @location(${a.location}) ${a.name} : ${a.info.wgsl},`,
    );
    const src = `struct ${structName} {\n${lines.concat(this.#instanceWgslFields()).join('\n')}\n};`;
    this.#wgslCache.set(structName, src);
    return src;
  }

  /**
   * The generated WGSL struct for the per-instance attributes alone.
   *
   * The same field text as the tail of {@link wgslStruct}, as a standalone
   * struct. It exists for reading and for tooling — an inspector, a test, or a
   * documentation build that wants the instance half without the vertex half.
   * A material does not need it: the fields are already in `VertexIn`.
   */
  instanceWgslStruct(structName = 'InstanceIn'): string {
    const cacheKey = `instance:${structName}`;
    const cached = this.#wgslCache.get(cacheKey);
    if (cached !== undefined) return cached;
    const lines = this.#instanceWgslFields();
    const src = lines.length === 0
      ? `struct ${structName} {\n};`
      : `struct ${structName} {\n${lines.join('\n')}\n};`;
    this.#wgslCache.set(cacheKey, src);
    return src;
  }

  #instanceWgslFields(): readonly string[] {
    return this.instanceAttributes.map(
      (a) => `  @location(${a.location}) ${a.name} : ${a.info.wgsl},`,
    );
  }

  /**
   * The `GPUVertexBufferLayout` for this layout, as a single interleaved buffer.
   *
   * One buffer, one `setVertexBuffer` call per draw, regardless of attribute
   * count. This is the layout that keeps per-draw CPU cost at the floor.
   *
   * It describes the **vertex** buffer only. A draw that binds a per-instance
   * buffer needs {@link gpuLayouts}, whose second element is the instance slot;
   * a pipeline built from `gpuLayout()` alone has no slot 1 and cannot read one.
   */
  gpuLayout(): GPUVertexBufferLayout {
    return {
      arrayStride: this.stride,
      stepMode: 'vertex',
      attributes: this.attributes.map((a) => ({
        shaderLocation: a.location,
        offset: a.offset,
        format: a.format,
      })),
    };
  }

  /**
   * Every vertex buffer layout this layout needs, in `setVertexBuffer` slot
   * order: the vertex buffer first, the per-instance buffer second.
   *
   * This is the canonical accessor. The array is the pipeline's `vertex.buffers`
   * and the slot index of each element is the index passed to
   * `setVertexBuffer`, so the two cannot be mixed up. A layout with no
   * per-instance attributes returns a single element, which is byte-for-byte
   * what {@link gpuLayout} returns.
   *
   * Cached and frozen: the renderer calls this once per draw item, and a
   * descriptor that could be mutated into a wrong pipeline is worse than an
   * allocation.
   */
  gpuLayouts(): readonly GPUVertexBufferLayout[] {
    if (this.#gpuLayouts !== null) return this.#gpuLayouts;
    const out: GPUVertexBufferLayout[] = [this.gpuLayout()];
    if (this.instanceAttributeCount > 0) {
      out.push(Object.freeze({
        arrayStride: this.instanceStride,
        stepMode: 'instance',
        attributes: this.instanceAttributes.map((a) => ({
          shaderLocation: a.location,
          offset: a.offset,
          format: a.format,
        })),
      }));
    }
    this.#gpuLayouts = Object.freeze(out);
    return this.#gpuLayouts;
  }

  /** The interleaved typed array the mesh uploads. */
  allocate(count: number): Float32Array {
    return new Float32Array(this.byteLength(count) >> 2);
  }

  /**
   * Verifies that `other` is compatible with this layout.
   *
   * Layouts must match exactly, in both names and formats — a partial match is
   * a validation error at draw time in WebGPU, and an inscrutable one, so
   * apse rejects it at bind time with the specific difference named.
   *
   * The per-instance half is compared **only when both sides declare one**. A
   * mesh is not the source of instance data, and `Material.updateMesh(mesh)`
   * hands a mesh to exactly this method; a mesh layout with no per-instance
   * attributes is not a mismatch, it is simply a different subject. Two layouts
   * that both carry an instance buffer — a material's and a `GpuInstances`' —
   * are compared in full, and a disagreement in the transform attributes is
   * caught here rather than as a misread matrix.
   */
  assertCompatible(other: VertexLayout, context: string): void {
    if (other === this || other.key === this.key) return;
    assertAttributeMatch(this.attributes, other.attributes, other, context, 'attribute');
    if (this.instanceAttributeCount > 0 && other.instanceAttributeCount > 0) {
      assertAttributeMatch(
        this.instanceAttributes,
        other.instanceAttributes,
        other,
        context,
        'instance attribute',
        (name) => other.instanceAttribute(name),
      );
    }
  }
}

/**
 * One half of {@link VertexLayout.assertCompatible}, shared by the vertex and
 * per-instance halves so both fail with the same code, the same message shape,
 * and the same "here is what I expected" listing.
 */
function assertAttributeMatch(
  mine0: readonly ResolvedAttribute[],
  theirs0: readonly ResolvedAttribute[],
  other: VertexLayout,
  context: string,
  what: 'attribute' | 'instance attribute',
  lookup?: (name: string) => ResolvedAttribute | undefined,
): void {
  const find = lookup ?? ((name: string): ResolvedAttribute | undefined => other.attribute(name));
  const mine = new Set(mine0.map((a) => a.name));
  const theirs = new Set(theirs0.map((a) => a.name));
  for (const attr of mine0) {
    const match = find(attr.name);
    if (match === undefined) {
      fail('ATTRIBUTE_MISSING',
        `${context} is missing ${what} "${attr.name}".`, {
        why: 'The pipeline layout declares this attribute at a specific @location; the vertex buffer has no data there.',
        fix: `Add "${attr.name}" to the geometry, or use a material built for ${context}'s layout. Expected: ${[...mine].join(', ')}. Got: ${[...theirs].join(', ') || 'nothing'}.`,
      });
    }
    if (match.format !== attr.format) {
      fail('LAYOUT_MISMATCH',
        `Attribute "${attr.name}" is "${attr.format}" on the material but "${match.format}" on ${context}.`, {
        why: 'A vertex attribute\'s WGSL type is fixed by its vertex format. Reading float32x3 data as vec4<f32> reads past the end of the attribute.',
        fix: 'Share one layout constant between the mesh and the material, or match the two formats.',
      });
    }
  }
  for (const name of theirs) {
    if (!mine.has(name)) {
      fail('ATTRIBUTE_MISSING',
        `${context} supplies ${what} "${name}", which the material does not declare.`, {
        why: 'A vertex buffer may be a superset of what the shader reads, but apse generates the struct from the material, so the byte offsets would be computed from the wrong order.',
        fix: `Remove "${name}" from the geometry, or declare it on the material. The material declares: ${[...mine].join(', ')}.`,
      });
    }
  }
}

/** Builds a {@link VertexLayout}. */
export function layout(attributes: AttributeDefs, maxStride?: number, instanceAttributes?: AttributeDefs): VertexLayout {
  return new VertexLayout(attributes, maxStride, instanceAttributes);
}

/**
 * Layout interned by attribute map, so `layout({...})` with the same shape
 * always returns the same instance. Identity comparison then works as a
 * pipeline cache key, and no layout objects are duplicated.
 */
const interned = new Map<string, VertexLayout>();

export function layoutCached(
  attributes: AttributeDefs,
  maxStride?: number,
  instanceAttributes?: AttributeDefs,
): VertexLayout {
  const key = [
    Object.entries(attributes).map(([k, v2]) => `${k}:${v2}`).join('|'),
    Object.entries(instanceAttributes ?? {}).map(([k, v2]) => `${k}:${v2}`).join('|'),
  ].join('#inst');
  let found = interned.get(key);
  if (found === undefined) {
    found = new VertexLayout(attributes, maxStride, instanceAttributes);
    interned.set(key, found);
  }
  return found;
}

// ---------------------------------------------------------------------------
// Shared layouts
// ---------------------------------------------------------------------------

/**
 * The attribute map behind {@link STANDARD_LAYOUT}.
 *
 * Exported because instancing has to build a *combined* layout — a vertex half
 * and a per-instance half — and it should default to the same vertex half
 * every other mesh uses rather than to a second, subtly different copy.
 * Declared first because {@link STANDARD_LAYOUT} is built from it.
 */
export const STANDARD_ATTRIBUTES: AttributeDefs = Object.freeze({
  position: 'float32x3',
  normal: 'float32x3',
  uv: 'float32x2',
});

/** position + normal + uv, 32 bytes/vertex. The right default for most meshes. */
export const STANDARD_LAYOUT: VertexLayout = layout(STANDARD_ATTRIBUTES);

/** position + normal + uv + tangent + handedness, 48 bytes/vertex. */
export const TANGENT_LAYOUT: VertexLayout = layout({
  position: 'float32x3',
  normal: 'float32x3',
  uv: 'float32x2',
  tangent: 'float32x4',
});

/** position + uv, 20 bytes/vertex. For unlit or fully custom shading. */
export const POSITION_UV_LAYOUT: VertexLayout = layout({
  position: 'float32x3',
  uv: 'float32x2',
});

/** position only, 12 bytes/vertex. */
export const POSITION_LAYOUT: VertexLayout = layout({ position: 'float32x3' });
