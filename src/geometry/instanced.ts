/**
 * Per-instance data — one draw call for many copies of one mesh.
 *
 * ## Why a second vertex buffer and not a storage buffer
 *
 * The obvious way to hand a per-instance matrix to a shader is a storage buffer
 * indexed by `@builtin(instance_index)`. apse cannot do that: it targets the
 * WebGPU **compatibility** profile, where `maxStorageBuffersInVertexStage` is
 * **0**. A vertex stage that reads one compiles on a desktop and fails to
 * create a pipeline on the phone, with a validation error that names a limit
 * rather than the shader. That is exactly the class of bug this library exists
 * to make impossible.
 *
 * So per-instance data arrives as a second vertex buffer with
 * `stepMode: 'instance'`. The vertex stage advances it once per instance and
 * reuses it for every vertex in between. It works on every profile, costs no
 * bind group, and is one `setVertexBuffer(1, …)` per draw rather than per
 * instance — the CPU cost of a 5000-cube scene is the same as the cost of one
 * cube.
 *
 * ## Why four `float32x4`s and not one `mat4x4f`
 *
 * WebGPU has no matrix vertex format at all: `GPUVertexFormat` goes up to
 * `float32x4`, and that is the end of it. A `mat4x4f` in a vertex struct is
 * therefore only reachable from a storage buffer, which is the thing above.
 * Four `float32x4` attributes is not the second-best option here, it is the
 * only one.
 *
 * The useful consequence is on the CPU side: the four attributes are four
 * consecutive `vec4`s, so a flat run of 16 floats per instance is *already* in
 * buffer order. Instance `i` occupies bytes `[i*64, i*64+64)` and nothing has
 * to be repacked. `uploadInstances` hands the caller's array straight to
 * `writeBuffer` in the transform-only case.
 *
 * ## The two classes
 *
 *   {@link InstanceData}     pure CPU. Growable, mutable, no device — so a
 *                            particle system or a scatter plot can be built
 *                            and asserted on in a test with no GPU in sight.
 *   {@link GpuInstances}     the uploaded form. One `GPUBuffer` and a count.
 */

import { fail } from '../core/error.ts';
import { Resource } from '../core/resource.ts';
import { create, fromQuat, fromScale, fromTranslation, identity, mul } from '../math/mat4.ts';
import { STANDARD_ATTRIBUTES, layout, vertexFormat } from './layout.ts';
import type { AttributeDefs, VertexLayout } from './layout.ts';

// ---------------------------------------------------------------------------
// The instance attribute vocabulary
// ---------------------------------------------------------------------------

/** Floats in one column-major `mat4x4f`. */
export const TRANSFORM_STRIDE = 16;

/** Components in a per-instance colour: r, g, b, a. */
export const INSTANCE_COLOR_COMPONENTS = 4;

/** The attribute name of the per-instance colour. Also the WGSL field name. */
export const INSTANCE_COLOR = 'instanceColor';

/**
 * The four columns of the per-instance transform, as four vertex attributes.
 *
 * `float32x4` rather than `unorm8x4` for the colour, and rather than any
 * packed format: the CPU-side arrays are `Float32Array`s, and a quantise step
 * in the upload path is a place for a rounding bug to live for no bandwidth
 * worth having on a buffer that is 64 bytes per instance.
 */
export const TRANSFORM_ATTRIBUTES: AttributeDefs = Object.freeze({
  instanceTransform0: 'float32x4',
  instanceTransform1: 'float32x4',
  instanceTransform2: 'float32x4',
  instanceTransform3: 'float32x4',
});

/** A transform-only instance: 4 locations, 64 bytes, one `mat4x4f`. */
export const INSTANCE_ATTRIBUTES: AttributeDefs = TRANSFORM_ATTRIBUTES;

/** A transform plus a per-instance tint: 5 locations, 80 bytes. */
export const INSTANCE_ATTRIBUTES_COLORED: AttributeDefs = Object.freeze({
  ...TRANSFORM_ATTRIBUTES,
  [INSTANCE_COLOR]: 'float32x4',
});

/**
 * Floats one instance occupies in an interleaved transform+colour record:
 * 16 of matrix and 4 of rgba.
 *
 * Exported because it is the thing to check an allocation against — a buffer
 * of `n` coloured instances is `n * COLORED_INSTANCE_STRIDE * 4` bytes — and
 * because it is the number that appears in every "why is this 80 bytes" thread.
 */
export const COLORED_INSTANCE_STRIDE = TRANSFORM_STRIDE + INSTANCE_COLOR_COMPONENTS;

export interface InstancedLayoutOptions {
  /** The mesh's vertex attributes. Defaults to {@link STANDARD_ATTRIBUTES}. */
  readonly vertexAttributes?: AttributeDefs;
  /** Declare the per-instance colour attribute. Defaults to false. */
  readonly color?: boolean;
  /** `maxVertexBufferArrayStride`. Defaults to 2048. */
  readonly maxStride?: number;
}

/**
 * The combined layout an instanced draw needs: mesh attributes plus instance
 * attributes, with the instance locations continuing where the vertex ones
 * stop.
 *
 * With {@link STANDARD_ATTRIBUTES} and a transform, that is `@location(0..2)`
 * for the mesh and `@location(3..6)` for the transform — 7 of the 16 locations
 * the device allows. Adding the colour makes it 8.
 *
 * This is one description consumed three ways, exactly as a non-instanced
 * layout is: the WGSL `VertexIn` struct, both `GPUVertexBufferLayout` slots,
 * and the buffer the mesh uploads. The mesh is built with the *vertex* half
 * only, so it is passed `instancedLayout().gpuLayouts()[0]`'s counterpart —
 * that is, an ordinary `layout(…)` of the same vertex attributes.
 */
export function instancedLayout(opts: InstancedLayoutOptions = {}): VertexLayout {
  const { vertexAttributes = STANDARD_ATTRIBUTES, color = false, maxStride } = opts;
  const instances = color ? INSTANCE_ATTRIBUTES_COLORED : INSTANCE_ATTRIBUTES;
  return maxStride === undefined
    ? layout(vertexAttributes, 2048, instances)
    : layout(vertexAttributes, maxStride, instances);
}

// ---------------------------------------------------------------------------
// InstanceData
// ---------------------------------------------------------------------------

export interface InstanceDataOptions {
  /** Only used in error messages and tooling. Defaults to `'instances'`. */
  readonly name?: string;
  /**
   * Per-instance transforms, 16 floats each, column-major.
   *
   * Taken **by reference** when it is already a `Float32Array`, on the same
   * terms as `MeshData`'s interleaved source: you may write into it and re-upload
   * without telling apse. apse never resizes it or writes to it — and note that
   * `reserve()` therefore allocates a *new* array, after which this object is no
   * longer a view of yours. Re-read `data.transforms` after growing.
   */
  readonly transforms: Float32Array | ArrayLike<number>;
  /** Optional per-instance colour, 4 floats each (r,g,b,a). */
  readonly colors?: Float32Array | ArrayLike<number> | null;
  /**
   * Floats between the start of one transform and the next. Defaults to 16.
   *
   * Must be 16 or a multiple of 4 above it. Note that only 16 is expressible
   * as an {@link AttributeDefs} — padding between attributes is not something a
   * format map can say — so {@link uploadInstances} rejects anything wider.
   */
  readonly transformStride?: number;
}

/** The options a factory needs: everything except the transforms it builds. */
export type InstanceDataMeta = Omit<InstanceDataOptions, 'transforms' | 'transformStride'>;

export class InstanceData extends Resource {
  readonly name: string;
  readonly transformStride: number;
  /** The per-instance attribute layout implied by this data. */
  readonly layout: AttributeDefs;
  #transforms: Float32Array;
  #colors: Float32Array | null;
  #count: number;

  constructor(opts: InstanceDataOptions) {
    super();
    const { name = 'instances', transforms, colors = null, transformStride = TRANSFORM_STRIDE } = opts;
    const stride = assertStride(transformStride, name);
    const count = assertRun('transforms', transforms.length, stride);

    const colors32 = colors === null ? null : asFloats(colors);
    if (colors32 !== null) {
      const expected = count * INSTANCE_COLOR_COMPONENTS;
      if (colors32.length !== expected) {
        fail('INTERNAL_INVARIANT',
          `InstanceData "${name}" has ${colors32.length} colour components; ${count} instance${count === 1 ? '' : 's'} need exactly ${expected}.`, {
          why: `A per-instance colour is ${INSTANCE_COLOR_COMPONENTS} floats, so ${count} instances need exactly ${expected}. A short array writes past its end into the next record; a long one is silently ignored, and the instance you thought you coloured is not the instance that changed.`,
          fix: `Pass ${expected} numbers, or omit \`colors\` entirely.`,
          detail: { kind: 'numeric', field: 'colors', value: colors32.length, min: expected, max: expected },
        });
      }
    }

    this.name = name;
    this.transformStride = stride;
    this.layout = colors32 === null ? INSTANCE_ATTRIBUTES : INSTANCE_ATTRIBUTES_COLORED;
    this.#transforms = asFloats(transforms);
    this.#colors = colors32;
    this.#count = count;
  }

  /** How many instances this data holds. Grown by {@link reserve}. */
  get count(): number {
    return this.#count;
  }

  /**
   * The transform runs, `count * transformStride` floats, column-major.
   *
   * A getter because {@link reserve} replaces the array: a typed array cannot
   * grow, so growing means allocating a new one and copying. A caller holding
   * `data.transforms` across a `reserve()` is holding the old one — re-read the
   * property after growing.
   */
  get transforms(): Float32Array {
    return this.#transforms;
  }

  /** The colour runs, 4 floats per instance, or null when this data has none. */
  get colors(): Float32Array | null {
    return this.#colors;
  }

  static fromMatrices(matrices: ArrayLike<number>, opts: InstanceDataMeta = {}): InstanceData {
    assertFlatRun(matrices, 'fromMatrices');
    return new InstanceData({ ...opts, transforms: asFloats(matrices) });
  }

  /**
   * Builds the transforms from a translation, scale, and rotation per instance.
   *
   * The ergonomic path: 5000 placed cubes from 15000 numbers, with no matrix
   * maths in the caller. The composition is `T · R · S` — scale first, then
   * rotate, then translate — so the rotation is about the instance's own origin
   * and the scale is applied in the instance's own axes.
   *
   * `rotations` are **quaternions**, four floats each, not Euler angles: an
   * Euler triple has a gimbal singularity, a six-ordering ambiguity, and no
   * `slerp`. `scales` may be a single number for a uniform scale.
   */
  static fromTRS(
    translations: ArrayLike<number>,
    scales?: ArrayLike<number> | number,
    rotations?: ArrayLike<number>,
    opts: InstanceDataMeta = {},
  ): InstanceData {
    assertFlatRun(translations, 'fromTRS(translations)');
    const count = translations.length / 3;
    if (!Number.isInteger(count)) {
      fail('INTERNAL_INVARIANT',
        `fromTRS() got ${translations.length} translation components; a position is 3 floats.`, {
        why: 'The count of instances is the length divided by 3, so a length that is not a multiple of 3 means the walk below would read past the end of the array and write a matrix full of undefined.',
        fix: `Pass ${Math.ceil(translations.length / 3) * 3} numbers, one xyz triple per instance.`,
      });
    }

    if (scales !== undefined && typeof scales !== 'number') {
      assertFlatRun(scales, 'fromTRS(scales)');
      if (scales.length !== count * 3) {
        fail('INTERNAL_INVARIANT',
          `fromTRS() got ${scales.length} scale components for ${count} instances.`, {
          why: 'A scale is 3 floats (or a single number applied to all three axes), so a length that disagrees with the translation count describes a different set of instances.',
          fix: `Pass ${count * 3} numbers, one xyz triple per instance, or a single number for a uniform scale.`,
        });
      }
    }

    let rotationCount = 0;
    if (rotations !== undefined) {
      assertFlatRun(rotations, 'fromTRS(rotations)');
      rotationCount = rotations.length / 4;
      if (!Number.isInteger(rotationCount) || rotationCount !== count) {
        fail('INTERNAL_INVARIANT',
          `fromTRS() got ${rotations.length} rotation components — ${rotationCount} quaternion${rotationCount === 1 ? '' : 's'} — for ${count} instances.`, {
          why: 'A rotation is a quaternion: 4 floats (x, y, z, w), one per instance, all of them or none of them. A partial set would leave the remaining instances rotated by whatever happened to be in the scratch matrix.',
          fix: `Pass ${count * 4} numbers, or omit \`rotations\` for an unrotated set.`,
        });
      }
    }

    const out = new Float32Array(count * TRANSFORM_STRIDE);

    for (let i = 0, t = 0, c = 0; i < count; i++, t += 3, c += 3) {
      fromTranslation(_t, translations[t], translations[t + 1], translations[t + 2]);
      if (rotations === undefined) {
        identity(_r);
      } else {
        const q = i * 4;
        _q[0] = rotations[q];
        _q[1] = rotations[q + 1];
        _q[2] = rotations[q + 2];
        _q[3] = rotations[q + 3];
        fromQuat(_r, _q);
      }
      // The scale goes on the right of R, not on the finished product: `scale`
      // multiplies every column including the homogeneous one, so applied after
      // the translation it would scale the translation too, and applied to a
      // matrix that already has one it would scale the `w` of every vertex.
      if (scales !== undefined) {
        if (typeof scales === 'number') fromScale(_s, scales, scales, scales);
        else fromScale(_s, scales[c], scales[c + 1], scales[c + 2]);
        mul(_r, _r, _s);
      }
      mul(_o, _t, _r);
      const o = i * TRANSFORM_STRIDE;
      for (let k = 0; k < TRANSFORM_STRIDE; k++) out[o + k] = _o[k];
    }

    return new InstanceData({ ...opts, transforms: out });
  }

  /** Copies `m`'s `transformStride` floats into instance `index`. */
  setTransform(index: number, m: ArrayLike<number>): void {
    this.#assertIndex(index, 'setTransform');
    if (m.length < this.transformStride) {
      fail('INTERNAL_INVARIANT',
        `setTransform(${index}) on "${this.name}" got ${m.length} floats; a transform is ${this.transformStride}.`, {
        why: 'The transform is copied a full stride at a time, so a short source leaves the tail of the record holding whatever the previous instance wrote there.',
        fix: `Pass a length-${this.transformStride} matrix — \`mat4.create()\`, a node's \`world\`, or a 16-element view of a uniform block.`,
        detail: { kind: 'numeric', field: 'transform', value: m.length, min: this.transformStride, max: this.transformStride },
      });
    }
    const base = index * this.transformStride;
    for (let k = 0; k < this.transformStride; k++) this.#transforms[base + k] = m[k];
  }

  /**
   * Writes instance `index`'s colour.
   *
   * Fails on data built without a colour array, because silently dropping the
   * write is the outcome that produces "the tint does nothing" with nothing in
   * the log.
   */
  setColor(index: number, r: number, g: number, b: number, a = 1): void {
    const colors = this.#colors;
    if (colors === null) {
      fail('ATTRIBUTE_MISSING',
        `setColor(${index}) on "${this.name}", which has no per-instance colours and therefore no "${INSTANCE_COLOR}" attribute to write.`, {
        why: `A colour is the "${INSTANCE_COLOR}" attribute of the per-instance buffer. Without it in the data there is nowhere to write, so the value would be dropped and the instance would keep the colour it had.`,
        fix: `Pass \`colors\` when constructing the InstanceData (4 floats per instance), or build the set with fromMatrices({ ..., colors }). A material compiled for instance colours expects all of them to have one.`,
      });
    }
    this.#assertIndex(index, 'setColor');
    const base = index * INSTANCE_COLOR_COMPONENTS;
    colors[base] = r;
    colors[base + 1] = g;
    colors[base + 2] = b;
    colors[base + 3] = a;
  }

  /**
   * Grows to hold `count` instances, preserving everything already written.
   *
   * New instances start at identity, not at zero. A zeroed transform is
   * `[0,0,0,0]` in every column: `w` is 0, the perspective divide divides by it,
   * and the instance vanishes — or, worse, survives as a smear of garbage
   * vertices across the clip volume. Identity is the only value that draws
   * something, and it is what "not configured yet" should look like.
   *
   * A growth to a **smaller** count is refused rather than ignored. Shrinking
   * would mean either dropping instances the caller still believes are there, or
   * silently keeping a larger `count` than the caller asked for; both make a
   * later `setTransform` address the wrong record. Resize the mesh instead —
   * count is a property of the data, not a buffer that has to be preallocated.
   */
  reserve(count: number): void {
    if (!Number.isInteger(count) || count < 0) {
      fail('INTERNAL_INVARIANT',
        `reserve(${count}) on "${this.name}"; an instance count is a non-negative integer.`, {
        why: 'The count indexes every array in this object. A fractional or negative one produces a length no allocation can satisfy, and `new Float32Array(-1)` throws a bare RangeError from the engine rather than an apse error.',
        fix: 'Pass a whole number of instances, e.g. the length of the position array you are about to fill.',
      });
    }
    if (count < this.count) {
      fail('INTERNAL_INVARIANT',
        `reserve(${count}) on "${this.name}", which already holds ${this.count} instances.`, {
        why: 'Shrinking would drop instances the caller still holds transforms for, or keep a count larger than the one that was asked for. Either way a later setTransform addresses a record the caller is not looking at.',
        fix: `Build a new InstanceData with the smaller count, or leave this one alone — the buffer is allocated from \`count\` at upload time, so nothing is reserved in advance and a smaller set costs nothing.`,
        detail: { kind: 'numeric', field: 'count', value: count, min: this.count, max: Number.MAX_SAFE_INTEGER },
      });
    }
    if (count === this.#count) return;

    const stride = this.transformStride;
    const grown = new Float32Array(count * stride);
    grown.set(this.#transforms);
    // Identity in every new record: 1 on each diagonal, 0 elsewhere.
    for (let i = this.#count; i < count; i++) {
      const o = i * stride;
      grown[o] = 1;
      grown[o + 5] = 1;
      grown[o + 10] = 1;
      grown[o + 15] = 1;
    }
    this.#transforms = grown;

    if (this.#colors !== null) {
      const tinted = new Float32Array(count * INSTANCE_COLOR_COMPONENTS);
      tinted.set(this.#colors);
      // White, not transparent black: an unconfigured tint should look like the
      // material's own colour, not like an object that failed to render.
      for (let i = this.#count * INSTANCE_COLOR_COMPONENTS; i < tinted.length; i++) {
        tinted[i] = 1;
      }
      this.#colors = tinted;
    }
    this.#count = count;
  }

  #assertIndex(index: number, what: string): void {
    if (Number.isInteger(index) && index >= 0 && index < this.count) return;
    const empty = this.count === 0;
    fail('INTERNAL_INVARIANT',
      `${what}(${index}) on "${this.name}", which holds ${this.count} instance${empty ? '' : this.count === 1 ? '' : 's'}.`, {
      why: `The write is a fixed offset into a typed array of exactly ${this.count} record${empty ? '' : 's'}, so ${empty ? 'there is no valid index at all' : `valid indices are 0..${this.count - 1}`}. An out-of-range index either writes into the following record or past the end of the array, and neither produces a visible error.`,
      fix: empty
        ? `Call reserve(${index + 1}) before writing, or build the InstanceData with its instances already in it.`
        : `Check the index against count (${this.count}) — a loop bound of \`positions.length / 3\` is the usual source of an off-by-one.`,
      detail: { kind: 'numeric', field: 'instanceIndex', value: index, min: 0, max: this.count - 1 },
    });
  }

  protected onDispose(): void {
    // Nothing on the device. The typed arrays become collectable as soon as the
    // last reference drops, and plain memory is collectable deterministically —
    // which is the one thing a GPU buffer is not, and the reason apse does not
    // leave the upload to the collector.
  }
}

// ---------------------------------------------------------------------------
// GpuInstances
// ---------------------------------------------------------------------------

export interface GpuInstancesOptions {
  /** Only used in error messages and tooling. Defaults to the data's name. */
  readonly name?: string;
  /**
   * The mesh's vertex attributes, for the vertex half of {@link GpuInstances.layout}.
   *
   * Instance data knows nothing about the mesh it is drawn with, so the vertex
   * half of the combined layout defaults to {@link STANDARD_ATTRIBUTES}. Pass
   * the mesh's attributes to make the comparison exact when the mesh does not
   * use them.
   */
  readonly vertexAttributes?: AttributeDefs;
  /**
   * The per-instance attribute vocabulary, in place of the one the data implies.
   *
   * The point of the option is that the byte offsets of the record come from
   * *here* — apse packs transforms and colours through the offsets the layout
   * resolved, so a layout that declares the colour first gets the colour
   * written first. A packer with the order hard-coded would be correct for
   * exactly one of the two layouts and quietly wrong for the other.
   */
  readonly instanceAttributes?: AttributeDefs;
}

/** Resolved layout of one instance record: which floats sit where. */
interface RecordShape {
  /** Floats per record in the GPU buffer. */
  readonly floats: number;
  /** Float offset of each of the four transform columns, or −1 when absent. */
  readonly transform: readonly number[];
  /** Float offset of the colour, or −1 when there is none. */
  readonly color: number;
}

/**
 * Reads the record shape out of a layout, so the packer and the pipeline are
 * generated from one description.
 *
 * A missing transform column is a hard error rather than a silent zero: three
 * columns produce a `mat4x4f` whose fourth column is `w = 0`, every vertex
 * divides by it, and the instances vanish.
 */
function recordShape(instanceAttributes: AttributeDefs): RecordShape {
  const found: number[] = [-1, -1, -1, -1];
  let color = -1;
  let stride = 0;
  for (const [name, format] of Object.entries(instanceAttributes)) {
    const byteSize = vertexFormat(format, name).byteSize;
    stride = (stride + 3) & ~3;
    // In floats, because every consumer of this shape indexes a `Float32Array`
    // with it. `stride` itself is walked in bytes — that is what a buffer layout
    // is expressed in — and converted once here, rather than at each of the three
    // places that would otherwise each have to remember to.
    const at = stride >> 2;
    if (name === INSTANCE_COLOR) color = at;
    const column = Object.keys(TRANSFORM_ATTRIBUTES).indexOf(name);
    // Declaration order is the column order, so the two agree by construction
    // and a mismatch is impossible rather than checked for.
    if (column >= 0) found[column] = at;
    stride += byteSize;
  }

  if (found.some((o) => o < 0)) {
    fail('INTERNAL_INVARIANT',
      `A per-instance layout needs all four of ${Object.keys(TRANSFORM_ATTRIBUTES).join(', ')}; this one declares ${Object.keys(instanceAttributes).join(', ')}.`, {
      why: 'The transform is a `mat4x4f` and WebGPU has no matrix vertex format, so it arrives as four `float32x4` attributes. Fewer than four means the generated `mat4x4f(v.instanceTransform0, …)` gets a zero for a column, `w` is 0, and every instance collapses to a divide by zero rather than failing to compile.',
      fix: 'Use `INSTANCE_ATTRIBUTES` (four columns) or `INSTANCE_ATTRIBUTES_COLORED` (those four plus `instanceColor`). A custom instance vocabulary has to declare all four transform columns.',
    });
  }

  return { floats: stride >> 2, transform: found, color };
}

/**
 * The uploaded form of {@link InstanceData}: one `GPUBuffer` and a count.
 *
 * ## Why the CPU side is retained, unlike a mesh
 *
 * {@link GpuMesh} deliberately drops its `MeshData`, because a mesh is written
 * once and never again. An instance buffer is the opposite: a particle system
 * or a crowd writes into it every frame, and the only way to re-upload a *range*
 * of it is to still have the CPU half to read the range out of. So this retains
 * its {@link InstanceData} — which costs 64 bytes per instance of arrays the
 * application was building anyway, and saves a full re-upload every frame for
 * the one instance out of five thousand that moved.
 *
 * The record the packer writes is `data.transforms` itself in the
 * transform-only case, so nothing is duplicated. A coloured set needs a real
 * interleaved record, and that is the one case where the CPU holds a second
 * copy — 80 bytes per instance, against 80 bytes of device memory, and it buys
 * {@link markDirty}.
 */
export class GpuInstances extends Resource {
  readonly name: string;
  readonly buffer: GPUBuffer;
  readonly count: number;
  /** The combined layout: the mesh's vertex attributes plus these instances. */
  readonly layout: VertexLayout;
  /** Bytes per instance in `buffer`. */
  readonly instanceStride: number;
  /** Total bytes held on the device. */
  readonly byteLength: number;
  /** The CPU half, retained so a dirty range can be re-uploaded. */
  readonly data: InstanceData;

  readonly #queue: GPUQueue;
  /** The interleaved record. `data.transforms` itself when there is no colour. */
  readonly #record: Float32Array;
  readonly #floats: number;
  #lo = 0;
  #hi = 0;

  constructor(device: GPUDevice, data: InstanceData, opts: GpuInstancesOptions = {}) {
    super();
    const name = opts.name ?? data.name;
    if (data.transformStride !== TRANSFORM_STRIDE) {
      fail('INTERNAL_INVARIANT',
        `InstanceData "${data.name}" has a transform stride of ${data.transformStride} floats, which cannot be uploaded.`, {
        why: `The generated per-instance buffer layout is four consecutive \`float32x4\`s, so its stride is ${TRANSFORM_STRIDE} floats. A wider record needs padding between the attributes, and an AttributeDefs map — which is what the layout is generated from — has no way to say "then leave a gap". Uploading it anyway would reinterpret the padding as transform data.`,
        fix: 'Drop `transformStride` and let the record be 16 floats, or build and upload the buffer yourself with a hand-written GPUVertexBufferLayout whose arrayStride matches.',
      });
    }

    const attributes = opts.instanceAttributes ?? data.layout;
    const shape = recordShape(attributes);
    const perInstance = shape.floats;
    const byteLength = data.count * perInstance * 4;
    const max = device.limits.maxBufferSize;
    if (byteLength > max) {
      fail('MESH_DATA_TOO_LARGE',
        `Instance data "${name}" needs ${byteLength} bytes on one buffer, over the device limit of ${max}.`, {
        why: 'A WebGPU buffer cannot exceed `maxBufferSize`, and a per-instance buffer is one contiguous run: the vertex stage reads it by stepping, so it cannot be split across the limit.',
        fix: 'Draw fewer instances per buffer — split the set into several InstanceData uploads — or drop the per-instance colour, which is 16 of every 80 bytes.',
      });
    }

    // `writeBuffer` rather than `mappedAtCreation`, which is what `GpuMesh` uses.
    // A mesh is written once and never again; an instance buffer is expected to
    // be re-written every frame from a moving transform list, and `writeBuffer`
    // is the call designed for that — a mapped-at-creation buffer would be
    // unmap/re-map per frame at best, and a create/destroy pair per frame at
    // worst.
    const buffer = device.createBuffer({
      label: `${name}:instance`,
      // A zero-size buffer is legal but useless, and an empty instance set is
      // legal data, so the floor keeps `createBuffer` well-defined either way.
      size: byteLength === 0 ? 4 : byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const record = packRecord(data, shape, perInstance);
    // The cast is a lib-version artifact, not a runtime one: TypeScript 5.7 made
    // the typed-array classes generic over their buffer, and the installed
    // @webgpu/types still asks for a view over a plain `ArrayBuffer`. Every
    // array apse allocates is one, and the caller's is too unless they handed
    // us a view onto a shared buffer, which `writeBuffer` accepts at runtime.
    device.queue.writeBuffer(buffer, 0, record as Float32Array<ArrayBuffer>, 0, data.count * perInstance);

    this.name = name;
    this.buffer = buffer;
    this.count = data.count;
    this.layout = layout(opts.vertexAttributes ?? STANDARD_ATTRIBUTES, 2048, attributes);
    this.instanceStride = perInstance * 4;
    this.byteLength = byteLength;
    this.data = data;
    this.#queue = device.queue;
    this.#record = record;
    this.#floats = perInstance;
  }

  // -------------------------------------------------------------------------
  // Partial re-upload
  // -------------------------------------------------------------------------

  /**
   * Marks `count` instances starting at `first` for re-upload by {@link flush}.
   *
   * Coalesces: marking every instance of a 5000-strong set one at a time
   * produces a single `writeBuffer` covering all of them, which is the case
   * that matters, and marking one produces a 64-byte write, which is the case
   * that is worth having. Mark everything with {@link markAllDirty} instead of
   * looping, which is the same thing with a smaller constant factor.
   */
  markDirty(first: number, count: number): void {
    this.#lo = this.#lo === this.#hi ? first : Math.min(this.#lo, first);
    this.#hi = Math.max(this.#hi, first + count);
  }

  /** Marks the whole buffer. The path for a caller that wrote into `data.transforms` directly. */
  markAllDirty(): void {
    this.#lo = 0;
    this.#hi = this.count;
  }

  /** The dirty span as `[first, count)`, or null when nothing is marked. */
  get dirtyRange(): readonly [number, number] | null {
    return this.#hi > this.#lo ? [this.#lo, this.#hi - this.#lo] : null;
  }

  /**
   * Uploads the marked span and clears it. Returns true if it wrote anything.
   *
   * The write is `writeBuffer(buffer, first * instanceStride, record, first *
   * floats, count * floats)`, so the GPU only sees the bytes that changed and
   * the CPU only walks the records that changed.
   */
  flush(): boolean {
    const range = this.dirtyRange;
    if (range === null) return false;
    this.#queue.writeBuffer(
      this.buffer,
      range[0] * this.instanceStride,
      this.#record as Float32Array<ArrayBuffer>,
      range[0] * this.#floats,
      range[1] * this.#floats,
    );
    this.#lo = 0;
    this.#hi = 0;
    return true;
  }

  /**
   * The record bytes for one instance, as the layout would read them.
   *
   * Exposed for tooling and for tests: it is the CPU half of the same contract
   * `writeBuffer` fulfils, decoded through the layout's own offsets, so a test
   * can prove the upload and the pipeline agree without a GPU.
   */
  recordFloats(index: number): Float32Array {
    const out = new Float32Array(this.#floats);
    out.set(this.#record.subarray(index * this.#floats, (index + 1) * this.#floats));
    return out;
  }

  protected onDispose(): void {
    this.buffer.destroy();
  }
}

/**
 * Uploads {@link InstanceData} as a per-instance vertex buffer.
 *
 * The data is not retained and not copied: `writeBuffer` copies at the call, so
 * the CPU arrays are free the moment this returns. Keep them only if you
 * intend to re-upload, which is the point of writing into them in place.
 */
export function uploadInstances(
  device: GPUDevice,
  data: InstanceData,
  opts: GpuInstancesOptions = {},
): GpuInstances {
  return new GpuInstances(device, data, opts);
}

// ---------------------------------------------------------------------------
// Scratch and validation
// ---------------------------------------------------------------------------

/**
 * Builds the interleaved instance record, offsets taken from the layout.
 *
 * With no colour attribute the record *is* `data.transforms` — four consecutive
 * `float32x4`s, so a flat 16-float run is already in buffer order and the
 * upload is the caller's array with no copy at all. That is 320 KB a frame at
 * 5000 instances that a copy would cost, and it is why the transform-only
 * layout is the one apse generates by default.
 *
 * The coloured case allocates, because the record has to outlive this call:
 * it is what a dirty range is later re-uploaded from.
 */
function packRecord(data: InstanceData, shape: RecordShape, floats: number): Float32Array {
  if (shape.color < 0) return data.transforms;

  const count = data.count;
  const out = new Float32Array(count * floats);
  const t = data.transforms;
  const colors = data.colors as Float32Array;
  for (let i = 0, o = 0, s = 0, c = 0; i < count; i++, o += floats, s += TRANSFORM_STRIDE, c += INSTANCE_COLOR_COMPONENTS) {
    for (let k = 0; k < TRANSFORM_STRIDE; k++) out[o + shape.transform[k]] = t[s + k];
    for (let k = 0; k < INSTANCE_COLOR_COMPONENTS; k++) out[o + shape.color + k] = colors[c + k];
  }
  return out;
}

function asFloats(source: Float32Array | ArrayLike<number>): Float32Array {
  if (source instanceof Float32Array) return source;
  return Float32Array.from(source as ArrayLike<number>);
}

function assertStride(stride: number, name: string): number {
  if (Number.isInteger(stride) && stride >= TRANSFORM_STRIDE && stride % 4 === 0) return stride;
  fail('INTERNAL_INVARIANT',
    `InstanceData "${name}" declares a transform stride of ${stride} floats.`, {
    why: `A transform is ${TRANSFORM_STRIDE} floats, and a wider record only makes sense as padding of whole 4-byte attribute boundaries — WebGPU rejects a vertex attribute offset that is not 4-byte aligned.`,
    fix: `Use ${TRANSFORM_STRIDE}, or a multiple of 4 above it. Only ${TRANSFORM_STRIDE} can be uploaded through uploadInstances(); see that function for why.`,
    detail: { kind: 'numeric', field: 'transformStride', value: stride, min: TRANSFORM_STRIDE, max: 2048 },
  });
}

function assertRun(what: string, length: number, stride: number): number {
  if (length % stride === 0) return length / stride;
  fail('INTERNAL_INVARIANT',
    `${what} has ${length} floats, which is not a whole number of ${stride}-float records.`, {
    why: 'The instance count is this length divided by the record size, so a remainder means the count is fractional and every index computed from it addresses a record that does not exist. Rounding it down would silently drop the last instance; rounding up would read past the end.',
    fix: `Pass a multiple of ${stride} floats.`,
    detail: { kind: 'numeric', field: what, value: length, min: 0, max: stride },
  });
}

/**
 * Rejects a nested array where a flat run of floats is required.
 *
 * `fromMatrices` takes one flat run, not a list of matrices. A caller who
 * passes `Float32Array[]` would otherwise get a count of `matrices.length / 16`
 * and a walk through the arrays as if they were numbers — `NaN` for every
 * component, with no error anywhere.
 */
function assertFlatRun(source: ArrayLike<number>, what: string): void {
  if (source.length === 0) return;
  if (typeof source[0] === 'number') return;
  fail('INTERNAL_INVARIANT',
    `${what} was given an array of arrays, not a flat run of floats.`, {
    why: 'Every instance argument in apse is one contiguous typed array, which is what lets the upload be a single writeBuffer with no repacking. A nested array would have to be flattened first, and silently reading it as floats fills every matrix with NaN.',
    fix: 'Flatten it: `new Float32Array(matrices.flatMap((m) => [...m]))`, or build the runs one after another with `set()`.',
  });
}

// T · R · S composition scratch. Module-level so building 5000 instances
// allocates nothing but the result — this is a construction path, but a
// construction path that runs per scene load and per buffer rebuild.
const _t = create();
const _r = create();
const _s = create();
const _o = create();
const _q = new Float32Array(4);
