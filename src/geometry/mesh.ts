/**
 * Mesh data — the CPU side of geometry, and the GPU buffers that mirror it.
 *
 * Two classes, deliberately separated:
 *
 *   {@link MeshData}  pure data. Interleaved, layout-packed, with a bounding
 *                     sphere already computed. No GPU, no device, no lifecycle
 *                     beyond its own reference count. Every primitive in
 *                     `apse/geometry` returns one.
 *
 *   {@link GpuMesh}   the uploaded form. Owns two `GPUBuffer`s and nothing
 *                     else, so disposing it is two `destroy()` calls and
 *                     cannot leave a half-torn-down mesh.
 *
 * The split is what makes primitives testable without a device. `sphere()` in a
 * `bun test` process is the same code path the renderer runs, not a parallel
 * CPU-only implementation that is allowed to drift.
 */

import { fail } from '../core/error.ts';
import { Resource } from '../core/resource.ts';
import { create, set, type Sphere } from '../math/sphere.ts';
import type { PrimitiveTopology } from '../render/pipeline-state.ts';
import type { DrawableGeometry } from '../render/types.ts';
import { GpuInstances } from './instanced.ts';
import type { VertexLayout } from './layout.ts';

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/**
 * Where a mesh's vertex data comes from.
 *
 * `interleaved` is the fast path: bytes that are already in `layout.stride`
 * order, taken as-is with no copy. Use it when the data came out of a file, a
 * glTF, or another engine.
 *
 * `attributes` is the ergonomic path: one dense array per attribute, which this
 * module packs into a single interleaved buffer.
 */
export type MeshSource =
  | { readonly interleaved: Float32Array; readonly vertexCount: number }
  | {
      readonly attributes: Record<string, ArrayLike<number>>;
      readonly vertexCount: number;
    };

export interface MeshDataOptions {
  /** Only used in error messages and tooling. */
  readonly name?: string;
  readonly layout: VertexLayout;
  readonly vertices: MeshSource;
  /** `null` for a non-indexed mesh, in which case `vertexCount` is the draw length. */
  readonly indices?: Uint32Array | Uint16Array | null;
  /** Override the computed bounds. Only valid if you know better than the vertex data. */
  readonly boundingSphere?: ArrayLike<number>;
  /** Defaults to `'triangle-list'`, which is every primitive apse ships. */
  readonly topology?: PrimitiveTopology;
}

// ---------------------------------------------------------------------------
// Bounding sphere
// ---------------------------------------------------------------------------

// Module-level scratch. Bounding-sphere construction must allocate nothing:
// mesh construction happens in batches, and a per-mesh or per-vertex allocation
// here would hand the collector more garbage than the rest of the module.
const scratchSphere: Sphere = create();

/** Packs a math-layer `Sphere` into the `[x, y, z, r]` form meshes expose. */
function packSphere(s: Sphere, out: Float32Array): Float32Array {
  out[0] = s.center[0];
  out[1] = s.center[1];
  out[2] = s.center[2];
  out[3] = s.radius;
  return out;
}

/**
 * Two-pass bounding sphere: AABB centre, then the farthest vertex from it.
 *
 * The centre does not have to be the centroid. Any point inside the convex hull
 * gives a sphere that contains the mesh, and the AABB centre is found in one
 * pass with six comparisons per vertex instead of a divide per vertex.
 *
 * # Why this is computed here and not during culling
 *
 * The obvious alternative is to compute the bounds in the frustum test, which is
 * where the values are first read. That is the hottest loop in a renderer: it
 * runs for every object on every frame, including the ones that are culled. Work
 * done there is paid 60 times a second for a value that cannot change while the
 * vertex buffer is static — and paid again on every re-upload, every
 * re-allocation, and every inspector refresh.
 *
 * A skinned or morphed mesh is the case that genuinely needs dynamic bounds,
 * and it will be an explicit `updateBounds()` on `MeshData` when morph targets
 * exist. Until then the honest answer is: the mesh does not change, so neither
 * does its sphere, so this is a one-time cost paid in the constructor.
 *
 * No morph-target expansion is applied, because apse has no morph targets to
 * expand for. A rigged mesh calls back in rather than paying for it on every
 * static mesh in the scene.
 */
function computeBoundingSphere(
  out: Float32Array,
  vertexData: Float32Array,
  vertexCount: number,
  floatsPerVertex: number,
  positionOffset: number,
  name: string,
): Float32Array {
  if (vertexCount === 0) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    return out;
  }

  // Pass 1 — AABB.
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let base = positionOffset;
  for (let i = 0; i < vertexCount; i++) {
    const x = vertexData[base];
    const y = vertexData[base + 1];
    const z = vertexData[base + 2];
    // NaN fails every `<` and every `>`, so a mesh with one NaN vertex produces a
    // *finite* AABB that is missing it — and an under-estimated bound culls an
    // object that is still on screen, with nothing reported. The finiteness test
    // has to be explicit, which is why it is a separate branch and not a
    // comparison.
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      fail('INVALID_USAGE',
        `Mesh "${name}" has a non-finite position at vertex ${i}: (${x}, ${y}, ${z}).`, {
        why: 'The bounding sphere is a min/max reduction over the positions, and a NaN compares false against everything, so it passes through the reduction without moving either end. The sphere that comes out is finite, looks right, and does not contain the mesh — which culls a visible object on every frame with no error anywhere.',
        fix: `Find vertex ${i} and fix the number. \`Number.isFinite\` on the source array before constructing the MeshData is the cheap version of this check; apse raises it because the failure it prevents is a bound that silently drops geometry.`,
      });
    }
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
    base += floatsPerVertex;
  }

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  set(scratchSphere, cx, cy, cz, 0);

  // Pass 2 — farthest vertex from that centre.
  let maxSq = 0;
  base = positionOffset;
  for (let i = 0; i < vertexCount; i++) {
    const dx = vertexData[base] - cx;
    const dy = vertexData[base + 1] - cy;
    const dz = vertexData[base + 2] - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > maxSq) maxSq = d2;
    base += floatsPerVertex;
  }

  return packSphere(set(scratchSphere, cx, cy, cz, Math.sqrt(maxSq)), out);
}

// ---------------------------------------------------------------------------
// MeshData
// ---------------------------------------------------------------------------

/**
 * CPU-side geometry. Immutable after construction, and free of any GPU object.
 *
 * `vertexData` is already interleaved to `layout.stride`, so uploading it is a
 * single `writeBuffer` and the pipeline needs one `setVertexBuffer` call no
 * matter how many attributes the layout has.
 */
export class MeshData extends Resource {
  readonly name: string;
  readonly layout: VertexLayout;
  readonly vertexData: Float32Array;
  readonly indexData: Uint32Array | Uint16Array | null;
  readonly vertexCount: number;
  /** `[cx, cy, cz, r]`. Computed once, in the constructor. */
  readonly boundingSphere: Float32Array;
  readonly topology: PrimitiveTopology;
  /** True when `indexData` is non-null. */
  readonly indexed: boolean;

  #indexCount: number;

  constructor(opts: MeshDataOptions) {
    super();
    const {
      name = 'mesh',
      layout,
      vertices,
      indices = null,
      boundingSphere: override,
      topology = 'triangle-list',
    } = opts;

    const vertexCount = vertices.vertexCount;
    if (!Number.isInteger(vertexCount) || vertexCount <= 0) {
      fail('MESH_EMPTY',
        `Mesh "${name}" declares ${vertexCount} vertices.`, {
        why: 'A mesh with no vertices has no drawable data, so there is nothing to encode and nothing to draw.',
        fix: 'Populate the geometry before constructing a MeshData, or skip adding the object to the scene while it is still being built.',
      });
    }

    const position = layout.attribute('position');
    if (position === undefined) {
      fail('MESH_NO_POSITION',
        `Mesh "${name}" uses a layout with no "position" attribute.`, {
        why: 'Every material needs a position to place vertices, and without one there is nothing for the vertex stage to transform.',
        fix: 'Build the layout from `STANDARD_LAYOUT`, or declare at least `position: "float32x3"`.',
      });
    }
    // The bounds pass below reads three consecutive floats at the position
    // offset, and so does every primitive's writer. A two-component position
    // does not fail loudly here — it makes z the *next* attribute's first
    // component, so the bound describes a mesh that is not this one.
    if (position.info.components < 3) {
      fail('MESH_NO_POSITION',
        `Mesh "${name}" declares "position" with ${position.info.components} components, and a position is at least 3.`, {
        why: `Every consumer in apse reads a position as three floats at the offset the layout resolved: the bounding-sphere reduction here, and the primitives\' own scalar stores. "${position.format}" is ${position.info.components} wide, so the bounds pass would read the following attribute's first component as z and produce a finite, plausible, wrong bound.`,
        fix: 'Declare `position: "float32x3"`. Every layout apse ships does, and a position is not optional: without a correct one there is nothing to bound and nothing to place.',
      });
    }

    const floatsPerVertex = layout.stride >> 2;
    const positionOffset = position.offset >> 2;
    const vertexData = 'interleaved' in vertices
      ? validateInterleaved(name, layout, vertices.interleaved, vertexCount)
      : packAttributes(name, layout, vertices.attributes, vertexCount);

    // A non-indexed triangle-list is drawn three vertices at a time, so a count
    // that is not a multiple of 3 leaves a tail triangle the driver would have
    // to invent. It is a data error, not something to paper over.
    if (indices === null && topology === 'triangle-list' && vertexCount % 3 !== 0) {
      fail('MESH_INDEX_MISALIGNED',
        `Mesh "${name}" has ${vertexCount} vertices and no index buffer, which is not a multiple of 3.`, {
        why: 'A triangle-list draws whole triangles from consecutive vertices, so the last group would be an incomplete triangle the driver has to discard — or, worse, read past the end of the buffer.',
        fix: 'Supply an index buffer, or pad the vertex count up to a multiple of 3.',
      });
    }

    this.name = name;
    this.layout = layout;
    this.vertexData = vertexData;
    this.indexData = indices;
    this.vertexCount = vertexCount;
    this.topology = topology;
    this.indexed = indices !== null;
    this.#indexCount = indices === null ? vertexCount : indices.length;

    this.boundingSphere = override !== undefined
      ? copySphere(override, name)
      : computeBoundingSphere(new Float32Array(4), vertexData, vertexCount, floatsPerVertex, positionOffset, name);
  }

  /** Number of indices to draw, or the vertex count when the mesh is not indexed. */
  get indexCount(): number {
    return this.#indexCount;
  }

  /** Bytes the vertex buffer will occupy. */
  get vertexByteLength(): number {
    return this.layout.byteLength(this.vertexCount);
  }

  /** Bytes the index buffer will occupy, or 0 when the mesh is not indexed. */
  get indexByteLength(): number {
    return this.indexData === null ? 0 : this.indexData.byteLength;
  }

  protected onDispose(): void {
    // No GPU object to release. The typed arrays become collectable as soon as
    // the last reference drops, and GC is deterministic for plain memory —
    // the reason apse does not rely on it for GPU buffers does not apply here.
  }
}

function copySphere(source: ArrayLike<number>, name: string): Float32Array {
  if (source.length < 4) {
    fail('INVALID_USAGE',
      `Mesh "${name}" was given a ${source.length}-element bounding sphere; it is [x, y, z, r], four numbers.`, {
      why: 'Every consumer in apse reads the bounds as a packed vec4, and a short array silently reads undefined for the missing components.',
      fix: 'Pass 4 numbers: [centreX, centreY, centreZ, radius]. Omit the option entirely to have apse compute it.',
    });
  }
  // A bound is an over-estimate by contract, and an over-estimate is a number
  // that has to be a number. A NaN radius fails the sphere test every time and
  // a negative one fails it on the inside — neither culls the object, both leave
  // the frustum test to compute with a bound it cannot reason about.
  for (let k = 0; k < 4; k++) {
    if (!Number.isFinite(source[k])) {
      fail('INVALID_USAGE',
        `Mesh "${name}" has a non-finite bounding sphere: (${source[0]}, ${source[1]}, ${source[2]}, ${source[3]}).`, {
        why: 'The sphere is compared against a distance on every culled object of every frame. A non-finite component makes that comparison false, so the object is never culled and never reported wrong — it is simply outside the range where apse can reason about what it is doing.',
        fix: `The override must be four finite numbers. Omit \`boundingSphere\` to have apse compute it from the vertices, which is right unless you know better than the vertex data.`,
      });
    }
  }
  if (source[3] < 0) {
    fail('INVALID_USAGE',
      `Mesh "${name}" has a negative bounding-sphere radius: ${source[3]}.`, {
      why: 'The radius is a length. A negative one is not "a smaller bound", it is a bound that excludes the mesh\'s own vertices, and an under-estimated bound culls an object that is still on screen — the one failure mode of this value that is invisible.',
      fix: `Pass the distance from the centre to the farthest vertex, which is never negative. A zero radius is legal and means every vertex sits exactly on the centre.`,
      detail: { kind: 'numeric', field: 'boundingSphere.radius', value: source[3], min: 0 },
    });
  }
  return packSphere(set(create(), source[0], source[1], source[2], source[3]), new Float32Array(4));
}

/** The `interleaved` source is taken by reference, so only its size is checked. */
function validateInterleaved(
  name: string,
  layout: VertexLayout,
  data: Float32Array,
  vertexCount: number,
): Float32Array {
  const required = layout.byteLength(vertexCount) >> 2;
  if (data.length < required) {
    fail('INTERNAL_INVARIANT',
      `Mesh "${name}" has ${vertexCount} vertices of stride ${layout.stride}, which needs ${required} floats, but its interleaved buffer holds ${data.length}.`, {
      why: 'The layout is the authority on how many floats a vertex occupies. A short buffer means the two disagree, and every read past the end is a read of whatever happened to be allocated next.',
      fix: `Provide at least ${required} floats, or lower the vertex count. Declared attributes: ${layout.attributes.map((a) => a.name).join(', ')}.`,
    });
  }
  return data.length === required ? data : data.subarray(0, required);
}

function packAttributes(
  name: string,
  layout: VertexLayout,
  attributes: Record<string, ArrayLike<number>>,
  vertexCount: number,
): Float32Array {
  // Every attribute is checked against the same authority — vertexCount times
  // the format's component count — which is what makes inconsistent lengths
  // impossible to reach the hot loop.
  for (const attr of layout.attributes) {
    const src = attributes[attr.name];
    if (src === undefined) {
      fail('ATTRIBUTE_MISSING',
        `Mesh "${name}" declares layout attribute "${attr.name}" but the source has no data for it.`, {
        why: 'A pipeline built for this layout reads every one of its attributes at a fixed @location. An attribute with no data reads whatever bytes happen to sit at that offset.',
        fix: `Supply an array for every attribute the layout declares. The layout declares: ${layout.attributes.map((a) => a.name).join(', ')}.`,
      });
    }
    const expected = vertexCount * attr.info.components;
    if (src.length !== expected) {
      fail('INTERNAL_INVARIANT',
        `Mesh "${name}" attribute "${attr.name}" has ${src.length} components; the invariant is ${expected} (${vertexCount} vertices × ${attr.info.components} for ${attr.format}).`, {
        why: 'Interleaving walks each attribute in lockstep with the vertex count. A length that disagrees with vertexCount means the source arrays describe different meshes, and the extra vertices are read from another array or from nothing.',
        fix: `Give "${attr.name}" exactly ${expected} values, or correct the vertexCount. All attributes in a source must agree on the same count.`,
      });
    }
    // Both of apse's writers emit 32-bit floats, so a normalised or integer
    // format has no path to this buffer: the scalar store would write four f32
    // bytes where the GPU reads one `unorm8` and three zero bytes, and 1.0 would
    // arrive as 14/255. It compiles, it uploads, and it shades the wrong colour.
    requireFloat32(attr.name, attr.format, `The "attributes" source for mesh "${name}"`);
  }

  const out = layout.allocate(vertexCount);
  const floatsPerVertex = layout.stride >> 2;

  // One attribute at a time, walking the destination with a running index.
  // Scalar writes into the destination beat `.set()` with a temporary: there is
  // no intermediate to write twice, no bounds re-check per element, and the
  // loop stays in L1 because the destination stride is small and known.
  for (const attr of layout.attributes) {
    const src = attributes[attr.name];
    const base = attr.offset >> 2;
    const comps = attr.info.components;
    if (comps === 1) {
      for (let i = 0, d = base; i < vertexCount; i++, d += floatsPerVertex) out[d] = src[i];
    } else if (comps === 2) {
      for (let i = 0, k = 0, d = base; i < vertexCount; i++, k += 2, d += floatsPerVertex) {
        out[d] = src[k];
        out[d + 1] = src[k + 1];
      }
    } else if (comps === 3) {
      for (let i = 0, k = 0, d = base; i < vertexCount; i++, k += 3, d += floatsPerVertex) {
        out[d] = src[k];
        out[d + 1] = src[k + 1];
        out[d + 2] = src[k + 2];
      }
    } else {
      for (let i = 0, k = 0, d = base; i < vertexCount; i++, k += 4, d += floatsPerVertex) {
        out[d] = src[k];
        out[d + 1] = src[k + 1];
        out[d + 2] = src[k + 2];
        out[d + 3] = src[k + 3];
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Index buffers
// ---------------------------------------------------------------------------

/**
 * Allocates an index buffer of the narrowest type that can address
 * `vertexCount` vertices.
 *
 * 16-bit indices halve index bandwidth, which matters for the meshes that get
 * redrawn every frame; the threshold is 65535 because index 65535 itself is a
 * valid vertex reference.
 */
export function allocateIndices(
  vertexCount: number,
  indexCount: number,
): Uint16Array | Uint32Array {
  return vertexCount > 65535
    ? new Uint32Array(indexCount)
    : new Uint16Array(indexCount);
}

// ---------------------------------------------------------------------------
// VertexWriter — the path primitives use
// ---------------------------------------------------------------------------

/**
 * A write cursor over one mesh's interleaved buffer.
 *
 * Every primitive generates the same three things — position, normal, uv — and
 * every one of them may be absent from the target layout. Rather than have six
 * primitives each re-derive the offsets and each decide for itself what to skip,
 * this resolves that once, at construction, and the primitive's inner loop is
 * left with plain scalar stores:
 *
 * ```
 * const w = new VertexWriter(layout, count);
 * const d = w.data;
 * for (let i = 0, o = 0; i < count; i++, o += w.floatsPerVertex) {
 *   d[o] = x; d[o + 1] = y; d[o + 2] = z;
 *   if (w.hasNormal) { d[o + w.normal + 0] = nx; ... }
 *   if (w.hasUv)     { d[o + w.uv + 0] = u;  d[o + w.uv + 1] = v; }
 * }
 * ```
 *
 * That is one pass over the destination. The alternative — build a full
 * position/normal/uv source and then re-pack the subset the layout asks for —
 * writes the same bytes twice and computes the attributes that were going to be
 * discarded anyway. The result is identical; only this one is honest about it.
 */
export class VertexWriter {
  readonly data: Float32Array;
  readonly count: number;
  /** Floats per vertex, i.e. `layout.stride >> 2`. */
  readonly floatsPerVertex: number;
  /** Float offset of `position` within a vertex. Always present. */
  readonly position: number;
  /** Float offset of `normal` within a vertex, or -1 when the layout omits it. */
  readonly normal: number;
  /** Float offset of `uv` within a vertex, or -1 when the layout omits it. */
  readonly uv: number;
  readonly hasNormal: boolean;
  readonly hasUv: boolean;

  constructor(layout: VertexLayout, count: number) {
    const position = layout.attribute('position');
    if (position === undefined) {
      fail('MESH_NO_POSITION',
        'A primitive was built for a layout with no "position" attribute.', {
        why: 'There is no geometry without positions; the vertex stage would have nothing to transform.',
        fix: 'Use a layout that declares at least `position: "float32x3"`.',
      });
    }
    requireFloat32(position.name, position.format, 'A primitive');

    const normal = layout.attribute('normal');
    if (normal !== undefined) requireFloat32(normal.name, normal.format, 'A primitive');
    const uv = layout.attribute('uv');
    if (uv !== undefined) requireFloat32(uv.name, uv.format, 'A primitive');

    this.count = count;
    this.floatsPerVertex = layout.stride >> 2;
    this.data = layout.allocate(count);
    this.position = position.offset >> 2;
    this.normal = normal === undefined ? -1 : normal.offset >> 2;
    this.uv = uv === undefined ? -1 : uv.offset >> 2;
    this.hasNormal = normal !== undefined;
    this.hasUv = uv !== undefined;
  }
}

function requireFloat32(name: string, format: string, source: string): void {
  if (format === 'float32' || format === 'float32x2' || format === 'float32x3' || format === 'float32x4') return;
  fail('LAYOUT_MISMATCH',
    `${source} supplies "${name}" as 32-bit floats, but the layout declares "${format}".`, {
    why: 'Both of apse\'s CPU writers — VertexWriter and the `attributes` source — store their output straight into the interleaved buffer as f32. A normalised or packed format needs a per-component quantise step, which a scalar store cannot do, and there is no second path: the mismatched bytes still upload, the pipeline still validates, and the shader reads four f32 bytes as one byte and three zeroes.',
    fix: `Declare "${name}" as float32x2/float32x3/float32x4, which is every format apse's own geometry uses. If you genuinely need a packed attribute, build the interleaved buffer yourself at the layout's stride and pass it as an \`interleaved\` source — that path takes the bytes as they are.`,
  });
}

// ---------------------------------------------------------------------------
// GpuMesh
// ---------------------------------------------------------------------------

/**
 * {@link DrawableGeometry} plus the three fields a *sub-range* draw needs.
 *
 * `DrawableGeometry` as it stands today describes a whole mesh: `indexCount`
 * with `firstIndex` implicitly 0 and `baseVertex` implicitly 0. Those three
 * fields are the difference between "draw this mesh" and "draw these 36 indices
 * of this mesh, at these 24 vertices" — which is the whole mechanism a batched
 * mesh uses to address one of its sources without re-merging.
 *
 * Assignable to `DrawableGeometry`, so a renderer typed against the narrower
 * interface accepts it unchanged; the wider type is what a caller that wants to
 * read `firstIndex` needs.
 */
export interface DrawableGeometryRange extends DrawableGeometry {
  /** `GPUIndexFormat` of `indexBuffer`. Null when the geometry is not indexed. */
  readonly indexFormat: 'uint16' | 'uint32' | null;
  /** Index **elements** to skip before the first index. 0 for a whole mesh. */
  readonly firstIndex: number;
  /** Added to every index read. 0 for a whole mesh; a batched range uses it. */
  readonly baseVertex: number;
}

/**
 * A per-instance vertex buffer the mesh did not produce.
 *
 * A {@link GpuInstances} is the obvious source and the one apse's own
 * {@link uploadInstances} returns. This shape is for a buffer the caller
 * allocated and filled itself — an indirect-readback path, a storage of
 * thousands of transforms in a compute pass — and the only thing apse asks of
 * it is the buffer, the count, and (optionally) the layout to validate against.
 */
export interface InstanceBufferSource {
  readonly buffer: GPUBuffer;
  readonly count: number;
  /**
   * The combined layout. Optional but strongly recommended: without it apse
   * cannot check that the pipeline's slot 1 offsets are the ones this buffer
   * was written with, and a disagreement is a wrong matrix rather than an
   * error.
   */
  readonly layout?: VertexLayout;
}

export interface GpuMeshOptions {
  readonly name?: string;
  /**
   * The **combined** layout: the mesh's vertex attributes plus the per-instance
   * attributes, e.g. `instancedLayout()`.
   *
   * Needed whenever `instances` is a raw {@link InstanceBufferSource}, because
   * `MeshData` cannot know about instancing — it is the thing being instanced.
   * Omit it when `instances` is a `GpuInstances`, which already knows the
   * layout it was written for, and passing one is how the two are checked
   * against each other.
   */
  readonly layout?: VertexLayout;
  /** Per-instance data, bound to vertex slot 1 with `stepMode: 'instance'`. */
  readonly instances?: GpuInstances | InstanceBufferSource;
  /** First instance to draw. Only meaningful with `instances`. */
  readonly firstInstance?: number;
  /**
   * How many instances to draw. Defaults to all of them. A sub-range
   * (`firstInstance` + `instanceCount`) is checked against the buffer's real
   * size, so a range that runs off the end is an apse error and not a
   * validation error from the driver at the first draw.
   */
  readonly instanceCount?: number;
}

/**
 * An uploaded {@link MeshData}, plus the instance buffer it draws with. This is
 * the whole of the renderer's {@link DrawableGeometry} contract, and nothing
 * else is touched.
 *
 * ## The contract, field by field
 *
 * A renderer issues exactly one of these per draw item:
 *
 * ```txt
 *   setVertexBuffer(0, geometry.vertexBuffer)          // from gpuLayouts()[0]
 *   setIndexBuffer(geometry.indexBuffer, geometry.indexFormat)
 *   setVertexBuffer(1, geometry.instanceBuffer)        // only if non-null
 *   drawIndexed(geometry.indexCount, geometry.instanceCount,
 *               geometry.firstIndex, geometry.baseVertex, geometry.firstInstance)
 * ```
 *
 * and that is the *whole* API. Concretely, for a mesh with N instances:
 *
 *   - `instanceCount` is `N` — not `1`, and not a separate flag. The renderer's
 *     only question is "how many", and a `boolean instanced` next to it is a
 *     second source of truth for the same number.
 *   - `instanceBuffer` is non-null **iff** `instanceCount !== 1`. The pipeline
 *     built from `layout.gpuLayouts()` has one slot when the layout declares no
 *     per-instance attributes and two when it does; binding slot 1 on the former
 *     is a validation error, and omitting it on the latter makes the vertex
 *     stage read whatever happens to be in slot 1.
 *   - `firstIndex` and `baseVertex` are the index-buffer offset and the vertex
 *     offset of a sub-range draw, in **elements**, and are `0` for a whole
 *     mesh. They exist so a batched mesh can address one of its sources without
 *     being re-merged or re-uploaded.
 *   - `indexFormat` is `'uint16' | 'uint32' | null`, and `null` means the mesh
 *     is not indexed, in which case the renderer calls `draw` and not
 *     `drawIndexed`.
 *
 * Disposal destroys the buffers it owns and touches no shared state, so a mesh
 * that several objects reference is released exactly once, when the last
 * reference drops. An instance buffer passed as a `GpuInstances` is *not*
 * owned by the mesh and is never destroyed here.
 */
export class GpuMesh extends Resource implements DrawableGeometryRange {
  readonly name: string;
  readonly layout: VertexLayout;
  readonly vertexBuffer: GPUBuffer;
  readonly indexBuffer: GPUBuffer | null;
  readonly indexCount: number;
  readonly instanceCount: number;
  readonly firstInstance: number;
  /** Per-instance data for vertex slot 1, or null. See {@link GpuMesh}. */
  readonly instanceBuffer: GPUBuffer | null;
  readonly vertexCount: number;
  /** `GPUIndexFormat` of `indexBuffer`. Null when the mesh is not indexed. */
  readonly indexFormat: 'uint16' | 'uint32' | null;
  /** Index elements to skip before the first index. 0 for a whole mesh. */
  readonly firstIndex: number;
  /** Added to every index read. 0 for a whole mesh; a batched sub-range uses it. */
  readonly baseVertex: number;
  /** Bytes per instance in `instanceBuffer`, or 0 when there is none. */
  readonly instanceStride: number;
  /** Total bytes held on the device, excluding any instance buffer. */
  readonly byteLength: number;

  /** The layout the vertex data was packed with. See {@link flush}. */
  readonly #sourceLayout: VertexLayout;
  readonly #queue: GPUQueue;
  /** Dirty spans, half-open, in vertices and in index elements. */
  #vLo = 0;
  #vHi = 0;
  #iLo = 0;
  #iHi = 0;

  constructor(device: GPUDevice, data: MeshData, opts: GpuMeshOptions = {}) {
    super();
    const name = opts.name ?? data.name;
    const vertexBytes = data.layout.byteLength(data.vertexCount);
    const indexBytes = data.indexData === null ? 0 : data.indexData.byteLength;
    const max = device.limits.maxBufferSize;

    if (vertexBytes > max || indexBytes > max) {
      fail('MESH_DATA_TOO_LARGE',
        `Mesh "${name}" needs ${Math.max(vertexBytes, indexBytes)} bytes on one buffer, over the device limit of ${max}.`, {
        why: 'A WebGPU buffer cannot exceed the `maxBufferSize` device limit, and a mesh with one vertex buffer cannot be split across the limit — the whole vertex stream has to be contiguous for a single `setVertexBuffer` to describe it.',
        fix: 'Split the geometry into several meshes, or reduce the vertex count (lower the segment counts on the primitive that produced it).',
      });
    }

    const { layout, instanceBuffer, instanceCount, instanceStride, firstInstance } =
      resolveInstances(name, data, opts);

    // `mappedAtCreation` skips the staging buffer and the extra copy that
    // `writeBuffer` needs. This is a one-time construction cost either way, but
    // it is also the one place where a large mesh would otherwise be resident
    // twice.
    const vertexBuffer = device.createBuffer({
      label: `${name}:vertex`,
      size: align4(vertexBytes),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(data.vertexData);
    vertexBuffer.unmap();

    let indexBuffer: GPUBuffer | null = null;
    if (data.indexData !== null) {
      indexBuffer = device.createBuffer({
        label: `${name}:index`,
        size: align4(indexBytes),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      // `new Uint16Array(view).set(uint16)` and the u32 form both work over the
      // raw bytes; the source's own constructor is what decides the width.
      const view = new Uint8Array(indexBuffer.getMappedRange());
      view.set(new Uint8Array(data.indexData.buffer, data.indexData.byteOffset, data.indexData.byteLength));
      indexBuffer.unmap();
    }

    this.name = name;
    this.layout = layout;
    this.vertexBuffer = vertexBuffer;
    this.indexBuffer = indexBuffer;
    this.indexCount = data.indexCount;
    this.indexFormat = data.indexData === null ? null : data.indexData instanceof Uint32Array ? 'uint32' : 'uint16';
    this.vertexCount = data.vertexCount;
    this.instanceBuffer = instanceBuffer;
    this.instanceCount = instanceCount;
    this.firstInstance = firstInstance;
    this.firstIndex = 0;
    this.baseVertex = 0;
    this.instanceStride = instanceStride;
    this.byteLength = align4(vertexBytes) + (indexBuffer === null ? 0 : align4(indexBytes));
    this.#sourceLayout = data.layout;
    this.#queue = device.queue;
  }

  /** True when a per-instance buffer is bound. `instanceCount > 1` implies it. */
  get instanced(): boolean {
    return this.instanceBuffer !== null;
  }

  /** The layout's slot array, or `null` when this mesh cannot be drawn as it is. */
  get gpuLayouts(): readonly GPUVertexBufferLayout[] {
    return this.layout.gpuLayouts();
  }

  // -------------------------------------------------------------------------
  // Partial re-upload
  // -------------------------------------------------------------------------

  /**
   * Marks `count` vertices starting at `first` for re-upload by {@link flush}.
   *
   * Repeated calls coalesce into one span, so marking three instances one at a
   * time costs one `writeBuffer` of the run that covers all three, not three
   * writes. That coalescing is the entire reason this exists rather than a
   * `writeBuffer` per change: 5000 instances moving by different amounts every
   * frame is 5000 `writeBuffer` calls, and the driver serialises each of them.
   */
  markVertexDirty(first: number, count: number): void {
    this.#vLo = this.#vLo === this.#vHi ? first : Math.min(this.#vLo, first);
    this.#vHi = Math.max(this.#vHi, first + count);
  }

  /** Marks `count` index elements starting at `first`. Coalesces like the above. */
  markIndexDirty(first: number, count: number): void {
    this.#iLo = this.#iLo === this.#iHi ? first : Math.min(this.#iLo, first);
    this.#iHi = Math.max(this.#iHi, first + count);
  }

  /** Marks the whole vertex and index buffers. The "I wrote into my own arrays" path. */
  markAllDirty(): void {
    this.#vLo = 0;
    this.#vHi = this.vertexCount;
    this.#iLo = 0;
    this.#iHi = this.indexCount;
  }

  /** The dirty vertex span as `[first, count)`, or null when nothing is marked. */
  get vertexDirtyRange(): readonly [number, number] | null {
    return this.#vHi > this.#vLo ? [this.#vLo, this.#vHi - this.#vLo] : null;
  }

  /** The dirty index span as `[first, count)`, or null when nothing is marked. */
  get indexDirtyRange(): readonly [number, number] | null {
    return this.#iHi > this.#iLo ? [this.#iLo, this.#iHi - this.#iLo] : null;
  }

  /**
   * Uploads the marked spans from `data`, and clears them. Returns true if it
   * wrote anything, so a caller can tell a frame that touched the mesh from one
   * that did not without tracking it separately. A mark on a buffer this mesh
   * does not have — an index mark on a non-indexed mesh — is cleared and
   * reported as *not* written, because that is what it is.
   *
   * `data` is the same {@link MeshData} the mesh was built from, passed back in
   * because the mesh does not retain it. A different one is refused by identity
   * on the layout plus a count check: re-uploading a *different* mesh's bytes
   * over this one produces a buffer that is internally consistent and visibly
   * wrong, which is the worst of both.
   *
   * The check is the layout's identity and the two counts, not the object's, and
   * it cannot be more than that without retaining the `MeshData` — which is the
   * CPU-side copy of the whole vertex buffer that {@link GpuMesh} exists to avoid.
   * Two meshes that agree on all three are indistinguishable here and the write
   * is structurally safe: the stride, the offsets, and the lengths are the same.
   */
  flush(data: MeshData): boolean {
    if (data.layout !== this.#sourceLayout
      || data.vertexCount !== this.vertexCount
      || data.indexCount !== this.indexCount) {
      fail('INTERNAL_INVARIANT',
        `flush() on mesh "${this.name}" was given "${data.name}", which is not the data it was uploaded from.`, {
        why: 'A partial write copies a run of vertices out of the caller\'s array into a buffer whose stride and offsets were fixed at upload. Handing it a different mesh writes that mesh\'s bytes at this mesh\'s addresses, and the result is a valid GPU buffer describing geometry nobody authored.',
        fix: `Keep the MeshData you passed to upload() and pass that same object to flush(): \`mesh.flush(theDataYouUploaded)\`.`,
      });
    }

    let wrote = false;

    const vertices = this.vertexDirtyRange;
    if (vertices !== null) {
      const stride = this.#sourceLayout.stride >> 2;
      this.#queue.writeBuffer(
        this.vertexBuffer,
        vertices[0] * this.#sourceLayout.stride,
        // A lib-version artifact, not a runtime one: TypeScript 5.7 made the
        // typed-array classes generic over their buffer and the installed
        // @webgpu/types still asks for a view over a plain ArrayBuffer.
        data.vertexData as Float32Array<ArrayBuffer>,
        vertices[0] * stride,
        vertices[1] * stride,
      );
      this.#vLo = 0;
      this.#vHi = 0;
      wrote = true;
    }

    const indices = this.indexDirtyRange;
    const source = data.indexData;
    if (indices !== null) {
      if (source !== null) {
        // `writeBuffer` requires a whole number of 4-byte units, and a Uint16
        // index is 2 bytes: an odd-length span is a validation error. Widening
        // the span by one element is free — the extra index is a value the
        // caller already had in the array and the draw never reaches it.
        const odd = this.indexFormat === 'uint16' ? indices[1] & 1 : 0;
        const count = Math.min(indices[1] + odd, source.length);
        this.#queue.writeBuffer(
          this.indexBuffer!,
          indices[0] * source.BYTES_PER_ELEMENT,
          source as Uint16Array<ArrayBuffer>,
          indices[0],
          count,
        );
        wrote = true;
      }
      // Cleared either way: a mark on a buffer that does not exist must not
      // survive to be reported again on the next frame.
      this.#iLo = 0;
      this.#iHi = 0;
    }

    return wrote;
  }

  protected onDispose(): void {
    this.vertexBuffer.destroy();
    this.indexBuffer?.destroy();
  }
}

/**
 * Resolves the instance half of a {@link GpuMesh} and checks it against the
 * layout the pipeline will be built from.
 *
 * Split out of the constructor because the checks are the interesting part and
 * the constructor is mostly buffer creation: the two failure modes here are an
 * instance buffer bound to a pipeline with no slot 1 for it, and a draw that
 * reads past the end of the instance buffer. Both are silent or inscrutable at
 * the driver, and both are cheap to catch at upload.
 */
function resolveInstances(
  name: string,
  data: MeshData,
  opts: GpuMeshOptions,
): {
    layout: VertexLayout;
    instanceBuffer: GPUBuffer | null;
    instanceCount: number;
    instanceStride: number;
    firstInstance: number;
  } {
  const source = opts.instances;
  const explicit = opts.layout;

  if (source === undefined) {
    if (explicit !== undefined && explicit !== data.layout) {
      explicit.assertCompatible(data.layout, `mesh "${name}"`);
    }
    return {
      layout: explicit ?? data.layout,
      instanceBuffer: null,
      // A mesh with no instance buffer is one copy of itself. An `instanceCount`
      // with no buffer would be a count the vertex stage has nothing to step
      // through, so 1 is the only value that means anything here.
      instanceCount: opts.instanceCount ?? 1,
      instanceStride: 0,
      firstInstance: 0,
    };
  }

  const owned = source instanceof GpuInstances ? source : null;
  const instanceBuffer = owned !== null ? owned.buffer : (source as InstanceBufferSource).buffer;
  const available = owned !== null ? owned.count : (source as InstanceBufferSource).count;
  const instanceStride = owned !== null
    ? owned.instanceStride
    : (explicit?.instanceStride ?? (source as InstanceBufferSource).layout?.instanceStride ?? 0);

  // The layout is the single source for the pipeline's slot array, so this is
  // where the two halves have to be made to agree. `GpuInstances` knows the
  // layout it packed its own bytes for; a caller-supplied `layout` has to be
  // told to match, and a caller-supplied buffer with neither is refused rather
  // than assumed.
  const layout = explicit ?? owned?.layout ?? (source as InstanceBufferSource).layout;
  if (layout === undefined) {
    fail('INTERNAL_INVARIANT',
      `Mesh "${name}" was given an instance buffer with no layout to read it through.`, {
      why: 'The pipeline takes its vertex state from `layout.gpuLayouts()`, and slot 1 of that array is what tells the vertex stage how many bytes into the instance buffer to step and which bytes are which transform column. With no layout there is nothing to validate the buffer against, and a stride that is off by anything renders a wrong matrix rather than failing.',
      fix: `Pass \`layout: instancedLayout()\` (or the combined layout your material uses) alongside the buffer, or pass a \`GpuInstances\` from uploadInstances(), which already carries one.`,
    });
  }
  layout.assertCompatible(data.layout, `mesh "${name}"`);
  if (owned !== null) owned.layout.assertCompatible(layout, `mesh "${name}"`);
  if (!layout.instanced) {
    fail('INTERNAL_INVARIANT',
      `Mesh "${name}" was given an instance buffer but the layout declares no per-instance attributes.`, {
      why: 'A pipeline built from a layout with one vertex buffer slot has no slot 1, so `setVertexBuffer(1, …)` is a validation error that invalidates the whole command buffer and draws nothing this frame. It is also the case where `instanceCount > 1` silently draws the same mesh N times on top of itself.',
      fix: 'Build the combined layout with `instancedLayout()` (optionally `{ color: true }`), and pass that same layout to the material so the two vertex states match.',
    });
  }

  const firstInstance = opts.firstInstance ?? 0;
  const instanceCount = opts.instanceCount ?? available;
  if (!Number.isInteger(instanceCount) || instanceCount < 1
    || !Number.isInteger(firstInstance) || firstInstance < 0) {
    fail('INTERNAL_INVARIANT',
      `Mesh "${name}" was given firstInstance ${firstInstance} and instanceCount ${instanceCount}.`, {
      why: 'These two become `drawIndexed(…, firstInstance)` and the instance count, and the driver reads the buffer through both. A fractional or negative value produces a draw that covers no instances, or one that starts before the buffer does, with nothing reported.',
      fix: `Pass a whole, non-negative firstInstance and a whole instanceCount of at least 1. The buffer holds ${available} instance${available === 1 ? '' : 's'} of ${instanceStride} bytes each.`,
      detail: { kind: 'numeric', field: 'instanceCount', value: instanceCount, min: 1, max: available },
    });
  }

  // The one check the driver cannot make for us at a useful time: whether the
  // span the draw covers is inside the buffer. `buffer.size` is in bytes, and
  // the empty-set case is floored at 4 by the upload path, hence the `=== 0`.
  const last = (firstInstance + instanceCount) * instanceStride;
  if (instanceStride > 0 && last > instanceBuffer.size && instanceBuffer.size !== 0) {
    fail('INTERNAL_INVARIANT',
      `Mesh "${name}" draws instances ${firstInstance}..${firstInstance + instanceCount}, which needs ${last} bytes, but the instance buffer holds ${instanceBuffer.size}.`, {
      why: 'The vertex stage steps the instance buffer without a bound check. Reading past its end is undefined behaviour on the device, not a validation error: the draw succeeds, the counts are honoured, and the geometry comes out as whatever followed the allocation.',
      fix: `Draw at most ${Math.floor(instanceBuffer.size / instanceStride)} instances, or upload a bigger instance buffer. The layout says each record is ${instanceStride} bytes.`,
    });
  }

  return { layout, instanceBuffer, instanceCount, instanceStride, firstInstance };
}

function align4(bytes: number): number {
  return (bytes + 3) & ~3;
}

/**
 * Uploads a {@link MeshData} to the device.
 *
 * The `MeshData` is not retained: this copies into GPU memory and keeps no
 * reference, so the CPU-side arrays can be released as soon as the caller drops
 * them. Keeping a whole vertex buffer alive on the CPU for the lifetime of the
 * GPU buffer doubles the memory cost of a mesh for no benefit.
 *
 * Which is also why {@link GpuMesh.flush} takes the `MeshData` back as an
 * argument: a partial re-upload reads from the caller's array, and apse has
 * deliberately not kept one.
 */
export function upload(
  device: GPUDevice,
  data: MeshData,
  opts?: GpuMeshOptions,
): GpuMesh {
  return new GpuMesh(device, data, opts);
}
