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
      : computeBoundingSphere(new Float32Array(4), vertexData, vertexCount, floatsPerVertex, positionOffset);
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
    fail('INTERNAL_INVARIANT',
      `Mesh "${name}" was given a ${source.length}-element bounding sphere; the invariant is 4 floats (x, y, z, r).`, {
      why: 'Every consumer in apse reads the bounds as a packed vec4, and a short array silently reads undefined for the missing components.',
      fix: 'Pass 4 numbers: [centreX, centreY, centreZ, radius]. Omit the option entirely to have apse compute it.',
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
    requireFloat32(position.name, position.format);

    const normal = layout.attribute('normal');
    if (normal !== undefined) requireFloat32(normal.name, normal.format);
    const uv = layout.attribute('uv');
    if (uv !== undefined) requireFloat32(uv.name, uv.format);

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

function requireFloat32(name: string, format: string): void {
  if (format === 'float32' || format === 'float32x2' || format === 'float32x3' || format === 'float32x4') return;
  fail('LAYOUT_MISMATCH',
    `A primitive supplies "${name}" as 32-bit floats, but the layout declares "${format}".`, {
    why: 'The primitive writers store their output straight into the interleaved buffer as f32. A normalised or packed format needs a per-component quantise step, which a scalar store cannot do.',
    fix: `Declare "${name}" as float32x2/float32x3/float32x4, or build the attribute yourself through a MeshData \`attributes\` source, which packs any format.`,
  });
}

// ---------------------------------------------------------------------------
// GpuMesh
// ---------------------------------------------------------------------------

export interface GpuMeshOptions {
  readonly name?: string;
  readonly instanceCount?: number;
  readonly firstInstance?: number;
  /**
   * A layout carrying per-instance attributes, e.g. `instancedLayout()`.
   *
   * The override exists because the instance *layout* and the instance *data*
   * have to agree: the pipeline declares slots from the layout, and the buffer
   * is bound to slot 1. `MeshData` cannot know about instancing — it is the
   * thing being instanced — so the layout is supplied at upload.
   */
  readonly layout?: VertexLayout;
  /** Per-instance transforms, bound to slot 1. Requires an instanced `layout`. */
  readonly instances?: { readonly buffer: GPUBuffer; readonly count: number };
}

/**
 * An uploaded {@link MeshData}. Implements the renderer's
 * {@link DrawableGeometry} contract and nothing else.
 *
 * Disposal destroys the buffers it owns and touches no shared state, so a mesh
 * that several objects reference is released exactly once, when the last
 * reference drops.
 */
export class GpuMesh extends Resource implements DrawableGeometry {
  readonly name: string;
  readonly layout: VertexLayout;
  readonly vertexBuffer: GPUBuffer;
  readonly indexBuffer: GPUBuffer | null;
  readonly indexCount: number;
  readonly instanceCount: number;
  readonly firstInstance: number;
  /** `GPUIndexFormat` of `indexBuffer`. Null when the mesh is not indexed. */
  readonly indexFormat: 'uint16' | 'uint32' | null;
  /** Per-instance transforms for vertex slot 1, or null. See `GpuMeshOptions`. */
  readonly instanceBuffer: GPUBuffer | null;
  readonly vertexCount: number;
  /** Total bytes held on the device. */
  readonly byteLength: number;

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
    // An instanced layout comes from the caller, not from the MeshData: the data
    // is what is being instanced, so it cannot describe its own instance slots.
    this.layout = opts.layout ?? data.layout;
    this.vertexBuffer = vertexBuffer;
    this.indexBuffer = indexBuffer;
    this.indexCount = data.indexCount;
    this.indexFormat = data.indexData === null ? null : data.indexData instanceof Uint32Array ? 'uint32' : 'uint16';
    this.vertexCount = data.vertexCount;
    this.instanceBuffer = opts.instances?.buffer ?? null;
    // An instanced mesh defaults to every instance. Passing an explicit count
    // with no buffer would draw one copy of the mesh and silently discard the
    // rest, so the two are resolved together.
    this.instanceCount = this.instanceBuffer !== null
      ? opts.instances!.count
      : (opts.instanceCount ?? 1);
    this.firstInstance = opts.firstInstance ?? 0;
    this.byteLength = align4(vertexBytes) + (indexBuffer === null ? 0 : align4(indexBytes));
  }

  protected onDispose(): void {
    this.vertexBuffer.destroy();
    this.indexBuffer?.destroy();
  }
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
 */
export function upload(
  device: GPUDevice,
  data: MeshData,
  opts?: GpuMeshOptions,
): GpuMesh {
  return new GpuMesh(device, data, opts);
}
