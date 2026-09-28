/**
 * Static batching — many meshes, one buffer, one draw call.
 *
 * ## Why this is a performance feature and not a convenience
 *
 * A draw call is not free and it is not the vertex data. On a modern GPU the
 * per-draw cost is the fixed-function setup: the pipeline is already bound, the
 * vertex and index buffers are already bound, and everything else is a handful
 * of registers. What is *not* free is the CPU work of getting there, and at a
 * thousand objects a scene spends more time in `drawIndexed` bookkeeping than
 * the GPU spends rasterising.
 *
 * three.js users reach for `BufferGeometryUtils.mergeGeometries` for the same
 * reason, and apse had nothing: a scene of two thousand small meshes was two
 * thousand draw calls, and no amount of instancing helps because each of those
 * two thousand is a *different* mesh.
 *
 * ## What is merged, and what is not
 *
 * Vertices and indices. The merged buffer is one interleaved run, so a draw is
 * `drawIndexed(mergedIndexCount, 1)` with no per-source state at all.
 *
 * The per-source sub-ranges are kept as well, because the second half of
 * batching is being able to draw *part* of a batch. Merging is the expensive,
 * structural operation; drawing one of the two thousand sources back out of it
 * has to be free, and it is: a range is an index offset and a vertex offset,
 * which is exactly the argument list of `drawIndexed`. See
 * {@link BatchedRange} and {@link GpuBatchedMesh.sub}.
 *
 * ## Index rebasing
 *
 * Each source's indices are relative to its own vertex 0, and the merged buffer
 * has one vertex 0. So source *k*'s index `j` is written as `j + firstVertex` of
 * source *k*, and the rebased run is recorded as the range that a draw targets.
 * A rebasing bug is not a visible artefact in isolation — it is another source's
 * geometry appearing in the wrong place, which is exactly the kind of thing a
 * screenshot review misses.
 */

import { fail } from '../core/error.ts';
import { Resource } from '../core/resource.ts';
import { create as createSphere, set as setSphere, transform as transformSphere } from '../math/sphere.ts';
import type { PrimitiveTopology } from '../render/pipeline-state.ts';
import type { DrawableGeometry } from '../render/types.ts';
import type { VertexLayout } from './layout.ts';
import { TANGENT } from './layout.ts';
import { GpuMesh, MeshData, allocateIndices } from './mesh.ts';
import type { DrawableGeometryRange } from './mesh.ts';
import type { GpuMeshOptions } from './mesh.ts';

/** One mesh going into a batch, and the transform to bake into it. */
export interface BatchSource {
  readonly mesh: MeshData;
  /**
   * A column-major 4×4 to bake into the vertices, or `null`/omitted to copy
   * them unchanged.
   *
   * Baking is what makes the batch static: after the merge the vertices *are*
   * in world space, so the whole batch draws under one identity transform and
   * costs one matrix multiply in the vertex shader rather than two.
   */
  readonly matrix?: ArrayLike<number> | null;
}

/**
 * One source's span of the merged buffers.
 *
 * `firstIndex` and `firstVertex` are exactly the third and fourth arguments of
 * `drawIndexed`, in **elements**. A renderer that reads them issues
 * `drawIndexed(range.indexCount, 1, range.firstIndex, range.firstVertex, 0)`.
 */
export interface BatchedRange {
  /** The source mesh's name, carried through for tooling. */
  readonly name: string;
  /** First index in the merged index buffer, in elements. */
  readonly firstIndex: number;
  /** Index elements this source contributes. */
  readonly indexCount: number;
  /** Added to every index read: the source's vertex 0 in the merged buffer. */
  readonly firstVertex: number;
  /** Vertices this source contributes. */
  readonly vertexCount: number;
  /** `[cx, cy, cz, r]`, after any baked matrix. Conservative. */
  readonly boundingSphere: Float32Array;
}

export interface BatchOptions {
  /** Only used in error messages and tooling. Defaults to `'batch'`. */
  readonly name?: string;
}

/**
 * A merged {@link MeshData} plus the ranges its sources occupy.
 *
 * Pure CPU: no device, so a batch can be built in a test, in a build script, or
 * in a worker and asserted on before anything is uploaded.
 */
export class BatchedMesh extends Resource {
  readonly name: string;
  /** The merged geometry. One vertex buffer, one index buffer. */
  readonly mesh: MeshData;
  /** One range per source, in the order they were given. */
  readonly ranges: readonly BatchedRange[];

  constructor(name: string, mesh: MeshData, ranges: readonly BatchedRange[]) {
    super();
    this.name = name;
    this.mesh = mesh;
    this.ranges = Object.freeze(ranges);
  }

  /** The layout every source shared. */
  get layout(): VertexLayout {
    return this.mesh.layout;
  }

  get vertexCount(): number {
    return this.mesh.vertexCount;
  }

  get indexCount(): number {
    return this.mesh.indexCount;
  }

  /**
   * Source `i`'s span. Out of range is an `INTERNAL_INVARIANT` naming the count,
   * because an out-of-range range would be read as `undefined` and turned into
   * a `drawIndexed` with NaN arguments.
   */
  range(i: number): BatchedRange {
    const found = this.ranges[i];
    if (found === undefined) {
      fail('INTERNAL_INVARIANT',
        `Batch "${this.name}" has ${this.ranges.length} range${this.ranges.length === 1 ? '' : 's'}; range(${i}) does not exist.`, {
        why: 'A range is looked up by index and used as the argument list of a draw. A missing one is not a no-op — it is a draw over a span of the buffer that belongs to a different source, or over nothing.',
        fix: `Valid indices are 0..${this.ranges.length - 1}. Iterate \`batch.ranges\` rather than counting, and check \`index < batch.ranges.length\` if the index is computed.`,
      });
    }
    return found;
  }

  protected onDispose(): void {
    // No GPU object. The merged typed arrays are collectable as soon as the last
    // reference drops, which is the same reasoning as `MeshData`.
  }
}

/**
 * Merges many {@link MeshData} into one.
 *
 * Every source must share the source layout's **byte** layout and its topology.
 * Vertex count, index count, layout, topology and index width are all derived
 * from the inputs, so the merged `MeshData` is constructed rather than
 * accumulated — there is no way for the counts in the index buffer and the
 * counts in the vertex buffer to disagree.
 *
 * A non-indexed source is given a generated index run `[0, 1, 2, …]`, which
 * costs 2 bytes per vertex and makes the whole batch indexed. That is the right
 * trade: a batch is only ever drawn with `drawIndexed`, and an unindexed source
 * inside it would otherwise need a second draw path.
 */
export function mergeMeshes(sources: readonly BatchSource[], opts: BatchOptions = {}): BatchedMesh {
  const name = opts.name ?? 'batch';
  if (sources.length === 0) {
    fail('MESH_EMPTY',
      `mergeMeshes() was given no sources, so "${name}" would be a batch of nothing.`, {
      why: 'A batch of zero meshes has no vertex data and no index range, and the object built from it would report a draw length of 0 — which is a successful, silent, permanently invisible draw rather than a mistake.',
      fix: 'Return the single source as-is instead of batching it, or check the collection before calling.',
    });
  }

  const layout = sources[0].mesh.layout;
  const topology: PrimitiveTopology = sources[0].mesh.topology;
  let totalVertices = 0;
  let totalIndices = 0;
  for (const [i, source] of sources.entries()) {
    const mesh = source.mesh;
    if (mesh.layout.key !== layout.key) {
      fail('LAYOUT_MISMATCH',
        `mergeMeshes(): source ${i} ("${mesh.name}") uses layout "${mesh.layout.key}" but source 0 uses "${layout.key}".`, {
        why: 'The merged buffer is one interleaved run at one stride. Two meshes with different layouts would have to be re-packed to a common one, and picking either layout silently drops the attributes the other one had — a normal, a uv, a colour.',
        fix: 'Build every source with the same layout, usually by passing the same `layout` option to every primitive call. `key` compares names, formats and order, so two layouts built from the same attribute map always match.',
      });
    }
    if (mesh.topology !== topology) {
      fail('MESH_INDEX_MISALIGNED',
        `mergeMeshes(): source ${i} ("${mesh.name}") is a "${mesh.topology}" but source 0 is a "${topology}".`, {
        why: 'A triangle-list and a triangle-strip read the same index buffer completely differently, and a strip additionally needs `stripIndexFormat` set on the pipeline. Merging them into one draw would decode one of them as noise.',
        fix: 'Merge meshes that share a topology. Draw strips separately, or convert them to triangle-lists first.',
      });
    }
    totalVertices += mesh.vertexCount;
    totalIndices += mesh.indexCount;
  }

  if (totalVertices > 0xffff) {
    fail('MESH_DATA_TOO_LARGE',
      `mergeMeshes() would produce ${totalVertices} vertices, over the 65535 a 16-bit index buffer can address.`, {
      why: 'apse allocates the narrowest index type that fits, so above 65535 vertices the index buffer becomes 32-bit — which is correct, but doubles the index bandwidth of a batch that was probably built from small meshes.',
      fix: 'Merge fewer meshes per batch (the usual answer: one batch per material and per roughly-static group), or accept the 32-bit index buffer.',
    });
  }

  const merged = layout.allocate(totalVertices);
  const indices = allocateIndices(totalVertices, totalIndices);

  const position = layout.attribute('position')!;
  const normal = layout.attribute('normal');
  const tangent = layout.attribute(TANGENT);
  const stride = layout.stride >> 2;
  const pos = position.offset >> 2;
  const nrm = normal === undefined ? -1 : normal.offset >> 2;
  const tan = tangent === undefined ? -1 : tangent.offset >> 2;
  // The tangent is a four-component attribute, and the fourth is the
  // handedness: a sign, not a coordinate. Rotating it like a position is the
  // mistake, so it is carried through as an unchanged ±1 below.
  const tangentComps = tangent === undefined ? 0 : tangent.info.components;

  const ranges: BatchedRange[] = [];
  const sphereA = createSphere();
  const sphereB = createSphere();
  let firstVertex = 0;
  let firstIndex = 0;
  let ii = 0;

  for (const source of sources) {
    const mesh = source.mesh;
    const matrix = source.matrix ?? null;
    const n = mesh.vertexCount;

    if (matrix === null) {
      merged.set(mesh.vertexData.subarray(0, n * stride), firstVertex * stride);
    } else {
      bake(merged, firstVertex * stride, stride, pos, nrm, tan, tangentComps,
        mesh.vertexData, n, matrix, mesh.name, name);
    }

    if (mesh.indexData === null) {
      for (let k = 0; k < n; k++) indices[ii++] = firstVertex + k;
    } else {
      const src = mesh.indexData;
      for (let k = 0; k < src.length; k++) indices[ii++] = src[k] + firstVertex;
    }

    // The source's own bounds, mapped through the same matrix the vertices
    // were, so a range can be culled or drawn on its own.
    const local = setSphere(sphereA, mesh.boundingSphere[0], mesh.boundingSphere[1], mesh.boundingSphere[2], mesh.boundingSphere[3]);
    const world = matrix === null
      ? setSphere(sphereB, local.center[0], local.center[1], local.center[2], local.radius)
      : transformSphere(sphereB, local, asMatrix(matrix, mesh.name, name));

    ranges.push({
      name: mesh.name,
      firstIndex,
      indexCount: mesh.indexCount,
      firstVertex,
      vertexCount: n,
      boundingSphere: new Float32Array([world.center[0], world.center[1], world.center[2], world.radius]),
    });

    firstVertex += n;
    firstIndex += mesh.indexCount;
  }

  if (ii !== totalIndices) {
    fail('INTERNAL_INVARIANT',
      `mergeMeshes() wrote ${ii} indices for "${name}" but planned ${totalIndices}.`, {
      why: 'The index count is the sum of the source index counts, computed before the loop. A mismatch means a source\'s `indexCount` and its `indexData` length disagree, which `MeshData` is supposed to make impossible.',
      fix: 'Internal error in apse. Please report the sources you passed to mergeMeshes().',
    });
  }

  const bounds = unionBounds(ranges, name);
  return new BatchedMesh(name, new MeshData({
    name,
    layout,
    vertices: { interleaved: merged, vertexCount: totalVertices },
    indices,
    boundingSphere: bounds,
  }), ranges);
}

/**
 * A batch uploaded to the device: the whole thing in one draw, and each of its
 * sources in one sub-range draw.
 */
export class GpuBatchedMesh extends Resource implements DrawableGeometryRange {
  readonly name: string;
  readonly layout: VertexLayout;
  readonly vertexBuffer: GPUBuffer;
  readonly indexBuffer: GPUBuffer | null;
  readonly indexCount: number;
  readonly instanceCount: number;
  readonly firstInstance: number;
  readonly instanceBuffer: GPUBuffer | null;
  readonly indexFormat: 'uint16' | 'uint32' | null;
  /** 0: a whole-batch draw starts at the beginning of the index buffer. */
  readonly firstIndex: number;
  /** 0: a whole-batch draw has no vertex rebasing. */
  readonly baseVertex: number;
  /** Total bytes held on the device. */
  readonly byteLength: number;
  /** The batched CPU description, for the ranges and the source names. */
  readonly batch: BatchedMesh;

  readonly #whole: GpuMesh;

  constructor(device: GPUDevice, batch: BatchedMesh, opts: GpuMeshOptions = {}) {
    super();
    this.batch = batch;
    this.#whole = new GpuMesh(device, batch.mesh, opts);
    this.name = opts.name ?? batch.name;
    this.layout = this.#whole.layout;
    this.vertexBuffer = this.#whole.vertexBuffer;
    this.indexBuffer = this.#whole.indexBuffer;
    this.indexCount = this.#whole.indexCount;
    this.instanceCount = this.#whole.instanceCount;
    this.firstInstance = this.#whole.firstInstance;
    this.instanceBuffer = this.#whole.instanceBuffer;
    this.indexFormat = this.#whole.indexFormat;
    this.firstIndex = 0;
    this.baseVertex = 0;
    this.byteLength = this.#whole.byteLength;
  }

  /**
   * A {@link DrawableGeometry} over one source of the batch, sharing all three
   * buffers.
   *
   * This is a view, not a copy: no new `GPUBuffer`, no new upload, and no
   * re-merge. It is a plain object literal of eleven fields, which is why
   * drawing one of two thousand sources out of a batch is free.
   */
  sub(i: number): DrawableGeometryRange {
    const range = this.batch.range(i);
    return {
      layout: this.layout,
      vertexBuffer: this.vertexBuffer,
      indexBuffer: this.indexBuffer,
      indexCount: range.indexCount,
      instanceCount: 1,
      firstInstance: 0,
      instanceBuffer: null,
      indexFormat: this.indexFormat,
      firstIndex: range.firstIndex,
      baseVertex: range.firstVertex,
    };
  }

  protected onDispose(): void {
    this.#whole.dispose();
  }
}

/** Uploads a {@link BatchedMesh}. One vertex buffer, one index buffer. */
export function uploadBatch(
  device: GPUDevice,
  batch: BatchedMesh,
  opts: GpuMeshOptions = {},
): GpuBatchedMesh {
  return new GpuBatchedMesh(device, batch, opts);
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/** Scratch, module-level. `mergeMeshes` is a construction path; this is its floor. */
const _m = new Float32Array(16);
const _n = new Float32Array(9);

function asMatrix(m: ArrayLike<number>, meshName: string, batchName: string): Float32Array {
  if (m.length < 16) {
    fail('INTERNAL_INVARIANT',
      `mergeMeshes(): source "${meshName}" in "${batchName}" was given a ${m.length}-element matrix; a 4x4 needs 16.`, {
      why: 'The matrix is read as a column-major 4×4 at 16 fixed offsets, so a shorter array reads `undefined` for the tail and the vertices are transformed by NaN — which is a mesh that disappears, with no error at the point of the mistake.',
      fix: 'Pass a `mat4.create()`-shaped array: 16 numbers, column-major, the same convention as every other matrix in apse.',
    });
  }
  for (let k = 0; k < 16; k++) _m[k] = m[k];
  return _m;
}

/**
 * Transforms one source's vertices into the merged buffer, with its normals
 * through the inverse transpose.
 *
 * The inverse transpose is not a nicety. Under a non-uniform scale, the upper
 * 3×3 maps a direction to a direction of the wrong *length and angle*, and
 * feeding that to a fragment shader changes the lighting: a sphere squashed to a
 * 2:1:1 ellipsoid shades like a sphere twice as long as it is wide. The tangent
 * goes through the plain 3×3 — it is a direction, not a covector — and its
 * handedness flips when the matrix mirrors, which is a negative determinant.
 *
 * The result is then **renormalised**, and that is not redundant. The inverse
 * transpose of a non-uniform scale is not an orthonormal matrix: it is a linear
 * map, so a unit normal comes out the right *direction* at the wrong length —
 * a +X normal under a 2:1:1 scale comes out with length 0.5. A shader that
 * normalises it recovers the direction, but a shader that does not — and a
 * half-written one does not — shades the surface with a short normal, which is a
 * different amount of light rather than a different direction. The divide is
 * three per vertex, once, at bake time.
 */
function bake(
  out: Float32Array,
  outBase: number,
  stride: number,
  pos: number,
  nrm: number,
  tan: number,
  tangentComps: number,
  source: Float32Array,
  count: number,
  matrix: ArrayLike<number>,
  meshName: string,
  batchName: string,
): void {
  const m = asMatrix(matrix, meshName, batchName);
  const a00 = m[0], a10 = m[1], a20 = m[2];
  const a01 = m[4], a11 = m[5], a21 = m[6];
  const a02 = m[8], a12 = m[9], a22 = m[10];
  const det = a00 * (a11 * a22 - a12 * a21)
    - a01 * (a10 * a22 - a12 * a20)
    + a02 * (a10 * a21 - a11 * a20);
  if (!Number.isFinite(det) || det === 0) {
    fail('INTERNAL_INVARIANT',
      `mergeMeshes(): the matrix for "${meshName}" in "${batchName}" has a singular basis (determinant ${det}).`, {
      why: 'The normals have to go through the inverse transpose of the upper 3×3, and a singular one has no inverse. Every vertex would come out with a NaN normal, the bounds would come out NaN with it, and the object would be culled on every frame with nothing reported.',
      fix: 'Do not bake a matrix that collapses an axis. A zero scale on one axis is a degenerate transform, not a placement — build the geometry with the right dimensions instead.',
    });
  }
  const inv = 1 / det;
  // Cofactor matrix over the determinant. Transposed relative to the plain
  // inverse, which is what makes it the transpose.
  _n[0] = (a11 * a22 - a12 * a21) * inv;
  _n[1] = (a12 * a20 - a10 * a22) * inv;
  _n[2] = (a10 * a21 - a11 * a20) * inv;
  _n[3] = (a02 * a21 - a01 * a22) * inv;
  _n[4] = (a00 * a22 - a02 * a20) * inv;
  _n[5] = (a01 * a20 - a00 * a21) * inv;
  _n[6] = (a01 * a12 - a02 * a11) * inv;
  _n[7] = (a02 * a10 - a00 * a12) * inv;
  _n[8] = (a00 * a11 - a01 * a10) * inv;
  const mirror = det < 0;

  for (let i = 0; i < count; i++) {
    const s = i * stride;
    const d = outBase + i * stride;
    const x = source[s + pos], y = source[s + pos + 1], z = source[s + pos + 2];
    out[d + pos] = m[0] * x + m[4] * y + m[8] * z + m[12];
    out[d + pos + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    out[d + pos + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    if (nrm >= 0) {
      const nx = source[s + nrm], ny = source[s + nrm + 1], nz = source[s + nrm + 2];
      const rx = _n[0] * nx + _n[3] * ny + _n[6] * nz;
      const ry = _n[1] * nx + _n[4] * ny + _n[7] * nz;
      const rz = _n[2] * nx + _n[5] * ny + _n[8] * nz;
      // A source normal of zero is already broken geometry; the guard keeps it
      // zero rather than turning it into a NaN that would spread through every
      // product the fragment stage does with it.
      const len = Math.sqrt(rx * rx + ry * ry + rz * rz);
      const k = len > 0 && Number.isFinite(len) ? 1 / len : 0;
      out[d + nrm] = rx * k;
      out[d + nrm + 1] = ry * k;
      out[d + nrm + 2] = rz * k;
    }
    if (tan >= 0) {
      const tx = source[s + tan], ty = source[s + tan + 1], tz = source[s + tan + 2];
      const bx = m[0] * tx + m[4] * ty + m[8] * tz;
      const by = m[1] * tx + m[5] * ty + m[9] * tz;
      const bz = m[2] * tx + m[6] * ty + m[10] * tz;
      const len = Math.sqrt(bx * bx + by * by + bz * bz);
      const k = len > 0 ? 1 / len : 0;
      out[d + tan] = bx * k;
      out[d + tan + 1] = by * k;
      out[d + tan + 2] = bz * k;
      if (tangentComps > 3) {
        out[d + tan + 3] = mirror ? -source[s + tan + 3] : source[s + tan + 3];
      }
    }
  }
}

/**
 * One conservative sphere around every range.
 *
 * The centre is the midpoint of the ranges' AABB, and the radius is the largest
 * distance from that centre to any range sphere's surface. Both steps are
 * deliberately loose: the alternative — the smallest enclosing sphere — is an
 * optimisation nobody needs, and an under-estimate here is an object that is
 * culled while it is still on screen.
 */
function unionBounds(ranges: readonly BatchedRange[], name: string): Float32Array {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const r of ranges) {
    const [x, y, z, rad] = r.boundingSphere;
    if (x - rad < minX) minX = x - rad;
    if (y - rad < minY) minY = y - rad;
    if (z - rad < minZ) minZ = z - rad;
    if (x + rad > maxX) maxX = x + rad;
    if (y + rad > maxY) maxY = y + rad;
    if (z + rad > maxZ) maxZ = z + rad;
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  let radius = 0;
  for (const r of ranges) {
    const d = Math.hypot(r.boundingSphere[0] - cx, r.boundingSphere[1] - cy, r.boundingSphere[2] - cz);
    if (d + r.boundingSphere[3] > radius) radius = d + r.boundingSphere[3];
  }
  if (!Number.isFinite(radius)) {
    fail('INTERNAL_INVARIANT',
      `mergeMeshes(): the combined bounds for "${name}" are not finite.`, {
      why: 'Every source contributed a finite sphere — `MeshData` refuses a non-finite bound — so a non-finite union means a baked matrix moved a centre off to infinity, and the merged mesh would be culled on every frame and never drawn.',
      fix: 'Check the matrices passed to mergeMeshes(): a non-finite entry in one produces a non-finite centre, and the vertex shader would have received the same value.',
    });
  }
  return new Float32Array([cx, cy, cz, radius]);
}
