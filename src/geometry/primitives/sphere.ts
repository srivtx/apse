/**
 * UV sphere.
 *
 * The workhorse: smooth normals, spherical UVs, and a topology that is uniform
 * enough to displace on the GPU without special cases. `heightSegments` rows of
 * `widthSegments` columns, plus a duplicated seam column and pole row so the u
 * wrap and the v poles do not have to be stitched.
 */

import { fail } from '../../core/error.ts';
import { STANDARD_LAYOUT, type VertexLayout } from '../layout.ts';
import { allocateIndices, MeshData, VertexWriter } from '../mesh.ts';

export interface SphereOptions {
  readonly radius?: number;
  readonly widthSegments?: number;
  readonly heightSegments?: number;
  readonly layout?: VertexLayout;
}

/** Below 3 there is no triangle to build; above 512 the mesh is not the bottleneck. */
const MIN_SEGMENTS = 3;
const MAX_SEGMENTS = 512;

function clampSegments(value: number, field: string): number {
  if (!Number.isFinite(value)) {
    fail('OPTION_UNKNOWN',
      `sphere() received ${field}: ${value}, which is not a finite number.`, {
      why: 'The segment counts drive every loop bound and every index computation in the generator, so a NaN or Infinity here silently produces an empty or unbounded buffer.',
      fix: `Pass a number in [${MIN_SEGMENTS}, ${MAX_SEGMENTS}].`,
    });
  }
  return Math.min(MAX_SEGMENTS, Math.max(MIN_SEGMENTS, Math.floor(value)));
}

/**
 * A UV sphere of `radius`, centred on the origin.
 *
 * `2 + (widthSegments + 1) * (heightSegments - 1)` vertices and
 * `widthSegments * (heightSegments - 1) * 2` triangles. Defaults are 32 × 16:
 * 497 vertices, 960 triangles — a smooth silhouette at phone size for a fifth
 * of a 32 × 32 sphere's cost.
 *
 * # The poles are single vertices, not rows
 *
 * The textbook parameterisation makes row 0 and row `heightSegments` each
 * `widthSegments + 1` copies of the same point, so the polar quads collapse and
 * half of every one of them is a zero-area triangle. Those are still indexed
 * and still rasterised — rejected on area, after the vertex stage has already
 * run for them. This generator emits one vertex per pole and fans the first and
 * last bands around it, so the sphere has no degenerate triangles at all, the
 * seam vertices are the only duplicated positions, and the whole surface is a
 * closed 2-manifold.
 *
 * Normals are the analytic unit vector for the sphere point, not
 * `normalize(position)`. They are the same value mathematically, but computing
 * them directly means the stored normal is unit to within f32 rounding, with
 * no division in the generation loop and no pole singularity.
 *
 * UVs: `u` wraps 0..1 around the equator, `v` runs 0 at the +Y pole to 1 at
 * the -Y pole. A pole's `u` is 0.5, so the texture converges to a point there
 * rather than tearing across a zero-width row.
 */
export function sphere(opts: SphereOptions = {}): MeshData {
  const { radius = 1, layout = STANDARD_LAYOUT } = opts;
  const widthSegments = clampSegments(opts.widthSegments ?? 32, 'widthSegments');
  const heightSegments = clampSegments(opts.heightSegments ?? 16, 'heightSegments');
  const wSeg = widthSegments;
  const hSeg = heightSegments;

  // Vertex 0 is the north pole, the interior rows follow, the last vertex is
  // the south pole. Interior rows are `wSeg + 1` columns so the u seam has two
  // coincident columns to interpolate between.
  const rowStride = wSeg + 1;
  const rowCount = hSeg - 1;
  const vertexCount = 2 + rowCount * rowStride;
  const southPole = vertexCount - 1;
  const indexCount = wSeg * (hSeg - 1) * 6;

  const w = new VertexWriter(layout, vertexCount);
  const out = w.data;
  const indices = allocateIndices(vertexCount, indexCount);
  const stride = w.floatsPerVertex;
  const pos = w.position;
  const nrm = w.normal;
  const uv = w.uv;
  const hasNormal = w.hasNormal;
  const hasUv = w.hasUv;

  const write = (index: number, nx: number, ny: number, nz: number, u: number, v: number): void => {
    const o = index * stride;
    out[o + pos] = nx * radius;
    out[o + pos + 1] = ny * radius;
    out[o + pos + 2] = nz * radius;
    if (hasNormal) {
      out[o + nrm] = nx;
      out[o + nrm + 1] = ny;
      out[o + nrm + 2] = nz;
    }
    if (hasUv) {
      out[o + uv] = u;
      out[o + uv + 1] = v;
    }
  };

  write(0, 0, 1, 0, 0.5, 0);
  write(southPole, 0, -1, 0, 0.5, 1);

  let base = 1;
  for (let iy = 1; iy <= rowCount; iy++) {
    // theta sweeps from PI/hSeg down to PI - PI/hSeg, so v runs 0 -> 1.
    const v = 1 - iy / hSeg;
    const theta = (1 - v) * Math.PI;
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);
    for (let ix = 0; ix <= wSeg; ix++) {
      const u = ix / wSeg;
      const phi = u * Math.PI * 2;
      const cosP = Math.cos(phi);
      const sinP = Math.sin(phi);
      // Unit normal for this direction; the position is the same vector scaled.
      write(base++, -cosP * sinT, cosT, sinP * sinT, u, v);
    }
  }

  // cross(∂u, ∂v) points inward for this parameterisation, so the quads step
  // down the column first: a, a+row, a+row+1, a+1. The pole fans inherit that
  // winding, which is why the north fan is (pole, a, a+1) and the south fan is
  // (a, pole, a+1) — opposite orders around a ring that is traversed the same
  // way.
  let ii = 0;
  const rowBase = (iy: number): number => 1 + (iy - 1) * rowStride;
  for (let ix = 0; ix < wSeg; ix++) {
    const first = rowBase(1) + ix;
    indices[ii++] = 0;
    indices[ii++] = first;
    indices[ii++] = first + 1;
  }
  for (let iy = 1; iy < rowCount; iy++) {
    const row = rowBase(iy);
    for (let ix = 0; ix < wSeg; ix++) {
      const a = row + ix;
      const b = a + rowStride;
      const c = b + 1;
      const e = a + 1;
      indices[ii++] = a;
      indices[ii++] = b;
      indices[ii++] = c;
      indices[ii++] = a;
      indices[ii++] = c;
      indices[ii++] = e;
    }
  }
  const lastRow = rowBase(rowCount);
  for (let ix = 0; ix < wSeg; ix++) {
    indices[ii++] = lastRow + ix;
    indices[ii++] = southPole;
    indices[ii++] = lastRow + ix + 1;
  }

  if (ii !== indexCount) {
    fail('INTERNAL_INVARIANT',
      `sphere() wrote ${ii} indices but planned ${indexCount}.`, {
      why: 'The index count is derived from the clamped segment counts before the loop runs; a mismatch means the clamping and the arithmetic have diverged.',
      fix: 'Internal error in apse. Please report it with the arguments you passed to sphere().',
    });
  }

  return new MeshData({
    name: 'sphere',
    layout,
    vertices: { interleaved: out, vertexCount },
    indices,
  });
}
