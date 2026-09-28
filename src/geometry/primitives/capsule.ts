/**
 * Capsule — a cylinder with hemispherical caps, +Y up.
 *
 * ## Why it is not two primitives welded together
 *
 * The obvious construction is `sphere()` on top of `cylinder()` on top of
 * `sphere()`, and it is wrong in a way that only shows up under a light. The
 * join between the hemisphere's equator and the cylinder's rim is two
 * coincident rings with the same positions and *different normals* — the
 * sphere's normal there is horizontal and the cylinder's is horizontal too, so
 * that part is fine — but the **uvs** are not: the sphere's v runs to 0.5 at
 * its equator and the cylinder's runs 0..1 over its own height, so a texture
 * stretches by a factor of two across the seam. And the vertices are duplicated
 * at the join, which is what a tangent generator needs in order to produce a
 * continuous frame, and which a merge would have to reconcile.
 *
 * So it is generated as one surface of revolution. A capsule is topologically a
 * sphere, and generating it as one is what keeps the u wrap, the v run and the
 * normal field continuous across the whole thing.
 *
 * ## The profile
 *
 * A capsule is the profile of a capsule, lathed. Walking from the north pole
 * down:
 *
 * ```txt
 *   y = h/2 + r                  north pole, radius 0
 *   arc to the equator           r·sin, h/2 + r·cos        (capSegments - 1 rows)
 *   straight down the side       r, h/2 … -h/2             (heightSegments rows)
 *   arc to the south pole        r·sin, -h/2 - r·cos       (capSegments - 1 rows)
 *   y = -h/2 - r                 south pole, radius 0
 * ```
 *
 * Both poles are **one vertex**, as in {@link sphere}. A polar ring of
 * coincident vertices is a band of zero-area triangles the rasteriser has to
 * reject, and on a capsule the poles are the two most visible points on the
 * silhouette.
 */

import { fail } from '../../core/error.ts';
import { STANDARD_LAYOUT, type VertexLayout } from '../layout.ts';
import { allocateIndices, MeshData, VertexWriter } from '../mesh.ts';

export interface CapsuleOptions {
  readonly radius?: number;
  /** Length of the **cylindrical section**, excluding the two caps. */
  readonly height?: number;
  /** Segments around the circumference. 3..512. */
  readonly radialSegments?: number;
  /** Rows along the cylindrical section. At least 1. */
  readonly heightSegments?: number;
  /**
   * Segments per hemisphere, pole to equator. 2..512.
   *
   * The minimum is 2 rather than 1 because 1 means *no* arc at all: the profile
   * goes straight from the pole to the wall, and the shape is a double cone
   * rather than a capsule. Two segments is the coarsest thing that is still
   * actually round.
   */
  readonly capSegments?: number;
  readonly layout?: VertexLayout;
}

const MIN_SEGMENTS = 3;
const MAX_SEGMENTS = 512;

function clamp(value: number, field: string, min: number): number {
  if (!Number.isFinite(value)) {
    fail('OPTION_UNKNOWN',
      `capsule() received ${field}: ${value}, which is not a finite number.`, {
      why: 'The segment counts drive every loop bound and every index computation in the generator, so a NaN or Infinity here silently produces an empty or unbounded buffer.',
      fix: `Pass a number in [${min}, ${MAX_SEGMENTS}].`,
    });
  }
  return Math.min(MAX_SEGMENTS, Math.max(min, Math.floor(value)));
}

/**
 * A capsule centred on the origin along Y.
 *
 * `2 + (2·capSegments + heightSegments − 2)·(radialSegments + 1)` vertices and
 * `2·radialSegments·(2·capSegments + heightSegments − 2)` triangles. At the
 * defaults — 16 radial, 4 cap, 1 height — that is 121 vertices and 224
 * triangles: a closed, smooth, correctly-normalled, correctly-UV'd solid for
 * less than a 16×8 sphere costs, and a shape a game needs constantly.
 *
 * The surface is a closed 2-manifold, so its Euler characteristic is 2 — the
 * same as the sphere it is topologically. `height: 0` is a sphere of `radius`,
 * and is allowed: it collapses the cylindrical section to nothing rather than
 * failing, because it is a legitimate degenerate request with a well-defined
 * answer.
 */
export function capsule(opts: CapsuleOptions = {}): MeshData {
  const {
    radius = 0.5,
    height = 1,
    layout = STANDARD_LAYOUT,
  } = opts;
  const rs = clamp(opts.radialSegments ?? 16, 'radialSegments', MIN_SEGMENTS);
  const capSeg = clamp(opts.capSegments ?? 4, 'capSegments', 2);
  // A zero-height capsule is a sphere, and the only way to get one is a single
  // wall row: more than one would be several coincident rings, which is a band
  // of zero-area triangles rather than a sphere.
  const hSeg = height === 0 ? 1 : Math.max(1, Math.floor(opts.heightSegments ?? 1));
  const half = height * 0.5;

  // Interior rings, walked north to south. `pole` marks the single-vertex rows.
  interface Row { readonly radius: number; readonly y: number; readonly ny: number; readonly pole: boolean }
  const rows: Row[] = [{ radius: 0, y: half + radius, ny: 1, pole: true }];
  const quarter = Math.PI * 0.5;
  for (let k = 1; k < capSeg; k++) {
    const a = (k / capSeg) * quarter;
    const s = Math.sin(a), c = Math.cos(a);
    rows.push({ radius: radius * s, y: half + radius * c, ny: c, pole: false });
  }
  for (let k = 0; k < hSeg; k++) {
    rows.push({ radius, y: half - (k / hSeg) * height, ny: 0, pole: false });
  }
  for (let k = capSeg - 1; k >= 1; k--) {
    const a = (k / capSeg) * quarter;
    const s = Math.sin(a), c = Math.cos(a);
    rows.push({ radius: radius * s, y: -half - radius * c, ny: -c, pole: false });
  }
  rows.push({ radius: 0, y: -half - radius, ny: -1, pole: true });

  // v is arc length along the profile rather than row index, so a texture does
  // not stretch by the ratio of a cap's arc to the side's height.
  const arcs: number[] = [0];
  for (let i = 1; i < rows.length; i++) {
    const a = rows[i - 1], b = rows[i];
    arcs.push(arcs[i - 1] + Math.hypot(b.radius - a.radius, b.y - a.y));
  }
  const totalArc = arcs[arcs.length - 1];

  const ringRows = rows.length - 2;
  const vertexCount = 2 + ringRows * (rs + 1);
  // Ring-to-ring quads, plus a fan at each pole.
  const triangles = rs * 2 * ringRows;
  const indexCount = triangles * 3;
  const rowStride = rs + 1;
  const southPole = vertexCount - 1;

  const w = new VertexWriter(layout, vertexCount);
  const out = w.data;
  const indices = allocateIndices(vertexCount, indexCount);
  const stride = w.floatsPerVertex;
  const pos = w.position, nrm = w.normal, uv = w.uv;
  const hasNormal = w.hasNormal, hasUv = w.hasUv;

  // Row 0 and the last row are the poles; everything between them is
  // `ringRows` rings of `rs + 1` columns, in the order they were pushed.
  const firstRing = 1;
  const ringAt = (i: number): number => firstRing + i * rowStride;

  const write = (vertex: number, px: number, py: number, pz: number, nx: number, ny: number, nz: number, u: number, v: number): void => {
    const o = vertex * stride;
    out[o + pos] = px; out[o + pos + 1] = py; out[o + pos + 2] = pz;
    if (hasNormal) { out[o + nrm] = nx; out[o + nrm + 1] = ny; out[o + nrm + 2] = nz; }
    if (hasUv) { out[o + uv] = u; out[o + uv + 1] = v; }
  };

  write(0, 0, half + radius, 0, 0, 1, 0, 0.5, 0);
  write(southPole, 0, -half - radius, 0, 0, -1, 0, 0.5, 1);

  for (let i = 0; i < ringRows; i++) {
    const row = rows[i + 1];
    const v = totalArc > 0 ? arcs[i + 1] / totalArc : 0;
    for (let k = 0; k <= rs; k++) {
      const u = k / rs;
      const phi = u * Math.PI * 2;
      const cosP = Math.cos(phi), sinP = Math.sin(phi);
      // The side of the profile has ny = 0, so the normal is purely radial and
      // the same formula serves the cap and the wall. Everywhere else it is
      // (sin a · cos φ, cos a, sin a · sin φ) with the profile's own y term.
      const radial = row.ny === 0 ? 1 : Math.sqrt(Math.max(0, 1 - row.ny * row.ny));
      write(
        ringAt(i) + k,
        row.radius * cosP, row.y, row.radius * sinP,
        radial * cosP, row.ny, radial * sinP,
        u, v,
      );
    }
  }

  // cross(∂u, ∂v) points inward for this parameterisation, so the quads step
  // down the column first, exactly as `sphere()` does — and the pole fans
  // inherit the same winding, which is why the north fan is (pole, a, a+1).
  let ii = 0;
  for (let k = 0; k < rs; k++) {
    indices[ii++] = 0;
    indices[ii++] = ringAt(0) + k;
    indices[ii++] = ringAt(0) + k + 1;
  }
  for (let i = 0; i < ringRows - 1; i++) {
    const row = ringAt(i);
    for (let k = 0; k < rs; k++) {
      const a = row + k;
      const b = a + rowStride;
      const c = b + 1;
      const e = a + 1;
      indices[ii++] = a; indices[ii++] = b; indices[ii++] = c;
      indices[ii++] = a; indices[ii++] = c; indices[ii++] = e;
    }
  }
  const last = ringAt(ringRows - 1);
  for (let k = 0; k < rs; k++) {
    indices[ii++] = last + k;
    indices[ii++] = southPole;
    indices[ii++] = last + k + 1;
  }

  if (ii !== indexCount) {
    fail('INTERNAL_INVARIANT',
      `capsule() wrote ${ii} indices but planned ${indexCount}.`, {
      why: 'The counts are derived from the clamped segment counts and the profile row count before the loops run; a mismatch means the profile and the index arithmetic have diverged.',
      fix: 'Internal error in apse. Please report it with the arguments you passed to capsule().',
    });
  }

  return new MeshData({
    name: 'capsule',
    layout,
    vertices: { interleaved: out, vertexCount },
    indices,
  });
}
