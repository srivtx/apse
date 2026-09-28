/**
 * View frustum as six planes, for culling.
 *
 * ## Storage and sign convention
 *
 * A frustum is a `Float32Array` of 24 floats: six planes of four floats each,
 * in the order **left, right, bottom, top, near, far**. A plane `(a, b, c, d)`
 * is unit-length in `abc` and a point `p` is on the **inside** when
 *
 *     a·p.x + b·p.y + c·p.z + d >= 0
 *
 * The normals therefore all point inward. Normalising matters: it makes
 * `dot(plane, centre)` a true signed distance in world units, which is what
 * lets {@link containsSphere} compare against a radius directly instead of
 * against a distance to an arbitrary plane origin.
 *
 * ## Extraction
 *
 * The Gribb–Hartmann method, using the rows of the view-projection matrix. The
 * six planes come from sums and differences of a row with the `w` row, and
 * only the near/far pair depends on the clip-space depth convention: WebGPU
 * puts NDC depth in `[0, 1]`, so the near plane *is* the `z` row and the far
 * plane is `w - z`. Under OpenGL's `[-1, 1]` the near plane would be
 * `w + z`; getting that wrong does not error, it culls everything in front of
 * the camera, so it is worth stating.
 *
 * ## Cost
 *
 * One extraction and six dot products per object, with no allocation. This is
 * the whole reason a bounding sphere is worth keeping: the test is exact for
 * spheres and costs the same regardless of how much geometry is inside.
 */

import { fail } from '../core/error.ts';
import type { Aabb, Sphere } from './sphere.ts';

/** Floats per plane. */
const PLANE_STRIDE = 4;

/** Floats in a full frustum. */
export const FRUSTUM_STRIDE = 24;

/** Allocates a frustum whose planes are all zero, ready for {@link setFromViewProjection}. */
export function create(): Float32Array {
  return new Float32Array(FRUSTUM_STRIDE);
}

/** Plane indices into a 24-float frustum, in storage order. */
export const PLANE_LEFT = 0;
export const PLANE_RIGHT = 1;
export const PLANE_BOTTOM = 2;
export const PLANE_TOP = 3;
export const PLANE_NEAR = 4;
export const PLANE_FAR = 5;

/**
 * Writes the six inward-facing planes of `m`, a view-projection matrix, into
 * `out` (24 floats, each plane unit-length).
 *
 * `m` must be a full projection, not a view: extracting from a view matrix
 * yields the six planes of an unbounded box, which culls nothing and looks
 * correct. Pass `viewProj`, not `view`.
 *
 * `out` must not alias `m`.
 */
export function setFromViewProjection(out: Float32Array, m: Float32Array): Float32Array {
  // Matrix rows in the mathematical sense. Column-major storage means row r is
  // strided by 4 starting at index r.
  const w0 = m[3];
  const w1 = m[7];
  const w2 = m[11];
  const w3 = m[15];

  // left = w + x
  normalizePlane(out, 0, w0 + m[0], w1 + m[4], w2 + m[8], w3 + m[12]);
  // right = w - x
  normalizePlane(out, 1, w0 - m[0], w1 - m[4], w2 - m[8], w3 - m[12]);
  // bottom = w + y
  normalizePlane(out, 2, w0 + m[1], w1 + m[5], w2 + m[9], w3 + m[13]);
  // top = w - y
  normalizePlane(out, 3, w0 - m[1], w1 - m[5], w2 - m[9], w3 - m[13]);
  // near = z, because WebGPU depth is [0, 1] and w - z is the far plane.
  normalizePlane(out, 4, m[2], m[6], m[10], m[14]);
  // far = w - z
  normalizePlane(out, 5, w0 - m[2], w1 - m[6], w2 - m[10], w3 - m[14]);
  return out;
}

/**
 * True when any part of `s` is inside the frustum.
 *
 * The test is `dot(plane, centre) >= -radius` for all six planes. Since
 * `dot` is a signed distance, that is exactly "the plane does not separate the
 * sphere from the camera", and a sphere that straddles a plane still passes it
 * — which is what a cull test wants. It is conservative, never wrong: a fully
 * outside sphere cannot pass, because the distance from its centre to any plane
 * it violates is greater than the radius.
 *
 * Six early-outs, no allocation.
 */
export function containsSphere(planes: Float32Array, s: Sphere): boolean {
  const x = s.center[0];
  const y = s.center[1];
  const z = s.center[2];
  const r = -s.radius;

  for (let i = 0; i < FRUSTUM_STRIDE; i += PLANE_STRIDE) {
    if (planes[i] * x + planes[i + 1] * y + planes[i + 2] * z + planes[i + 3] < r) return false;
  }
  return true;
}

/**
 * True when any part of `a` is inside the frustum, by the positive-vertex test.
 *
 * A box is inside a plane exactly when its corner furthest along the plane
 * normal is: for axis *i*, take `max[i]` if the normal points that way and
 * `min[i]` otherwise. If that one corner is behind a plane, the whole box is.
 * Eight corner tests per plane would give the same answer 64 times over.
 */
export function containsAabb(planes: Float32Array, a: Aabb): boolean {
  const minX = a.min[0];
  const minY = a.min[1];
  const minZ = a.min[2];
  const maxX = a.max[0];
  const maxY = a.max[1];
  const maxZ = a.max[2];

  for (let i = 0; i < FRUSTUM_STRIDE; i += PLANE_STRIDE) {
    const nx = planes[i];
    const ny = planes[i + 1];
    const nz = planes[i + 2];
    const d = planes[i + 3];
    if (nx * (nx > 0 ? maxX : minX) +
        ny * (ny > 0 ? maxY : minY) +
        nz * (nz > 0 ? maxZ : minZ) + d < 0) {
      return false;
    }
  }
  return true;
}

/**
 * Verifies that `f` is usable as a frustum: a `Float32Array` of 24 floats.
 *
 * Same boundary-check role as `assertVec3`; the per-plane reads in
 * {@link containsSphere} are unchecked by design.
 */
export function assertFrustum(f: Float32Array, label: string): void {
  if (f instanceof Float32Array && f.length === FRUSTUM_STRIDE) return;
  fail('INTERNAL_INVARIANT',
    `${label} is ${f instanceof Float32Array ? `a Float32Array of ${f.length} components` : typeof f}, not a Float32Array of ${FRUSTUM_STRIDE}.`, {
      why: 'A frustum is six planes of four floats. A short buffer reads `undefined` for the missing planes, and `undefined < -radius` is false — so the cull test silently passes everything, and nothing is ever culled.',
      fix: 'Use a length-24 Float32Array from `frustum.create()` or `frustum.setFromViewProjection(...)`.',
    });
}

/** Writes one plane at `index`, scaled so the normal is unit length. */
function normalizePlane(out: Float32Array, index: number, a: number, b: number, c: number, d: number): void {
  const len = Math.sqrt(a * a + b * b + c * c);
  const inv = len === 0 ? 0 : 1 / len;
  const o = index * PLANE_STRIDE;
  out[o] = a * inv;
  out[o + 1] = b * inv;
  out[o + 2] = c * inv;
  out[o + 3] = d * inv;
}
