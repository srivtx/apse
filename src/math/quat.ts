/**
 * Rotations as unit quaternions.
 *
 * A quaternion is a `Float32Array` of length 4, laid out `[x, y, z, w]` — the
 * same order WGSL uses for `vec4<f32>`, so a quaternion can be written into a
 * uniform block with no shuffling.
 *
 * ## Why there is no Euler support
 *
 * There is no `euler.ts` in apse, and no Euler *storage* anywhere in the scene
 * graph. Euler angles have three properties that make them the wrong type for
 * a retained scene graph:
 *
 *   1. They are not closed under composition. `rx(90) * ry(90)` is not a single
 *      Euler triple, so every parent-relative rotation is a matrix or a
 *      quaternion, and storing angles alongside it means keeping two
 *      representations of one value.
 *   2. Interpolating them is wrong. The straight line between two Euler
 *      triples does not follow the shortest arc, so a spinning object banks the
 *      wrong way unless the angles are re-ordered at every keyframe.
 *   3. The conversion is not free. A bidirectional sync — which is what a
 *      `rotation: [x, y, z]` property requires — costs a trigonometry-heavy
 *      conversion on every write to `rotation`, plus a setter and a getter
 *      closure allocated per object, and it forces a dirty flag and a
 *      recomposition path through the graph.
 *
 * A quaternion is 4 floats, composes in place with 16 multiplies, interpolates
 * correctly with {@link slerp}, and needs no sync layer. The authoring
 * convenience that Euler would have provided is covered once, explicitly, by
 * {@link fromEulerXYZ} — a pure function that converts three angles into a
 * quaternion at the moment of authoring. Nothing in apse stores the angles.
 */

import { fail } from '../core/error.ts';

/**
 * Below this dot product, {@link slerp} falls back to normalised lerp.
 *
 * `sin(θ)` for the half-angle goes to zero as the two quaternions converge, so
 * the spherical formula divides by a quantity that has no useful limit. The
 * nlerp path is continuous across that boundary and is the better answer there
 * anyway.
 */
const SLERP_LINEAR_EPSILON = 0.9995;

/** Allocates a quaternion. Defaults to identity. The only allocating function here. */
export function create(x = 0, y = 0, z = 0, w = 1): Float32Array {
  const out = new Float32Array(4);
  out[0] = x;
  out[1] = y;
  out[2] = z;
  out[3] = w;
  return out;
}

/** Allocates a copy of `q`. */
export function clone(q: Float32Array): Float32Array {
  const out = new Float32Array(4);
  out[0] = q[0];
  out[1] = q[1];
  out[2] = q[2];
  out[3] = q[3];
  return out;
}

/** Writes the identity rotation `(0, 0, 0, 1)`. */
export function identity(out: Float32Array): Float32Array {
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;
  out[3] = 1;
  return out;
}

/** Writes the four components. `out` may alias `q`. */
export function set(out: Float32Array, x: number, y: number, z: number, w: number): Float32Array {
  out[0] = x;
  out[1] = y;
  out[2] = z;
  out[3] = w;
  return out;
}

/** Copies `q` into `out`. Aliasing `out === q` is a no-op. */
export function copy(out: Float32Array, q: Float32Array): Float32Array {
  out[0] = q[0];
  out[1] = q[1];
  out[2] = q[2];
  out[3] = q[3];
  return out;
}

/** Dot product. Unaffected by the double cover. */
export function dot(a: Float32Array, b: Float32Array): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/** Euclidean norm. 1 for a unit quaternion. */
export function length(q: Float32Array): number {
  return Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

/**
 * Writes the rotation of `angle` radians about `axis`.
 *
 * `axis` is normalised here rather than required to arrive normalised, because
 * a caller computing an axis from a cross product is exactly the caller most
 * likely to get the scale wrong, and an unnormalised axis silently produces a
 * quaternion whose *shape* is right but whose magnitude is not — which then
 * fails much later, in {@link toMat4}, with no obvious cause.
 *
 * `angle` is in radians, always. apse has no degrees anywhere.
 *
 * `out` must not alias `axis` unless the axis is already unit length.
 */
export function setAxisAngle(out: Float32Array, axis: Float32Array, angle: number): Float32Array {
  const x = axis[0];
  const y = axis[1];
  const z = axis[2];
  const len = Math.sqrt(x * x + y * y + z * z);
  if (len === 0 || !Number.isFinite(len)) {
    fail('INTERNAL_INVARIANT',
      `setAxisAngle was given a rotation axis of length ${len}.`, {
        why: 'A zero-length axis has no direction, so the rotation it names is undefined. The usual cause is normalising a zero cross product, which happens when an up vector is parallel to the view direction.',
      fix: 'Guard the cross product before the call, or pick a different axis. `setAxisAngle(out, [1, 0, 0], angle)` is always well-defined.',
    });
  }
  const half = angle * 0.5;
  const s = Math.sin(half) / len;
  out[0] = x * s;
  out[1] = y * s;
  out[2] = z * s;
  out[3] = Math.cos(half);
  return out;
}

/**
 * Writes the rotation equivalent to applying `rx`, then `ry`, then `rz` about
 * the *parent* axes, in radians.
 *
 * This is the one sanctioned path from angles to a rotation, and it is a
 * pure function: the angles are an input, not a storage format. Nothing in apse
 * holds on to them. The composition is `q = qz * qy * qx`, which is the same
 * as a column-major model matrix `Rz * Ry * Rx` built with
 * {@link mat4.fromQuat} on the result.
 *
 * The expanded product is written out rather than routed through three
 * {@link setAxisAngle} calls and two {@link mul}s, because it is a pure
 * function of three scalars and the intermediate quaternions would be six
 * redundant stores. The signs are checkable by expanding the Hamilton product
 * by hand: `x` picks up `-cx*sy*sz` and `y` picks up `+sx*cy*sz`, because
 * `qy * qx` is `(cy*sx, sy*cx, -sy*sx, cy*cx)` and the `qz` product flips the
 * sign of the second cross term in each. A test asserts the result against
 * `quat.mul` over random triples, which is what catches a sign here.
 *
 * Use this when a loader, an editor, or a test hands you three angles. Do not
 * use it as an animation storage format — that is the failure mode the rest of
 * this module exists to prevent.
 *
 * `out` must not alias a previously-written result being read as input; it may
 * alias nothing, since all three angles are scalars.
 */
export function fromEulerXYZ(out: Float32Array, x: number, y: number, z: number): Float32Array {
  const hx = x * 0.5;
  const hy = y * 0.5;
  const hz = z * 0.5;
  const sx = Math.sin(hx);
  const cx = Math.cos(hx);
  const sy = Math.sin(hy);
  const cy = Math.cos(hy);
  const sz = Math.sin(hz);
  const cz = Math.cos(hz);
  out[0] = sx * cy * cz - cx * sy * sz;
  out[1] = cx * sy * cz + sx * cy * sz;
  out[2] = cx * cy * sz - sx * sy * cz;
  out[3] = cx * cy * cz + sx * sy * sz;
  return out;
}

/**
 * Hamilton product `a * b`: the rotation that applies `b` first, then `a`.
 *
 * The product is not commutative, and getting the argument order wrong is the
 * most common bug in quaternion code. `mul(out, a, b)` means "parent `a`, child
 * `b`" — the same order as `mat4.mul(out, parentMatrix, childMatrix)`.
 *
 * The result is the composition of two unit quaternions and is already unit,
 * up to float error. No normalisation is performed.
 *
 * `out` must not alias `a` or `b`.
 */
export function mul(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  const ax = a[0];
  const ay = a[1];
  const az = a[2];
  const aw = a[3];
  const bx = b[0];
  const by = b[1];
  const bz = b[2];
  const bw = b[3];
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

/**
 * Writes `q / |q|`, restoring the unit invariant after accumulation.
 *
 * A zero-length quaternion writes identity. Every function here assumes unit
 * input, and drift compounds: a graph that multiplies a few thousand times
 * without a renormalise is measurably non-unit, and the error shows up as a
 * slowly shrinking object rather than as a crash.
 */
export function normalize(out: Float32Array, q: Float32Array): Float32Array {
  const x = q[0];
  const y = q[1];
  const z = q[2];
  const w = q[3];
  const len = Math.sqrt(x * x + y * y + z * z + w * w);
  if (len === 0 || !Number.isFinite(len)) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 1;
    return out;
  }
  const inv = 1 / len;
  out[0] = x * inv;
  out[1] = y * inv;
  out[2] = z * inv;
  out[3] = w * inv;
  return out;
}

/**
 * Writes `−q`.
 *
 * `-q` is the *same rotation* as `q` — the unit quaternions are a double cover
 * of the rotation group. This matters for two things: comparing two rotations
 * for equality (compare `|dot|`, not `dot`) and for keeping a hemisphere
 * convention when slerping. It is the only cheap way to move a quaternion
 * between the two sheets.
 */
export function negate(out: Float32Array, q: Float32Array): Float32Array {
  out[0] = -q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
  return out;
}

/**
 * Writes the conjugate `(−x, −y, −z, w)`.
 *
 * For a unit quaternion this is the inverse, and it is the cheap way to apply
 * a rotation in the opposite direction: `q⁻¹ = q*`, so a conjugated rotation
 * costs three negations instead of an inversion. Use {@link invert} when `q` is
 * not known to be unit.
 */
export function conjugate(out: Float32Array, q: Float32Array): Float32Array {
  out[0] = -q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = q[3];
  return out;
}

/**
 * Writes `q⁻¹ = q* / |q|²`, valid for non-unit `q`.
 *
 * A zero-length quaternion has no inverse; identity is written rather than
 * NaN, for the reason given in {@link normalize}.
 */
export function invert(out: Float32Array, q: Float32Array): Float32Array {
  const x = q[0];
  const y = q[1];
  const z = q[2];
  const w = q[3];
  const sq = x * x + y * y + z * z + w * w;
  if (sq === 0) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 1;
    return out;
  }
  const inv = 1 / sq;
  out[0] = -x * inv;
  out[1] = -y * inv;
  out[2] = -z * inv;
  out[3] = w * inv;
  return out;
}

/**
 * Constant-angular-velocity interpolation from `a` to `b`.
 *
 * Maintains the unit invariant: the result is normalised, so it can be fed
 * straight back into {@link mul} without drift.
 *
 * Both cases that break naive slerp are handled:
 *
 *   - **Antipodal / negative dot.** `a` and `b` can name the same rotation with
 *     opposite signs (`q` and `−q`). Interpolating the raw signs takes the long
 *     way round — 350° instead of 10°. `b` is negated when `dot < 0` so the
 *     path is always the short one.
 *   - **Coincident.** When the two are (nearly) the same rotation, `sin(θ)` in
 *     the slerp formula underflows toward zero and the division produces NaN.
 *     The result falls back to a normalised lerp above
 *     `SLERP_LINEAR_EPSILON`, which is continuous with slerp across the
 *     boundary.
 *
 * `out` must not alias `a` or `b`. `t` is unclamped.
 */
export function slerp(out: Float32Array, a: Float32Array, b: Float32Array, t: number): Float32Array {
  let ax = a[0];
  let ay = a[1];
  let az = a[2];
  let aw = a[3];
  let bx = b[0];
  let by = b[1];
  let bz = b[2];
  let bw = b[3];

  let cosom = ax * bx + ay * by + az * bz + aw * bw;

  if (cosom < 0) {
    cosom = -cosom;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  }

  if (cosom > SLERP_LINEAR_EPSILON) {
    // Too close to distinguish the great circle from a straight line.
    out[0] = ax + (bx - ax) * t;
    out[1] = ay + (by - ay) * t;
    out[2] = az + (bz - az) * t;
    out[3] = aw + (bw - aw) * t;
    return normalize(out, out);
  }

  const omega = Math.acos(cosom);
  const sinom = Math.sin(omega);
  const wa = Math.sin((1 - t) * omega) / sinom;
  const wb = Math.sin(t * omega) / sinom;

  out[0] = ax * wa + bx * wb;
  out[1] = ay * wa + by * wb;
  out[2] = az * wa + bz * wb;
  out[3] = aw * wa + bw * wb;
  return out;
}

/**
 * Self-normalising, `t`-clamped {@link slerp} — the safe form for public and
 * tooling use, where the inputs are not known to be unit and `t` may come from
 * a scrub bar.
 *
 * Use {@link slerp} in a per-frame animation loop, where both quaternions came
 * from {@link normalize} and `t` is already in range, and this one anywhere the
 * cost does not matter or the inputs are untrusted. They produce identical
 * results for unit inputs and `t` in `[0, 1]`.
 */
export function slerpQuat(out: Float32Array, a: Float32Array, b: Float32Array, t: number): Float32Array {
  const u = t < 0 ? 0 : t > 1 ? 1 : t;
  return slerp(out, a, b, u);
}

/**
 * Rotates the vector `v` by the unit quaternion `q` (the axis-angle `v' = q v q⁻¹`).
 *
 * `q` **must** be unit. A non-unit `q` scales the result by `|q|²`, so the
 * error is a silent size change rather than a wrong direction. Renormalise
 * after composition if `q` came from repeated {@link mul}.
 *
 * `out` must not alias `v`.
 */
export function rotateVec3(out: Float32Array, v: Float32Array, q: Float32Array): Float32Array {
  const qx = q[0];
  const qy = q[1];
  const qz = q[2];
  const qw = q[3];
  const vx = v[0];
  const vy = v[1];
  const vz = v[2];

  // t = 2 * (q.xyz × v), so v' = v + w*t + q.xyz × t.
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);

  out[0] = vx + qw * tx + qy * tz - qz * ty;
  out[1] = vy + qw * ty + qz * tx - qx * tz;
  out[2] = vz + qw * tz + qx * ty - qy * tx;
  return out;
}

/**
 * Writes the 4×4 rotation matrix of `q` as a **column-major** `mat4x4f`,
 * matching the storage apse uses everywhere: translation is the last column
 * (`out[12..14]`), and `out[15] === 1`.
 *
 * This is the single source of truth for quaternion→matrix in apse;
 * `mat4.fromQuat` is this function. The matrix carries no scale — a rotation
 * quaternion cannot express one — so the basis columns are exactly unit
 * length. A model matrix with scale is `T * R * S` with the scale on the
 * right, not folded into `q`.
 *
 * `q` must be unit, for the same reason as in {@link rotateVec3}.
 * `out` must not alias `q`.
 */
export function toMat4(out: Float32Array, q: Float32Array): Float32Array {
  const x = q[0];
  const y = q[1];
  const z = q[2];
  const w = q[3];

  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;

  out[0] = 1 - (yy + zz);
  out[1] = xy + wz;
  out[2] = xz - wy;
  out[3] = 0;
  out[4] = xy - wz;
  out[5] = 1 - (xx + zz);
  out[6] = yz + wx;
  out[7] = 0;
  out[8] = xz + wy;
  out[9] = yz - wx;
  out[10] = 1 - (xx + yy);
  out[11] = 0;
  out[12] = 0;
  out[13] = 0;
  out[14] = 0;
  out[15] = 1;
  return out;
}

/**
 * Verifies that `q` is usable as a quaternion: a `Float32Array` of exactly 4
 * components.
 *
 * Like {@link assertVec3}, this is an explicit boundary check rather than part
 * of the per-call cost. Reach for it where a rotation arrives from outside
 * apse — a loader, an editor, a network payload.
 */
export function assertQuat(q: Float32Array, label: string): void {
  if (q instanceof Float32Array && q.length === 4) return;
  fail('INTERNAL_INVARIANT',
    `${label} is ${describe(q)}, not a Float32Array of 4 components.`, {
      why: 'A quaternion rotation is undefined unless all four components are present. A short buffer makes `w` read as `undefined`, which becomes NaN in every downstream matrix and blanks the frame rather than failing here.',
      fix: 'Pass a length-4 Float32Array, e.g. from `quat.create()` or `quat.setAxisAngle(...)`.',
    });
}

function describe(q: unknown): string {
  if (q instanceof Float32Array) return `a Float32Array of ${q.length} components`;
  if (Array.isArray(q)) return `a plain array of ${q.length} numbers`;
  return `${typeof q} \`${String(q)}\``;
}
