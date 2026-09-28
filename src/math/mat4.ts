/**
 * 4×4 matrices, column-major, over `Float32Array`.
 *
 * ## Storage
 *
 * `m[0..3]` is **column 0**, `m[4..7]` is column 1, and so on. A matrix element
 * at row `r`, column `c` lives at `m[c * 4 + r]`. The translation is therefore
 * the *last* column, `m[12], m[13], m[14]`, and `m[15]` is the homogeneous 1.
 *
 * This is not a preference. WGSL and GLSL both store `mat4x4<f32>` column by
 * column, and WebGPU uniform blocks are laid out by WGSL's address-space rules.
 * A matrix written here is therefore **byte-identical** to the same matrix
 * constructed in a shader — `mat4.mul(m, a, b)` in WGSL and `mat4.mul(out, a, b)`
 * here produce the same 64 bytes, with no transpose anywhere. A row-major
 * library would need a 16-element shuffle on every uniform write, per object,
 * per frame.
 *
 * The consequence to remember: `m[1]` is *not* the second element of row 0, it
 * is the first element of column 1. This trips up everyone once.
 *
 * ## Conventions
 *
 * Vectors are column vectors. A point is transformed as `M * p`; a direction as
 * `M * (p, 0)`. Clip space follows WebGPU, not OpenGL: after the perspective
 * divide, `x` and `y` are in `[-1, 1]` and **`z` is in `[0, 1]`**, with the
 * near plane at 0. {@link perspective} and {@link orthographic} are built for
 * that range; a depth-compare of `less` with a cleared depth of 0 is therefore
 * correct with no bias and no remap.
 *
 * ## Cost
 *
 * Nothing here allocates except {@link create} and {@link clone}. Every other
 * function writes into a caller-supplied `out`. {@link mul} computes all
 * sixteen results in locals before storing any of them, so `out` may alias
 * either input.
 */

import { fail } from '../core/error.ts';
import { toMat4 } from './quat.ts';

/** Allocates a zeroed `mat4`. The only allocating function here. */
export function create(): Float32Array {
  return new Float32Array(16);
}

/** Allocates a copy of `m`. */
export function clone(m: Float32Array): Float32Array {
  return new Float32Array(m);
}

/** Writes the identity matrix. */
export function identity(out: Float32Array): Float32Array {
  out[0] = 1;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  out[4] = 0;
  out[5] = 1;
  out[6] = 0;
  out[7] = 0;
  out[8] = 0;
  out[9] = 0;
  out[10] = 1;
  out[11] = 0;
  out[12] = 0;
  out[13] = 0;
  out[14] = 0;
  out[15] = 1;
  return out;
}

/** Copies `m` into `out`. Aliasing `out === m` is a no-op. */
export function copy(out: Float32Array, m: Float32Array): Float32Array {
  out.set(m);
  return out;
}

/** Writes zeros. */
export function zero(out: Float32Array): Float32Array {
  out.fill(0);
  return out;
}

/**
 * Writes the product `a * b` — the transform that applies `b` first, then `a`.
 *
 * Argument order matches {@link quat.mul} and the usual parent/child
 * convention: `mat4.mul(out, parent, child)`. All sixteen results are computed
 * into locals before any store, so `out` may alias `a` or `b`.
 *
 * The `aXY` locals below are named **column X, row Y** — `a01` is `a[1]`,
 * `a10` is `a[4]`. That is the same naming the Gribb-Hartmann derivation in
 * `frustum.ts` uses, and it is the opposite of what most matrix libraries
 * write. Read the storage map above the code once and it stops mattering.
 */
export function mul(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  const a00 = a[0];
  const a01 = a[1];
  const a02 = a[2];
  const a03 = a[3];
  const a10 = a[4];
  const a11 = a[5];
  const a12 = a[6];
  const a13 = a[7];
  const a20 = a[8];
  const a21 = a[9];
  const a22 = a[10];
  const a23 = a[11];
  const a30 = a[12];
  const a31 = a[13];
  const a32 = a[14];
  const a33 = a[15];

  const b00 = b[0];
  const b01 = b[1];
  const b02 = b[2];
  const b03 = b[3];
  const b10 = b[4];
  const b11 = b[5];
  const b12 = b[6];
  const b13 = b[7];
  const b20 = b[8];
  const b21 = b[9];
  const b22 = b[10];
  const b23 = b[11];
  const b30 = b[12];
  const b31 = b[13];
  const b32 = b[14];
  const b33 = b[15];

  out[0] = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
  out[1] = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
  out[2] = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;
  out[3] = a03 * b00 + a13 * b01 + a23 * b02 + a33 * b03;
  out[4] = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
  out[5] = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
  out[6] = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;
  out[7] = a03 * b10 + a13 * b11 + a23 * b12 + a33 * b13;
  out[8] = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
  out[9] = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
  out[10] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
  out[11] = a03 * b20 + a13 * b21 + a23 * b22 + a33 * b23;
  out[12] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
  out[13] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
  out[14] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
  out[15] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
  return out;
}

/**
 * Writes the transpose of `m`. `out` may alias `m`.
 *
 * Needed for a normal matrix, which is the inverse *transpose* of the model
 * matrix — a transform with non-uniform scale does not rotate normals
 * correctly on its own.
 */
export function transpose(out: Float32Array, m: Float32Array): Float32Array {
  const a01 = m[1];
  const a02 = m[2];
  const a03 = m[3];
  const a12 = m[6];
  const a13 = m[7];
  const a23 = m[11];
  out[0] = m[0];
  out[1] = m[4];
  out[2] = m[8];
  out[3] = m[12];
  out[4] = a01;
  out[5] = m[5];
  out[6] = m[9];
  out[7] = m[13];
  out[8] = a02;
  out[9] = a12;
  out[10] = m[10];
  out[11] = m[14];
  out[12] = a03;
  out[13] = a13;
  out[14] = a23;
  out[15] = m[15];
  return out;
}

/** Determinant of `m`. Zero means singular: no inverse exists. */
export function determinant(m: Float32Array): number {
  const a00 = m[0];
  const a01 = m[1];
  const a02 = m[2];
  const a03 = m[3];
  const a10 = m[4];
  const a11 = m[5];
  const a12 = m[6];
  const a13 = m[7];
  const a20 = m[8];
  const a21 = m[9];
  const a22 = m[10];
  const a23 = m[11];
  const a30 = m[12];
  const a31 = m[13];
  const a32 = m[14];
  const a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  return b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
}

/**
 * Writes the inverse of `m`.
 *
 * A singular matrix is a hard failure rather than a silently-NaN result: the
 * inverse of a view-projection feeds ray reconstruction, and NaN there turns
 * into discarded pixels rather than a visible error. The usual causes are an
 * object scaled to zero on some axis, or a projection built with `near === 0`.
 *
 * `out` must not alias `m`.
 */
export function invert(out: Float32Array, m: Float32Array): Float32Array {
  const a00 = m[0];
  const a01 = m[1];
  const a02 = m[2];
  const a03 = m[3];
  const a10 = m[4];
  const a11 = m[5];
  const a12 = m[6];
  const a13 = m[7];
  const a20 = m[8];
  const a21 = m[9];
  const a22 = m[10];
  const a23 = m[11];
  const a30 = m[12];
  const a31 = m[13];
  const a32 = m[14];
  const a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (det === 0 || !Number.isFinite(det)) {
    fail('INTERNAL_INVARIANT',
      `A matrix handed to invert() is not invertible (determinant ${det}).`, {
        why: 'Every matrix apse builds itself is invertible, so a singular one arrived from a caller. The usual causes are a scale of zero on any axis — which collapses the transform — or a projection matrix built with `near: 0`.',
        fix: 'Check for a zero or negative scale on the node whose matrix this is, and check the near/far you passed to `perspective`/`orthographic`. A zeroed scale is the overwhelmingly common case.',
      });
  }
  det = 1 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

/**
 * Writes `M * p` for a point, including the perspective divide.
 *
 * The divide is what makes this a *point* operation: after it, the result is in
 * the same space `p` was in — normalised device coordinates, or view space for
 * a view matrix. Use {@link transformDirection} for anything that should not be
 * divided (a normal, a velocity, a look-at target offset).
 *
 * `w === 0` produces infinities rather than an error; that is a point on the
 * camera plane, which is a caller bug the divide cannot repair.
 *
 * `out` must not alias `v`.
 */
export function transformPoint(out: Float32Array, v: Float32Array, m: Float32Array): Float32Array {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  const inv = 1 / w;
  out[0] = (m[0] * x + m[4] * y + m[8] * z + m[12]) * inv;
  out[1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) * inv;
  out[2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) * inv;
  return out;
}

/**
 * Writes `M * (v, 0)`: a direction, so the translation is ignored and there is
 * no divide.
 *
 * Scale *is* applied — that is the correct behaviour for a direction, and it is
 * why a non-uniformly scaled object gives the wrong answer if you rotate a
 * normal with this. For normals under non-uniform scale, use the inverse
 * transpose: `mat4.mul(out, mat4.transpose(t, model), mat4.invert(t, model))`
 * applied to the normal, or pre-transform it into world space once per frame
 * rather than per vertex.
 *
 * `out` must not alias `v`.
 */
export function transformDirection(out: Float32Array, v: Float32Array, m: Float32Array): Float32Array {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  out[0] = m[0] * x + m[4] * y + m[8] * z;
  out[1] = m[1] * x + m[5] * y + m[9] * z;
  out[2] = m[2] * x + m[6] * y + m[10] * z;
  return out;
}

/**
 * Writes the full `M * v` for a `vec4`, with no divide. Use this to carry a
 * clip-space position or a `vec4` attribute through a matrix.
 *
 * `out` must not alias `v`.
 */
export function transformVec4(out: Float32Array, v: Float32Array, m: Float32Array): Float32Array {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  const w = v[3];
  out[0] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
  out[1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
  out[2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
  out[3] = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
  return out;
}

/** Writes the translation column of `m` into `out`. `out` must not alias `m`. */
export function translation(out: Float32Array, m: Float32Array): Float32Array {
  out[0] = m[12];
  out[1] = m[13];
  out[2] = m[14];
  return out;
}

/** Writes a pure translation. `out` must not alias `v`. */
export function fromTranslation(out: Float32Array, x: number, y: number, z: number): Float32Array {
  identity(out);
  out[12] = x;
  out[13] = y;
  out[14] = z;
  return out;
}

/** Writes a pure scale. `out` must not alias `v`. */
export function fromScale(out: Float32Array, x: number, y: number, z: number): Float32Array {
  zero(out);
  out[0] = x;
  out[5] = y;
  out[10] = z;
  out[15] = 1;
  return out;
}

/**
 * Writes a pure rotation as a `mat4` — the same conversion as
 * {@link quat.toMat4}, exposed here so model-matrix code does not have to
 * import the quaternion module.
 *
 * `q` must be a unit quaternion; the result has no scale.
 */
export function fromQuat(out: Float32Array, q: Float32Array): Float32Array {
  return toMat4(out, q);
}

/** Writes a right-handed rotation of `angle` radians about the X axis. */
export function fromRotationX(out: Float32Array, angle: number): Float32Array {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  identity(out);
  out[5] = c;
  out[6] = s;
  out[9] = -s;
  out[10] = c;
  return out;
}

/** Writes a right-handed rotation of `angle` radians about the Y axis. */
export function fromRotationY(out: Float32Array, angle: number): Float32Array {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  identity(out);
  out[0] = c;
  out[2] = -s;
  out[8] = s;
  out[10] = c;
  return out;
}

/** Writes a right-handed rotation of `angle` radians about the Z axis. */
export function fromRotationZ(out: Float32Array, angle: number): Float32Array {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  identity(out);
  out[0] = c;
  out[1] = s;
  out[4] = -s;
  out[5] = c;
  return out;
}

/**
 * Post-multiplies a rotation about the node's local X axis: `out = out * Rx`.
 *
 * Post-multiplying is the point. Composing on the right applies the rotation
 * in the *node's own* space, so a node spun every frame spins about the axis
 * its parent gave it, and a chain of these calls reproduces a TRS order without
 * rebuilding the matrix. To rotate about the parent axis instead, premultiply.
 *
 * Only the first three columns move: column *j* of the product is
 * `R·(out column j)`, and `Rx` leaves column 0 and column 3 alone.
 *
 * `out` is both the input and the output.
 */
export function rotateX(out: Float32Array, angle: number): Float32Array {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  const m10 = out[4];
  const m11 = out[5];
  const m12 = out[6];
  const m13 = out[7];
  const m20 = out[8];
  const m21 = out[9];
  const m22 = out[10];
  const m23 = out[11];
  out[4] = m10 * c + m20 * s;
  out[5] = m11 * c + m21 * s;
  out[6] = m12 * c + m22 * s;
  out[7] = m13 * c + m23 * s;
  out[8] = m20 * c - m10 * s;
  out[9] = m21 * c - m11 * s;
  out[10] = m22 * c - m12 * s;
  out[11] = m23 * c - m13 * s;
  return out;
}

/** Post-multiplies a rotation about the node's local Y axis. See {@link rotateX}. */
export function rotateY(out: Float32Array, angle: number): Float32Array {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  const m00 = out[0];
  const m01 = out[1];
  const m02 = out[2];
  const m03 = out[3];
  const m20 = out[8];
  const m21 = out[9];
  const m22 = out[10];
  const m23 = out[11];
  out[0] = m00 * c - m20 * s;
  out[1] = m01 * c - m21 * s;
  out[2] = m02 * c - m22 * s;
  out[3] = m03 * c - m23 * s;
  out[8] = m00 * s + m20 * c;
  out[9] = m01 * s + m21 * c;
  out[10] = m02 * s + m22 * c;
  out[11] = m03 * s + m23 * c;
  return out;
}

/** Post-multiplies a rotation about the node's local Z axis. See {@link rotateX}. */
export function rotateZ(out: Float32Array, angle: number): Float32Array {
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  const m00 = out[0];
  const m01 = out[1];
  const m02 = out[2];
  const m03 = out[3];
  const m10 = out[4];
  const m11 = out[5];
  const m12 = out[6];
  const m13 = out[7];
  out[0] = m00 * c + m10 * s;
  out[1] = m01 * c + m11 * s;
  out[2] = m02 * c + m12 * s;
  out[3] = m03 * c + m13 * s;
  out[4] = m10 * c - m00 * s;
  out[5] = m11 * c - m01 * s;
  out[6] = m12 * c - m02 * s;
  out[7] = m13 * c - m03 * s;
  return out;
}

/** Post-multiplies a local translation: `out = out * T`. `out` is in and out. */
export function translate(out: Float32Array, x: number, y: number, z: number): Float32Array {
  const m00 = out[0];
  const m01 = out[1];
  const m02 = out[2];
  const m10 = out[4];
  const m11 = out[5];
  const m12 = out[6];
  const m20 = out[8];
  const m21 = out[9];
  const m22 = out[10];
  out[12] = m00 * x + m10 * y + m20 * z + out[12];
  out[13] = m01 * x + m11 * y + m21 * z + out[13];
  out[14] = m02 * x + m12 * y + m22 * z + out[14];
  return out;
}

/** Post-multiplies a local scale: `out = out * S`. `out` is in and out. */
export function scale(out: Float32Array, x: number, y: number, z: number): Float32Array {
  for (let c = 0; c < 4; c++) {
    const i = c * 4;
    out[i] *= x;
    out[i + 1] *= y;
    out[i + 2] *= z;
  }
  return out;
}

/**
 * Writes the right-handed view matrix: the transform from world space into the
 * space where the camera sits at the origin looking down **−Z**.
 *
 * The basis is built directly in column-major storage, so the translation
 * lands in `out[12..14]` already oriented. Those three floats are
 * `−dot(axis, eye)` for the three view axes — with the world axes aligned to
 * the view axes that is simply `−eye`, which is why the eye's world position
 * is *not* stored literally. The invariant that actually matters:
 * `mat4.transformPoint(out, eye, view)` is `(0, 0, 0)`.
 *
 * `up` need not be normalised and need not be perpendicular to the view
 * direction; it is orthogonalised against the view axis. `eye === target` is
 * degenerate and produces a zero basis rather than NaN.
 */
export function lookAt(out: Float32Array, eye: Float32Array, target: Float32Array, up: Float32Array): Float32Array {
  const ex = eye[0];
  const ey = eye[1];
  const ez = eye[2];

  // Camera looks down -Z, so +Z points from the target back to the eye.
  let zx = ex - target[0];
  let zy = ey - target[1];
  let zz = ez - target[2];
  let len = Math.sqrt(zx * zx + zy * zy + zz * zz);
  if (len === 0) {
    zx = 0;
    zy = 0;
    zz = 1;
  } else {
    const inv = 1 / len;
    zx *= inv;
    zy *= inv;
    zz *= inv;
  }

  // X = up × Z, re-orthogonalised so a non-perpendicular `up` still yields a
  // right-handed orthonormal basis.
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.sqrt(xx * xx + xy * xy + xz * xz);
  if (len === 0) {
    xx = 1;
    xy = 0;
    xz = 0;
  } else {
    const inv = 1 / len;
    xx *= inv;
    xy *= inv;
    xz *= inv;
  }

  // Y = Z × X, already unit because Z and X are.
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx;
  out[1] = yx;
  out[2] = zx;
  out[3] = 0;
  out[4] = xy;
  out[5] = yy;
  out[6] = zy;
  out[7] = 0;
  out[8] = xz;
  out[9] = yz;
  out[10] = zz;
  out[11] = 0;
  out[12] = -(xx * ex + xy * ey + xz * ez);
  out[13] = -(yx * ex + yy * ey + yz * ez);
  out[14] = -(zx * ex + zy * ey + zz * ez);
  out[15] = 1;
  return out;
}

/**
 * Writes a perspective projection for WebGPU's clip space: `x, y` in
 * `[-1, 1]`, **`z` in `[0, 1]`**, near at 0 and far at 1.
 *
 * `fovY` is the full vertical field of view in radians, measured on the short
 * side. `aspect` is width / height. The matrix is right-handed and maps a
 * point at `z = -near` to depth 0 and one at `z = -far` to depth 1, so the
 * camera's forward direction is negative Z in view space — the same convention
 * {@link lookAt} produces. A default `depthCompare` of `'less'` against a
 * cleared depth of 0 is then correct with no remap and no bias.
 *
 * `far` may be `Infinity`, which produces the infinite-far projection (the
 * horizon maps to depth 1 and the far plane is never reached). That is the
 * right choice for a skydome or an infinite ground plane, and the usual cause
 * of a wrong horizon when `far` is instead set to a large finite number.
 */
export function perspective(out: Float32Array, fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovY * 0.5);
  out[0] = f / aspect;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  out[4] = 0;
  out[5] = f;
  out[6] = 0;
  out[7] = 0;
  out[8] = 0;
  out[9] = 0;
  out[11] = -1;
  out[12] = 0;
  out[13] = 0;
  out[15] = 0;
  if (far === Infinity) {
    out[10] = -1;
    out[14] = -near;
  } else {
    const inv = 1 / (near - far);
    out[10] = far * inv;
    out[14] = far * near * inv;
  }
  return out;
}

/**
 * Writes an orthographic projection for the same WebGPU clip space as
 * {@link perspective}: `z` in `[0, 1]`, near at 0 and far at 1.
 *
 * The near/far distances are measured along the camera's forward direction, so
 * both are positive for a volume in front of the camera.
 */
export function orthographic(
  out: Float32Array,
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
): Float32Array {
  const lr = 1 / (left - right);
  const bt = 1 / (bottom - top);
  const nf = 1 / (near - far);
  out[0] = -2 * lr;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  out[4] = 0;
  out[5] = -2 * bt;
  out[6] = 0;
  out[7] = 0;
  out[8] = 0;
  out[9] = 0;
  out[10] = nf;
  out[11] = 0;
  out[12] = (left + right) * lr;
  out[13] = (top + bottom) * bt;
  out[14] = near * nf;
  out[15] = 1;
  return out;
}

/**
 * Verifies that `m` is usable as a `mat4`: a `Float32Array` of exactly 16
 * components.
 *
 * Explicit boundary check, never called from the operations above — see
 * {@link assertVec3} for the reasoning. Reach for it where a matrix enters
 * apse from a loader, an editor, or a test fixture.
 */
export function assertMat4(m: Float32Array, label: string): void {
  if (m instanceof Float32Array && m.length === 16) return;
  fail('INTERNAL_INVARIANT',
    `${label} is ${describe(m)}, not a Float32Array of 16 components.`, {
      why: 'apse reads and writes matrices at fixed indices, column by column. A wrong-length buffer leaves the trailing columns reading `undefined`, which becomes NaN in the product and blanking the frame rather than failing here.',
      fix: 'Pass a length-16 Float32Array, e.g. from `mat4.create()` or `mat4.identity(...)`. A 16-element view of a uniform block is fine.',
    });
}

function describe(m: unknown): string {
  if (m instanceof Float32Array) return `a Float32Array of ${m.length} components`;
  if (Array.isArray(m)) return `a plain array of ${m.length} numbers`;
  return `${typeof m} \`${String(m)}\``;
}
