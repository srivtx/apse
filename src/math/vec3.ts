/**
 * 3-component vectors.
 *
 * A vector is a plain `Float32Array` of length 3 — not a class, not
 * `{x, y, z}`. That choice is load-bearing: a `vec3<f32>` in a WGSL uniform
 * block is three consecutive f32s at a 16-byte boundary, so a length-3
 * `Float32Array` view over a uniform block *is* the GPU-side value. No
 * marshalling, no per-write object churn, no `.x/.y/.z` property loads in the
 * frame loop.
 *
 * Every operation writes into a caller-supplied `out` and returns it. `out`
 * may alias an input except where the doc says otherwise. The only functions
 * that allocate are {@link create} and {@link clone}; everything else is safe
 * to call per object per frame.
 *
 * Fast paths are **unchecked by design**: lengths are not re-verified on every
 * call, because a `v.length !== 3` check per vector per frame costs more than
 * the math. Use {@link assertVec3} at the boundaries where vectors enter apse
 * from outside — mesh loading, scene authoring, deserialisation.
 */

import { fail } from '../core/error.ts';

/** Allocates a new `vec3`. The only allocating constructor-style function here. */
export function create(x = 0, y = 0, z = 0): Float32Array {
  const out = new Float32Array(3);
  out[0] = x;
  out[1] = y;
  out[2] = z;
  return out;
}

/** Allocates a copy of `v`. Truncates or zero-pads to length 3. */
export function clone(v: Float32Array): Float32Array {
  const out = new Float32Array(3);
  out[0] = v[0];
  out[1] = v[1];
  out[2] = v[2];
  return out;
}

/** Writes `(0, 0, 0)`. */
export function zero(out: Float32Array): Float32Array {
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;
  return out;
}

/** Writes the three components. `out` may alias any input. */
export function set(out: Float32Array, x: number, y: number, z: number): Float32Array {
  out[0] = x;
  out[1] = y;
  out[2] = z;
  return out;
}

/** Copies `v` into `out`. Aliasing `out === v` is a no-op. */
export function copy(out: Float32Array, v: Float32Array): Float32Array {
  out[0] = v[0];
  out[1] = v[1];
  out[2] = v[2];
  return out;
}

/** Component-wise sum. `out` may alias either input. */
export function add(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  const ax = a[0];
  const ay = a[1];
  const az = a[2];
  out[0] = ax + b[0];
  out[1] = ay + b[1];
  out[2] = az + b[2];
  return out;
}

/** Component-wise difference. `out` may alias either input. */
export function sub(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  const ax = a[0];
  const ay = a[1];
  const az = a[2];
  out[0] = ax - b[0];
  out[1] = ay - b[1];
  out[2] = az - b[2];
  return out;
}

/** Uniform scale. `out` may alias `a`. */
export function scale(out: Float32Array, a: Float32Array, s: number): Float32Array {
  const ax = a[0];
  const ay = a[1];
  const az = a[2];
  out[0] = ax * s;
  out[1] = ay * s;
  out[2] = az * s;
  return out;
}

/** Writes `a + b * s`. The accumulate-into-a-term form; avoids a temp vector. */
export function addScaled(out: Float32Array, a: Float32Array, b: Float32Array, s: number): Float32Array {
  out[0] = a[0] + b[0] * s;
  out[1] = a[1] + b[1] * s;
  out[2] = a[2] + b[2] * s;
  return out;
}

/** Component-wise negation. `out` may alias `a`. */
export function negate(out: Float32Array, a: Float32Array): Float32Array {
  out[0] = -a[0];
  out[1] = -a[1];
  out[2] = -a[2];
  return out;
}

/** Component-wise minimum. NaN in either input propagates, per IEEE-754. */
export function min(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  out[0] = a[0] < b[0] ? a[0] : b[0];
  out[1] = a[1] < b[1] ? a[1] : b[1];
  out[2] = a[2] < b[2] ? a[2] : b[2];
  return out;
}

/** Component-wise maximum. NaN in either input propagates, per IEEE-754. */
export function max(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  out[0] = a[0] > b[0] ? a[0] : b[0];
  out[1] = a[1] > b[1] ? a[1] : b[1];
  out[2] = a[2] > b[2] ? a[2] : b[2];
  return out;
}

/** Dot product. */
export function dot(a: Float32Array, b: Float32Array): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Writes the right-handed cross product `a × b`. `out` may alias either input. */
export function cross(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  const ax = a[0];
  const ay = a[1];
  const az = a[2];
  const bx = b[0];
  const by = b[1];
  const bz = b[2];
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}

/** Euclidean length. */
export function length(v: Float32Array): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/** Squared length. Compare against a squared threshold to skip the sqrt. */
export function lengthSquared(v: Float32Array): number {
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

/** Distance between two points. */
export function distance(a: Float32Array, b: Float32Array): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Writes `v / |v|`.
 *
 * A zero-length input writes zeros rather than NaN. Normalising a direction
 * that does not exist has no meaningful answer, and zeros keep the damage
 * local to the one vector instead of poisoning every value computed from it.
 * Callers that care should branch on {@link length} first — for culling and
 * intersection code that branch is usually already there.
 */
export function normalize(out: Float32Array, v: Float32Array): Float32Array {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  const len = Math.sqrt(x * x + y * y + z * z);
  if (len === 0) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    return out;
  }
  const inv = 1 / len;
  out[0] = x * inv;
  out[1] = y * inv;
  out[2] = z * inv;
  return out;
}

/** Linear interpolation. `t` is unclamped, so it also extrapolates. */
export function lerp(out: Float32Array, a: Float32Array, b: Float32Array, t: number): Float32Array {
  const ax = a[0];
  const ay = a[1];
  const az = a[2];
  out[0] = ax + (b[0] - ax) * t;
  out[1] = ay + (b[1] - ay) * t;
  out[2] = az + (b[2] - az) * t;
  return out;
}

/**
 * Verifies that `v` is usable as a `vec3`: a `Float32Array` of exactly 3
 * components.
 *
 * Deliberately *not* called by the operations above. The math is written for
 * unchecked buffers so the per-call cost stays near zero; this is the explicit
 * check for the places where a vector arrives from outside apse, where a
 * wrong-length array would otherwise read `undefined` and write NaN.
 */
export function assertVec3(v: Float32Array, label: string): void {
  if (v instanceof Float32Array && v.length === 3) return;
  fail('INTERNAL_INVARIANT',
    `${label} is ${describe(v)}, not a Float32Array of 3 components.`, {
      why: 'apse writes vectors component by component into fixed-length buffers. A wrong-length or non-f32 buffer reads `undefined`, which becomes NaN in the matrix multiply downstream and blanks the frame rather than failing here.',
      fix: 'Pass a length-3 Float32Array, e.g. from `vec3.create(x, y, z)`. If this value came from a packed buffer, take a 3-element view of it rather than the whole array.',
    });
}

function describe(v: unknown): string {
  if (v instanceof Float32Array) return `a Float32Array of ${v.length} components`;
  if (Array.isArray(v)) return `a plain array of ${v.length} numbers`;
  return `${typeof v} \`${String(v)}\``;
}
