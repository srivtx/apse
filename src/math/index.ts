/**
 * The math module, as five namespaces.
 *
 * These are exported as namespace objects rather than flattened into one list
 * because the short names collide by design: every type needs `create`,
 * `identity`, `copy`, `add`, `scale`, `normalize`, `invert`, and `transform`.
 * Flattening would force `mat4Scale` on half of them and `vec3Scale` on the
 * rest, which is the prefix-soup naming this module exists to avoid.
 *
 *     import { mat4, quat, vec3 } from 'apse/math';
 *
 *     const view = mat4.lookAt(mat4.create(), eye, target, vec3.create(0, 1, 0));
 *     const viewProj = mat4.mul(mat4.create(), proj, view);
 *
 * Both import styles are supported and equivalent in size; the namespaces are
 * a naming decision, not a namespace-objects-are-slow decision — every one of
 * these is a plain frozen object of function references.
 *
 * ## The three rules
 *
 * 1. **No classes.** Vectors, matrices, and quaternions are `Float32Array`s.
 *    A `Float32Array` of the right length *is* the GPU-side value, so a uniform
 *    write is a `copy`-free direct upload rather than a marshalling step.
 * 2. **Column-major.** `m[c * 4 + r]`, matching WGSL and GLSL memory layout, so
 *    no transpose is needed anywhere. See `mat4.ts` for the consequences.
 * 3. **No Euler angles.** Rotation is a quaternion, everywhere, always. See
 *    `quat.ts` for why, and for the one conversion that does exist.
 *
 * Allocation is confined to the `create` and `clone` constructors. Everything
 * else writes into a caller-owned output buffer and returns it, so a scene
 * update allocates nothing at all.
 */

export * as vec3 from './vec3.ts';
export * as mat4 from './mat4.ts';
export * as quat from './quat.ts';
export * as sphere from './sphere.ts';
export * as frustum from './frustum.ts';

export type { Aabb, Sphere } from './sphere.ts';
