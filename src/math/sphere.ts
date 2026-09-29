/**
 * Bounding volumes: a sphere and an axis-aligned box.
 *
 * Both are a plain object holding borrowed `Float32Array`s plus a radius, not a
 * class. They are the input to frustum culling, and culling runs per object per
 * frame, so neither a property load through a prototype chain nor a per-call
 * allocation is acceptable there.
 *
 * A sphere is preferred over a box for culling because it survives rotation
 * without being re-fitted: a sphere is rotation-invariant, so the only case
 * that ever makes it stale is a non-uniform scale, handled exactly in
 * {@link transform}. A box is kept for the cases where the tight fit matters
 * more than the rotation invariance — mesh bounds, ray-box tests, tiled
 * selection.
 */

/**
 * A sphere. `center` is a length-3 `Float32Array`; apse does not copy it, so
 * the owner must keep it alive and must not resize it in place.
 */
export interface Sphere {
  center: Float32Array;
  /** Always non-negative. A negative radius is a caller bug, not a clamped value. */
  radius: number;
}

/**
 * An axis-aligned box. `min` and `max` are length-3 `Float32Array`s, borrowed
 * on the same terms as {@link Sphere.center}. `min` must be component-wise less
 * than or equal to `max`; nothing in apse enforces that, because enforcing it
 * costs a comparison in the cull path.
 */
export interface Aabb {
  min: Float32Array;
  max: Float32Array;
}

/**
 * Allocates a sphere at the origin with radius 1.
 *
 * Takes an optional centre to borrow instead of allocating, so a caller with a
 * per-instance centre array can make a sphere per instance without allocating
 * per instance.
 */
export function create(center: Float32Array = new Float32Array(3), radius = 1): Sphere {
  return { center, radius };
}

/** Allocates an empty box, ready for {@link aabbSet}. */
export function aabbCreate(): Aabb {
  return { min: new Float32Array(3), max: new Float32Array(3) };
}

/** Writes centre and radius. `out.center` is borrowed, not replaced. */
export function set(out: Sphere, x: number, y: number, z: number, radius: number): Sphere {
  out.center[0] = x;
  out.center[1] = y;
  out.center[2] = z;
  out.radius = radius;
  return out;
}

/** Writes the box corners. `out.min` and `out.max` are borrowed. */
export function aabbSet(out: Aabb, minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): Aabb {
  out.min[0] = minX;
  out.min[1] = minY;
  out.min[2] = minZ;
  out.max[0] = maxX;
  out.max[1] = maxY;
  out.max[2] = maxZ;
  return out;
}

/** Copies the bounds of `a` into `out`. `out.center` keeps pointing at its own array. */
export function copy(out: Sphere, a: Sphere): Sphere {
  out.center[0] = a.center[0];
  out.center[1] = a.center[1];
  out.center[2] = a.center[2];
  out.radius = a.radius;
  return out;
}

/** Copies the corners of `a` into `out`. */
export function aabbCopy(out: Aabb, a: Aabb): Aabb {
  out.min[0] = a.min[0];
  out.min[1] = a.min[1];
  out.min[2] = a.min[2];
  out.max[0] = a.max[0];
  out.max[1] = a.max[1];
  out.max[2] = a.max[2];
  return out;
}

/**
 * Writes the sphere `m` maps `s` to: the transformed centre, and the radius
 * scaled by the **largest** axis scale of `m`.
 *
 * Taking the maximum of the three basis-column lengths rather than a single
 * average is what makes the result conservative. Under a non-uniform scale the
 * sphere is mapped to an ellipsoid, and no sphere contains it; the tight choice
 * is the ellipsoid's largest semi-axis, so the cull test stays sound — the
 * worst case over a sheared basis is an over-estimate, which costs a draw of
 * something already invisible rather than the culling of something visible.
 *
 * In practice `m` is a model matrix, whose basis columns are exactly the scale
 * factors, so `m[0..2]`, `m[4..6]`, `m[8..10]` are read directly.
 *
 * `out.center` must not alias `s.center`; `s` is read before any store.
 */
export function transform(out: Sphere, s: Sphere, m: Float32Array): Sphere {
  const cx = s.center[0];
  const cy = s.center[1];
  const cz = s.center[2];
  const inv = 1 / (m[3] * cx + m[7] * cy + m[11] * cz + m[15]);

  out.center[0] = (m[0] * cx + m[4] * cy + m[8] * cz + m[12]) * inv;
  out.center[1] = (m[1] * cx + m[5] * cy + m[9] * cz + m[13]) * inv;
  out.center[2] = (m[2] * cx + m[6] * cy + m[10] * cz + m[14]) * inv;

  // Column lengths are the scale on each basis axis. `sqrt` is monotonic, so
  // the longest column is the one with the largest *squared* length: comparing
  // the squares and taking a single square root at the end is the same answer
  // bit for bit, for two fewer square roots.
  const lx = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
  const ly = m[4] * m[4] + m[5] * m[5] + m[6] * m[6];
  const lz = m[8] * m[8] + m[9] * m[9] + m[10] * m[10];
  out.radius = s.radius * Math.sqrt(lx > ly ? (lx > lz ? lx : lz) : ly > lz ? ly : lz);
  return out;
}

/**
 * Writes the smallest sphere containing both `a` and `b`.
 *
 * When one already contains the other it is copied through unchanged, so
 * repeatedly unioning a growing set does not inflate the bound the way a naive
 * centre/radius average does. Otherwise the new centre sits on the line between
 * the two centres, placed so that both spheres are exactly enclosed.
 *
 * `out` must not alias `a` or `b`; the centre and radius are read before any
 * store, but writing through an aliased `out` would be a silent no-op.
 */
export function union(out: Sphere, a: Sphere, b: Sphere): Sphere {
  const dx = b.center[0] - a.center[0];
  const dy = b.center[1] - a.center[1];
  const dz = b.center[2] - a.center[2];
  const d2 = dx * dx + dy * dy + dz * dz;
  const ar = a.radius;
  const br = b.radius;

  // Containment tested squared, to keep the sqrt out of the common case. The
  // radius comparison matters: (r - other)² is only the containment condition
  // when r is the larger radius, and skipping it would report containment for
  // two spheres that merely overlap.
  if (ar >= br && d2 <= (ar - br) * (ar - br)) return copy(out, a);
  if (br >= ar && d2 <= (br - ar) * (br - ar)) return copy(out, b);

  const d = Math.sqrt(d2);
  const r = (d + ar + br) * 0.5;
  // Unreachable as d === 0, which both branches above already claimed.
  const k = d === 0 ? 0 : (r - ar) / d;

  out.center[0] = a.center[0] + dx * k;
  out.center[1] = a.center[1] + dy * k;
  out.center[2] = a.center[2] + dz * k;
  out.radius = r;
  return out;
}

/** True when `p` is inside or on the surface of `s`. */
export function containsPoint(s: Sphere, p: Float32Array): boolean {
  const dx = p[0] - s.center[0];
  const dy = p[1] - s.center[1];
  const dz = p[2] - s.center[2];
  return dx * dx + dy * dy + dz * dz <= s.radius * s.radius;
}

/** True when `p` is inside or on the surface of `a`. */
export function aabbContainsPoint(a: Aabb, p: Float32Array): boolean {
  for (let i = 0; i < 3; i++) {
    const v = p[i];
    if (v < a.min[i] || v > a.max[i]) return false;
  }
  return true;
}
