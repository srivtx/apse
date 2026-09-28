/**
 * Rounded box — a box with the edges and corners filleted.
 *
 * ## Why a projection and not a bevel
 *
 * The construction is the one every rounded-box generator uses: take a
 * subdivided box, and move every vertex onto the surface of the same box
 * shrunk by the corner radius, along the direction from that inner point.
 *
 * ```txt
 *   core_i = clamp(p_i, -(h_i − r), h_i − r)
 *   n      = normalize(p − core)
 *   p'     = core + r · n
 * ```
 *
 * Three exact properties fall out of it, and they are why the bevel
 * alternative is not worth having:
 *
 *   - The six flat faces are **exactly** the faces of a box of half-extent
 *     `h`. The projection is the identity there, so a rounded box has the same
 *     silhouette as its box, and the only difference is a fillet along the
 *     edges — which is what "rounded" is supposed to mean.
 *   - Each of the eight corners is a **sphere octant** of radius `r` about the
 *     inner corner, and each of the twelve edges is a quarter cylinder. They
 *     are not approximated; the positions satisfy the equations exactly.
 *   - The faces' own vertex grids **weld**: the corner of the +X face and the
 *     corner of the +Y face project to the same point, because they are the
 *     same point on the same surface.
 *
 * ## The sampling, which is the part that is easy to get wrong
 *
 * A uniform subdivision wastes almost all of its triangles on the flat faces,
 * where every quad is coplanar with its neighbour and none of them change the
 * shape. So the parameter is remapped: the rounded bands at each end of every
 * axis get `segments` intervals each, and the flat middle gets **one** — a
 * plane is exactly a quad. At `segments: 2` that is 3 intervals per axis and a
 * 4 × 4 grid per face; at `segments: 6` it is 13 and a 14 × 14 grid, and the
 * corner sphere has the resolution you would expect from 6 segments rather than
 * from a uniform grid that spent most of them on a face that did not need them.
 *
 * The cost of that choice is that the vertex count is quadratic in `segments`
 * with a larger constant than a uniform grid of the same nominal density. It is
 * still a fraction of a `sphere()` of comparable smoothness, and it is the
 * correct place to spend them.
 */

import { fail } from '../../core/error.ts';
import { STANDARD_LAYOUT, type VertexLayout } from '../layout.ts';
import { allocateIndices, MeshData, VertexWriter } from '../mesh.ts';
import { BOX_FACES, type BoxFace } from './box.ts';

export interface RoundedBoxOptions {
  readonly width?: number;
  readonly height?: number;
  readonly depth?: number;
  /**
   * Segments across each rounded band. The flat middle of each face is one quad
   * regardless, so `segments: 2` is a 4 × 4 grid per face and `segments: 6` a
   * 14 × 14 one. Clamped to [1, 64], default 3.
   */
  readonly segments?: number;
  /**
   * Corner radius. Must be positive and strictly less than half the smallest
   * extent, so that the flat middle of every face has a real width.
   */
  readonly radius?: number;
  readonly layout?: VertexLayout;
}

const MAX_SEGMENTS = 64;

/**
 * Remaps a face sample index onto a **unit** box axis, concentrating the samples
 * in the rounded bands.
 *
 * `segments` intervals into each end band and one across the flat middle, so
 * `side = 2 * segments + 2` samples and `2 * segments + 1` intervals. The two
 * middle samples land exactly on `±(1 − r)`, which is where the fillet meets the
 * face and where the surface is tangent-continuous across it.
 *
 * The axis is the unit one on purpose: the extents belong in the projection, not
 * in the parameterisation, and a remap that took the extent would have to be
 * re-derived per face.
 */
function axisUnit(k: number, segments: number, r: number): number {
  const flat = 1 - r;
  if (k <= segments) return -1 + (k / segments) * r;
  if (k === segments + 1) return flat;
  return flat + ((k - segments - 1) / segments) * r;
}

/**
 * A box of `width × height × depth` with edges and corners filleted by `radius`,
 * centred on the origin.
 *
 * `6 · (2 · segments + 2)²` vertices and `12 · (2 · segments + 1)²` triangles —
 * 384 vertices and 588 triangles at the default `segments: 3`, and 216 and 300 at
 * `segments: 2`. The mesh is
 * closed, consistently wound, and after welding by position it is a
 * topological sphere: the 8 corners, 12 edges and 6 faces are the same three
 * pieces a box has, only curved.
 *
 * At `radius: 0` this is `box()` with a different vertex count and the same
 * shape, so a zero radius is refused rather than quietly producing a denser box:
 * a caller who wrote `radius: 0` wanted a box, and `box()` is the cheaper way to
 * say it.
 */
export function roundedBox(opts: RoundedBoxOptions = {}): MeshData {
  const {
    width = 1,
    height = 1,
    depth = 1,
    radius = 0.1,
    layout = STANDARD_LAYOUT,
  } = opts;

  const seg = Math.min(MAX_SEGMENTS, Math.max(1, Math.floor(opts.segments ?? 3)));
  const hx = width * 0.5, hy = height * 0.5, hz = depth * 0.5;

  if (!Number.isFinite(radius) || radius <= 0) {
    fail('OPTION_UNKNOWN',
      `roundedBox() received radius: ${radius}.`, {
      why: 'The corner radius is the distance from the inner box to the surface, and the projection is `inner + radius · normalize(p − inner)`. A zero radius makes every vertex on a flat face collapse onto the inner box, so the solid shrinks to its inner face and the caller gets a smaller box than they asked for.',
      fix: 'Pass a positive radius, or use `box()` for sharp edges. A radius of half the smallest extent is the largest that still leaves a flat face.',
    });
  }
  const smallest = Math.min(hx, hy, hz);
  if (!(radius < smallest) || !Number.isFinite(smallest)) {
    fail('OPTION_UNKNOWN',
      `roundedBox() cannot fillet a ${width} × ${height} × ${depth} box by ${radius}.`, {
      why: 'The fillet is applied from the inside out: each axis keeps a flat middle of half-width `extent/2 − radius`, and that has to be positive. At or past half the smallest extent the flat collapses, the projection stops being injective, and the two ends of an axis meet — a mesh with zero-area triangles in a band and a shape nobody asked for.',
      fix: `Use a radius below ${smallest} (half of the smallest extent), or make the box bigger on that axis.`,
      detail: { kind: 'numeric', field: 'radius', value: radius, min: 0, max: smallest },
    });
  }

  // The unit radius, so the remap is a pure function of the parameter and the
  // extents enter only where they belong: in the final projection.
  const ru = radius / smallest;
  const side = 2 * seg + 2;
  const perFace = side * side;
  const vertexCount = perFace * BOX_FACES.length;
  const intervals = side - 1;
  const indexCount = BOX_FACES.length * intervals * intervals * 6;

  const w = new VertexWriter(layout, vertexCount);
  const out = w.data;
  const indices = allocateIndices(vertexCount, indexCount);
  const stride = w.floatsPerVertex;
  const pos = w.position, nrm = w.normal, uv = w.uv;
  const hasNormal = w.hasNormal, hasUv = w.hasUv;

  let ii = 0;
  let base = 0;
  for (let f = 0; f < BOX_FACES.length; f++) {
    const face: BoxFace = BOX_FACES[f];
    // The grid is parameterised about the face's **centre**, not about a corner:
    // the remap is symmetric about zero (see {@link axisUnit}), so the origin has
    // to be too, and the face centre is the only point of the face whose two
    // tangent offsets are both zero. The tangents are scaled by the half-extent
    // so the parameter runs -1..1 across the whole face.
    const nx0 = face.n[0], ny0 = face.n[1], nz0 = face.n[2];
    const ox = nx0 * hx, oy = ny0 * hy, oz = nz0 * hz;
    const ux = face.u[0] * hx, uy = face.u[1] * hy, uz = face.u[2] * hz;
    const vx = face.v[0] * hx, vy = face.v[1] * hy, vz = face.v[2] * hz;

    for (let j = 0; j < side; j++) {
      const tv = axisUnit(j, seg, ru);
      for (let i = 0; i < side; i++) {
        const tu = axisUnit(i, seg, ru);
        // The pre-projection point: the face plane at the remapped parameter.
        const px = ox + ux * tu + vx * tv;
        const py = oy + uy * tu + vy * tv;
        const pz = oz + uz * tu + vz * tv;

        // The inner box: each axis clamped to its own flat half-width. The
        // clamp is per axis and in absolute units, so the same expression works
        // for a face whose `u` runs down X and one whose `u` runs down Z.
        const cx = clampAxis(px, hx, radius);
        const cy = clampAxis(py, hy, radius);
        const cz = clampAxis(pz, hz, radius);
        let dx = px - cx, dy = py - cy, dz = pz - cz;
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        let nx: number, ny: number, nz: number;
        if (len > 1e-12) {
          nx = dx / len; ny = dy / len; nz = dz / len;
          dx = cx + nx * radius; dy = cy + ny * radius; dz = cz + nz * radius;
        } else {
          // The projection is the identity in the flat middle, so this is the
          // face normal: the direction of the axis the face is perpendicular to.
          nx = face.n[0]; ny = face.n[1]; nz = face.n[2];
          dx = px; dy = py; dz = pz;
        }

        const o = base * stride;
        out[o + pos] = dx; out[o + pos + 1] = dy; out[o + pos + 2] = dz;
        if (hasNormal) { out[o + nrm] = nx; out[o + nrm + 1] = ny; out[o + nrm + 2] = nz; }
        // The grid is symmetric about the origin, so the uv is the remap
        // shifted into 0..1 — the same 0..1-per-face contract `box()` has.
        if (hasUv) { out[o + uv] = tu * 0.5 + 0.5; out[o + uv + 1] = tv * 0.5 + 0.5; }
        base++;
      }
    }

    for (let j = 0; j < intervals; j++) {
      const row = f * perFace + j * side;
      for (let i = 0; i < intervals; i++) {
        const a = row + i;
        const b = a + 1;
        const c = a + side + 1;
        const q = a + side;
        indices[ii++] = a;
        indices[ii++] = b;
        indices[ii++] = c;
        indices[ii++] = a;
        indices[ii++] = c;
        indices[ii++] = q;
      }
    }
  }

  if (ii !== indexCount) {
    fail('INTERNAL_INVARIANT',
      `roundedBox() wrote ${ii} indices but planned ${indexCount}.`, {
      why: 'The index count is six faces of `intervals²` quads, computed before the loop runs; a mismatch means the grid arithmetic and the count have diverged.',
      fix: 'Internal error in apse. Please report it with the arguments you passed to roundedBox().',
    });
  }

  return new MeshData({
    name: 'roundedBox',
    layout,
    vertices: { interleaved: out, vertexCount },
    indices,
  });
}

function clampAxis(p: number, half: number, r: number): number {
  const flat = half - r;
  if (p < -flat) return -flat;
  if (p > flat) return flat;
  return p;
}
