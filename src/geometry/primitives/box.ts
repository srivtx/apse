/**
 * Box — the axis-aligned cube, subdivided per face.
 *
 * Hard edges, so each of the six faces gets its own four corners and its own
 * normal. A cube with shared vertices is smaller but shades like a sphere, and
 * that is the whole reason apse's box is 24 vertices instead of 8.
 *
 * With the default `segments: 1` this is 24 vertices / 36 indices, which is the
 * cheapest way to get a closed, correctly-normalled, correctly-UV'd solid.
 */

import { fail } from '../../core/error.ts';
import { STANDARD_LAYOUT, type VertexLayout } from '../layout.ts';
import { allocateIndices, MeshData, VertexWriter } from '../mesh.ts';

export interface BoxOptions {
  readonly width?: number;
  readonly height?: number;
  readonly depth?: number;
  /**
   * Subdivisions per face, per axis. `1` gives the flat 24-vertex box; higher
   * values matter for vertex-lit or subdivided-shading boxes, and are wasted
   * work for anything else.
   */
  readonly segments?: number;
  readonly layout?: VertexLayout;
}

type Vec3 = readonly [number, number, number];

interface Face {
  readonly n: Vec3;
  /** Unit tangent, scaled by the box extent on the axis it points down. */
  readonly u: Vec3;
  /** Unit bitangent. `cross(u, v) === n`, so the quad order below faces out. */
  readonly v: Vec3;
  /** The (u=0, v=0) corner of the face. */
  readonly o: Vec3;
}

// Six faces, each wound CCW as seen from outside. `cross(u, v) === n` for every
// one of them, which is the single invariant the index generation relies on.
const FACES: readonly Face[] = [
  // +X — looking at the right face, u runs -Z and v runs +Y.
  { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0], o: [1, -1, 1] },
  // -X
  { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0], o: [-1, -1, -1] },
  // +Y — the top face, u runs +X and v runs -Z.
  { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1], o: [-1, 1, 1] },
  // -Y
  { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1], o: [-1, -1, -1] },
  // +Z — the front face, u runs +X and v runs +Y.
  { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0], o: [-1, -1, 1] },
  // -Z
  { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0], o: [1, -1, -1] },
];

/**
 * A box centred on the origin.
 *
 * 24 vertices and 36 indices at `segments: 1`; `segments: n` gives
 * `6 * (n + 1)²` vertices and `12 * n²` triangles. UVs run 0..1 across each
 * face independently, so the same texture lands on all six.
 */
export function box(opts: BoxOptions = {}): MeshData {
  const {
    width = 1,
    height = 1,
    depth = 1,
    segments = 1,
    layout = STANDARD_LAYOUT,
  } = opts;

  const seg = Math.max(1, Math.floor(segments));
  const extent: Vec3 = [width, height, depth];
  const side = seg + 1;
  const perFace = side * side;
  const vertexCount = perFace * FACES.length;
  const indexCount = FACES.length * seg * seg * 6;

  const w = new VertexWriter(layout, vertexCount);
  const out = w.data;
  const indices = allocateIndices(vertexCount, indexCount);
  const stride = w.floatsPerVertex;
  const pos = w.position;
  const nrm = w.normal;
  const uv = w.uv;

  let ii = 0;
  let base = 0;
  for (let f = 0; f < FACES.length; f++) {
    const face = FACES[f];
    const nx = face.n[0];
    const ny = face.n[1];
    const nz = face.n[2];
    const ox = face.o[0] * width * 0.5;
    const oy = face.o[1] * height * 0.5;
    const oz = face.o[2] * depth * 0.5;
    const ux = face.u[0] * extent[0];
    const uy = face.u[1] * extent[1];
    const uz = face.u[2] * extent[2];
    const vx = face.v[0] * extent[0];
    const vy = face.v[1] * extent[1];
    const vz = face.v[2] * extent[2];

    for (let j = 0; j <= seg; j++) {
      const tv = j / seg;
      const cj = ox + vx * tv;
      const dj = oy + vy * tv;
      const ej = oz + vz * tv;
      for (let i = 0; i <= seg; i++) {
        const tu = i / seg;
        const o = base * stride;
        out[o + pos] = cj + ux * tu;
        out[o + pos + 1] = dj + uy * tu;
        out[o + pos + 2] = ej + uz * tu;
        if (w.hasNormal) {
          out[o + nrm] = nx;
          out[o + nrm + 1] = ny;
          out[o + nrm + 2] = nz;
        }
        if (w.hasUv) {
          out[o + uv] = tu;
          out[o + uv + 1] = tv;
        }
        base++;
      }
    }

    for (let j = 0; j < seg; j++) {
      const row = f * perFace + j * side;
      for (let i = 0; i < seg; i++) {
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
      `box() wrote ${ii} indices but planned ${indexCount}.`, {
      why: 'The index count is computed from segments and face count before the loop runs; a mismatch means the face table and the arithmetic have diverged.',
      fix: 'Internal error in apse. Please report it with the arguments you passed to box().',
    });
  }

  return new MeshData({
    name: 'box',
    layout,
    vertices: { interleaved: out, vertexCount },
    indices,
  });
}
