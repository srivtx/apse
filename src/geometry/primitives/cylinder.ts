/**
 * Cylinder, cone, and open tube — one generator, three shapes.
 *
 * They are the same swept surface, distinguished only by whether the two end
 * radii differ and whether the ends are closed, so sharing the generator is not
 * a shortcut: a cone and a cylinder differ by exactly the values they are given.
 */

import { fail } from '../../core/error.ts';
import { STANDARD_LAYOUT, type VertexLayout } from '../layout.ts';
import { allocateIndices, MeshData, VertexWriter } from '../mesh.ts';

export interface CylinderOptions {
  readonly radiusTop?: number;
  readonly radiusBottom?: number;
  readonly height?: number;
  readonly radialSegments?: number;
  readonly heightSegments?: number;
  /** False leaves both ends open. With equal radii that is a tube. */
  readonly capped?: boolean;
  readonly layout?: VertexLayout;
}

const MIN_RADIAL = 3;
const MAX_RADIAL = 512;

function clampRadial(value: number): number {
  if (!Number.isFinite(value)) {
    fail('OPTION_UNKNOWN',
      `cylinder() received radialSegments: ${value}, which is not a finite number.`, {
      why: 'The segment count drives every loop bound and every index computation in the generator, so a NaN or Infinity here silently produces an empty or unbounded buffer.',
      fix: `Pass a number in [${MIN_RADIAL}, ${MAX_RADIAL}].`,
    });
  }
  return Math.min(MAX_RADIAL, Math.max(MIN_RADIAL, Math.floor(value)));
}

/**
 * A cylinder centred on the origin, +Y up.
 *
 * `radiusTop: 0` gives a cone and `radiusBottom: 0` an inverted one. A zero
 * radius collapses that whole ring to a single apex vertex rather than
 * `radialSegments` coincident ones, so the cone is a real fan of triangles
 * instead of a band of zero-area triangles that the rasteriser has to reject.
 *
 * `capped: false` with equal radii is an open tube: the side only, no end
 * discs, no interior to shade.
 *
 * The side's normal is tilted by the slope, so a cone shades as a cone. For a
 * straight cylinder the slope is zero and the normal is the plain radial
 * vector. Caps carry their own ±Y normals rather than inheriting the side's,
 * because a cap is a hard edge from any viewing angle that sees both.
 */
export function cylinder(opts: CylinderOptions = {}): MeshData {
  const {
    radiusTop = 1,
    radiusBottom = 1,
    height = 1,
    radialSegments = 32,
    heightSegments = 1,
    capped = true,
    layout = STANDARD_LAYOUT,
  } = opts;

  const rs = clampRadial(radialSegments);
  const hs = Math.max(1, Math.floor(heightSegments));
  const half = height * 0.5;

  // A row whose radius is exactly 0 is an apex. Only the two extreme rows can
  // be, and only when the corresponding end radius is 0.
  const apexRow: boolean[] = new Array<boolean>(hs + 1);
  const ringSize: number[] = new Array<number>(hs + 1);
  let sideVertices = 0;
  for (let j = 0; j <= hs; j++) {
    const v = j / hs;
    const r = radiusBottom + (radiusTop - radiusBottom) * v;
    const apex = r === 0;
    apexRow[j] = apex;
    ringSize[j] = apex ? 1 : rs + 1;
    sideVertices += ringSize[j];
  }

  const topCap = capped && radiusTop > 0 ? rs + 1 : 0;
  const bottomCap = capped && radiusBottom > 0 ? rs + 1 : 0;

  // A quad between a normal row and an apex row is really one triangle: the
  // apex makes half of it zero-area. Counting them here rather than emitting
  // slivers and hoping the rasteriser drops them keeps the index buffer honest.
  let sideTriangles = 0;
  for (let j = 0; j < hs; j++) {
    sideTriangles += ((apexRow[j] ? 0 : 1) + (apexRow[j + 1] ? 0 : 1)) * rs;
  }
  const vertexCount = sideVertices + topCap + bottomCap;
  const indexCount = sideTriangles * 3 +
    (topCap > 0 ? rs * 3 : 0) +
    (bottomCap > 0 ? rs * 3 : 0);

  const w = new VertexWriter(layout, vertexCount);
  const out = w.data;
  const indices = allocateIndices(vertexCount, indexCount);
  const stride = w.floatsPerVertex;
  const pos = w.position;
  const nrm = w.normal;
  const uv = w.uv;
  const hasNormal = w.hasNormal;
  const hasUv = w.hasUv;

  // (j, i) -> vertex index, so the index loop needs no knowledge of which rows
  // collapsed to an apex.
  const map = new Int32Array((hs + 1) * (rs + 1));

  // Side normal tilt. cos² + sin² collapses to 1, so the normalisation factor
  // is the same for every angle and can be hoisted out of both loops.
  const slope = height === 0 ? 0 : (radiusBottom - radiusTop) / height;
  const nScale = 1 / Math.sqrt(1 + slope * slope);

  let base = 0;
  for (let j = 0; j <= hs; j++) {
    const v = j / hs;
    const y = (v - 0.5) * height;
    const r = radiusBottom + (radiusTop - radiusBottom) * v;
    const count = ringSize[j];
    const apex = apexRow[j];
    for (let i = 0; i < count; i++) {
      const o = base * stride;
      if (apex) {
        // Every u collapses onto the centre of the uv square, so the texture
        // converges at the tip instead of tearing across it.
        out[o + pos] = 0;
        out[o + pos + 1] = y;
        out[o + pos + 2] = 0;
        if (hasNormal) {
          out[o + nrm] = 0;
          out[o + nrm + 1] = y > 0 ? 1 : -1;
          out[o + nrm + 2] = 0;
        }
        if (hasUv) {
          out[o + uv] = 0.5;
          out[o + uv + 1] = v;
        }
      } else {
        const u = i / rs;
        const theta = u * Math.PI * 2;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        out[o + pos] = r * cosT;
        out[o + pos + 1] = y;
        out[o + pos + 2] = r * sinT;
        if (hasNormal) {
          out[o + nrm] = cosT * nScale;
          out[o + nrm + 1] = slope * nScale;
          out[o + nrm + 2] = sinT * nScale;
        }
        if (hasUv) {
          out[o + uv] = u;
          out[o + uv + 1] = v;
        }
      }
      base++;
    }
    for (let i = 0; i <= rs; i++) {
      // Every column of a normal row is its own vertex; every column of an
      // apex row is the single vertex at that row's radius-0 point.
      map[j * (rs + 1) + i] = apex ? base - count : base - count + i;
    }
  }

  let capBase = base;

  // Caps, after the side, each a centre plus a `rs` ring. A fan needs no seam
  // duplication: the uv it interpolates is the centre of the disc, so the wrap
  // closes on itself.
  for (let c = 0; c < 2; c++) {
    if (c === 0 ? topCap === 0 : bottomCap === 0) continue;
    const r = c === 0 ? radiusTop : radiusBottom;
    const y = c === 0 ? half : -half;
    const ny = c === 0 ? 1 : -1;
    const o = capBase * stride;
    out[o + pos] = 0;
    out[o + pos + 1] = y;
    out[o + pos + 2] = 0;
    if (hasNormal) {
      out[o + nrm] = 0;
      out[o + nrm + 1] = ny;
      out[o + nrm + 2] = 0;
    }
    if (hasUv) {
      out[o + uv] = 0.5;
      out[o + uv + 1] = 0.5;
    }
    capBase++;
    for (let i = 0; i < rs; i++) {
      const theta = (i / rs) * Math.PI * 2;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);
      const e = capBase * stride;
      out[e + pos] = r * cosT;
      out[e + pos + 1] = y;
      out[e + pos + 2] = r * sinT;
      if (hasNormal) {
        out[e + nrm] = 0;
        out[e + nrm + 1] = ny;
        out[e + nrm + 2] = 0;
      }
      if (hasUv) {
        out[e + uv] = 0.5 + 0.5 * cosT;
        out[e + uv + 1] = 0.5 + 0.5 * sinT;
      }
      capBase++;
    }
  }

  // cross(∂θ, ∂y) points inward for this parameterisation, so the side is
  // emitted a, a+row, a+row+1, a+1 — down the column first. Both halves keep
  // that winding where they survive: the quad's orientation is continuous into
  // the triangle that is left when the other half collapses to an apex.
  let ii = 0;
  for (let j = 0; j < hs; j++) {
    const nearApex = apexRow[j];
    const farApex = apexRow[j + 1];
    for (let i = 0; i < rs; i++) {
      const a = map[j * (rs + 1) + i];
      const b = map[(j + 1) * (rs + 1) + i];
      const c = map[(j + 1) * (rs + 1) + i + 1];
      const e = map[j * (rs + 1) + i + 1];
      if (!farApex) {
        indices[ii++] = a;
        indices[ii++] = b;
        indices[ii++] = c;
      }
      if (!nearApex) {
        indices[ii++] = a;
        indices[ii++] = c;
        indices[ii++] = e;
      }
    }
  }

  // Top cap, then bottom. The two windings are opposite, because the same
  // increasing angle is CCW seen from one end and CW seen from the other.
  for (let c = 0; c < 2; c++) {
    if (c === 0 ? topCap === 0 : bottomCap === 0) continue;
    const start = sideVertices + (c === 1 ? topCap : 0);
    const centre = start;
    for (let i = 0; i < rs; i++) {
      const a = centre;
      const b = start + 1 + (i + 1) % rs;
      const e = start + 1 + i;
      if (c === 0) {
        indices[ii++] = a;
        indices[ii++] = b;
        indices[ii++] = e;
      } else {
        indices[ii++] = a;
        indices[ii++] = e;
        indices[ii++] = b;
      }
    }
  }

  if (ii !== indexCount) {
    fail('INTERNAL_INVARIANT',
      `cylinder() wrote ${ii} indices but planned ${indexCount}.`, {
      why: 'The index count is derived from the clamped segment counts and the cap flags before the loop runs; a mismatch means one of them has drifted.',
      fix: 'Internal error in apse. Please report it with the arguments you passed to cylinder().',
    });
  }

  return new MeshData({
    name: 'cylinder',
    layout,
    vertices: { interleaved: out, vertexCount },
    indices,
  });
}
