/**
 * Torus.
 *
 * A tube swept around a circle. The one primitive here with a closed surface
 * in both directions, which is also why it is the best primitive in the set to
 * test topology against: every edge is shared by exactly two triangles and
 * every directed edge appears exactly once.
 */

import { fail } from '../../core/error.ts';
import { STANDARD_LAYOUT, type VertexLayout } from '../layout.ts';
import { allocateIndices, MeshData, VertexWriter } from '../mesh.ts';

export interface TorusOptions {
  /** Distance from the centre of the torus to the centre of the tube. */
  readonly radius?: number;
  /** Thickness of the tube. `radius - tube` is the inner radius, so keep tube < radius. */
  readonly tube?: number;
  /** Segments around the main ring. */
  readonly radialSegments?: number;
  /** Segments around the tube. */
  readonly tubularSegments?: number;
  readonly layout?: VertexLayout;
}

const MIN_SEGMENTS = 3;
const MAX_SEGMENTS = 512;

function clampSegments(value: number, field: string): number {
  if (!Number.isFinite(value)) {
    fail('OPTION_UNKNOWN',
      `torus() received ${field}: ${value}, which is not a finite number.`, {
      why: 'The segment counts drive every loop bound and every index computation in the generator, so a NaN or Infinity here silently produces an empty or unbounded buffer.',
      fix: `Pass a number in [${MIN_SEGMENTS}, ${MAX_SEGMENTS}].`,
    });
  }
  return Math.min(MAX_SEGMENTS, Math.max(MIN_SEGMENTS, Math.floor(value)));
}

/**
 * A torus in the XZ plane, centred on the origin.
 *
 * `(radialSegments + 1) * (tubularSegments + 1)` vertices and
 * `radialSegments * tubularSegments * 2` triangles, at 48 × 24 by default.
 *
 * Normals point away from the tube's own centre circle, not from the torus
 * centre — that is what makes the inner ring shade correctly, where a naive
 * "normalize(position)" would point every inner normal at the origin and light
 * the inside of the hole as if it were the outside.
 */
export function torus(opts: TorusOptions = {}): MeshData {
  const { radius = 1, tube = 0.4, layout = STANDARD_LAYOUT } = opts;
  const radialSegments = clampSegments(opts.radialSegments ?? 48, 'radialSegments');
  const tubularSegments = clampSegments(opts.tubularSegments ?? 24, 'tubularSegments');

  const rowStride = tubularSegments + 1;
  const vertexCount = rowStride * (radialSegments + 1);
  const indexCount = radialSegments * tubularSegments * 6;

  const w = new VertexWriter(layout, vertexCount);
  const out = w.data;
  const indices = allocateIndices(vertexCount, indexCount);
  const stride = w.floatsPerVertex;
  const pos = w.position;
  const nrm = w.normal;
  const uv = w.uv;
  const hasNormal = w.hasNormal;
  const hasUv = w.hasUv;

  let base = 0;
  for (let j = 0; j <= radialSegments; j++) {
    const v = j / radialSegments;
    const phi = v * Math.PI * 2;
    const cosP = Math.cos(phi);
    const sinP = Math.sin(phi);
    // The point on the main ring this row of the tube is swept around.
    const cx = radius * cosP;
    const cz = radius * sinP;
    for (let i = 0; i <= tubularSegments; i++) {
      const u = i / tubularSegments;
      const theta = u * Math.PI * 2;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);
      // Normal is (cos θ · cos φ, sin θ, cos θ · sin φ): unit by construction,
      // and pointing out of the tube in every direction.
      const nx = cosT * cosP;
      const ny = sinT;
      const nz = cosT * sinP;

      const o = base * stride;
      out[o + pos] = cx + tube * nx;
      out[o + pos + 1] = tube * ny;
      out[o + pos + 2] = cz + tube * nz;
      if (hasNormal) {
        out[o + nrm] = nx;
        out[o + nrm + 1] = ny;
        out[o + nrm + 2] = nz;
      }
      if (hasUv) {
        out[o + uv] = u;
        out[o + uv + 1] = v;
      }
      base++;
    }
  }

  // cross(∂u, ∂v) points outward for this parameterisation, so stepping along
  // the tube first — a, a+1, a+row+1, a+row — is the CCW-from-outside order.
  let ii = 0;
  for (let j = 0; j < radialSegments; j++) {
    const row = j * rowStride;
    for (let i = 0; i < tubularSegments; i++) {
      const a = row + i;
      const b = a + 1;
      const c = a + rowStride + 1;
      const e = a + rowStride;
      indices[ii++] = a;
      indices[ii++] = b;
      indices[ii++] = c;
      indices[ii++] = a;
      indices[ii++] = c;
      indices[ii++] = e;
    }
  }

  if (ii !== indexCount) {
    fail('INTERNAL_INVARIANT',
      `torus() wrote ${ii} indices but planned ${indexCount}.`, {
      why: 'The index count is derived from the clamped segment counts before the loop runs; a mismatch means the clamping and the arithmetic have diverged.',
      fix: 'Internal error in apse. Please report it with the arguments you passed to torus().',
    });
  }

  return new MeshData({
    name: 'torus',
    layout,
    vertices: { interleaved: out, vertexCount },
    indices,
  });
}
