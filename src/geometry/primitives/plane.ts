/**
 * Plane — a flat quad in the XZ plane, facing +Y.
 *
 * XZ rather than XY because this is the ground primitive: it drops onto y = 0
 * with no rotation, its normal is +Y, and it is the plane a shadow-catcher or
 * a height query wants.
 */

import { fail } from '../../core/error.ts';
import { STANDARD_LAYOUT, type VertexLayout } from '../layout.ts';
import { allocateIndices, MeshData, VertexWriter } from '../mesh.ts';

export interface PlaneOptions {
  readonly width?: number;
  readonly depth?: number;
  readonly widthSegments?: number;
  readonly depthSegments?: number;
  readonly layout?: VertexLayout;
  /** Mesh name, for error messages and tooling. Defaults to `'plane'`. */
  readonly name?: string;
}

/**
 * A plane centred on the origin, normal +Y, subdivided `widthSegments` ×
 * `depthSegments`.
 *
 * At the default 1 × 1 that is 4 vertices and 6 indices. UVs run `(0,0)` at the
 * -X/-Z corner to `(1,1)` at +X/+Z, with v increasing away from the camera's
 * default position, so a top-down image lands the right way up.
 */
export function plane(opts: PlaneOptions = {}): MeshData {
  const {
    width = 1,
    depth = 1,
    widthSegments = 1,
    depthSegments = 1,
    layout = STANDARD_LAYOUT,
    name = 'plane',
  } = opts;

  const ws = Math.max(1, Math.floor(widthSegments));
  const ds = Math.max(1, Math.floor(depthSegments));
  const rowStride = ws + 1;
  const vertexCount = rowStride * (ds + 1);
  const indexCount = ws * ds * 6;

  const w = new VertexWriter(layout, vertexCount);
  const out = w.data;
  const indices = allocateIndices(vertexCount, indexCount);
  const stride = w.floatsPerVertex;
  const pos = w.position;
  const nrm = w.normal;
  const uv = w.uv;
  const hasNormal = w.hasNormal;
  const hasUv = w.hasUv;

  const x0 = -width * 0.5;
  const z0 = -depth * 0.5;
  const sx = width / ws;
  const sz = depth / ds;

  let base = 0;
  for (let j = 0; j <= ds; j++) {
    const tv = j / ds;
    const z = z0 + sz * j;
    for (let i = 0; i <= ws; i++) {
      const tu = i / ws;
      const o = base * stride;
      out[o + pos] = x0 + sx * i;
      out[o + pos + 1] = 0;
      out[o + pos + 2] = z;
      if (hasNormal) {
        out[o + nrm] = 0;
        out[o + nrm + 1] = 1;
        out[o + nrm + 2] = 0;
      }
      if (hasUv) {
        out[o + uv] = tu;
        out[o + uv + 1] = 1 - tv;
      }
      base++;
    }
  }

  // Emitted as a, a+row, a+row+1, a+1: stepping along +Z before +X is CCW seen
  // from +Y, which is the side the normal points to.
  let ii = 0;
  for (let j = 0; j < ds; j++) {
    const row = j * rowStride;
    for (let i = 0; i < ws; i++) {
      const a = row + i;
      const b = a + rowStride;
      const c = b + 1;
      const e = a + 1;
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
      `plane() wrote ${ii} indices but planned ${indexCount}.`, {
      why: 'The index count is derived from the clamped segment counts before the loop runs; a mismatch means the clamping and the arithmetic have diverged.',
      fix: 'Internal error in apse. Please report it with the arguments you passed to plane().',
    });
  }

  return new MeshData({
    name,
    layout,
    vertices: { interleaved: out, vertexCount },
    indices,
  });
}
