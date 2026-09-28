/**
 * Grid — a ground-plane wireframe lattice.
 *
 * The same surface as `plane()`, with defaults chosen for a viewport-filling
 * ground: 10 × 10 cells over 10 × 10 world units, so one cell is one unit and
 * the whole thing reads as a floor without a texture.
 *
 * Intended for wireframe rendering (a material with `topology: 'line-list'`
 * drawing `indexCount`/`2` lines, or a fragment-stage grid function reading the
 * UVs) and for debug overlays. At the defaults that is 121 vertices and 200
 * triangles — the vertex count is `(widthSegments + 1) * (depthSegments + 1)`,
 * shared corner vertices, which is why it stays cheap as the cell count grows.
 */

import { STANDARD_LAYOUT, type VertexLayout } from '../layout.ts';
import type { MeshData } from '../mesh.ts';
import { plane } from './plane.ts';
export interface GridOptions {
  readonly width?: number;
  readonly depth?: number;
  readonly widthSegments?: number;
  readonly depthSegments?: number;
  readonly layout?: VertexLayout;
}

/**
 * A grid centred on the origin in the XZ plane, normal +Y.
 *
 * Defaults to 10 × 10 units over 10 × 10 segments: 121 vertices, 200
 * triangles, one unit per cell. Delegates to {@link plane} so the two can never
 * disagree about winding, orientation, or uv direction.
 */
export function grid(opts: GridOptions = {}): MeshData {
  const {
    width = 10,
    depth = 10,
    widthSegments = 10,
    depthSegments = 10,
    layout = STANDARD_LAYOUT,
  } = opts;

  return plane({ width, depth, widthSegments, depthSegments, layout, name: 'grid' });
}
