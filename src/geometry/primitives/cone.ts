/**
 * Cone — a cylinder with one radius at zero.
 *
 * `cone()` delegates to {@link cylinder} rather than implementing its own swept
 * surface, and that is not a shortcut: a cone *is* a cylinder with
 * `radiusTop: 0`, and a second generator for the same parameterisation is a
 * second set of things to get the winding, the slope-corrected normals and the
 * apex collapse wrong. The delegated mesh is byte-for-byte what
 * `cylinder({ radiusTop: 0, ... })` produces.
 *
 * The reason it is still worth having is the call site. A cone is a shape, not a
 * cylinder with an argument, and `cone({ radius: 1, height: 2 })` cannot be
 * expressed as `{ radiusBottom: 1, radiusTop: 0, height: 2 }` by someone who
 * does not already know which of the two radii is which.
 */

import { cylinder } from './cylinder.ts';
import type { CylinderOptions } from './cylinder.ts';

export interface ConeOptions {
  /** Radius of the base. Defaults to 1. */
  readonly radius?: number;
  readonly height?: number;
  readonly radialSegments?: number;
  /** False leaves the base open. Defaults to true. */
  readonly capped?: boolean;
  readonly layout?: CylinderOptions['layout'];
}

/**
 * A cone centred on the origin, base down at −Y, apex at +Y.
 *
 * `radialSegments + 2` vertices at `radialSegments` segments: the base ring
 * plus the collapsed apex row, which is one vertex rather than a ring of
 * coincident ones, so there are no zero-area triangles for the rasteriser to
 * reject. With the default `capped: true` the base adds a centre and a ring, so
 * `2 * (radialSegments + 2)` vertices and `3 * radialSegments` triangles.
 */
export function cone(opts: ConeOptions = {}) {
  const { radius = 1, height = 1, radialSegments = 32, capped = true, layout } = opts;
  return cylinder({ radiusTop: 0, radiusBottom: radius, height, radialSegments, capped, layout });
}
