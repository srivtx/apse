/**
 * Primitives — the six shapes apse ships, as pure functions.
 *
 * Every one returns a `MeshData`: no classes, no GPU, no device. A primitive is
 * a value you can build once, cache, and hand to as many meshes as you like, so
 * there is nothing here that needs instance state or a lifecycle.
 *
 * Every one takes a `layout` and honours its subset. A `POSITION_LAYOUT` sphere
 * skips the normal and uv stores entirely rather than computing them and
 * throwing them away, and the buffer is 12 bytes per vertex instead of 32.
 */

export { box, type BoxOptions } from './box.ts';
export { sphere, type SphereOptions } from './sphere.ts';
export { plane, type PlaneOptions } from './plane.ts';
export { torus, type TorusOptions } from './torus.ts';
export { cylinder, type CylinderOptions } from './cylinder.ts';
export { grid, type GridOptions } from './grid.ts';
