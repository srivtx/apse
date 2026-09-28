/**
 * Geometry module tests.
 *
 * The bar here is *shape*, not size. A box that has 24 vertices can still have
 * its normals pointing inwards, its uv's wrapped to 8, or its index buffer
 * wound the wrong way round — and every one of those renders as a plausible
 * looking object that is subtly wrong. So these tests assert what the geometry
 * *is*: that a normal points away from the surface, that a face is closed, that
 * a uv is where the texture says it is, and that a bounding sphere contains
 * every vertex it claims to.
 *
 * Sizes are asserted too, but as a tripwire for the subset-layout path rather
 * than as the point of the exercise.
 */

import { describe, expect, test } from 'bun:test';
import {
  box,
  capsule,
  cone,
  computeTangents,
  cylinder,
  grid,
  layout,
  mergeMeshes,
  MeshData,
  plane,
  POSITION_LAYOUT,
  roundedBox,
  sphere,
  STANDARD_LAYOUT,
  TANGENT_ATTRIBUTES,
  TANGENT_LAYOUT,
  torus,
  upload,
  uploadBatch,
  withTangents,
  type BatchedRange,
} from '../src/geometry/index.ts';
import { recordingDevice } from './fake-geometry-device.ts';
import { AseError } from '../src/core/error.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Reads one attribute of one vertex out of the interleaved buffer. */
function attr(mesh: MeshData, name: string, i: number): number[] {
  const a = mesh.layout.attribute(name);
  if (a === undefined) throw new Error(`layout has no attribute "${name}"`);
  const base = i * (mesh.layout.stride >> 2) + (a.offset >> 2);
  const out: number[] = [];
  for (let c = 0; c < a.info.components; c++) out.push(mesh.vertexData[base + c]);
  return out;
}

function pos(mesh: MeshData, i: number): number[] {
  return attr(mesh, 'position', i);
}

function nrm(mesh: MeshData, i: number): number[] {
  return attr(mesh, 'normal', i);
}

function uv(mesh: MeshData, i: number): number[] {
  return attr(mesh, 'uv', i);
}

function indicesOf(mesh: MeshData): number[] {
  if (mesh.indexData === null) throw new Error(`${mesh.name} is not indexed`);
  return Array.from(mesh.indexData);
}

function sub(a: number[], b: number[]): number[] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: number[], b: number[]): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function length(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

/** Indices of triangles with two coincident vertices, i.e. zero-area. */
function degenerateTriangles(mesh: MeshData): number[] {
  const idx = indicesOf(mesh);
  const bad: number[] = [];
  for (let t = 0; t < idx.length; t += 3) {
    const a = pos(mesh, idx[t]);
    const b = pos(mesh, idx[t + 1]);
    const c = pos(mesh, idx[t + 2]);
    if (length(sub(a, b)) < 1e-9 || length(sub(b, c)) < 1e-9 || length(sub(a, c)) < 1e-9) {
      bad.push(t / 3);
    }
  }
  return bad;
}

/**
 * Topology check: weld by position, then verify the triangulation.
 *
 * Every primitive that wraps in u needs two coincident columns for the seam's
 * uv to interpolate across, so a raw index buffer is topologically a slit
 * surface — correct geometry, deliberately split vertices. Welding is what
 * distinguishes "has a uv seam" from "has a crack", which is exactly the
 * distinction worth asserting.
 *
 * Asserted:
 *   - no edge is used by more than two triangles (no fold-over),
 *   - every directed edge has a reverse (consistent orientation everywhere),
 *   - the boundary is a union of disjoint loops, never a dangling chain,
 *   - the Euler characteristic matches the expected surface. A closed solid is
 *     2 (sphere-like); a plane is 1; an open tube is 0.
 *
 * `closed` additionally requires that there is no boundary at all.
 */
function assertManifold(mesh: MeshData, expectedChi: number, closed = false): void {
  const idx = indicesOf(mesh);
  expect(idx.length % 3, 'index count divisible by 3').toBe(0);

  const weld = new Map<string, number>();
  const remap: number[] = [];
  for (let i = 0; i < mesh.vertexCount; i++) {
    const p = pos(mesh, i);
    const key = p.map((c) => Math.round(c * 1e6)).join(',');
    let canonical = weld.get(key);
    if (canonical === undefined) {
      canonical = weld.size;
      weld.set(key, canonical);
    }
    remap.push(canonical);
  }

  const uses = new Map<string, number>();
  const directed = new Set<string>();
  for (let t = 0; t < idx.length; t += 3) {
    const v = [remap[idx[t]], remap[idx[t + 1]], remap[idx[t + 2]]];
    expect(v[0], `triangle ${t / 3} is not degenerate after welding`).not.toBe(v[1]);
    for (let e = 0; e < 3; e++) {
      const a = v[e];
      const b = v[(e + 1) % 3];
      directed.add(`${a}>${b}`);
      const key = a < b ? `${a}-${b}` : `${b}-${a}`;
      uses.set(key, (uses.get(key) ?? 0) + 1);
    }
  }

  const boundaryDegree = new Map<number, number>();
  let boundaryEdges = 0;
  for (const [key, count] of uses) {
    if (count > 2) throw new Error(`edge ${key} is used by ${count} triangles, expected at most 2`);
    if (count === 1) {
      boundaryEdges++;
      const [a, b] = key.split('-').map(Number);
      boundaryDegree.set(a, (boundaryDegree.get(a) ?? 0) + 1);
      boundaryDegree.set(b, (boundaryDegree.get(b) ?? 0) + 1);
    }
  }
  if (closed && boundaryEdges !== 0) {
    throw new Error(`${mesh.name} has ${boundaryEdges} boundary edges, expected a closed surface`);
  }
  for (const [v, degree] of boundaryDegree) {
    if (degree !== 2) throw new Error(`boundary vertex ${v} has degree ${degree}, so the boundary is a chain, not a loop`);
  }

  // A boundary edge legitimately has no reverse, so only interior edges are
  // required to be traversed in both directions.
  for (const [key, count] of uses) {
    if (count !== 2) continue;
    const [a, b] = key.split('-');
    if (!directed.has(`${a}>${b}`) || !directed.has(`${b}>${a}`)) {
      throw new Error(`interior edge ${key} is not traversed in both directions — inconsistent winding`);
    }
  }
  const chi = weld.size - uses.size + idx.length / 3;
  expect(chi, `Euler characteristic of ${mesh.name}`).toBe(expectedChi);
  if (closed) {
    // 3F = 2E and every edge twice-traversed only hold with no boundary.
    expect(idx.length, '3F = 2E').toBe(uses.size * 2);
    expect(directed.size, 'two directed edges per undirected edge').toBe(uses.size * 2);
  }
}

function assertClosedManifold(mesh: MeshData): void {
  assertManifold(mesh, 2, true);
}

// ---------------------------------------------------------------------------
// box
// ---------------------------------------------------------------------------

describe('box', () => {
  test('has 24 vertices, 36 indices, and a 32-byte stride', () => {
    const m = box();
    expect(m.vertexCount).toBe(24);
    expect(m.indexCount).toBe(36);
    expect(m.indexData?.length).toBe(36);
    expect(m.indexed).toBe(true);
    expect(m.layout).toBe(STANDARD_LAYOUT);
    expect(m.layout.stride).toBe(32);
    expect(m.vertexData.length).toBe(24 * 8);
  });

  test('normals are unit length and point outward', () => {
    const m = box();
    for (let i = 0; i < m.vertexCount; i++) {
      const n = nrm(m, i);
      const p = pos(m, i);
      expect(length(n), `normal ${i} length`).toBeCloseTo(1, 6);
      // A unit cube centred at the origin puts every vertex half a unit off its
      // face plane, so the outward direction is exactly sign(dot).
      expect(dot(n, p), `normal ${i} vs position`).toBeCloseTo(0.5, 6);
      expect(dot(n, p), `normal ${i} faces outward`).toBeGreaterThan(0);
    }
  });

  test('uvs are 0..1 per face and the 24 vertices are 6 faces of 4 distinct corners', () => {
    const m = box();
    const byNormal = new Map<string, number[]>();
    for (let i = 0; i < m.vertexCount; i++) {
      const n = nrm(m, i);
      const key = n.join(',');
      const list = byNormal.get(key);
      if (list === undefined) byNormal.set(key, [i]);
      else list.push(i);
    }
    expect(byNormal.size, 'distinct face normals').toBe(6);

    for (const [key, list] of byNormal) {
      expect(list.length, `vertices on face ${key}`).toBe(4);
      const n = key.split(',').map(Number);
      const corners = new Set<string>();
      const faceUvs = new Set<string>();
      for (const i of list) {
        const p = pos(m, i);
        // Each corner sits on the face plane at half the extent.
        expect(dot(n, p), `corner on the plane of face ${key}`).toBeCloseTo(0.5, 6);
        corners.add(p.join(','));
        const t = uv(m, i);
        for (const c of t) {
          expect(c, `uv of face ${key} in [0,1]`).toBeGreaterThanOrEqual(0);
          expect(c, `uv of face ${key} in [0,1]`).toBeLessThanOrEqual(1);
        }
        faceUvs.add(t.join(','));
      }
      expect(corners.size, `distinct corners on face ${key}`).toBe(4);
      expect(faceUvs.size, `distinct uvs on face ${key}`).toBe(4);
    }

    // The four corners of a face must span the two axes tangent to it.
    const side = [1, 0, 0];
    const list = byNormal.get(side.join(',')) ?? [];
    expect(list.length).toBe(4);
    const tangents = new Set(list.map((i) => pos(m, i).slice(1).join(',')));
    expect(tangents.size, 'corners spread across both tangent axes').toBe(4);
  });

  test('index buffer addresses all 24 vertices with correct winding', () => {
    const m = box();
    const idx = indicesOf(m);
    expect(new Set(idx).size, 'every vertex is referenced').toBe(24);
    for (let t = 0; t < idx.length; t += 3) {
      // A face of the box is a planar quad, so the geometric normal of its
      // first triangle must equal the stored normal of its first vertex.
      const a = pos(m, idx[t]);
      const b = pos(m, idx[t + 1]);
      const c = pos(m, idx[t + 2]);
      const face = cross(sub(b, a), sub(c, a));
      const n = nrm(m, idx[t]);
      expect(dot(face, n), `triangle ${t / 3} wound CCW as seen from outside`).toBeGreaterThan(0);
    }
  });

  test('honours non-unit extents', () => {
    const m = box({ width: 2, height: 4, depth: 6 });
    expect(m.vertexCount).toBe(24);
    let maxX = 0;
    let maxY = 0;
    let maxZ = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      maxX = Math.max(maxX, Math.abs(p[0]));
      maxY = Math.max(maxY, Math.abs(p[1]));
      maxZ = Math.max(maxZ, Math.abs(p[2]));
    }
    expect(maxX).toBeCloseTo(1, 6);
    expect(maxY).toBeCloseTo(2, 6);
    expect(maxZ).toBeCloseTo(3, 6);
  });

  test('subdivides when asked, and stays closed', () => {
    const m = box({ segments: 3 });
    expect(m.vertexCount).toBe(6 * 16);
    expect(m.indexCount).toBe(6 * 9 * 6);
    expect(degenerateTriangles(m)).toEqual([]);
    for (let i = 0; i < m.vertexCount; i++) {
      expect(length(nrm(m, i))).toBeCloseTo(1, 6);
    }
  });

  test('skips normal and uv entirely for POSITION_LAYOUT', () => {
    const m = box({ layout: POSITION_LAYOUT });
    expect(m.layout.stride).toBe(12);
    expect(m.layout.has('normal')).toBe(false);
    expect(m.vertexCount).toBe(24);
    expect(m.vertexData.length).toBe(24 * 3);
    // Positions must be identical to the full-layout build.
    const full = box();
    for (let i = 0; i < m.vertexCount; i++) {
      expect(pos(m, i).join(',')).toBe(pos(full, i).join(','));
    }
  });
});

// ---------------------------------------------------------------------------
// sphere
// ---------------------------------------------------------------------------

describe('sphere', () => {
  test('normals are unit length to 1e-6', () => {
    const m = sphere();
    for (let i = 0; i < m.vertexCount; i++) {
      expect(length(nrm(m, i)), `normal ${i}`).toBeCloseTo(1, 6);
      expect(Math.abs(length(nrm(m, i)) - 1), `normal ${i} error`).toBeLessThan(1e-6);
    }
  });

  test('normals point along the position, and the radius is exact', () => {
    const m = sphere({ radius: 3 });
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      const n = nrm(m, i);
      const radial = length(p);
      expect(radial, `position ${i} on the surface`).toBeCloseTo(3, 5);
      expect(dot([p[0] / radial, p[1] / radial, p[2] / radial], n), `normal ${i} is radial`).toBeCloseTo(1, 6);
    }
  });

  test('has the documented counts, no degenerate triangles, and a closed surface', () => {
    const m = sphere({ widthSegments: 16, heightSegments: 8 });
    // 2 poles + 7 interior rows of 17.
    expect(m.vertexCount).toBe(2 + 7 * 17);
    // w * (h - 1) * 2 triangles: two fans plus (h - 2) interior bands.
    expect(m.indexCount).toBe(16 * 7 * 6);
    expect(m.indexCount / 3).toBe(224);
    expect(degenerateTriangles(m)).toEqual([]);
    assertClosedManifold(m);
  });

  test('defaults to 497 vertices and 960 triangles', () => {
    const m = sphere();
    expect(m.vertexCount).toBe(2 + 33 * 15);
    expect(m.indexCount / 3).toBe(960);
  });

  test('the poles are single vertices at ±Y, and the fans wind outward', () => {
    const m = sphere({ widthSegments: 16, heightSegments: 8 });
    const rowStride = 17;
    const firstRing = 1;
    expect(pos(m, 0).join(',')).toBe('0,1,0');
    expect(pos(m, m.vertexCount - 1).join(',')).toBe('0,-1,0');
    expect(nrm(m, 0).join(',')).toBe('0,1,0');
    expect(uv(m, 0)).toEqual([0.5, 0]);
    expect(uv(m, m.vertexCount - 1)).toEqual([0.5, 1]);

    // Every fan triangle must face away from the origin, which is the whole
    // point of collapsing the pole rows: a polar triangle that faces inward is
    // invisible from outside and shades the inside of the sphere.
    const idx = indicesOf(m);
    for (let t = 0; t < idx.length; t += 3) {
      const a = pos(m, idx[t]);
      const b = pos(m, idx[t + 1]);
      const c = pos(m, idx[t + 2]);
      const face = cross(sub(b, a), sub(c, a));
      if (length(face) < 1e-9) continue;
      const centre = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
      expect(dot(face, centre), `triangle ${t / 3} faces outward`).toBeGreaterThan(0);
    }
    // Sanity: the first 16 triangles are the north fan, fanning out from the
    // single pole vertex along the first ring.
    expect(idx.slice(0, 3)).toEqual([0, firstRing, firstRing + 1]);
    expect(idx.slice(45, 48)).toEqual([0, firstRing + 15, firstRing + 16]);
    // The last 16 are the south fan, running the other way round.
    const south = m.vertexCount - 1;
    expect(idx.slice(-3)).toEqual([firstRing + rowStride * 6 + 15, south, firstRing + rowStride * 6 + 16]);
  });

  test('uvs wrap in u and run pole to pole in v', () => {
    const wSeg = 8;
    const hSeg = 4;
    const m = sphere({ widthSegments: wSeg, heightSegments: hSeg });
    const rowStride = wSeg + 1;
    for (let iy = 1; iy <= hSeg - 1; iy++) {
      const row = 1 + (iy - 1) * rowStride;
      for (let ix = 0; ix <= wSeg; ix++) {
        expect(uv(m, row + ix), `u ${iy}.${ix}`).toEqual([ix / wSeg, 1 - iy / hSeg]);
      }
    }
    // v = 0 at the +Y pole, v = 1 at the -Y pole.
    expect(pos(m, 0)[1]).toBeCloseTo(1, 6);
    expect(pos(m, m.vertexCount - 1)[1]).toBeCloseTo(-1, 6);
  });

  test('clamps segment counts to [3, 512]', () => {
    // Minimum 3 x 3: 2 poles + 2 rings of 4.
    expect(sphere({ widthSegments: 1, heightSegments: 0 }).vertexCount).toBe(2 + 4 * 2);
    expect(sphere({ widthSegments: 1, heightSegments: 0 }).indexCount / 3).toBe(3 * 2 * 2);
    // Maximum 512 x 512.
    expect(sphere({ widthSegments: 9999, heightSegments: 9999 }).vertexCount).toBe(2 + 513 * 511);
  });

  test('POSITION_LAYOUT gives 12 bytes per vertex and the same positions', () => {
    const m = sphere({ widthSegments: 12, heightSegments: 6, layout: POSITION_LAYOUT });
    const full = sphere({ widthSegments: 12, heightSegments: 6 });
    expect(m.layout.stride).toBe(12);
    expect(m.vertexData.length).toBe(m.vertexCount * 3);
    expect(m.vertexCount).toBe(full.vertexCount);
    for (let i = 0; i < m.vertexCount; i++) {
      expect(pos(m, i).join(',')).toBe(pos(full, i).join(','));
    }
    // Still valid geometry: closed, non-degenerate, and bounded.
    expect(degenerateTriangles(m)).toEqual([]);
    expect(m.boundingSphere[3]).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// plane
// ---------------------------------------------------------------------------

describe('plane', () => {
  test('defaults to 4 vertices, 6 indices, in the XZ plane facing +Y', () => {
    const m = plane();
    expect(m.vertexCount).toBe(4);
    expect(m.indexCount).toBe(6);
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      const n = nrm(m, i);
      expect(p[1], 'on the XZ plane').toBe(0);
      expect(n.join(',')).toBe('0,1,0');
    }
    const xs = new Set([pos(m, 0)[0], pos(m, 1)[0], pos(m, 2)[0], pos(m, 3)[0]]);
    expect(xs.size).toBe(2);
  });

  test('winds CCW seen from +Y', () => {
    const m = plane();
    const idx = indicesOf(m);
    for (let t = 0; t < idx.length; t += 3) {
      const face = cross(
        sub(pos(m, idx[t + 1]), pos(m, idx[t])),
        sub(pos(m, idx[t + 2]), pos(m, idx[t])),
      );
      expect(face[1], `triangle ${t / 3} faces +Y`).toBeGreaterThan(0);
    }
  });

  test('uvs run 0..1 across the quad', () => {
    const m = plane();
    for (let i = 0; i < m.vertexCount; i++) {
      for (const c of uv(m, i)) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(1);
      }
    }
  });

  test('subdivides without degenerating', () => {
    const m = plane({ width: 8, depth: 4, widthSegments: 8, depthSegments: 4 });
    expect(m.vertexCount).toBe(9 * 5);
    expect(m.indexCount).toBe(8 * 4 * 6);
    expect(degenerateTriangles(m)).toEqual([]);
    // A plane is an open surface: one boundary loop, Euler characteristic 1.
    assertManifold(m, 1);
  });
});

// ---------------------------------------------------------------------------
// torus
// ---------------------------------------------------------------------------

describe('torus', () => {
  test('is a closed, consistently wound surface', () => {
    const m = torus({ radialSegments: 12, tubularSegments: 8 });
    expect(m.vertexCount).toBe(13 * 9);
    expect(m.indexCount).toBe(12 * 8 * 6);
    expect(m.indexCount % 3).toBe(0);
    expect(degenerateTriangles(m)).toEqual([]);
    // A torus is genus 1, so its Euler characteristic is 0, not 2. Asserting
    // the exact value pins the topology: a sphere-with-a-hole-sawn-out has the
    // same counts for a different reason.
    assertManifold(m, 0, true);
  });

  test('normals point away from the tube centre, not from the origin', () => {
    const radius = 1;
    const tube = 0.4;
    const m = torus({ radius, tube, radialSegments: 16, tubularSegments: 12 });
    for (let i = 0; i < m.vertexCount; i++) {
      expect(length(nrm(m, i)), `normal ${i} unit`).toBeCloseTo(1, 6);
      // The tube centre for this vertex is the point on the main ring at the
      // same angle: project the position onto the ring.
      const p = pos(m, i);
      const ringLen = Math.hypot(p[0], p[2]);
      if (ringLen < 1e-6) continue;
      const centre = [(p[0] / ringLen) * radius, 0, (p[2] / ringLen) * radius];
      expect(dot(nrm(m, i), sub(p, centre)), 'normal points out of the tube').toBeCloseTo(tube, 5);
    }
  });

  test('outer-ring triangles face away from the torus centroid', () => {
    // Scoped to the outer half on purpose. On the inner half the surface faces
    // *towards* the origin — that is the inside of the hole, and a normal there
    // pointing outwards from the centroid would be the bug, not the fix.
    // |centroid| > radius is exactly that split: |p|² = r² + 2·r·tube·cosθ + t².
    const radius = 1;
    const m = torus({ radius, tube: 0.4, radialSegments: 16, tubularSegments: 12 });
    const idx = indicesOf(m);
    let checked = 0;
    for (let t = 0; t < idx.length; t += 3) {
      const a = pos(m, idx[t]);
      const b = pos(m, idx[t + 1]);
      const c = pos(m, idx[t + 2]);
      const centre = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
      if (length(centre) <= radius) continue;
      const face = cross(sub(b, a), sub(c, a));
      expect(dot(face, centre), `outer triangle ${t / 3} faces away from the centroid`).toBeGreaterThan(0);
      checked++;
    }
    expect(checked, 'some triangles were on the outer half').toBeGreaterThan(0);
  });

  test('reaches radius + tube from the origin', () => {
    const m = torus({ radius: 2, tube: 0.5, radialSegments: 32, tubularSegments: 16 });
    let max = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      max = Math.max(max, length(pos(m, i)));
    }
    expect(max).toBeCloseTo(2.5, 5);
    expect(m.boundingSphere[3]).toBeCloseTo(2.5, 5);
  });

  test('clamps segment counts', () => {
    expect(torus({ radialSegments: 1, tubularSegments: 1 }).vertexCount).toBe(4 * 4);
  });
});

// ---------------------------------------------------------------------------
// cylinder
// ---------------------------------------------------------------------------

describe('cylinder', () => {
  test('defaults to a capped cylinder of the documented size', () => {
    const m = cylinder();
    // 2 side rows of 33, plus a centre + 32 ring for each cap.
    expect(m.vertexCount).toBe(2 * 33 + 2 * 33);
    expect(m.indexCount).toBe(32 * 6 + 2 * 32 * 3);
    expect(m.vertexCount).toBe(132);
    expect(m.indexCount).toBe(384);
    expect(degenerateTriangles(m)).toEqual([]);
  });

  test('caps get their own ±Y normals and the side gets radial ones', () => {
    const m = cylinder({ radialSegments: 8 });
    const top = [];
    const side = [];
    for (let i = 0; i < m.vertexCount; i++) {
      const n = nrm(m, i);
      expect(length(n)).toBeCloseTo(1, 6);
      if (Math.abs(n[1]) > 0.5) top.push(n[1]);
      else {
        side.push(n);
        expect(n[1], 'side normal is horizontal').toBe(0);
        const p = pos(m, i);
        expect(dot(n, [p[0], 0, p[2]]), 'side normal is radial').toBeGreaterThan(0);
      }
    }
    expect(top.length, 'cap vertices').toBe(2 * 9);
    expect(new Set(top), 'caps point +Y and -Y').toEqual(new Set([1, -1]));
    expect(side.length, 'side vertices').toBe(2 * 9);
  });

  test('a cone (radiusTop 0) has no zero-area triangles', () => {
    // The apex collapses the whole top ring to one vertex, so every side
    // triangle is a real triangle rather than a sliver.
    const m = cylinder({ radiusTop: 0, radiusBottom: 1, height: 2, radialSegments: 16 });
    expect(degenerateTriangles(m)).toEqual([]);
    let apex = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      if (pos(m, i)[1] > 0.999) apex++;
    }
    expect(apex, 'exactly one apex vertex').toBe(1);
    expect(degenerateTriangles(m)).toEqual([]);
  });

  test('an inverted cone (radiusBottom 0) has no zero-area triangles', () => {
    const m = cylinder({ radiusTop: 1, radiusBottom: 0, height: 2, radialSegments: 16 });
    expect(degenerateTriangles(m)).toEqual([]);
  });

  test('capped false with equal radii is an open tube', () => {
    const tube = cylinder({ capped: false, radialSegments: 16 });
    const open = cylinder({ radiusTop: 1, radiusBottom: 1, height: 1, radialSegments: 16, heightSegments: 1, capped: false });
    expect(tube.vertexCount).toBe(open.vertexCount);
    // 2 rows of 17, and no cap centre vertices.
    expect(open.vertexCount).toBe(2 * 17);
    expect(open.indexCount).toBe(16 * 6);
    const capped = cylinder({ radialSegments: 16 });
    expect(capped.vertexCount).toBe(open.vertexCount + 2 * 17);
    // The side is a cylinder, not a torus: the end rims are not shared.
    expect(degenerateTriangles(open)).toEqual([]);
  });

  test('tilts the side normal for a cone', () => {
    const m = cylinder({ radiusTop: 0, radiusBottom: 1, height: 1, radialSegments: 8 });
    let sawTilt = false;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      if (p[1] < 0.4) continue;
      const n = nrm(m, i);
      expect(n[1], 'a cone side normal points up and out').toBeGreaterThan(0);
      sawTilt = true;
    }
    expect(sawTilt).toBe(true);
  });

  test('subdivides vertically without degenerating', () => {
    const m = cylinder({ heightSegments: 4, radialSegments: 8 });
    expect(m.vertexCount).toBe(5 * 9 + 2 * 9);
    expect(degenerateTriangles(m)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// grid
// ---------------------------------------------------------------------------

describe('grid', () => {
  test('defaults to 121 vertices and 200 triangles', () => {
    const m = grid();
    expect(m.vertexCount).toBe(121);
    expect(m.indexCount).toBe(600);
    expect(m.indexCount / 3).toBe(200);
    expect(m.name).toBe('grid');
  });

  test('spans 10x10 units of XZ with shared corner vertices', () => {
    const m = grid();
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      expect(p[1], 'flat').toBe(0);
      expect(nrm(m, i).join(',')).toBe('0,1,0');
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minZ = Math.min(minZ, p[2]);
      maxZ = Math.max(maxZ, p[2]);
    }
    expect(minX).toBeCloseTo(-5, 5);
    expect(maxX).toBeCloseTo(5, 5);
    expect(minZ).toBeCloseTo(-5, 5);
    expect(maxZ).toBeCloseTo(5, 5);
    // 11 x 11 lattice: 11 distinct x coordinates, not 121.
    const xs = new Set<string>();
    for (let i = 0; i < m.vertexCount; i++) xs.add(String(pos(m, i)[0]));
    expect(xs.size).toBe(11);
  });

  test('scales with the requested size', () => {
    const m = grid({ width: 4, depth: 2, widthSegments: 4, depthSegments: 2 });
    expect(m.vertexCount).toBe(5 * 3);
    expect(m.indexCount).toBe(4 * 2 * 6);
  });
});

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

describe('MeshData packing', () => {
  // A deliberately awkward layout: two attributes that need 4-byte alignment
  // after a 12-byte one, and a format at the end, so a packer that assumes
  // "position, normal, uv, stride 32" cannot pass this by accident.
  const CUSTOM = layout({ position: 'float32x3', uv: 'float32x2', colour: 'float32x4' });
  const FLOATS = CUSTOM.stride >> 2;

  const POSITIONS = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1];
  const UVS = [0, 0, 1, 0, 0.5, 1, 0.25, 0.75];
  const COLOURS = [
    1, 0, 0, 1,
    0, 1, 0, 1,
    0, 0, 1, 1,
    1, 1, 0, 0.5,
  ];

  test('interleaves the attributes form at the documented offsets', () => {
    expect(CUSTOM.stride, 'stride accounts for 4-byte alignment').toBe(36);
    expect(CUSTOM.attribute('position')?.offset).toBe(0);
    expect(CUSTOM.attribute('uv')?.offset).toBe(12);
    expect(CUSTOM.attribute('colour')?.offset).toBe(20);

    const m = new MeshData({
      layout: CUSTOM,
      vertices: {
        attributes: { position: POSITIONS, uv: UVS, colour: COLOURS },
        vertexCount: 4,
      },
      indices: new Uint16Array([0, 1, 2, 0, 2, 3]),
    });

    expect(m.vertexData.length).toBe(4 * FLOATS);
    for (let i = 0; i < 4; i++) {
      for (let c = 0; c < 3; c++) {
        expect(m.vertexData[i * FLOATS + 0 + c], `position ${i}.${c}`).toBe(POSITIONS[i * 3 + c]);
      }
      for (let c = 0; c < 2; c++) {
        expect(m.vertexData[i * FLOATS + 3 + c], `uv ${i}.${c}`).toBe(UVS[i * 2 + c]);
      }
      for (let c = 0; c < 4; c++) {
        expect(m.vertexData[i * FLOATS + 5 + c], `colour ${i}.${c}`).toBe(COLOURS[i * 4 + c]);
      }
    }
  });

  test('leaves no gap bytes in the buffer', () => {
    const m = new MeshData({
      layout: CUSTOM,
      vertices: {
        attributes: { position: POSITIONS, uv: UVS, colour: COLOURS },
        vertexCount: 4,
      },
      indices: new Uint16Array([0, 1, 2, 0, 2, 3]),
    });
    // Every float in the buffer belongs to some attribute — no uninitialised
    // holes, which is what an off-by-one in the offset arithmetic would cause.
    const accounted = new Array<number>(m.vertexData.length).fill(0);
    for (const a of CUSTOM.attributes) {
      for (let i = 0; i < 4; i++) {
        for (let c = 0; c < a.info.components; c++) {
          accounted[i * FLOATS + (a.offset >> 2) + c] = 1;
        }
      }
    }
    expect(accounted.every((v) => v === 1)).toBe(true);
  });

  test('takes an interleaved source by reference, trimmed to the stride', () => {
    const interleaved = new Float32Array(4 * FLOATS + 5);
    for (let i = 0; i < 4 * FLOATS; i++) interleaved[i] = i;
    const m = new MeshData({
      layout: CUSTOM,
      vertices: { interleaved, vertexCount: 4 },
      indices: new Uint16Array([0, 1, 2]),
    });
    expect(m.vertexData.length, 'trimmed to the stride').toBe(4 * FLOATS);
    // The source is taken as-is: no copy, no re-ordering, no normalisation.
    for (let i = 0; i < 4 * FLOATS; i++) expect(m.vertexData[i]).toBe(i);
    expect(m.vertexData.byteOffset, 'a view onto the same memory').toBe(interleaved.byteOffset);
  });

  test('rejects attributes whose lengths disagree', () => {
    const err = (() => {
      try {
        new MeshData({
          layout: CUSTOM,
          vertices: {
            attributes: { position: POSITIONS, uv: [0, 0, 1], colour: COLOURS },
            vertexCount: 4,
          },
        });
        return null;
      } catch (e) {
        return e as AseError;
      }
    })();
    expect(err).toBeInstanceOf(AseError);
    expect(err?.code).toBe('INTERNAL_INVARIANT');
    expect(err?.message).toContain('uv');
    expect(err?.message).toContain('3');
    expect(err?.message).toContain('8');
    expect(err?.why).toBeTruthy();
    expect(err?.fix).toBeTruthy();
  });

  test('rejects an attribute the layout declares but the source omits', () => {
    expect(() => new MeshData({
      layout: CUSTOM,
      vertices: { attributes: { position: POSITIONS, uv: UVS }, vertexCount: 4 },
    })).toThrow(/colour/);
  });

  test('rejects an interleaved buffer that is too short', () => {
    expect(() => new MeshData({
      layout: CUSTOM,
      vertices: { interleaved: new Float32Array(4 * FLOATS - 1), vertexCount: 4 },
    })).toThrow(/floats/);
  });

  test('rejects a non-indexed triangle mesh with a ragged vertex count', () => {
    expect(() => new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { interleaved: new Float32Array(4 * 3), vertexCount: 4 },
    })).toThrow(/multiple of 3/);

    // The same count is fine when indexed, or under a strip topology.
    expect(() => new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { interleaved: new Float32Array(4 * 3), vertexCount: 4 },
      indices: new Uint16Array([0, 1, 2]),
    })).not.toThrow();
    expect(() => new MeshData({
      layout: POSITION_LAYOUT,
      topology: 'triangle-strip',
      vertices: { interleaved: new Float32Array(4 * 3), vertexCount: 4 },
    })).not.toThrow();
  });

  test('rejects an empty mesh and a layout with no position', () => {
    expect(() => new MeshData({
      layout: STANDARD_LAYOUT,
      vertices: { interleaved: new Float32Array(0), vertexCount: 0 },
    })).toThrow(/0 vertices/);
    expect(() => new MeshData({
      layout: layout({ uv: 'float32x2' }),
      vertices: { attributes: { uv: [0, 0, 1, 1, 0, 1] }, vertexCount: 3 },
    })).toThrow(/position/);
  });

  test('indexCount falls back to the vertex count when unindexed', () => {
    const m = new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { interleaved: new Float32Array(3 * 3), vertexCount: 3 },
    });
    expect(m.indexed).toBe(false);
    expect(m.indexData).toBeNull();
    expect(m.indexCount).toBe(3);
  });

  test('widens the index buffer past 65535 vertices', () => {
    const m = new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { interleaved: new Float32Array(70000 * 3), vertexCount: 70000 },
      indices: new Uint32Array([69999, 0, 1]),
    });
    expect(m.indexData).toBeInstanceOf(Uint32Array);
    expect(m.indexCount).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Bounding sphere and lifetime
// ---------------------------------------------------------------------------

describe('bounding sphere', () => {
  test('box(1,1,1) at the origin has radius sqrt(3)/2', () => {
    const m = box();
    expect(m.boundingSphere.length).toBe(4);
    expect(m.boundingSphere[0]).toBeCloseTo(0, 6);
    expect(m.boundingSphere[1]).toBeCloseTo(0, 6);
    expect(m.boundingSphere[2]).toBeCloseTo(0, 6);
    expect(m.boundingSphere[3]).toBeCloseTo(Math.sqrt(3) / 2, 6);
  });

  test('contains every vertex of a unit sphere', () => {
    const m = sphere({ radius: 1, widthSegments: 32, heightSegments: 16 });
    expect(m.boundingSphere[0]).toBeCloseTo(0, 6);
    expect(m.boundingSphere[3]).toBeCloseTo(1, 5);
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      const d = Math.hypot(
        p[0] - m.boundingSphere[0],
        p[1] - m.boundingSphere[1],
        p[2] - m.boundingSphere[2],
      );
      expect(d, `vertex ${i} inside the sphere`).toBeLessThanOrEqual(m.boundingSphere[3] + 1e-6);
    }
  });

  test('is centred on the AABB of a translated, non-centred mesh', () => {
    const l = layout({ position: 'float32x3' });
    const m = new MeshData({
      layout: l,
      vertices: {
        attributes: { position: [10, 0, 0, 12, 4, 0, 10, 0, 6] },
        vertexCount: 3,
      },
    });
    expect(m.boundingSphere[0]).toBeCloseTo(11, 6);
    expect(m.boundingSphere[1]).toBeCloseTo(2, 6);
    expect(m.boundingSphere[2]).toBeCloseTo(3, 6);
    expect(m.boundingSphere[3]).toBeCloseTo(Math.sqrt(1 + 4 + 9), 6);
  });

  test('grid is bounded by its own corners, not its cell count', () => {
    const m = grid();
    expect(m.boundingSphere[3]).toBeCloseTo(Math.sqrt(50), 5);
  });

  test('an explicit override is accepted and packed', () => {
    const m = new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { interleaved: new Float32Array(3 * 3), vertexCount: 3 },
      boundingSphere: [1, 2, 3, 4],
    });
    expect(Array.from(m.boundingSphere)).toEqual([1, 2, 3, 4]);
  });
});

describe('lifetime', () => {
  test('MeshData disposes, is idempotent, and refuses re-referencing', () => {
    const m = box();
    expect(m.disposed).toBe(false);
    expect(m.refCount).toBe(1);
    m.dispose();
    expect(m.disposed).toBe(true);
    expect(m.refCount).toBe(0);
    m.dispose();
    expect(m.disposed).toBe(true);
    expect(() => m.ref()).toThrow(/disposed/);
  });

  test('the last unref releases', () => {
    const m = torus();
    m.ref();
    expect(m.refCount).toBe(2);
    m.unref();
    expect(m.disposed).toBe(false);
    m.unref();
    expect(m.disposed).toBe(true);
  });

  test('unref on an already-disposed resource is a no-op', () => {
    const m = plane();
    m.dispose();
    expect(() => m.unref()).not.toThrow();
    expect(m.disposed).toBe(true);
  });
});

describe('upload', () => {
  test('rejects a mesh larger than the device buffer limit', () => {
    // Only the limit check runs before any GPU call, so a stub device is enough
    // to exercise the real validation path without a device.
    const device = { limits: { maxBufferSize: 16 } } as unknown as GPUDevice;
    try {
      upload(device, box());
      throw new Error('expected a throw');
    } catch (e) {
      const err = e as AseError;
      expect(err.code).toBe('MESH_DATA_TOO_LARGE');
      expect(err.message).toContain('16');
    }
  });
});

// ---------------------------------------------------------------------------
// cone
// ---------------------------------------------------------------------------

describe('cone', () => {
  test('is the cylinder with a zero top radius, byte for byte', () => {
    // The delegated definition is the point: one swept-surface generator, so
    // there is no second one to get the apex collapse or the slope normal wrong.
    const a = cone({ radius: 2, height: 3, radialSegments: 16 });
    const b = cylinder({ radiusTop: 0, radiusBottom: 2, height: 3, radialSegments: 16 });
    expect(a.vertexCount).toBe(b.vertexCount);
    expect(a.indexCount).toBe(b.indexCount);
    expect(Array.from(a.vertexData)).toEqual(Array.from(b.vertexData));
    expect(Array.from(a.indexData!)).toEqual(Array.from(b.indexData!));
  });

  test('has a single apex vertex and no zero-area triangles', () => {
    const m = cone({ radius: 1, height: 2, radialSegments: 16 });
    let apex = 0;
    for (let i = 0; i < m.vertexCount; i++) if (pos(m, i)[1] > 0.999) apex++;
    expect(apex, 'exactly one apex vertex').toBe(1);
    expect(degenerateTriangles(m)).toEqual([]);
  });

  test('is a closed surface whose side normals tilt with the slope', () => {
    const m = cone({ radius: 1, height: 2, radialSegments: 12 });
    assertClosedManifold(m);
    let tilted = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      const n = nrm(m, i);
      expect(length(n), `normal ${i} unit`).toBeCloseTo(1, 6);
      // Above the midpoint the wall leans inward, so its normal has a positive
      // y component; on a straight cylinder it would be exactly zero.
      if (p[1] > 0.4) {
        expect(n[1], `side normal ${i} tilts up`).toBeGreaterThan(0);
        tilted++;
      }
    }
    expect(tilted, 'some vertices are on the sloping wall').toBeGreaterThan(0);
  });

  test('the base disc is capped and faces -Y', () => {
    const capped = cone({ radialSegments: 8 });
    const open = cone({ radialSegments: 8, capped: false });
    // A cap is a centre plus a ring of `radialSegments` — 9 more vertices and
    // `radialSegments` more triangles.
    expect(capped.vertexCount).toBe(open.vertexCount + 9);
    expect(capped.indexCount).toBe(open.indexCount + 8 * 3);
    let down = 0;
    for (let i = 0; i < capped.vertexCount; i++) {
      if (nrm(capped, i)[1] < -0.5) down++;
    }
    expect(down, 'base-cap vertices').toBe(9);
    expect(degenerateTriangles(open)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// capsule
// ---------------------------------------------------------------------------

describe('capsule', () => {
  test('has the documented counts and a closed surface', () => {
    const rs = 12, capSeg = 4, hSeg = 1;
    const m = capsule({ radialSegments: rs, capSegments: capSeg, heightSegments: hSeg });
    const rings = 2 * capSeg + hSeg - 2;
    expect(m.vertexCount).toBe(2 + rings * (rs + 1));
    expect(m.indexCount / 3).toBe(2 * rs * rings);
    expect(degenerateTriangles(m)).toEqual([]);
    // Topologically a sphere, so the same Euler characteristic the sphere has.
    assertClosedManifold(m);
  });

  test('is taller than it is wide by exactly the two caps, and its bounds say so', () => {
    const radius = 0.5, height = 4;
    const m = capsule({ radius, height, radialSegments: 16, capSegments: 6 });
    let minY = Infinity, maxY = -Infinity, maxR = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
      maxR = Math.max(maxR, Math.hypot(p[0], p[2]));
    }
    expect(maxY - minY).toBeCloseTo(height + 2 * radius, 6);
    expect(maxY).toBeCloseTo(height / 2 + radius, 6);
    expect(minY).toBeCloseTo(-height / 2 - radius, 6);
    expect(maxR).toBeCloseTo(radius, 6);
    // The bound is the pole, not the equator: |pole| = h/2 + r.
    expect(m.boundingSphere[3]).toBeCloseTo(height / 2 + radius, 5);
    expect(m.boundingSphere[0]).toBeCloseTo(0, 6);
  });

  test('the poles are single vertices and the fans face outward', () => {
    const m = capsule({ radius: 1, height: 1, radialSegments: 12, capSegments: 4 });
    let north = 0, south = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      if (p[1] > 1.499) north++;
      if (p[1] < -1.499) south++;
    }
    expect([north, south], 'one vertex per pole').toEqual([1, 1]);
    const idx = indicesOf(m);
    for (let t = 0; t < idx.length; t += 3) {
      const a = pos(m, idx[t]), b = pos(m, idx[t + 1]), c = pos(m, idx[t + 2]);
      const face = cross(sub(b, a), sub(c, a));
      const centre = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
      expect(dot(face, centre), `triangle ${t / 3} faces away from the axis`).toBeGreaterThan(0);
    }
  });

  test('normals are unit everywhere and horizontal on the cylindrical section', () => {
    const m = capsule({ radius: 1, height: 2, radialSegments: 12, capSegments: 4, heightSegments: 3 });
    let wall = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const n = nrm(m, i);
      expect(length(n), `normal ${i} unit`).toBeCloseTo(1, 6);
      if (n[1] === 0) {
        wall++;
        const p = pos(m, i);
        // A wall normal is radial, so it has no y at all and points outwards.
        expect(dot(n, [p[0], 0, p[2]]), `wall normal ${i} is radial`).toBeGreaterThan(0);
      }
    }
    expect(wall, 'the wall has at least one row').toBeGreaterThan(0);
  });

  test('height 0 is a sphere, not a stack of coincident rings', () => {
    // The interesting part is that it does not degenerate: more than one wall
    // row at zero height would be several coincident rings, which is a band of
    // zero-area triangles rather than a sphere.
    const m = capsule({ radius: 0.75, height: 0, radialSegments: 12, capSegments: 4, heightSegments: 8 });
    expect(m.boundingSphere[3]).toBeCloseTo(0.75, 5);
    expect(degenerateTriangles(m)).toEqual([]);
    assertClosedManifold(m);
  });

  test('v runs pole to pole and u wraps', () => {
    const m = capsule({ radialSegments: 8, capSegments: 4, heightSegments: 1 });
    expect(uv(m, 0)).toEqual([0.5, 0]);
    expect(uv(m, m.vertexCount - 1)).toEqual([0.5, 1]);
    for (let i = 0; i < m.vertexCount; i++) {
      const t = uv(m, i);
      expect(t[0], `u ${i}`).toBeGreaterThanOrEqual(0);
      expect(t[0], `u ${i}`).toBeLessThanOrEqual(1);
      expect(t[1], `v ${i}`).toBeGreaterThanOrEqual(0);
      expect(t[1], `v ${i}`).toBeLessThanOrEqual(1);
    }
  });

  test('clamps its segment counts and honours POSITION_LAYOUT', () => {
    expect(capsule({ radialSegments: 1, capSegments: 0, heightSegments: 0 }).vertexCount)
      .toBe(capsule({ radialSegments: 3, capSegments: 2, heightSegments: 1 }).vertexCount);
    const m = capsule({ layout: POSITION_LAYOUT, radialSegments: 8, capSegments: 3 });
    expect(m.layout.stride).toBe(12);
    expect(m.vertexData.length).toBe(m.vertexCount * 3);
    expect(degenerateTriangles(m)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// rounded box
// ---------------------------------------------------------------------------

describe('roundedBox', () => {
  /** The exact farthest point of a filleted box: the corner along a body diagonal. */
  const cornerDistance = (radius: number, half: number): number =>
    Math.SQRT2 * 0 + Math.sqrt(3) * (half - radius) + radius;

  test('has the documented counts and a closed surface', () => {
    const seg = 3;
    const side = 2 * seg + 2;
    const m = roundedBox({ segments: seg });
    expect(m.vertexCount).toBe(6 * side * side);
    expect(m.indexCount / 3).toBe(12 * (side - 1) * (side - 1));
    expect(degenerateTriangles(m)).toEqual([]);
    assertClosedManifold(m);
  });

  test('the corners are sphere octants of exactly the requested radius', () => {
    const radius = 0.2;
    const m = roundedBox({ radius, segments: 4 });
    const s = m.layout.stride >> 2;
    let sawOctant = false;
    for (let i = 0; i < m.vertexCount; i++) {
      const o = i * s;
      const p = [m.vertexData[o], m.vertexData[o + 1], m.vertexData[o + 2]];
      // Distance from the inner box's corner to the point, on the three axes that
      // the point is rounded on. On the octant that is exactly the radius.
      const cx = Math.min(0.5 - radius, Math.max(-(0.5 - radius), p[0]));
      const cy = Math.min(0.5 - radius, Math.max(-(0.5 - radius), p[1]));
      const cz = Math.min(0.5 - radius, Math.max(-(0.5 - radius), p[2]));
      const d = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
      if (d < 1e-9) continue; // a vertex in the flat middle: distance 0 by construction
      expect(d, `vertex ${i} lies on the fillet surface`).toBeCloseTo(radius, 5);
      sawOctant = true;
    }
    expect(sawOctant, 'some vertices are on the fillet').toBe(true);
  });

  test('reaches the same far corner a sharp box does, minus the fillet', () => {
    for (const radius of [0.05, 0.1, 0.25]) {
      const m = roundedBox({ radius });
      let max = 0;
      for (let i = 0; i < m.vertexCount; i++) {
        const p = pos(m, i);
        max = Math.max(max, length(p));
      }
      expect(max, `radius ${radius}`).toBeCloseTo(cornerDistance(radius, 0.5), 5);
      // And the bound is that vertex, because the AABB centre is the origin.
      expect(m.boundingSphere[3], `bound for radius ${radius}`).toBeCloseTo(max, 5);
    }
  });

  test('the silhouette is the sharp box\'s: every axis still reaches half its extent', () => {
    // The defining property of a fillet rather than a chamfer or a squashed
    // box: the flat faces are untouched, so the outline is the same rectangle.
    const m = roundedBox({ radius: 0.3, segments: 3 });
    let maxX = 0, maxY = 0, maxZ = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      maxX = Math.max(maxX, Math.abs(p[0]));
      maxY = Math.max(maxY, Math.abs(p[1]));
      maxZ = Math.max(maxZ, Math.abs(p[2]));
      expect(length(nrm(m, i)), `normal ${i} unit`).toBeCloseTo(1, 6);
    }
    expect(maxX).toBeCloseTo(0.5, 6);
    expect(maxY).toBeCloseTo(0.5, 6);
    expect(maxZ).toBeCloseTo(0.5, 6);
    // And the flat middle of the +Y face is exactly on the plane, with the face
    // normal: the projection is the identity where the fillet is not. There is
    // no vertex at the face's centre and there cannot be one — the grid spends
    // `segments` intervals on each rounded band and exactly one on the flat
    // middle, because a plane is a quad. So the flat middle is the four vertices
    // of that quad, and the centre of the face is interior to a triangle.
    let onTopFace = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      if (p[1] !== 0.5) continue;
      expect(nrm(m, i).join(','), `vertex ${i} on the +Y plane is planar and outward`).toBe('0,1,0');
      onTopFace++;
    }
    expect(onTopFace, 'the +Y face has one quad of flat middle').toBe(4);
  });

  test('every triangle winds CCW as seen from outside', () => {
    const m = roundedBox({ segments: 2 });
    const idx = indicesOf(m);
    for (let t = 0; t < idx.length; t += 3) {
      const a = pos(m, idx[t]), b = pos(m, idx[t + 1]), c = pos(m, idx[t + 2]);
      const face = cross(sub(b, a), sub(c, a));
      expect(dot(face, nrm(m, idx[t])), `triangle ${t / 3}`).toBeGreaterThan(0);
    }
  });

  test('honours non-uniform extents and a POSITION_LAYOUT', () => {
    const radius = 0.2;
    const m = roundedBox({ width: 4, height: 2, depth: 1, radius, segments: 2 });
    const reached = [0, 0, 0];
    let far = 0;
    for (let i = 0; i < m.vertexCount; i++) {
      const p = pos(m, i);
      for (let a = 0; a < 3; a++) reached[a] = Math.max(reached[a], Math.abs(p[a]));
      far = Math.max(far, length(p));
    }
    // Every axis reaches its own half extent — the silhouette of the sharp box,
    // which is the whole claim of a fillet over a chamfer.
    expect(reached[0]).toBeCloseTo(2, 6);
    expect(reached[1]).toBeCloseTo(1, 6);
    expect(reached[2]).toBeCloseTo(0.5, 6);
    // The bound is the farthest vertex. It is *not* the unit box's corner
    // distance: for a non-cube the farthest point of a filleted box is where the
    // direction from the origin leaves the corner sphere, which is not on the
    // body diagonal, and a polyhedral approximation stops just short of it even
    // on a cube. So the two things that have to hold are asserted instead — the
    // bound is tight (not an under-estimate) and it has not forgotten the fillet.
    expect(m.boundingSphere[3], 'the bound is the farthest vertex, not an under-estimate').toBeCloseTo(far, 5);
    expect(m.boundingSphere[3], 'and it includes the fillet on every axis')
      .toBeGreaterThan(Math.hypot(2 - radius, 1 - radius, 0.5 - radius));
    const flat = roundedBox({ layout: POSITION_LAYOUT, segments: 2 });
    expect(flat.vertexData.length).toBe(flat.vertexCount * 3);
  });

  test('the sampling puts a vertex exactly on the fillet boundary on every axis', () => {
    // The property the remap exists for, and the one a single shared radius
    // fraction broke on a non-cube: the middle sample lands exactly on `±(h − r)`,
    // which is where the projection is the identity and the fillet meets the flat
    // face tangentially. With one fraction for the whole box, a 4 × 2 × 1 box put
    // its long axes' samples out in the flat middle instead and left the 4-unit
    // edges with a single facet each, while the 1-unit edges got `segments`.
    const radius = 0.2;
    const m = roundedBox({ width: 4, height: 2, depth: 1, radius, segments: 3 });
    const half = [2, 1, 0.5];
    for (let a = 0; a < 3; a++) {
      const boundary = half[a] - radius;
      let onBoundary = 0;
      for (let i = 0; i < m.vertexCount; i++) {
        if (Math.abs(Math.abs(pos(m, i)[a]) - boundary) < 1e-6) onBoundary++;
      }
      expect(onBoundary, `axis ${a} has vertices on its ${boundary} fillet boundary`).toBeGreaterThan(0);
    }
  });

  test('a radius at or past half the smallest extent is refused, with the number', () => {
    const e = err(() => roundedBox({ width: 1, height: 1, depth: 1, radius: 0.5 }));
    expect(e.code).toBe('OPTION_UNKNOWN');
    expect(e.message).toContain('0.5');
    expect(() => roundedBox({ radius: 0.75 })).toThrow(/0.375|fillet/);
    expect(() => roundedBox({ radius: 0 })).toThrow(/positive/);
    expect(() => roundedBox({ radius: -1 })).toThrow(/positive/);
  });
});

// ---------------------------------------------------------------------------
// Bounds audit
// ---------------------------------------------------------------------------

describe('bounds: the audit', () => {
  test('a single non-finite position is caught, not silently averaged away', () => {
    // The bug this exists for: NaN fails every `<` and every `>`, so a mesh with
    // one NaN vertex produces a *finite* AABB that is missing it — a bound that
    // does not contain the mesh, which culls a visible object with no error.
    const floats = new Float32Array(9);
    floats.set([0, 0, 0, 1, 0, 0, NaN, 1, 0]);
    const e = err(() => new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { interleaved: floats, vertexCount: 3 },
    }));
    // `INVALID_USAGE` and not `INTERNAL_INVARIANT`: the NaN is in the caller's
    // array, so this is blame `caller`. `INTERNAL_INVARIANT` is the one code
    // classified `library`, and filing a user's NaN under it says apse is broken
    // when it is not.
    expect(e.code).toBe('INVALID_USAGE');
    expect(e.message).toContain('non-finite');
    expect(e.message).toContain('vertex 2');
  });

  test('an Infinity position is caught the same way', () => {
    const floats = new Float32Array(9);
    floats.set([0, 0, 0, Infinity, 0, 0, 0, 1, 0]);
    const e = err(() => new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { interleaved: floats, vertexCount: 3 },
    }));
    expect(e.code).toBe('INVALID_USAGE');
    expect(e.message).toContain('vertex 1');
  });

  test('a two-component position is refused, because the bounds pass would read the wrong floats', () => {
    const e = err(() => new MeshData({
      layout: layout({ position: 'float32x2', uv: 'float32x2' }),
      vertices: { attributes: { position: [0, 0, 1, 1], uv: [0, 0, 1, 1] }, vertexCount: 2 },
    }));
    expect(e.code).toBe('MESH_NO_POSITION');
    expect(e.message).toContain('2 components');
  });

  test('a bounding-sphere override has to be finite and non-negative', () => {
    // A caller-supplied value, so `INVALID_USAGE`: a negative or non-finite
    // radius is a wrong argument, not a disagreement between apse's own tables.
    const base = { layout: POSITION_LAYOUT, vertices: { interleaved: new Float32Array(9), vertexCount: 3 } };
    const negative = err(() => new MeshData({ ...base, boundingSphere: [0, 0, 0, -1] }));
    expect(negative.code).toBe('INVALID_USAGE');
    expect(negative.message).toContain('-1');
    expect(err(() => new MeshData({ ...base, boundingSphere: [0, 0, 0, NaN] })).code)
      .toBe('INVALID_USAGE');
    expect(err(() => new MeshData({ ...base, boundingSphere: [Infinity, 0, 0, 1] })).code)
      .toBe('INVALID_USAGE');
    expect(err(() => new MeshData({ ...base, boundingSphere: [0, 0, 1] })).code)
      .toBe('INVALID_USAGE');
  });

  test('every primitive\'s bound contains every one of its own vertices', () => {
    // Conservative means never an under-estimate, on all nine: a bound that is
    // tight to 1e-7 is fine, a bound that is 1e-7 short drops the object.
    for (const [name, m] of Object.entries({
      box: box(),
      sphere: sphere({ widthSegments: 16, heightSegments: 8 }),
      plane: plane({ widthSegments: 4, depthSegments: 4 }),
      torus: torus({ radialSegments: 16, tubularSegments: 12 }),
      cylinder: cylinder({ radialSegments: 16 }),
      cone: cone({ radialSegments: 16 }),
      capsule: capsule({ radialSegments: 16 }),
      roundedBox: roundedBox({ segments: 2 }),
      grid: grid(),
    })) {
      for (let i = 0; i < m.vertexCount; i++) {
        const p = pos(m, i);
        const d = Math.hypot(p[0] - m.boundingSphere[0], p[1] - m.boundingSphere[1], p[2] - m.boundingSphere[2]);
        expect(d, `${name} vertex ${i} is inside the bound`).toBeLessThanOrEqual(m.boundingSphere[3] + 1e-6);
      }
    }
  });

  test('the attributes source refuses a packed format rather than writing the wrong bytes', () => {
    // A `Float32Array` destination and an `unorm8x4` attribute are a mismatch
    // that compiles, uploads, and shades: the GPU reads four f32 bytes as one
    // byte plus three zero bytes, so 1.0 arrives as 14/255.
    const e = err(() => new MeshData({
      layout: layout({ position: 'float32x3', colour: 'unorm8x4' }),
      vertices: {
        attributes: { position: [0, 0, 0, 1, 0, 0, 0, 1, 0], colour: [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1] },
        vertexCount: 3,
      },
    }));
    expect(e.code).toBe('LAYOUT_MISMATCH');
    expect(e.message).toContain('colour');
    expect(e.message).toContain('unorm8x4');
  });
});

// ---------------------------------------------------------------------------
// Tangents
// ---------------------------------------------------------------------------

describe('computeTangents', () => {
  test('a flat quad gets the tangent the uv gradient says it should', () => {
    // A unit quad in XZ with u running +X and v running +Z: the tangent is
    // +X, exactly, and the handedness is +1 because cross(N, T) · B is positive
    // with N = +Y, T = +X, B = +Z.
    const m = plane({ layout: TANGENT_LAYOUT });
    const r = withTangents(m);
    expect(r.degenerateTriangles).toBe(0);
    expect(r.fallbackVertices).toBe(0);
    for (let i = 0; i < r.mesh.vertexCount; i++) {
      const t = tangent(r.mesh, i);
      expect(Math.abs(t[0]), `tangent ${i} x`).toBeCloseTo(1, 5);
      expect(Math.abs(t[1]), `tangent ${i} y`).toBeCloseTo(0, 5);
      expect(Math.abs(t[2]), `tangent ${i} z`).toBeCloseTo(0, 5);
      expect(t[3], `handedness ${i}`).toBe(1);
    }
  });

  test('every tangents that are unit, orthogonal to the normal, and ±1', () => {
    // The property that actually matters, over all nine primitives: a normal map
    // multiplies a non-unit tangent by a non-orthogonal frame and the surface
    // lights as if it were somewhere else.
    for (const [name, m] of Object.entries({
      box: box({ layout: TANGENT_LAYOUT }),
      sphere: sphere({ layout: TANGENT_LAYOUT, widthSegments: 16, heightSegments: 8 }),
      torus: torus({ layout: TANGENT_LAYOUT, radialSegments: 12, tubularSegments: 8 }),
      cylinder: cylinder({ layout: TANGENT_LAYOUT, radialSegments: 12 }),
      cone: cone({ layout: TANGENT_LAYOUT, radialSegments: 12 }),
      capsule: capsule({ layout: TANGENT_LAYOUT, radialSegments: 12 }),
      roundedBox: roundedBox({ layout: TANGENT_LAYOUT, segments: 2 }),
      plane: plane({ layout: TANGENT_LAYOUT }),
      grid: grid({ layout: TANGENT_LAYOUT }),
    })) {
      const r = withTangents(m);
      expect(r.degenerateTriangles, `${name} has no singular uv triangles`).toBe(0);
      expect(r.fallbackVertices, `${name} has no invented tangents`).toBe(0);
      for (let i = 0; i < r.mesh.vertexCount; i++) {
        const t = tangent(r.mesh, i);
        const n = nrm(r.mesh, i);
        expect(Math.abs(length(t) - 1), `${name} tangent ${i} unit`).toBeLessThan(1e-5);
        expect(Math.abs(dot(t, n)), `${name} tangent ${i} ⟂ normal`).toBeLessThan(1e-5);
        expect(t[3] === 1 || t[3] === -1, `${name} handedness ${i}`).toBe(true);
      }
    }
  });

  test('a sphere\'s tangent is the direction of increasing longitude', () => {
    // Analytic, and the reason this test exists: a wrong tangent direction is
    // invisible until a normal map is on it.
    //
    // The analytic answer is the direction of increasing u, which for `sphere()`'s
    // own parameterisation — column k at `(−cos φ, ·, sin φ)` with `φ = 2πu` — is
    // `(sin φ, 0, cos φ)`. The sign of the x component is the whole content of
    // the test: a negated tangent still satisfies "unit, orthogonal to the
    // normal, handedness ±1" and still passes every other test in this file, and
    // lights a normal map onto the wrong axis.
    const r = withTangents(sphere({ layout: TANGENT_LAYOUT, widthSegments: 24, heightSegments: 12 }));
    // The seam and the poles are the vertices where "the direction of
    // increasing u" is a matter of convention; every other vertex has an exact
    // answer. The seam is *both* of its columns: u = 0 and u = 1 are the same
    // point, and the second one is as contaminated by the pole's u = 0.5 as the
    // first.
    let checked = 0;
    for (let i = 0; i < r.mesh.vertexCount; i++) {
      const p = pos(r.mesh, i);
      if (Math.abs(p[1]) > 0.999) continue; // poles
      const uvHere = uv(r.mesh, i);
      if (uvHere[0] === 0 || uvHere[0] === 1) continue; // the seam columns
      const phi = uvHere[0] * Math.PI * 2;
      const t = tangent(r.mesh, i);
      const want = [Math.sin(phi), 0, Math.cos(phi)];
      // Component-exact only on the equator. The polar rows cannot be: a pole is
      // one vertex whose `u` is 0.5 by convention, so the fan triangle's uv
      // deltas are not the row's own and the accumulated tangent leans a few
      // degrees. That is a property of a single-pole uv sphere, not of the
      // solver, and it decays toward the equator — so it is a direction
      // assertion everywhere and a component assertion where the direction is
      // exact.
      if (Math.abs(p[1]) < 1e-6) {
        expect(t[0], `vertex ${i} x`).toBeCloseTo(want[0], 6);
        expect(t[1], `vertex ${i} y`).toBeCloseTo(0, 6);
        expect(t[2], `vertex ${i} z`).toBeCloseTo(want[2], 6);
      }
      const alignment = dot(t, want);
      expect(alignment, `vertex ${i} points the right way, not the wrong way`)
        .toBeGreaterThan(0.999);
      checked++;
    }
    expect(checked, 'a real number of vertices were checked').toBeGreaterThan(100);
  });

  test('a duplicated uv is skipped and counted, never divided by', () => {
    // Two triangles' worth of vertices with every uv equal: the 2×2 uv system
    // is singular, `1/det` is Infinity, and an Infinity that lands in a tangent
    // survives normalisation as NaN across a whole patch of the mesh.
    const flat = layout({ position: 'float32x3', normal: 'float32x3', uv: 'float32x2' });
    const data = flat.allocate(3);
    const s = flat.stride >> 2;
    for (let i = 0; i < 3; i++) {
      const o = i * s;
      data[o] = i; data[o + 1] = 0; data[o + 2] = 0;
      data[o + 3] = 0; data[o + 4] = 1; data[o + 5] = 0;
      data[o + 6] = 0; data[o + 7] = 0;
    }
    const r = computeTangents({
      layout: flat, vertexData: data, vertexCount: 3, indices: new Uint16Array([0, 1, 2]),
    });
    expect(r.triangles).toBe(1);
    expect(r.degenerateTriangles, 'the whole triangle is skipped').toBe(1);
    expect(r.fallbackVertices, 'and all three of its vertices are invented').toBe(3);
    for (let i = 0; i < 3; i++) {
      const t = r.tangents.subarray(i * 4, i * 4 + 4);
      for (const c of t) expect(Number.isFinite(c), `tangent ${i} is finite`).toBe(true);
      expect(Math.abs(length([t[0], t[1], t[2]]) - 1)).toBeLessThan(1e-5);
      expect(t[3]).toBe(1);
    }
  });

  test('a vertex no triangle reaches gets an invented tangent, not a NaN', () => {
    const flat = layout({ position: 'float32x3', normal: 'float32x3', uv: 'float32x2' });
    const m = new MeshData({
      layout: flat,
      vertices: { attributes: { position: [0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 5, 5], normal: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], uv: [0, 0, 1, 0, 0, 1, 9, 9] }, vertexCount: 4 },
      indices: new Uint16Array([0, 1, 2]),
    });
    const r = computeTangents({ layout: flat, vertexData: m.vertexData, vertexCount: 4, indices: m.indexData });
    expect(r.degenerateTriangles).toBe(0);
    expect(r.fallbackVertices, 'exactly the orphan').toBe(1);
    const orphan = r.tangents.subarray(12, 16);
    for (const c of orphan) expect(Number.isFinite(c)).toBe(true);
    // The fallback is perpendicular to the normal, which is what makes it usable.
    expect(orphan[1]).toBeCloseTo(0, 6);
    expect(Math.hypot(orphan[0], orphan[1], orphan[2])).toBeCloseTo(1, 6);
  });

  test('a tangent parallel to the normal falls back rather than normalising a zero', () => {
    // A uv that varies along the normal direction: the solver's tangent is
    // parallel to the normal, so Gram-Schmidt produces the zero vector and
    // normalising it would be 0/0.
    //
    // The triangle has to be a right triangle in uv — `[[0,0],[1,0],[0,1]]` has
    // det 1. Three collinear uvs are a *singular* uv triangle, which is the case
    // above this one and is counted as degenerate; a uv that is merely
    // pathological in 3D is not.
    const flat = layout({ position: 'float32x3', normal: 'float32x3', uv: 'float32x2' });
    const m = new MeshData({
      layout: flat,
      vertices: {
        attributes: {
          position: [0, 0, 0, 0, 1, 0, 0, 2, 0],
          normal: [0, 1, 0, 0, 1, 0, 0, 1, 0],
          uv: [0, 0, 1, 0, 0, 1],
        },
        vertexCount: 3,
      },
      indices: new Uint16Array([0, 1, 2]),
    });
    const r = computeTangents({ layout: flat, vertexData: m.vertexData, vertexCount: 3, indices: m.indexData });
    expect(r.degenerateTriangles, 'the uv triangle is not singular, it is just vertical').toBe(0);
    expect(r.fallbackVertices, 'all three vertices fall back').toBe(3);
    for (let i = 0; i < 3; i++) {
      const t = r.tangents.subarray(i * 4, i * 4 + 4);
      expect(Number.isFinite(t[0])).toBe(true);
      expect(Math.hypot(t[0], t[1], t[2])).toBeCloseTo(1, 6);
      expect(t[1], 'and it is perpendicular to the +Y normal').toBeCloseTo(0, 6);
    }
  });

  test('a mirrored uv produces a negative handedness', () => {
    // The same quad twice, with v running the other way. The tangent direction
    // is the same and the handedness is the other sign — which is the entire
    // reason the fourth component exists: without it a mirrored uv lights the
    // wrong side of every normal map and nothing in the geometry says so.
    const quad = (mirror: boolean): MeshData => {
      const src = plane({ layout: TANGENT_LAYOUT });
      const out = src.vertexData.slice();
      const s = TANGENT_LAYOUT.stride >> 2;
      const at = TANGENT_LAYOUT.attribute('uv')!.offset >> 2;
      if (mirror) {
        for (let i = 0; i < src.vertexCount; i++) out[i * s + at + 1] = 1 - out[i * s + at + 1];
      }
      return new MeshData({
        layout: TANGENT_LAYOUT,
        vertices: { interleaved: out, vertexCount: src.vertexCount },
        indices: src.indexData,
      });
    };
    const plain = withTangents(quad(false));
    const flipped = withTangents(quad(true));
    expect(tangent(plain.mesh, 0)[3]).toBe(1);
    expect(tangent(flipped.mesh, 0)[3], 'mirrored v flips the handedness').toBe(-1);
    // The tangent direction is unchanged, which proves the sign is not a
    // by-product of a different solve.
    expect(tangent(flipped.mesh, 0)[0]).toBeCloseTo(tangent(plain.mesh, 0)[0], 5);
    expect(flipped.degenerateTriangles).toBe(0);
  });

  test('an unindexed triangle-list works, with the triangles read in order', () => {
    const m = new MeshData({
      layout: layout({ position: 'float32x3', normal: 'float32x3', uv: 'float32x2' }),
      vertices: {
        attributes: {
          position: [0, 0, 0, 1, 0, 0, 0, 1, 0],
          normal: [0, 1, 0, 0, 1, 0, 0, 1, 0],
          uv: [0, 0, 1, 0, 0, 1],
        },
        vertexCount: 3,
      },
    });
    expect(m.indexed).toBe(false);
    const r = computeTangents({ layout: m.layout, vertexData: m.vertexData, vertexCount: 3 });
    expect(r.triangles).toBe(1);
    expect(r.degenerateTriangles).toBe(0);
    expect(r.fallbackVertices).toBe(0);
    expect(r.tangents[0]).toBeCloseTo(1, 5);
  });

  test('a layout with no uv, no normal or no position is refused by name', () => {
    const p = new Float32Array(9);
    expect(err(() => computeTangents({ layout: layout({ position: 'float32x3' }), vertexData: p, vertexCount: 3 }))
      .message).toContain('uv');
    expect(err(() => computeTangents({ layout: layout({ position: 'float32x3', uv: 'float32x2' }), vertexData: p, vertexCount: 3 }))
      .message).toContain('normal');
    expect(err(() => computeTangents({ layout: layout({ uv: 'float32x2', normal: 'float32x3' }), vertexData: p, vertexCount: 3 }))
      .message).toContain('position');
  });
});

describe('withTangents', () => {
  test('a layout with no tangent attribute is refused, naming TANGENT_LAYOUT', () => {
    const e = err(() => withTangents(box()));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain('tangent');
    // The layout constant is the *fix*, not the diagnosis, so it is asserted on
    // the field whose job is the corrective action — the same split every other
    // apse test makes.
    expect(e.fix).toContain('TANGENT_LAYOUT');
  });

  test('a vec3 tangent is refused: there is nowhere to put the handedness', () => {
    const noHandedness = layout({ ...TANGENT_ATTRIBUTES, tangent: 'float32x3' });
    const m = box({ layout: noHandedness });
    const e = err(() => withTangents(m));
    expect(e.code).toBe('LAYOUT_MISMATCH');
    expect(e.message).toContain('float32x4');
  });

  test('the input mesh is untouched and the result shares its bounds exactly', () => {
    const m = box({ layout: TANGENT_LAYOUT });
    const before = Array.from(m.vertexData);
    const r = withTangents(m);
    expect(Array.from(m.vertexData), 'the source is not written through').toEqual(before);
    // Adding a tangent cannot move a vertex, and a bound that moved by a
    // rounding error is a bound that can drop a visible object.
    expect(Array.from(r.mesh.boundingSphere)).toEqual(Array.from(m.boundingSphere));
    expect(Array.from(r.mesh.indexData!)).toEqual(Array.from(m.indexData!));
    expect(r.mesh.layout).toBe(m.layout);
    expect(r.mesh.name).toBe('box:tangent');
  });

  test('only the tangent attribute differs from the input', () => {
    const m = box({ layout: TANGENT_LAYOUT });
    const r = withTangents(m);
    const at = TANGENT_LAYOUT.attribute('tangent')!.offset >> 2;
    const s = TANGENT_LAYOUT.stride >> 2;
    let differing = 0;
    for (let i = 0; i < r.mesh.vertexData.length; i++) {
      if (i % s === at || (i % s) > at) continue;
      expect(r.mesh.vertexData[i], `float ${i} is unchanged`).toBe(m.vertexData[i]);
    }
    for (let i = 0; i < m.vertexCount; i++) {
      const t = tangent(r.mesh, i);
      if (t[0] !== 0 || t[1] !== 0 || t[2] !== 0) differing++;
    }
    expect(differing, 'and the tangents are actually written').toBe(m.vertexCount);
  });

  test('the layout is the single source: 48 bytes a vertex, tangent at 32', () => {
    // 12 + 12 + 8 + 16, in declaration order. The tangent is the last attribute
    // and `float32x4`, so it starts where the uv ends — 32, not 36. WebGPU
    // requires a vertex offset to be a multiple of 4, not of 16, so a `vec3` uv
    // would not buy alignment here; the packed bandwidth it would save is 4 bytes
    // against the 4 bytes of quantisation the whole attribute source refuses.
    expect(TANGENT_LAYOUT.stride).toBe(48);
    expect(TANGENT_LAYOUT.attribute('tangent')?.offset).toBe(32);
    expect(TANGENT_LAYOUT.wgslStruct()).toContain('@location(3) tangent : vec4<f32>,');
    const r = withTangents(box({ layout: TANGENT_LAYOUT }));
    expect(r.mesh.vertexData.length).toBe(r.mesh.vertexCount * 12);
  });
});

// ---------------------------------------------------------------------------
// mergeMeshes
// ---------------------------------------------------------------------------

describe('mergeMeshes', () => {
  const at = (originX: number) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, originX, 0, 0, 1]);

  test('rebases every source onto one vertex buffer and one index buffer', () => {
    const a = box();
    const b = box({ width: 2 });
    const batch = mergeMeshes([{ mesh: a }, { mesh: b, matrix: at(10) }]);
    expect(batch.mesh.vertexCount).toBe(a.vertexCount + b.vertexCount);
    expect(batch.mesh.indexCount).toBe(a.indexCount + b.indexCount);
    expect(batch.mesh.layout).toBe(a.layout);
    const idx = indicesOf(batch.mesh);
    expect(Math.max(...idx), 'no index runs past the merged vertex count').toBe(batch.mesh.vertexCount - 1);
    // Source 0's first two triangles are its own, unshifted, and source 1's are
    // the same two shifted by source 0's vertex count.
    const head = 6;
    expect(idx.slice(0, head)).toEqual(Array.from(a.indexData!).slice(0, head));
    expect(idx.slice(a.indexCount, a.indexCount + head))
      .toEqual(Array.from(b.indexData!).slice(0, head).map((v) => v + a.vertexCount));
  });

  test('keeps one range per source, in order, with the draw arguments in it', () => {
    const parts = [box(), sphere({ widthSegments: 8, heightSegments: 4 }), plane()];
    const batch = mergeMeshes(parts.map((m) => ({ mesh: m })), { name: 'kit' });
    expect(batch.name).toBe('kit');
    expect(batch.ranges).toHaveLength(3);
    let firstVertex = 0, firstIndex = 0;
    parts.forEach((m, i) => {
      const r: BatchedRange = batch.range(i);
      expect(r.name).toBe(m.name);
      expect(r.firstVertex).toBe(firstVertex);
      expect(r.firstIndex).toBe(firstIndex);
      expect(r.indexCount).toBe(m.indexCount);
      expect(r.vertexCount).toBe(m.vertexCount);
      firstVertex += m.vertexCount;
      firstIndex += m.indexCount;
    });
  });

  test('the combined bound contains every source, at every offset', () => {
    const batch = mergeMeshes([
      { mesh: box(), matrix: at(0) },
      { mesh: box(), matrix: at(100) },
      { mesh: box(), matrix: at(-50) },
    ]);
    const b = batch.mesh.boundingSphere;
    for (const r of batch.ranges) {
      const c = r.boundingSphere[0];
      expect([0, 100, -50].some((x) => Math.abs(c - x) < 1e-5), `range centre ${c}`).toBe(true);
    }
    for (let i = 0; i < batch.mesh.vertexCount; i++) {
      const p = pos(batch.mesh, i);
      const d = Math.hypot(p[0] - b[0], p[1] - b[1], p[2] - b[2]);
      expect(d, `merged vertex ${i} inside the combined bound`).toBeLessThanOrEqual(b[3] + 1e-6);
    }
    // Three boxes of half-extent 0.5 spanning -50..100: a bound that forgot the
    // offset one would be off by 50.
    expect(b[3]).toBeGreaterThan(50);
  });

  test('bakes the matrix into the vertices, with the normal through the inverse transpose', () => {
    const uniform = mergeMeshes([{ mesh: box(), matrix: at(5) }]);
    const full = box();
    for (let i = 0; i < full.vertexCount; i++) {
      expect(pos(uniform.mesh, i)[0], `vertex ${i} moved`).toBeCloseTo(pos(full, i)[0] + 5, 5);
    }
    // A non-uniform scale: a 2:1:1 box. The plain 3x3 would leave the normals
    // the wrong length and the wrong angle, which is a shading bug that no
    // geometry test can see.
    const squash = new Float32Array([2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const source = box();
    const baked = mergeMeshes([{ mesh: source, matrix: squash }]);
    for (let i = 0; i < source.vertexCount; i++) {
      // Compared against the *source* mesh's normal rather than against the baked
      // position: a 2:1:1 scale puts the +Y face's outer column at x = 1, exactly
      // where the +X face is, so a position test cannot say which face a vertex is
      // on. A box's normals are all axis vectors, and a diagonal scale's inverse
      // transpose maps an axis vector to the same axis — so the baked normal has to
      // come back as the source's, and unit. The renormalisation is the point: the
      // raw product of invTranspose · (1,0,0) is (0.5, 0, 0), and a stored
      // 0.5-length normal is a mesh that shades with the wrong amount of light.
      const before = nrm(source, i);
      const after = nrm(baked.mesh, i);
      expect(length(after), `normal ${i} unit`).toBeCloseTo(1, 5);
      expect(after[0], `normal ${i} x`).toBeCloseTo(before[0], 5);
      expect(after[1], `normal ${i} y`).toBeCloseTo(before[1], 5);
      expect(after[2], `normal ${i} z`).toBeCloseTo(before[2], 5);
    }
    // A box's own normals are all axis-aligned, so a box cannot distinguish the
    // inverse transpose from the plain 3x3 by direction — only by length, which
    // the renormalisation above undoes. The source that *does* distinguish them is
    // a sphere: its normals run through every direction, so a 2:1:1 scale moves
    // them all, and only the inverse transpose moves them the right way.
    const ball = sphere({ widthSegments: 16, heightSegments: 8 });
    const spun = mergeMeshes([{ mesh: ball, matrix: squash }]);
    // invTranspose of diag(2,1,1) is diag(1/2,1,1): scaling a direction by a matrix
    // divides it by the scale squared, and the transpose undoes the exchange of
    // axes.
    const invTranspose = (n: number[]): number[] => {
      const raw = [0.5 * n[0], n[1], n[2]];
      const len = Math.hypot(raw[0], raw[1], raw[2]);
      return [raw[0] / len, raw[1] / len, raw[2] / len];
    };
    const plain3x3 = (n: number[]): number[] => {
      const raw = [2 * n[0], n[1], n[2]];
      const len = Math.hypot(raw[0], raw[1], raw[2]);
      return [raw[0] / len, raw[1] / len, raw[2] / len];
    };
    let plain3x3WouldMatch = 0;
    for (let i = 0; i < ball.vertexCount; i++) {
      const before = nrm(ball, i);
      const after = nrm(spun.mesh, i);
      const want = invTranspose(before);
      expect(after[0], `sphere normal ${i} x is the inverse transpose's`).toBeCloseTo(want[0], 5);
      expect(after[1], `sphere normal ${i} y is the inverse transpose's`).toBeCloseTo(want[1], 5);
      expect(after[2], `sphere normal ${i} z is the inverse transpose's`).toBeCloseTo(want[2], 5);
      // Counted rather than claimed: the plain 3x3 leaves a sphere's normal
      // pointing along the unscaled sphere, which is a different direction
      // wherever the normal has an x or a z component.
      if (dot(after, plain3x3(before)) > 0.9999) plain3x3WouldMatch++;
    }
    expect(plain3x3WouldMatch, 'a plain 3x3 would have disagreed somewhere')
      .toBeLessThan(ball.vertexCount);
  });

  test('a mirroring matrix flips the tangent handedness', () => {
    const mirror = new Float32Array([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const m = withTangents(box({ layout: TANGENT_LAYOUT }));
    const baked = mergeMeshes([{ mesh: m.mesh, matrix: mirror }]);
    const at = TANGENT_LAYOUT.attribute('tangent')!.offset >> 2;
    const s = TANGENT_LAYOUT.stride >> 2;
    let flipped = 0;
    for (let i = 0; i < m.mesh.vertexCount; i++) {
      const before = m.mesh.vertexData[i * s + at + 3];
      const after = baked.mesh.vertexData[i * s + at + 3];
      if (before === -after) flipped++;
    }
    expect(flipped, 'every handedness flipped').toBe(m.mesh.vertexCount);
  });

  test('a non-indexed source is given a generated index run', () => {
    const flat = new MeshData({
      layout: STANDARD_LAYOUT,
      vertices: {
        attributes: {
          position: [0, 0, 0, 1, 0, 0, 0, 1, 0],
          normal: [0, 1, 0, 0, 1, 0, 0, 1, 0],
          uv: [0, 0, 1, 0, 0, 1],
        },
        vertexCount: 3,
      },
    });
    const batch = mergeMeshes([{ mesh: box() }, { mesh: flat }]);
    expect(batch.mesh.indexed).toBe(true);
    // The generated run starts where the box's 36 indices end, and carries the
    // box's 24 vertices as its base: source 1's vertex `k` is vertex `24 + k`.
    expect(indicesOf(batch.mesh).slice(36, 39)).toEqual([24, 25, 26]);
  });

  test('refuses mismatched layouts, topologies and a singular matrix, by name', () => {
    const a = box();
    const e = err(() => mergeMeshes([{ mesh: a }, { mesh: box({ layout: TANGENT_LAYOUT }) }]));
    expect(e.code).toBe('LAYOUT_MISMATCH');
    expect(e.message).toContain('tangent');

    const strip = new MeshData({
      layout: STANDARD_LAYOUT,
      vertices: { attributes: { position: [0, 0, 0, 1, 0, 0, 0, 1, 0], normal: [0, 1, 0, 0, 1, 0, 0, 1, 0], uv: [0, 0, 1, 0, 0, 1] }, vertexCount: 3 },
      topology: 'triangle-strip',
    });
    expect(err(() => mergeMeshes([{ mesh: a }, { mesh: strip }])).code).toBe('MESH_INDEX_MISALIGNED');

    const singular = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const s = err(() => mergeMeshes([{ mesh: a, matrix: singular }]));
    expect(s.code).toBe('INTERNAL_INVARIANT');
    expect(s.message).toContain('singular');

    expect(err(() => mergeMeshes([])).code).toBe('MESH_EMPTY');
    expect(err(() => mergeMeshes([{ mesh: a, matrix: new Float32Array(4) }])).message).toContain('16');
  });

  test('more than 65535 vertices is refused, with the number', () => {
    // A `plane({ 3, 3 })` is 16 vertices, so the count has to be well past
    // 65535/16 for the check to be reached at all — 3000 of them is 48000 and
    // merges happily, which is the point of the threshold rather than a smaller
    // number.
    const one = plane({ widthSegments: 3, depthSegments: 3 });
    const over = Array.from({ length: 4200 }, () => ({ mesh: one }));
    const e = err(() => mergeMeshes(over));
    expect(e.code).toBe('MESH_DATA_TOO_LARGE');
    expect(e.message).toContain('65535');
    expect(e.message).toContain(String(one.vertexCount * 4200));
    // One fewer is fine, which is what makes 65535 the threshold and not 65536:
    // index 65535 itself is a valid vertex reference.
    expect(mergeMeshes(over.slice(0, 4095)).mesh.vertexCount).toBe(4095 * one.vertexCount);
  });

  test('an out-of-range range names the count', () => {
    const batch = mergeMeshes([{ mesh: box() }]);
    const e = err(() => batch.range(1));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('1 range');
    // The valid range is the corrective action, so it is asserted on `fix`.
    expect(e.fix).toContain('0..0');
  });
});

describe('uploadBatch', () => {
  test('the whole batch is one draw, and a source is a sub-range of the same buffers', () => {
    const fake = recordingDevice();
    const parts = [box(), sphere({ widthSegments: 8, heightSegments: 4 })];
    const batch = mergeMeshes(parts.map((m) => ({ mesh: m })));
    const gpu = uploadBatch(fake.device, batch);

    // Two buffers. Not one per source: that is the entire point.
    expect(fake.buffers).toHaveLength(2);
    expect(gpu.indexFormat).toBe('uint16');
    expect(gpu.firstIndex).toBe(0);
    expect(gpu.baseVertex).toBe(0);
    expect(gpu.instanceCount).toBe(1);

    const sub = gpu.sub(1);
    // A view, not a copy: the same three GPU objects.
    expect(sub.vertexBuffer).toBe(gpu.vertexBuffer);
    expect(sub.indexBuffer).toBe(gpu.indexBuffer);
    expect(sub.layout).toBe(gpu.layout);
    // And the two numbers that make it a *sub*-range draw.
    expect(sub.indexCount).toBe(parts[1].indexCount);
    expect(sub.firstIndex).toBe(parts[0].indexCount);
    expect(sub.baseVertex).toBe(parts[0].vertexCount);
    expect(sub.instanceCount).toBe(1);
  });

  test('the encoded plan is one drawIndexed for the batch and one per sub-range', () => {
    const fake = recordingDevice();
    const parts = [box(), sphere({ widthSegments: 8, heightSegments: 4 })];
    const batch = mergeMeshes(parts.map((m) => ({ mesh: m })));
    const gpu = uploadBatch(fake.device, batch);

    // Exactly the sequence in the GpuMesh contract, recorded rather than assumed.
    const enc = fake.encoder();
    enc.setVertexBuffer(0, gpu.vertexBuffer);
    enc.setIndexBuffer(gpu.indexBuffer!, gpu.indexFormat!);
    enc.drawIndexed(gpu.indexCount, gpu.instanceCount, gpu.firstIndex, gpu.baseVertex, gpu.firstInstance);
    expect(enc.draws).toHaveLength(1);
    expect(enc.draws[0]).toMatchObject({
      kind: 'drawIndexed', vertexCount: batch.mesh.indexCount, instanceCount: 1, firstIndex: 0, baseVertex: 0,
    });

    const enc2 = fake.encoder();
    const s = gpu.sub(1);
    enc2.setVertexBuffer(0, s.vertexBuffer);
    enc2.setIndexBuffer(s.indexBuffer!, s.indexFormat!);
    enc2.drawIndexed(s.indexCount, s.instanceCount, s.firstIndex, s.baseVertex, s.firstInstance);
    expect(enc2.draws[0]).toMatchObject({
      vertexCount: parts[1].indexCount, instanceCount: 1,
      firstIndex: parts[0].indexCount, baseVertex: parts[0].vertexCount,
    });
    // Slot 0 only: a batch has no instance buffer, and binding slot 1 on a
    // one-slot pipeline is a validation error.
    expect(enc2.vertexBindings.map((b) => b.slot)).toEqual([0]);
  });

  test('disposing the batch destroys exactly the two buffers it owns', () => {
    const fake = recordingDevice();
    const batch = mergeMeshes([{ mesh: box() }, { mesh: box() }]);
    const gpu = uploadBatch(fake.device, batch);
    gpu.dispose();
    expect(fake.buffers.every((b) => b.destroyed)).toBe(true);
    expect(gpu.disposed).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// GpuMesh: the draw contract and partial re-upload
// ---------------------------------------------------------------------------

describe('GpuMesh draw contract', () => {
  test('a plain mesh reports the whole-draw defaults', () => {
    const fake = recordingDevice();
    const m = upload(fake.device, box());
    expect(m.instanceCount).toBe(1);
    expect(m.firstInstance).toBe(0);
    expect(m.instanceBuffer).toBeNull();
    expect(m.instanced).toBe(false);
    expect(m.instanceStride).toBe(0);
    expect(m.firstIndex).toBe(0);
    expect(m.baseVertex).toBe(0);
    expect(m.indexFormat).toBe('uint16');
    expect(m.indexCount).toBe(36);
    // One vertex buffer slot, and the layout is the single source for it.
    expect(m.gpuLayouts).toHaveLength(1);
    expect(m.layout.gpuLayouts()).toHaveLength(1);
  });

  test('an unindexed mesh reports indexFormat null, so the renderer calls draw', () => {
    const fake = recordingDevice();
    const flat = new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { attributes: { position: [0, 0, 0, 1, 0, 0, 0, 1, 0] }, vertexCount: 3 },
    });
    const m = upload(fake.device, flat);
    expect(m.indexBuffer).toBeNull();
    expect(m.indexFormat).toBeNull();
    expect(m.indexCount).toBe(3);
  });

  test('the vertex and index buffers are two VERTEX|INDEX|COPY_DST buffers', () => {
    const fake = recordingDevice();
    upload(fake.device, box());
    expect(fake.buffers).toHaveLength(2);
    const [vertex, index] = fake.buffers;
    expect(vertex.label).toBe('box:vertex');
    expect(index.label).toBe('box:index');
    expect(vertex.size).toBe(24 * 32);
    expect(index.size).toBe(36 * 2);
    // `mappedAtCreation` is the one-time fast path, and it is why the fake can
    // read the uploaded bytes back at all.
    expect(fake.descriptors.map((d) => d.mappedAtCreation)).toEqual([true, true]);
  });

  test('a partial vertex write covers exactly the marked span and nothing else', () => {
    const fake = recordingDevice();
    const data = box();
    const m = upload(fake.device, data);
    fake.writes.length = 0;

    data.vertexData[0] = 99;
    m.markVertexDirty(0, 1);
    expect(m.flush(data)).toBe(true);
    expect(fake.writes).toHaveLength(1);
    expect(fake.writes[0].offset, 'byte offset of vertex 0').toBe(0);
    expect(fake.writes[0].elements, 'one vertex of 8 floats').toBe(8);
    // The rest of the buffer is untouched, which is the property that makes a
    // partial write worth having.
    const read = fake.readFloats(m.vertexBuffer);
    expect(read[0]).toBe(99);
    expect(read[8]).toBe(data.vertexData[8]);

    fake.writes.length = 0;
    expect(m.flush(data), 'nothing marked, nothing written').toBe(false);
    expect(fake.writes).toHaveLength(0);
  });

  test('three separate marks coalesce into one write covering the whole run', () => {
    const fake = recordingDevice();
    const data = box();
    const m = upload(fake.device, data);
    fake.writes.length = 0;
    m.markVertexDirty(10, 1);
    m.markVertexDirty(2, 1);
    m.markVertexDirty(6, 1);
    expect(m.vertexDirtyRange).toEqual([2, 9]);
    m.flush(data);
    expect(fake.writes).toHaveLength(1);
    expect(fake.writes[0].offset).toBe(2 * 32);
    expect(fake.writes[0].elements).toBe(9 * 8);
    expect(m.vertexDirtyRange, 'and the span is cleared').toBeNull();
  });

  test('an odd-length index span is widened to a whole number of 4-byte units', () => {
    // A Uint16 index is 2 bytes, and `writeBuffer` requires a multiple of 4, so
    // three indices is a validation error and four is not.
    const fake = recordingDevice();
    const data = box();
    const m = upload(fake.device, data);
    fake.writes.length = 0;
    m.markIndexDirty(0, 3);
    m.flush(data);
    expect(fake.writes[0].offset).toBe(0);
    expect(fake.writes[0].elements, 'widened from 3 to 4').toBe(4);
  });

  test('markAllDirty covers both buffers, and flush takes them together', () => {
    const fake = recordingDevice();
    const data = box();
    const m = upload(fake.device, data);
    fake.writes.length = 0;
    m.markAllDirty();
    expect(m.vertexDirtyRange).toEqual([0, data.vertexCount]);
    expect(m.indexDirtyRange).toEqual([0, data.indexCount]);
    m.flush(data);
    expect(fake.writes.map((w) => w.elements)).toEqual([data.vertexCount * 8, data.indexCount]);
  });

  test('flush refuses a different mesh, by identity on the layout and the counts', () => {
    const fake = recordingDevice();
    const data = box();
    const m = upload(fake.device, data);

    // The check has two discriminators, and both are exercised: the layout's
    // identity, and the two counts. Note what is *not* here — `box({ width: 2 })`.
    // It is a different mesh by every measure a person would use, and it is
    // indistinguishable to `flush`, which compares the layout and the counts and
    // deliberately does not retain the `MeshData` to compare object identity.
    // Asserting that a rejection happens for an indistinguishable mesh would be
    // asserting a check the class does not make and cannot make without the copy
    // it exists to avoid.
    const otherLayout = box({ layout: POSITION_LAYOUT });
    m.markVertexDirty(0, 1);
    const byLayout = err(() => m.flush(otherLayout));
    expect(byLayout.code).toBe('INTERNAL_INVARIANT');
    expect(byLayout.fix).toContain('upload()');

    // Same layout, more vertices: the counts are the only thing left.
    const otherCounts = box({ segments: 2 });
    m.markVertexDirty(0, 1);
    const byCount = err(() => m.flush(otherCounts));
    expect(byCount.code).toBe('INTERNAL_INVARIANT');
    expect(byCount.fix).toContain('upload()');

    // And the data it *was* uploaded from is still accepted, so the marks above
    // did not leave the mesh in a state that rejects its own data.
    expect(m.flush(data)).toBe(true);
  });

  test('a layout override must still describe the data that was packed', () => {
    const fake = recordingDevice();
    // A combined instanced layout over a position-only mesh: the vertex half has
    // to agree, or the offsets the pipeline reads are not the ones written.
    const bad = layout({ position: 'float32x3', normal: 'float32x3', uv: 'float32x2' }, 2048, {
      instanceTransform0: 'float32x4',
    });
    const e = err(() => upload(fake.device, box({ layout: POSITION_LAYOUT }), { layout: bad }));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain('normal');
  });

  test('an unindexed mesh tolerates an index mark', () => {
    const fake = recordingDevice();
    const flat = new MeshData({
      layout: POSITION_LAYOUT,
      vertices: { attributes: { position: [0, 0, 0, 1, 0, 0, 0, 1, 0] }, vertexCount: 3 },
    });
    const m = upload(fake.device, flat);
    expect(() => m.markIndexDirty(0, 3)).not.toThrow();
    expect(m.flush(flat)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Local helpers used by the blocks above
// ---------------------------------------------------------------------------

function err(fn: () => void): AseError {
  try {
    fn();
  } catch (e) {
    if (e instanceof AseError) return e;
    throw e;
  }
  throw new Error('expected an AseError, nothing was thrown');
}

/** The `tangent` attribute of one vertex of a tangent-carrying mesh. */
function tangent(mesh: MeshData, i: number): number[] {
  return attr(mesh, 'tangent', i);
}
