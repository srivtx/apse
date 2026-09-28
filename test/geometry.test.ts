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
  cylinder,
  grid,
  layout,
  MeshData,
  plane,
  POSITION_LAYOUT,
  sphere,
  STANDARD_LAYOUT,
  torus,
  upload,
} from '../src/geometry/index.ts';
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
