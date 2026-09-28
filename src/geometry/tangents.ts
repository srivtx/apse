/**
 * Tangent generation — what a normal map needs and apse had no way to produce.
 *
 * ## Why a tangent is not optional
 *
 * A normal map stores a *tangent-space* perturbation: a direction in the two
 * dimensions the texture is parameterised by. Shading it needs those two
 * dimensions to exist at every vertex, as a direction, and the answer to "which
 * way is +U here" is a per-vertex quantity that depends on the uv layout — the
 * same 3D point has a different tangent depending on which mesh it belongs to
 * and how that mesh is unwrapped. It cannot be derived in the fragment shader,
 * and it cannot be guessed from the normal.
 *
 * So a material that samples a normal map needs a `tangent` vertex attribute,
 * and if the geometry layer cannot produce one, the material library has a hard
 * ceiling: no normal-mapped anything, ever. That is what this module lifts.
 *
 * ## The algorithm
 *
 * Per triangle, solve the 2×2 system that maps uv deltas onto position deltas:
 *
 * ```txt
 *   [du1 dv1] [T]   [e1]
 *   [du2 dv2] [B] = [e2]
 * ```
 *
 * giving `T = (e1·dv2 − e2·dv1) / det` and `B = (e2·du1 − e1·du2) / det`. Both
 * are accumulated per vertex, orthonormalised against the vertex normal at the
 * end, and the handedness is `sign(dot(cross(N, T), B))` — which is MikkTSpace's
 * answer and the reason a mirrored uv does not light the wrong side of every
 * normal map.
 *
 * Accumulating rather than averaging is the point. A vertex shared by six
 * triangles has six different tangent frames, and taking any one of them makes
 * a curved surface visibly crease at the seams. The sum is the least-squares
 * fit, which is what makes a sphere's tangent field continuous.
 *
 * ## The degenerate cases, which are most of the interesting ones
 *
 *   - **`det == 0`** — a zero-area uv triangle, which is what duplicated uv
 *     coordinates at a seam produce. The triangle is *skipped and counted*,
 *     never divided by. This is the one that produces garbage: a `1/0` here is
 *     an `Infinity` that lands in every vertex of the triangle and survives
 *     normalisation as a NaN tangent on a whole patch of the mesh.
 *   - **a vertex touched by no triangle**, or only by skipped ones: its
 *     accumulated tangent is the zero vector, and normalisation gives NaN. It is
 *     replaced with any vector perpendicular to its normal, and counted.
 *   - **no uv attribute**: refused by name. A tangent with no uv is a random
 *     direction, and a random direction is worse than none because it is
 *     plausible.
 *   - **no normal attribute**: refused by name, for the same reason — the
 *     orthonormalisation has nothing to be orthogonal *to*.
 */

import { fail } from '../core/error.ts';
import { TANGENT, type VertexLayout } from './layout.ts';
import { MeshData } from './mesh.ts';

/** Floats in a tangent: x, y, z, and the ±1 handedness. */
export const TANGENT_FLOATS = 4;

/** What {@link computeTangents} produced, and what it had to skip to do it. */
export interface TangentBasis {
  /**
   * `vertexCount * 4` floats, in the same vertex order as the input: `xyz` is a
   * unit tangent, `w` is the handedness, exactly `+1` or `-1`.
   */
  readonly tangents: Float32Array;
  /** Triangles that contributed nothing, because their uv triangle is singular. */
  readonly degenerateTriangles: number;
  /** Vertices whose tangent had to be invented, because nothing accumulated. */
  readonly fallbackVertices: number;
  /** Triangles that were read, for a ratio. */
  readonly triangles: number;
}

export interface TangentInput {
  readonly layout: VertexLayout;
  /** The interleaved buffer, `layout.stride` bytes per vertex. */
  readonly vertexData: Float32Array;
  readonly vertexCount: number;
  /** Null for a non-indexed triangle-list, where the triangles are implied. */
  readonly indices?: Uint32Array | Uint16Array | null;
  /** Only used in error messages. */
  readonly name?: string;
}

/**
 * Computes a tangent basis from positions, uvs and indices.
 *
 * Takes the three arrays rather than a {@link MeshData} so it works on data that
 * is not a mesh yet — a merge in progress, a mesh still being written — and
 * returns a plain `Float32Array` of `vertexCount * 4` floats rather than
 * writing into a vertex buffer, so the caller decides where it lands.
 *
 * Allocates the result and nothing else: no per-vertex scratch, no per-triangle
 * temporaries. At 200,000 triangles that is the difference between one array
 * and half a million short-lived ones.
 */
export function computeTangents(input: TangentInput): TangentBasis {
  const { layout, vertexData, vertexCount, indices = null, name = 'geometry' } = input;

  const position = layout.attribute('position');
  if (position === undefined) {
    fail('MESH_NO_POSITION',
      `computeTangents() on "${name}" was given a layout with no "position" attribute.`, {
      why: 'A tangent frame is built out of the position deltas across a triangle. Without positions there is nothing to build it from, and the result would be a direction with no surface to lie on.',
      fix: 'Use a layout built from `STANDARD_ATTRIBUTES` or `TANGENT_ATTRIBUTES`.',
    });
  }
  const uv = layout.attribute('uv');
  if (uv === undefined) {
    fail('ATTRIBUTE_MISSING',
      `computeTangents() on "${name}" was given a layout with no "uv" attribute.`, {
      why: 'The tangent is the direction of increasing u, and u only exists as a vertex attribute. Without it there is no definition to solve for — a tangent produced some other way is a guess, and a guessed tangent shades a normal map onto the wrong axis.',
      fix: 'Add `uv: "float32x2"` to the layout. `STANDARD_ATTRIBUTES` and `TANGENT_ATTRIBUTES` both have it.',
    });
  }
  const normal = layout.attribute('normal');
  if (normal === undefined) {
    fail('ATTRIBUTE_MISSING',
      `computeTangents() on "${name}" was given a layout with no "normal" attribute.`, {
      why: 'The accumulated tangent is made perpendicular to the vertex normal before it is stored — that is the Gram-Schmidt step, and it is what keeps the tangent frame orthogonal as the surface curves. Without a normal there is nothing to be perpendicular to, and the result is the raw sum of six disagreeing triangles.',
      fix: 'Add `normal: "float32x3"` to the layout, or use `TANGENT_ATTRIBUTES` / `STANDARD_ATTRIBUTES`.',
    });
  }
  if (position.info.components !== 3 || uv.info.components !== 2 || normal.info.components !== 3) {
    fail('ATTRIBUTE_MISSING',
      `computeTangents() needs position as 3 components, normal as 3 and uv as 2.`, {
      why: 'The solver reads a fixed number of floats at each attribute\'s offset, and the layout is what those offsets come from. A shorter attribute would have the next attribute read as a coordinate.',
      fix: 'Declare `position: "float32x3"`, `normal: "float32x3"`, `uv: "float32x2"`.',
    });
  }

  const stride = layout.stride >> 2;
  const p = position.offset >> 2;
  const n = normal.offset >> 2;
  const t = uv.offset >> 2;

  const triangles = indices === null
    ? Math.floor(vertexCount / 3)
    : Math.floor(indices.length / 3);
  const tangents = new Float32Array(vertexCount * TANGENT_FLOATS);
  // Two scratch runs, three floats per vertex: the accumulated tangent and the
  // accumulated bitangent. Separate from `tangents`, which is four floats per
  // vertex and is the *result* — a stride mismatch between an accumulator and its
  // output is invisible until a tangent comes out somewhere else entirely.
  const accumulated = new Float32Array(vertexCount * 3);
  const bitangents = new Float32Array(vertexCount * 3);
  // Which vertices a real triangle contributed to. Recoverable from the arrays
  // only if no triangle ever cancels exactly, which is not a thing to rely on.
  const contributed = new Uint8Array(vertexCount);

  let degenerateTriangles = 0;
  const readIndex = indices === null
    ? (k: number): number => k
    : (k: number): number => indices[k];

  for (let tri = 0; tri < triangles; tri++) {
    const i0 = readIndex(tri * 3);
    const i1 = readIndex(tri * 3 + 1);
    const i2 = readIndex(tri * 3 + 2);
    const o0 = i0 * stride, o1 = i1 * stride, o2 = i2 * stride;

    const e1x = vertexData[o1 + p] - vertexData[o0 + p];
    const e1y = vertexData[o1 + p + 1] - vertexData[o0 + p + 1];
    const e1z = vertexData[o1 + p + 2] - vertexData[o0 + p + 2];
    const e2x = vertexData[o2 + p] - vertexData[o0 + p];
    const e2y = vertexData[o2 + p + 1] - vertexData[o0 + p + 1];
    const e2z = vertexData[o2 + p + 2] - vertexData[o0 + p + 2];

    const du1 = vertexData[o1 + t] - vertexData[o0 + t];
    const dv1 = vertexData[o1 + t + 1] - vertexData[o0 + t + 1];
    const du2 = vertexData[o2 + t] - vertexData[o0 + t];
    const dv2 = vertexData[o2 + t + 1] - vertexData[o0 + t + 1];

    const det = du1 * dv2 - du2 * dv1;
    // The one line in this module that must not be skipped. `1/0` is Infinity
    // and `1/NaN` is NaN, and both of them survive into the accumulated tangent
    // of all three vertices, where normalisation turns them into a patch of
    // NaN tangents that a normal map multiplies through. Skipping and counting
    // is the only honest answer for a triangle with no uv area.
    if (det === 0 || !Number.isFinite(det)) {
      degenerateTriangles++;
      continue;
    }
    const r = 1 / det;

    const tx = (e1x * dv2 - e2x * dv1) * r;
    const ty = (e1y * dv2 - e2y * dv1) * r;
    const tz = (e1z * dv2 - e2z * dv1) * r;
    const bx = (e2x * du1 - e1x * du2) * r;
    const by = (e2y * du1 - e1y * du2) * r;
    const bz = (e2z * du1 - e1z * du2) * r;

    accumulate(accumulated, i0, tx, ty, tz);
    accumulate(accumulated, i1, tx, ty, tz);
    accumulate(accumulated, i2, tx, ty, tz);
    accumulate(bitangents, i0, bx, by, bz);
    accumulate(bitangents, i1, bx, by, bz);
    accumulate(bitangents, i2, bx, by, bz);
    contributed[i0] = 1;
    contributed[i1] = 1;
    contributed[i2] = 1;
  }

  let fallbackVertices = 0;
  for (let i = 0; i < vertexCount; i++) {
    const o = i * TANGENT_FLOATS;
    const b = i * 3;
    const nx = vertexData[i * stride + n];
    const ny = vertexData[i * stride + n + 1];
    const nz = vertexData[i * stride + n + 2];
    const tx = accumulated[i * 3], ty = accumulated[i * 3 + 1], tz = accumulated[i * 3 + 2];

    if (contributed[i] === 0 || !Number.isFinite(tx + ty + tz)) {
      fallback(tangents, i, nx, ny, nz);
      fallbackVertices++;
      continue;
    }

    // Gram-Schmidt against the normal. Without it the tangent of a sphere drifts
    // away from the surface: the accumulation is only the least-squares fit in
    // *uv* space, and the uv parameterisation of a curved surface is not
    // orthogonal to the surface normal.
    const dot = tx * nx + ty * ny + tz * nz;
    let ax = tx - nx * dot;
    let ay = ty - ny * dot;
    let az = tz - nz * dot;
    let len = Math.sqrt(ax * ax + ay * ay + az * az);
    if (!(len > 1e-12) || !Number.isFinite(len)) {
      // The accumulated tangent was parallel to the normal — which happens when
      // the uv is degenerate along the surface, the same failure as a zero-area
      // uv triangle but spread over a whole vertex.
      fallback(tangents, i, nx, ny, nz);
      fallbackVertices++;
      continue;
    }
    ax /= len; ay /= len; az /= len;

    // Handedness: `+1` when the uv basis agrees with the frame implied by the
    // normal, `-1` when it is mirrored. Stored as the fourth component so the
    // fragment stage can flip the bitangent instead of lighting the back of
    // every normal map.
    const cx = ny * az - nz * ay;
    const cy = nz * ax - nx * az;
    const cz = nx * ay - ny * ax;
    const w = (cx * bitangents[b] + cy * bitangents[b + 1] + cz * bitangents[b + 2]) < 0 ? -1 : 1;

    tangents[o] = ax;
    tangents[o + 1] = ay;
    tangents[o + 2] = az;
    tangents[o + 3] = w;
  }

  return { tangents, degenerateTriangles, fallbackVertices, triangles };
}

/** Adds one triangle's tangent to a vertex's running sum. */
function accumulate(out: Float32Array, vertex: number, x: number, y: number, z: number): void {
  const o = vertex * 3;
  out[o] += x;
  out[o + 1] += y;
  out[o + 2] += z;
}

/**
 * Any unit vector perpendicular to `(nx, ny, nz)`, and `+1` for handedness.
 *
 * Crosses the normal with whichever cardinal axis it is *least* aligned with:
 * crossing with the most aligned one gives a near-zero vector, which is the same
 * failure as a degenerate triangle in a place that is much harder to notice.
 */
function fallback(out: Float32Array, vertex: number, nx: number, ny: number, nz: number): void {
  const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
  let ux = 0, uy = 0, uz = 0;
  if (ax <= ay && ax <= az) ux = 1;
  else if (ay <= az) uy = 1;
  else uz = 1;
  let tx = ny * uz - nz * uy;
  let ty = nz * ux - nx * uz;
  let tz = nx * uy - ny * ux;
  const len = Math.sqrt(tx * tx + ty * ty + tz * tz);
  const o = vertex * TANGENT_FLOATS;
  // A zero normal — a mesh that declares one and leaves it at zero — leaves `len`
  // at zero too. Rather than divide, store the first axis: an unnormalised
  // tangent on an unnormalised normal is a mesh the caller already got wrong, and
  // a NaN here would poison every downstream product.
  if (len > 1e-12 && Number.isFinite(len)) {
    tx /= len; ty /= len; tz /= len;
  } else {
    tx = 1; ty = 0; tz = 0;
  }
  out[o] = tx;
  out[o + 1] = ty;
  out[o + 2] = tz;
  out[o + 3] = 1;
}

/**
 * Returns `mesh` with its `tangent` attribute filled in.
 *
 * The result is a new {@link MeshData} over a copy of the interleaved buffer
 * with the tangents written at the layout's own offset. Positions, normals, uvs
 * and indices are byte-identical to the input, and so are the bounds — which is
 * the point of passing them through rather than recomputing: adding a tangent
 * cannot change where the mesh is, and a bound that moved by a rounding error
 * is a bound that can drop a visible object.
 *
 * `layout` must declare a `tangent` of `float32x4`. Anything else is refused by
 * name: a `vec3` tangent has nowhere to put the handedness, and a packed one
 * cannot be written by a scalar store.
 */
export function withTangents(mesh: MeshData, opts: WithTangentsOptions = {}): TangentMeshResult {
  const tangent = mesh.layout.attribute(TANGENT);
  if (tangent === undefined) {
    fail('ATTRIBUTE_MISSING',
      `withTangents() needs "${mesh.name}" to be built with a "${TANGENT}" attribute, and its layout has none.`, {
      why: 'The tangent is written into the interleaved buffer at the offset the layout resolved for it. A layout without that attribute has no offset to write to, and the mesh would come out of the function with its geometry intact and no tangents in it — a normal-mapped material reading zeroes.',
      fix: `Build the mesh with a layout that has one: \`TANGENT_LAYOUT\` (position, normal, uv, tangent: float32x4, 48 bytes per vertex), or \`layout({ ...TANGENT_ATTRIBUTES })\` if you need a different vertex half.`,
    });
  }
  if (tangent.format !== 'float32x4') {
    fail('LAYOUT_MISMATCH',
      `withTangents() needs "${TANGENT}" as "float32x4", but "${mesh.name}" declares "${tangent.format}".`, {
      why: 'The fourth component is the handedness — a +1 or a −1, not a coordinate. A vec3 tangent has nowhere to put it, so a mirrored uv normal map lights the wrong side of the surface, and a packed format cannot be written by the four scalar stores that produce it.',
      fix: 'Declare the tangent as `tangent: "float32x4"`, which is what `TANGENT_ATTRIBUTES` and `TANGENT_LAYOUT` do.',
    });
  }

  const basis = computeTangents({
    layout: mesh.layout,
    vertexData: mesh.vertexData,
    vertexCount: mesh.vertexCount,
    indices: mesh.indexData,
    name: mesh.name,
  });

  const data = mesh.vertexData.slice();
  const stride = mesh.layout.stride >> 2;
  const at = tangent.offset >> 2;
  for (let i = 0; i < mesh.vertexCount; i++) {
    const s = i * TANGENT_FLOATS;
    const d = i * stride + at;
    data[d] = basis.tangents[s];
    data[d + 1] = basis.tangents[s + 1];
    data[d + 2] = basis.tangents[s + 2];
    data[d + 3] = basis.tangents[s + 3];
  }

  const name = opts.name ?? `${mesh.name}:tangent`;
  return {
    mesh: new MeshData({
      name,
      layout: mesh.layout,
      vertices: { interleaved: data, vertexCount: mesh.vertexCount },
      indices: mesh.indexData,
      // Passed through rather than recomputed: identical positions must give an
      // identical bound, and "identical" here means bit-identical.
      boundingSphere: mesh.boundingSphere,
      topology: mesh.topology,
    }),
    ...basis,
  };
}

export interface WithTangentsOptions {
  /** Mesh name for the result. Defaults to `` `${mesh.name}:tangent` ``. */
  readonly name?: string;
}

/** What {@link withTangents} returns: the mesh, and the diagnostics. */
export interface TangentMeshResult extends TangentBasis {
  readonly mesh: MeshData;
}

/** The attribute name, re-exported so a batcher or a loader need not import layout.ts. */
export { TANGENT };
