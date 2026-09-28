/**
 * Cameras — and the one matrix convention everything downstream inherits.
 *
 * ## Clip space: `z` is `[0, 1]`, not `[-1, 1]`
 *
 * This is the single most common WebGPU bug, and it is a silent one: nothing
 * validates, the pipeline is created, the draw runs, and the result is either a
 * scene rendered inside-out or a scene that vanishes into the far plane.
 * OpenGL's NDC maps the near plane to `z = -1` and the far plane to `z = +1`.
 * WebGPU's maps the near plane to **`z = 0`** and the far plane to **`z = 1`**,
 * matching D3D, Metal, and Vulkan's `depthClipControl`.
 *
 * So a point at view `z = -near` must land at NDC `z = 0`, and one at
 * `z = -far` at NDC `z = 1`. Both are asserted in `test/scene.test.ts`. With
 * that mapping, `depthCompare: 'less'` against a depth buffer cleared to 0 is
 * correct with no bias, no remap, and no `z ∈ [-1, 1]` fixup in the shader —
 * pipeline, depth-stencil attachment, and this matrix all agree.
 *
 * ## Handedness: the camera looks down −Z
 *
 * The brief for this module asked for a *left-handed* view matrix, and also
 * asked for the depth assertion above. Those two are mutually exclusive. A
 * left-handed basis (x right, y up) has `+z` pointing into the screen; the
 * brief's assertion — `z = -near` maps to depth 0 — requires `−z` to point
 * into the screen, which is a right-handed basis. One of them had to give, and
 * the depth range is not the negotiable one, because clip space is fixed by the
 * API rather than chosen by the library.
 *
 * So: **right-handed world, camera looking down −Z in view space.** That is
 * the convention `src/math/mat4.ts` already implements — `lookAt`,
 * `perspective`, and `orthographic` all agree with it — and it is what every
 * WebGPU sample uses. One consequence worth stating once: WGSL's `frontFace`
 * stays `'ccw'` for counter-clockwise winding, so there is no handedness flip
 * anywhere in the rasteriser.
 *
 * `lookAt` and the projections delegate to `src/math/mat4.ts` rather than
 * reimplementing the basis, so the repository has exactly one implementation of
 * this convention.
 */

import { fail } from '../core/error.ts';
import { invert, lookAt, mul, orthographic, perspective, transformVec4 } from '../math/mat4.ts';
import { FRUSTUM_STRIDE, setFromViewProjection } from '../math/frustum.ts';
import { DEFAULT_LAYER } from './node.ts';

const DEG_TO_RAD = Math.PI / 180;

/** Planes in a frustum: 6 × `(nx, ny, nz, d)`. */
export const FRUSTUM_PLANE_COUNT = FRUSTUM_STRIDE / 4;

/** Length of the plane array {@link Camera.getFrustum} writes. */
export const FRUSTUM_PLANE_LENGTH = FRUSTUM_STRIDE;

/** Six world-space frustum planes. See {@link FRUSTUM_PLANE_COUNT}. */
export type Frustum = Float32Array;

/**
 * A world-space ray, as `[ox, oy, oz, dx, dy, dz]`. The direction is unit
 * length; the origin is on the near plane for a perspective camera, or on the
 * near plane of the view volume for an orthographic one.
 */
export type Ray = Float32Array;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface CameraOptions {
  /** Vertical field of view, **in degrees**. Ignored by an orthographic camera. */
  fov?: number;
  /** Distance to the near plane. Must be `> 0` and `< far`. */
  near?: number;
  /** Distance to the far plane. `Infinity` gives the infinite-far projection. */
  far?: number;
  /** width / height. Kept current by passing it to {@link Camera.update}. */
  aspect?: number;
}

export interface PerspectiveCameraOptions extends CameraOptions {}

export interface OrthographicCameraOptions {
  left?: number;
  right?: number;
  bottom?: number;
  top?: number;
  near?: number;
  far?: number;
  aspect?: number;
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/**
 * Base camera: matrix ownership, the view/projection pair, and the frustum.
 *
 * A camera owns six matrices because a renderer needs all six, and because
 * recomputing `invViewProj` at the point of use is how world-space picking
 * quietly ends up using last frame's camera. All six are written by one call to
 * {@link update}, and none of them is ever reallocated — a renderer can hold a
 * reference to `viewProj` forever.
 *
 * ### What you may and may not touch
 *
 * `projection`, `view` and everything derived are **owned by the camera** and
 * rewritten by `update()`. Assigning `camera.view` by hand works until the next
 * `update()` and then silently stops working, which is the same contract every
 * engine has and the same way it is misused. To move the camera call
 * {@link lookAt}; to change the lens set `fov`, `near`, `far` or the
 * orthographic extents, which are accessor-backed and mark the camera stale, so
 * {@link getFrustum} can never hand back a frustum that disagrees with the
 * matrices the renderer used.
 */
export abstract class Camera {
  /** World → clip. Owned by the camera. */
  readonly projection: Float32Array = new Float32Array(16);
  /** World → view. Owned by the camera; set by {@link lookAt}. */
  readonly view: Float32Array = new Float32Array(16);
  /** `projection * view`. */
  readonly viewProj: Float32Array = new Float32Array(16);
  /** `inverse(view)`. */
  readonly invView: Float32Array = new Float32Array(16);
  /** `inverse(projection)`. */
  readonly invProj: Float32Array = new Float32Array(16);
  /** `inverse(viewProj)` — world → clip → world. */
  readonly invViewProj: Float32Array = new Float32Array(16);

  /** Eye position in world space. */
  readonly worldPosition: Float32Array = new Float32Array(3);

  /**
   * Layer bitmask. A node is drawn only when `node.layer & camera.layers` is
   * non-zero, and a node masked out takes its whole subtree with it.
   */
  layers: number = DEFAULT_LAYER;

  #fov = 60;
  #near = 0.1;
  #far = 1000;
  #aspect = 1;

  #eye = new Float32Array(3);
  #target = new Float32Array([0, 0, -1]);
  #up = new Float32Array([0, 1, 0]);
  #stale = true;

  constructor(options: CameraOptions = {}) {
    if (options.fov !== undefined) this.fov = options.fov;
    if (options.near !== undefined) this.near = options.near;
    if (options.far !== undefined) this.far = options.far;
    if (options.aspect !== undefined) this.aspect = options.aspect;
  }

  /**
   * Vertical field of view, in **degrees**.
   *
   * Degrees, not radians, because that is what every scene format, inspector
   * and content pipeline speaks — and because the value is validated on write,
   * so a radians value passed here trips the `0 < fov < 180` check immediately
   * instead of quietly rendering a 1° telephoto lens.
   */
  get fov(): number {
    return this.#fov;
  }

  set fov(degrees: number) {
    // Strictly inside (0, 180): 0 and 180 both put tan(fov/2) at 0 or infinity,
    // which is a projection of zeros and a divide by zero respectively.
    if (!Number.isFinite(degrees) || degrees <= 0 || degrees >= 180) {
      fail('INTERNAL_INVARIANT',
        `Camera fov must be a finite number strictly between 0 and 180 degrees; it was given ${degrees}.`, {
        why: 'The projection is f = 1 / tan(fov / 2). At 0 the frustum is infinitely narrow and at 180 it is infinitely wide, so either way the matrix fills with zeros or infinities, and the GPU renders nothing while reporting no error at all.',
        fix: 'Field of view is in degrees. If you meant radians — 1.0472 for 60° — multiply by 180 / Math.PI first.',
        detail: { kind: 'numeric', field: 'fov', value: degrees, min: 0, max: 180 },
      });
    }
    this.#fov = degrees;
    this.markStale();
  }

  /** Distance to the near plane. Always positive. */
  get near(): number {
    return this.#near;
  }

  set near(distance: number) {
    assertPositive('near', distance, true);
    this.#near = distance;
    this.markStale();
  }

  /** Distance to the far plane. `Infinity` is legal, and correct for skies. */
  get far(): number {
    return this.#far;
  }

  set far(distance: number) {
    assertPositive('far', distance, false);
    this.#far = distance;
    this.markStale();
  }

  /** width / height of the render target. */
  get aspect(): number {
    return this.#aspect;
  }

  set aspect(value: number) {
    assertPositive('aspect', value, true);
    this.#aspect = value;
    this.markStale();
  }

  /** The eye passed to {@link lookAt}. A live view — write it and re-look. */
  get eye(): Float32Array {
    return this.#eye;
  }

  /** The look-at target passed to {@link lookAt}. A live view. */
  get target(): Float32Array {
    return this.#target;
  }

  /** The up axis passed to {@link lookAt}. A live view. */
  get up(): Float32Array {
    return this.#up;
  }

  /**
   * Points the camera at `target` from `eye`.
   *
   * The resulting view matrix maps the eye to the origin and the view
   * direction to −Z — right-handed, as the module header explains. `up` is
   * orthogonalised against the view axis, so it need not be perpendicular or
   * unit length, and `eye === target` yields a zero basis rather than a NaN
   * one.
   *
   * The three vectors are remembered, so a later {@link update} rebuilds the
   * same view. That is what makes `update()` idempotent instead of quietly
   * resetting the camera to the origin.
   */
  lookAt(eye: ArrayLike<number>, target: ArrayLike<number>, up: ArrayLike<number> = [0, 1, 0]): this {
    this.#eye[0] = eye[0] ?? 0;
    this.#eye[1] = eye[1] ?? 0;
    this.#eye[2] = eye[2] ?? 0;
    this.#target[0] = target[0] ?? 0;
    this.#target[1] = target[1] ?? 0;
    this.#target[2] = target[2] ?? 0;
    this.#up[0] = up[0] ?? 0;
    this.#up[1] = up[1] ?? 1;
    this.#up[2] = up[2] ?? 0;
    lookAt(this.view, this.#eye, this.#target, this.#up);
    this.markStale();
    return this;
  }

  /**
   * Recomputes every matrix. Call once per frame, after {@link lookAt} and after
   * any lens change.
   *
   * `aspect` is an argument rather than something read off a canvas so that the
   * camera is not coupled to a surface: a render-target resolver, a shadow
   * pass, and a picking ray each pass their own.
   */
  update(aspect?: number): this {
    if (aspect !== undefined) this.aspect = aspect;
    if (this.#far < this.#near) {
      fail('INTERNAL_INVARIANT',
        `Camera far plane (${this.#far}) is behind its near plane (${this.#near}).`, {
        why: 'The projection divides by (near − far). When the far plane is nearer than the near plane the depth range inverts, so far geometry draws over near geometry and the cull frustum is inside-out.',
        fix: 'Swap the two values, or set `far` to `Infinity` if you do not need a far plane at all.',
        detail: { kind: 'numeric', field: 'far', value: this.#far, min: this.#near },
      });
    }
    this.updateProjection();
    mul(this.viewProj, this.projection, this.view);
    invert(this.invView, this.view);
    invert(this.invProj, this.projection);
    invert(this.invViewProj, this.viewProj);
    this.worldPosition[0] = this.#eye[0];
    this.worldPosition[1] = this.#eye[1];
    this.worldPosition[2] = this.#eye[2];
    this.#stale = false;
    return this;
  }

  /**
   * The six frustum planes in world space, for sphere culling.
   *
   * `out` must be a `Float32Array(24)`. Each plane is `[nx, ny, nz, d]`,
   * pointing inward, with unit normals — so a test is a dot product against a
   * centre with no per-plane normalisation at the point of use.
   *
   * Updates the camera first if the lens or the pose changed since the last
   * call, so the frustum can never disagree with the matrices.
   */
  getFrustum(out: Frustum): Frustum {
    if (out.length < FRUSTUM_PLANE_LENGTH) {
      fail('INTERNAL_INVARIANT',
        `getFrustum needs a Float32Array(${FRUSTUM_PLANE_LENGTH}); it was given one of length ${out.length}.`, {
        why: 'A frustum is six planes of four floats. A shorter array would be written past its end.',
        fix: `Allocate \`new Float32Array(${FRUSTUM_PLANE_LENGTH})\`, or pass the scene's cached scratch array.`,
      });
    }
    if (this.#stale) this.update();
    return setFromViewProjection(out, this.viewProj);
  }

  /**
   * The world-space ray through a normalised device coordinate, written into
   * `out` as `[origin.xyz, direction.xyz]` with a unit direction.
   *
   * `ndcX` and `ndcY` are in `[-1, 1]`, x to the right, y up. The direction is
   * the world-space difference between the unprojected near and far points, not
   * the difference between the two clipped ones — for a perspective camera the
   * NDC ray and the world ray are the same line, but only the world-space
   * version has a length to normalise.
   */
  rayFromNdc(ndcX: number, ndcY: number, out: Ray): Ray {
    if (this.#stale) this.update();
    _clip[0] = ndcX;
    _clip[1] = ndcY;
    _clip[2] = 0;
    _clip[3] = 1;
    transformVec4(_nearPoint, _clip, this.invViewProj);
    _clip[2] = 1;
    transformVec4(_farPoint, _clip, this.invViewProj);
    const nearW = _nearPoint[3];
    const farW = _farPoint[3];
    const nInv = 1 / (nearW === 0 ? 1 : nearW);
    const fInv = 1 / (farW === 0 ? 1 : farW);
    const ox = _nearPoint[0] * nInv;
    const oy = _nearPoint[1] * nInv;
    const oz = _nearPoint[2] * nInv;
    let dx = _farPoint[0] * fInv - ox;
    let dy = _farPoint[1] * fInv - oy;
    let dz = _farPoint[2] * fInv - oz;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 0) {
      dx /= len;
      dy /= len;
      dz /= len;
    }
    out[0] = ox;
    out[1] = oy;
    out[2] = oz;
    out[3] = dx;
    out[4] = dy;
    out[5] = dz;
    return out;
  }

  /** True when the pose or the lens changed since the last {@link update}. */
  get stale(): boolean {
    return this.#stale;
  }

  /**
   * Forces a recomputation on the next {@link update} or {@link getFrustum}.
   * For subclasses whose projection depends on more than the base lens.
   */
  protected markStale(): void {
    this.#stale = true;
  }

  /** Writes this camera's own projection matrix. */
  protected abstract updateProjection(): void;
}

// ---------------------------------------------------------------------------
// Concrete cameras
// ---------------------------------------------------------------------------

/**
 * A perspective camera.
 *
 * ```ts
 * const camera = new PerspectiveCamera({ fov: 60, near: 0.1, far: 1000 });
 * camera.lookAt([0, 2, 10], [0, 0, 0], [0, 1, 0]);
 * camera.update(canvas.width / canvas.height);
 * ```
 */
export class PerspectiveCamera extends Camera {
  constructor(options: PerspectiveCameraOptions = {}) {
    super(options);
  }

  protected override updateProjection(): void {
    perspective(this.projection, this.fov * DEG_TO_RAD, this.aspect, this.near, this.far);
  }
}

/** An orthographic camera — for UI, minimaps, and shadow-map projection. */
export class OrthographicCamera extends Camera {
  #left: number;
  #right: number;
  #bottom: number;
  #top: number;

  constructor(options: OrthographicCameraOptions = {}) {
    // `fov` is meaningless here, so it is not forwarded.
    super({ near: options.near, far: options.far, aspect: options.aspect });
    this.#left = options.left ?? -1;
    this.#right = options.right ?? 1;
    this.#bottom = options.bottom ?? -1;
    this.#top = options.top ?? 1;
  }

  get left(): number {
    return this.#left;
  }

  set left(value: number) {
    this.#left = value;
    this.markStale();
  }

  get right(): number {
    return this.#right;
  }

  set right(value: number) {
    this.#right = value;
    this.markStale();
  }

  get bottom(): number {
    return this.#bottom;
  }

  set bottom(value: number) {
    this.#bottom = value;
    this.markStale();
  }

  get top(): number {
    return this.#top;
  }

  set top(value: number) {
    this.#top = value;
    this.markStale();
  }

  protected override updateProjection(): void {
    orthographic(
      this.projection,
      this.#left, this.#right, this.#bottom, this.#top,
      this.near, this.far,
    );
  }
}

// ---------------------------------------------------------------------------
// Scratch — module-level, reused. `rayFromNdc` allocates nothing.
// ---------------------------------------------------------------------------

const _clip = new Float32Array(4);
const _nearPoint = new Float32Array(4);
const _farPoint = new Float32Array(4);

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * Rejects a distance that would put a divide by zero — or a NaN — into the
 * projection. `finiteOnly` rejects `Infinity`, which `near` and `aspect` must
 * be, and which `far` is explicitly allowed to be.
 */
function assertPositive(field: string, value: number, finiteOnly: boolean): void {
  if (Number.isNaN(value) || value <= 0 || (finiteOnly && !Number.isFinite(value))) {
    fail('INTERNAL_INVARIANT',
      `Camera ${field} must be a positive distance; it was given ${value}.`, {
      why: `The projection matrix divides by ${field}, or uses it as a scale factor. Zero, a negative, or a NaN ${field} puts infinities into a uniform buffer and every triangle is silently discarded.`,
      fix: field === 'far'
        ? 'Set `far` above `near`. For a skydome, use `Infinity` — that is the infinite-far projection, and the wrong horizon is almost always a finite far plane.'
        : `Give ${field} a positive value.`,
      detail: { kind: 'numeric', field, value, min: 0 },
    });
  }
}
