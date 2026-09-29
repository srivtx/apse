/**
 * Math module tests.
 *
 * The bar these hold to is round-trip and invariant, not "matches a literal
 * from a blog post": an inverse that round-trips, a projection that puts the
 * near plane at depth 0, a look-at that puts the eye at the origin. A test
 * that hard-codes 16 floats only proves the function has not changed since
 * someone typed them in; these prove the transform is actually correct.
 */

import { describe, expect, test } from 'bun:test';
import { frustum, mat4, quat, sphere, vec3 } from '../src/math/index.ts';

const EPS = 1e-5;

function expectClose(actual: number, expected: number, message = ''): void {
  if (!(Math.abs(actual - expected) <= EPS)) {
    throw new Error(`${message} expected ${expected}, got ${actual}`.trim());
  }
}

function expectVec3Close(actual: Float32Array, expected: readonly [number, number, number], message = ''): void {
  for (let i = 0; i < 3; i++) {
    expectClose(actual[i], expected[i], `${message} component ${i}:`);
  }
}

function expectMat4Close(actual: Float32Array, expected: Float32Array, message = ''): void {
  for (let i = 0; i < 16; i++) {
    expectClose(actual[i], expected[i], `${message} element ${i}:`);
  }
}

function expectIdentity(actual: Float32Array, message = ''): void {
  expectMat4Close(actual, mat4.identity(mat4.create()), message);
}

const DEG = Math.PI / 180;

describe('mat4.invert', () => {
  test('round-trips to identity', () => {
    const m = mat4.mul(mat4.create(), mat4.fromTranslation(mat4.create(), 3, -4, 5), mat4.fromRotationY(mat4.create(), 0.7));
    mat4.scale(m, 2, 3, 4);

    const out = mat4.create();
    mat4.mul(out, m, mat4.invert(out, m));

    expectIdentity(out, 'm * inverse(m):');
  });

  test('round-trips a general view-projection', () => {
    const proj = mat4.perspective(mat4.create(), 60 * DEG, 16 / 9, 0.1, 500);
    const view = mat4.lookAt(mat4.create(), vec3.create(4, 6, 9), vec3.create(0, 1, 0), vec3.create(0, 1, 0));
    const viewProj = mat4.mul(mat4.create(), proj, view);

    const out = mat4.create();
    mat4.mul(out, viewProj, mat4.invert(out, viewProj));

    expectIdentity(out, 'viewProj * inverse(viewProj):');
  });

  test('inverts the identity', () => {
    const out = mat4.create();
    mat4.invert(out, mat4.identity(mat4.create()));
    expectIdentity(out);
  });

  test('rejects a singular matrix instead of writing NaN', () => {
    const singular = mat4.fromScale(mat4.create(), 1, 0, 1);
    expect(() => mat4.invert(mat4.create(), singular)).toThrow(/not invertible/);
  });

  test('agrees with transpose twice', () => {
    const m = mat4.mul(mat4.create(), mat4.fromRotationZ(mat4.create(), 0.3), mat4.fromTranslation(mat4.create(), 1, 2, 3));
    const out = mat4.create();
    mat4.transpose(out, m);
    mat4.transpose(out, out);
    expectMat4Close(out, m, 'transposed twice:');
  });

  test('determinant of a scale is the product of the axes', () => {
    expectClose(mat4.determinant(mat4.fromScale(mat4.create(), 2, 3, 4)), 24, 'det(scale):');
    expectClose(mat4.determinant(mat4.identity(mat4.create())), 1, 'det(identity):');
    // A rotation is length-preserving, so it is always volume-preserving.
    expectClose(mat4.determinant(mat4.fromRotationX(mat4.create(), 1.1)), 1, 'det(rotation):');
  });
});

describe('mat4.mul', () => {
  test('applies the right operand first', () => {
    const translate = mat4.fromTranslation(mat4.create(), 1, 0, 0);
    const scale = mat4.fromScale(mat4.create(), 2, 2, 2);

    const out = mat4.create();
    mat4.mul(out, translate, scale);

    // (1, 0, 0) must scale to (2, 0, 0) and then translate to (3, 0, 0).
    const p = mat4.transformPoint(vec3.create(), vec3.create(1, 0, 0), out);
    expectVec3Close(p, [3, 0, 0], 'translate * scale applied to a point:');
  });

  test('is the identity when multiplied by the identity', () => {
    const m = mat4.mul(mat4.create(), mat4.fromRotationY(mat4.create(), 0.9), mat4.fromTranslation(mat4.create(), 7, -2, 3));
    const out = mat4.create();
    mat4.mul(out, m, mat4.identity(mat4.create()));
    expectMat4Close(out, m, 'right identity:');

    mat4.mul(out, mat4.identity(mat4.create()), m);
    expectMat4Close(out, m, 'left identity:');
  });

  test('agrees with the post-multiplying helpers', () => {
    const angle = 0.42;
    const direct = mat4.mul(mat4.create(), mat4.fromTranslation(mat4.create(), 5, 0, 0), mat4.fromRotationZ(mat4.create(), angle));

    const inPlace = mat4.fromTranslation(mat4.create(), 5, 0, 0);
    mat4.rotateZ(inPlace, angle);

    expectMat4Close(inPlace, direct, 'translate * rotateZ:');
  });
});

describe('mat4.lookAt', () => {
  test('puts the eye at the origin of view space', () => {
    const eye = vec3.create(4, 6, 9);
    const view = mat4.lookAt(mat4.create(), eye, vec3.create(0, 1, 0), vec3.create(0, 1, 0));

    expectVec3Close(mat4.transformPoint(vec3.create(), eye, view), [0, 0, 0], 'eye in view space:');
  });

  test('puts the target on the negative Z axis', () => {
    const eye = vec3.create(0, 0, 5);
    const view = mat4.lookAt(mat4.create(), eye, vec3.create(0, 0, 0), vec3.create(0, 1, 0));

    // The camera looks down -Z, so a target in front of it has negative view Z.
    expectVec3Close(mat4.transformPoint(vec3.create(), vec3.create(0, 0, 0), view), [0, 0, -5], 'target in view space:');
  });

  test('stores the translation in the last column', () => {
    const eye = vec3.create(0, 0, 5);
    const view = mat4.lookAt(mat4.create(), eye, vec3.create(0, 0, 0), vec3.create(0, 1, 0));

    // With the world axes aligned to the view axes, the translation column is
    // -eye: the view matrix undoes the camera's offset.
    expectClose(view[12], 0, 'view[12]:');
    expectClose(view[13], 0, 'view[13]:');
    expectClose(view[14], -5, 'view[14]:');
    expectClose(view[15], 1, 'view[15]:');
  });

  test('keeps up pointing up when the camera is level', () => {
    // Level means the view direction is horizontal, so world up is parallel to
    // the camera's up axis. A tilted camera would not preserve it.
    const view = mat4.lookAt(mat4.create(), vec3.create(0, 2, 5), vec3.create(0, 2, 0), vec3.create(0, 1, 0));
    const up = mat4.transformDirection(vec3.create(), vec3.create(0, 1, 0), view);
    expectVec3Close(up, [0, 1, 0], 'world up in view space:');
  });

  test('tilts correctly when the camera looks downward', () => {
    const view = mat4.lookAt(mat4.create(), vec3.create(0, 2, 5), vec3.create(0, 0, 0), vec3.create(0, 1, 0));

    // Whatever the world direction of the view ray, it is -Z in view space.
    // transformDirection applies scale but not normalisation, so the input has
    // to be a unit vector for the output to be one.
    const ray = vec3.normalize(vec3.create(), vec3.create(0, -2, -5));
    const forward = mat4.transformDirection(vec3.create(), ray, view);
    expectVec3Close(forward, [0, 0, -1], 'forward in view space:');

    // A camera tilted down must swing world up *backwards*, into +Z. Getting
    // this sign wrong is the classic lookAt bug and shows up as an upside-down
    // or mirrored world.
    const d = Math.sqrt(29);
    const up = mat4.transformDirection(vec3.create(), vec3.create(0, 1, 0), view);
    expectVec3Close(up, [0, 5 / d, 2 / d], 'world up in view space:');
  });

  test('is the inverse of the camera transform', () => {
    const eye = vec3.create(-2, 3, 7);
    const view = mat4.lookAt(mat4.create(), eye, vec3.create(0, 0, 0), vec3.create(0, 1, 0));
    const camera = mat4.invert(mat4.create(), view);

    const out = mat4.create();
    mat4.mul(out, view, camera);
    expectIdentity(out, 'lookAt * inverse(lookAt):');
  });
});

describe('mat4.perspective', () => {
  test('matches known values', () => {
    // 90 degrees vertical FOV, square aspect, near 1, far 11.
    const m = mat4.perspective(mat4.create(), 90 * DEG, 1, 1, 11);

    expectClose(m[0], 1, 'm[0] = 1/tan(fov/2) / aspect:');
    expectClose(m[5], 1, 'm[5] = 1/tan(fov/2):');
    expectClose(m[10], -1.1, 'm[10] = far / (near - far):');
    expectClose(m[11], -1, 'm[11] = -1, the perspective divide:');
    expectClose(m[14], -1.1, 'm[14] = far * near / (near - far):');
    expectClose(m[15], 0, 'm[15] must be 0, not 1:');
    expectClose(m[3], 0, 'm[3]:');
    expectClose(m[7], 0, 'm[7]:');
  });

  test('puts the near plane at depth 0 and the far plane at depth 1', () => {
    const m = mat4.perspective(mat4.create(), 60 * DEG, 16 / 9, 0.5, 250);

    const near = mat4.transformPoint(vec3.create(), vec3.create(0, 0, -0.5), m);
    const far = mat4.transformPoint(vec3.create(), vec3.create(0, 0, -250), m);
    expectClose(near[2], 0, 'near plane depth:');
    expectClose(far[2], 1, 'far plane depth:');
  });

  test('narrows the X scale as the aspect ratio widens', () => {
    const square = mat4.perspective(mat4.create(), 90 * DEG, 1, 1, 100);
    const wide = mat4.perspective(mat4.create(), 90 * DEG, 2, 1, 100);
    expectClose(wide[0], square[0] * 0.5, 'wide[0]:');
    expectClose(wide[5], square[5], 'Y is independent of aspect:');
  });

  test('maps the frustum edge to NDC x = 1', () => {
    const m = mat4.perspective(mat4.create(), 90 * DEG, 1, 1, 100);
    const corner = mat4.transformPoint(vec3.create(), vec3.create(1, 0, -1), m);
    expectClose(corner[0], 1, 'right frustum edge:');
  });

  test('supports an infinite far plane', () => {
    const m = mat4.perspective(mat4.create(), 90 * DEG, 1, 1, Infinity);
    expectClose(m[10], -1, 'm[10] with infinite far:');
    expectClose(m[14], -1, 'm[14] with infinite far:');

    const p = mat4.transformPoint(vec3.create(), vec3.create(0, 0, -1e6), m);
    expect(Number.isFinite(p[2])).toBe(true);
    expect(p[2]).toBeGreaterThan(0);
    expect(p[2]).toBeLessThan(1);
  });
});

describe('mat4.orthographic', () => {
  test('maps the box corners onto the NDC cube', () => {
    const m = mat4.orthographic(mat4.create(), -2, 2, -1, 1, 0.5, 40);

    const nearLeft = mat4.transformPoint(vec3.create(), vec3.create(-2, 0, -0.5), m);
    const farRight = mat4.transformPoint(vec3.create(), vec3.create(2, 0, -40), m);
    expectVec3Close(nearLeft, [-1, 0, 0], 'near left corner:');
    expectVec3Close(farRight, [1, 0, 1], 'far right corner:');
  });

  test('has no perspective divide', () => {
    const m = mat4.orthographic(mat4.create(), -1, 1, -1, 1, 1, 10);
    expectClose(m[11], 0, 'm[11]:');
    expectClose(m[15], 1, 'm[15]:');
  });
});

describe('mat4.transformDirection', () => {
  test('ignores translation and keeps length under a rigid transform', () => {
    const m = mat4.mul(
      mat4.create(),
      mat4.fromTranslation(mat4.create(), 100, 100, 100),
      mat4.fromRotationY(mat4.create(), 0.9),
    );
    const d = mat4.transformDirection(vec3.create(), vec3.create(1, 0, 0), m);

    expectClose(vec3.length(d), 1, 'rotated direction length:');
    expectVec3Close(d, [Math.cos(0.9), 0, -Math.sin(0.9)], 'rotated direction:');
  });
});

describe('quat.slerp', () => {
  const a = quat.setAxisAngle(quat.create(), vec3.create(0, 1, 0), 0);
  const b = quat.setAxisAngle(quat.create(), vec3.create(0, 1, 0), 90 * DEG);

  test('reproduces the endpoints', () => {
    const out = quat.create();
    quat.slerp(out, a, b, 0);
    for (let i = 0; i < 4; i++) expectClose(out[i], a[i], `endpoint a[${i}]:`);
    quat.slerp(out, a, b, 1);
    for (let i = 0; i < 4; i++) expectClose(out[i], b[i], `endpoint b[${i}]:`);
  });

  test('stays on the unit sphere at the midpoint', () => {
    const out = quat.create();
    quat.slerp(out, a, b, 0.5);
    expectClose(quat.length(out), 1, 'midpoint length:');

    // Halfway from 0 to 90 degrees about Y is 45 degrees.
    const expected = quat.setAxisAngle(quat.create(), vec3.create(0, 1, 0), 45 * DEG);
    for (let i = 0; i < 4; i++) {
      // A quaternion and its negation are the same rotation, so compare the
      // rotation, not the signs.
      const sign = quat.dot(out, expected) < 0 ? -1 : 1;
      expectClose(out[i] * sign, expected[i], `midpoint[${i}]:`);
    }
  });

  test('takes the short path when the quaternions are antipodal', () => {
    const flipped = quat.create();
    quat.negate(flipped, b);

    const out = quat.create();
    quat.slerp(out, a, flipped, 0.5);

    expectClose(quat.length(out), 1, 'antipodal midpoint length:');
    for (const component of out) expect(Number.isFinite(component)).toBe(true);

    const expected = quat.setAxisAngle(quat.create(), vec3.create(0, 1, 0), 45 * DEG);
    expect(quat.dot(out, expected)).toBeGreaterThan(0.999);
  });

  test('does not divide by zero for identical quaternions', () => {
    const out = quat.create();
    quat.slerp(out, b, b, 0.5);
    for (let i = 0; i < 4; i++) {
      expect(Number.isFinite(out[i])).toBe(true);
      expectClose(out[i], b[i], `identical[${i}]:`);
    }
  });

  test('slerpQuat clamps t and slerp does not', () => {
    const clamped = quat.create();
    const extrapolated = quat.create();
    quat.slerpQuat(clamped, a, b, 5);
    quat.slerp(extrapolated, a, b, 5);
    for (let i = 0; i < 4; i++) expectClose(clamped[i], b[i], `clamped[${i}]:`);
    expect(quat.dot(extrapolated, b)).toBeLessThan(0.999);
  });
});

describe('quat and mat4', () => {
  test('toMat4 agrees with mat4.fromQuat', () => {
    const cases: Float32Array[] = [
      quat.identity(quat.create()),
      quat.setAxisAngle(quat.create(), vec3.create(0, 1, 0), 0.9),
      quat.setAxisAngle(quat.create(), vec3.create(0.577, 0.577, 0.577), 2.1),
      quat.fromEulerXYZ(quat.create(), 0.3, -1.2, 2.7),
    ];

    for (const q of cases) {
      const fromMat4 = mat4.create();
      const fromQuatModule = mat4.create();
      mat4.fromQuat(fromMat4, q);
      quat.toMat4(fromQuatModule, q);
      expectMat4Close(fromQuatModule, fromMat4, 'fromQuat vs toMat4:');
    }
  });

  test('rotateVec3 agrees with the matrix', () => {
    const q = quat.setAxisAngle(quat.create(), vec3.create(0, 0, 1), 90 * DEG);
    const v = vec3.create(1, 0, 0);

    const rotated = quat.rotateVec3(vec3.create(), v, q);
    expectVec3Close(rotated, [0, 1, 0], 'quat rotation:');

    const byMatrix = mat4.transformPoint(vec3.create(), v, mat4.fromQuat(mat4.create(), q));
    expectVec3Close(byMatrix, [0, 1, 0], 'matrix rotation:');
  });

  test('produces a pure rotation with no scale', () => {
    const m = mat4.fromQuat(mat4.create(), quat.setAxisAngle(quat.create(), vec3.create(1, 1, 0), 1.3));
    expectClose(vec3.length(vec3.create(m[0], m[1], m[2])), 1, 'basis column 0 length:');
    expectClose(vec3.length(vec3.create(m[4], m[5], m[6])), 1, 'basis column 1 length:');
    expectClose(vec3.length(vec3.create(m[8], m[9], m[10])), 1, 'basis column 2 length:');
    expectClose(m[15], 1, 'homogeneous element:');
  });

  test('fromEulerXYZ applies X, then Y, then Z', () => {
    const q = quat.fromEulerXYZ(quat.create(), 30 * DEG, 0, 0);
    const byAngle = quat.setAxisAngle(quat.create(), vec3.create(1, 0, 0), 30 * DEG);
    for (let i = 0; i < 4; i++) expectClose(q[i], byAngle[i], `euler x [${i}]:`);
  });

  test('conjugate undoes the rotation of a unit quaternion', () => {
    const q = quat.normalize(quat.create(), quat.setAxisAngle(quat.create(), vec3.create(0.2, 0.9, -0.3), 1.7));
    expectClose(quat.length(q), 1, '|q|:');

    // The identity that matters is on the rotation, not on the dot product:
    // q and its conjugate are inverses, so applying both returns the vector.
    const v = vec3.create(1, -2, 3);
    const round = quat.rotateVec3(vec3.create(), quat.rotateVec3(vec3.create(), v, q), quat.conjugate(quat.create(), q));
    expectVec3Close(round, [1, -2, 3], 'q * q⁻¹ applied to a vector:');
  });

  test('negate preserves the rotation', () => {
    const q = quat.setAxisAngle(quat.create(), vec3.create(0, 1, 0), 0.6);
    const flipped = quat.negate(quat.create(), q);

    const a = mat4.fromQuat(mat4.create(), q);
    const b = mat4.fromQuat(mat4.create(), flipped);
    expectMat4Close(b, a, 'matrix of -q:');
  });

  test('invert handles a non-unit quaternion', () => {
    const q = quat.create(0, 0, 0, 4);
    const inverse = quat.invert(quat.create(), q);
    expectClose(inverse[3], 0.25, 'inverse of a scaled identity:');
  });
});

describe('sphere', () => {
  test('scales the radius by the largest axis, not naively', () => {
    const s = sphere.create(vec3.create(1, 0, 0), 2);
    const m = mat4.fromScale(mat4.create(), 1, 5, 1);
    const out = sphere.create();
    sphere.transform(out, s, m);

    expectClose(out.radius, 10, 'radius scaled by the max axis:');
    expectVec3Close(out.center, [1, 0, 0], 'center:');
  });

  test('transforms the centre through rotation and translation', () => {
    const s = sphere.create(vec3.create(1, 0, 0), 1);
    const m = mat4.mul(mat4.create(), mat4.fromTranslation(mat4.create(), 0, 3, 0), mat4.fromRotationZ(mat4.create(), 90 * DEG));
    const out = sphere.create();
    sphere.transform(out, s, m);

    expectVec3Close(out.center, [0, 4, 0], 'transformed center:');
    expectClose(out.radius, 1, 'radius under a rigid transform:');
  });

  test('union keeps a sphere that already contains the other', () => {
    const big = sphere.create(vec3.create(0, 0, 0), 10);
    const small = sphere.create(vec3.create(1, 0, 0), 1);
    const out = sphere.create();
    sphere.union(out, big, small);

    expectVec3Close(out.center, [0, 0, 0], 'center:');
    expectClose(out.radius, 10, 'radius:');
  });

  test('union encloses both spheres', () => {
    const a = sphere.create(vec3.create(-1, 0, 0), 2);
    const b = sphere.create(vec3.create(3, 0, 0), 1);
    const out = sphere.create();
    sphere.union(out, a, b);

    expect(sphere.containsPoint(out, vec3.create(-3, 0, 0))).toBe(true);
    expect(sphere.containsPoint(out, vec3.create(4, 0, 0))).toBe(true);
    // The minimal enclosing sphere of two overlapping ones: r = (d + ra + rb) / 2.
    expectClose(out.radius, 3.5, 'radius:');
  });

  test('containsPoint includes the surface', () => {
    const s = sphere.create(vec3.create(0, 0, 0), 1);
    expect(sphere.containsPoint(s, vec3.create(0, 0, 0))).toBe(true);
    expect(sphere.containsPoint(s, vec3.create(1, 0, 0))).toBe(true);
    expect(sphere.containsPoint(s, vec3.create(1.01, 0, 0))).toBe(false);
  });
});

describe('frustum', () => {
  const proj = mat4.perspective(mat4.create(), 60 * DEG, 1, 1, 100);
  const view = mat4.lookAt(mat4.create(), vec3.create(0, 0, 10), vec3.create(0, 0, 0), vec3.create(0, 1, 0));
  const viewProj = mat4.mul(mat4.create(), proj, view);
  const planes = frustum.setFromViewProjection(frustum.create(), viewProj);

  test('writes six unit-length inward planes', () => {
    expect(planes.length).toBe(24);
    for (let i = 0; i < 6; i++) {
      const o = i * 4;
      const len = Math.sqrt(planes[o] * planes[o] + planes[o + 1] * planes[o + 1] + planes[o + 2] * planes[o + 2]);
      expectClose(len, 1, `plane ${i} normal length:`);
    }
  });

  test('contains a sphere in front of the camera', () => {
    expect(frustum.containsSphere(planes, sphere.create(vec3.create(0, 0, 0), 1))).toBe(true);
    expect(frustum.containsSphere(planes, sphere.create(vec3.create(0, 0, -5), 0.5))).toBe(true);
  });

  test('rejects a sphere behind the camera', () => {
    expect(frustum.containsSphere(planes, sphere.create(vec3.create(0, 0, 20), 1))).toBe(false);
  });

  test('rejects a sphere outside the horizontal field of view', () => {
    // The 60-degree vertical FOV makes the horizontal half-angle about 26.6
    // degrees at aspect 1, so x = 20 at z = 0 is far outside it.
    expect(frustum.containsSphere(planes, sphere.create(vec3.create(20, 0, 0), 1))).toBe(false);
  });

  test('rejects a sphere beyond the far plane', () => {
    expect(frustum.containsSphere(planes, sphere.create(vec3.create(0, 0, -200), 1))).toBe(false);
  });

  test('respects the radius at the frustum edge', () => {
    // Straddling the near plane: the centre is outside, a large sphere is not.
    const centre = vec3.create(0, 0, 11);
    expect(frustum.containsSphere(planes, sphere.create(centre, 0.5))).toBe(false);
    expect(frustum.containsSphere(planes, sphere.create(centre, 2))).toBe(true);
  });

  test('containsAabb agrees with containsSphere on the same bounds', () => {
    const box = sphere.aabbSet(sphere.aabbCreate(), -1, -1, -1, 1, 1, 1);
    const enclosing = sphere.create(vec3.create(0, 0, 0), Math.sqrt(3));
    expect(frustum.containsAabb(planes, box)).toBe(true);
    expect(frustum.containsSphere(planes, enclosing)).toBe(true);

    const far = sphere.aabbSet(sphere.aabbCreate(), -1, -1, -100, 1, 1, -99);
    expect(frustum.containsAabb(planes, far)).toBe(false);
  });
});

describe('assertions', () => {
  test('accept correctly sized buffers', () => {
    expect(() => vec3.assertVec3(vec3.create(), 'position')).not.toThrow();
    expect(() => mat4.assertMat4(mat4.create(), 'model')).not.toThrow();
  });

  test('reject a wrong-length buffer instead of letting NaN through', () => {
    expect(() => vec3.assertVec3(new Float32Array(2), 'position')).toThrow(/3 components/);
    expect(() => mat4.assertMat4(new Float32Array(4), 'model')).toThrow(/16 components/);
  });

  test('reject a plain array', () => {
    expect(() => vec3.assertVec3([0, 0, 0] as unknown as Float32Array, 'position')).toThrow();
  });
});

// ---------------------------------------------------------------------------
// Regression tests for bugs these functions used to have.
//
// Each one is here because the property is a *contract*, not a literal: the
// failing case is a class of input, and a test that only pinned the old output
// would pass again the moment someone "simplified" the arithmetic back.
// ---------------------------------------------------------------------------

/** Deterministic LCG, so a failure is reproducible rather than one-in-a-thousand. */
function rng(seed: number): () => number {
  let s = seed;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

/** A well-conditioned affine transform: a rotation, a scale in [0.5, 3], a translation. */
function wellConditioned(r: () => number): Float32Array {
  const q = quat.setAxisAngle(
    quat.create(),
    vec3.normalize(vec3.create(), vec3.create(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1)),
    r() * 6,
  );
  const s = 0.5 + r() * 2.5;
  return mat4.mul(
    mat4.create(),
    mat4.fromTranslation(mat4.create(), (r() * 2 - 1) * 50, (r() * 2 - 1) * 50, (r() * 2 - 1) * 50),
    mat4.mul(
      mat4.create(),
      quat.toMat4(mat4.create(), q),
      mat4.fromScale(mat4.create(), s, 0.5 + r() * 2.5, 0.5 + r() * 2.5),
    ),
  );
}

/** The shipped `normalMatrixOf` from `src/render/renderer.ts`, verbatim. The oracle. */
const IDENTITY_NORMAL = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
function referenceNormalMatrix(out: Float32Array, m: Float32Array): Float32Array {
  const a = m[0]!, b = m[1]!, c = m[2]!;
  const d = m[4]!, e = m[5]!, f = m[6]!;
  const g = m[8]!, h = m[9]!, i = m[10]!;
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (det === 0 || !Number.isFinite(det)) {
    for (let k = 0; k < 12; k++) out[k] = IDENTITY_NORMAL[k];
    return out;
  }
  const s = 1 / det;
  out[0] = A * s; out[1] = B * s; out[2] = C * s;
  out[4] = (c * h - b * i) * s; out[5] = (a * i - c * g) * s; out[6] = (b * g - a * h) * s;
  out[8] = (b * f - c * e) * s; out[9] = (c * d - a * f) * s; out[10] = (a * e - b * d) * s;
  out[3] = 0; out[7] = 0; out[11] = 0;
  return out;
}

describe('mat4.scale post-multiplies (regression)', () => {
  test('scales the basis columns, not the rows', () => {
    // A rotation makes the two sides distinguishable. `Rz(0.5)` has
    // column 0 = (cos, sin, 0) and column 1 = (-sin, cos, 0). Under `out * S`
    // each column is scaled by its own factor, so column 1 becomes
    // 3*(-sin, cos, 0). Under `S * out` the *rows* are scaled and column 1
    // would come out as (2*(-sin), 3*cos, 0) — a matrix that is no longer a
    // rotation, and whose determinant is not x*y*z.
    const m = mat4.fromRotationZ(mat4.create(), 0.5);
    const c = Math.cos(0.5);
    const s = Math.sin(0.5);
    mat4.scale(m, 2, 3, 4);
    expectClose(m[0], 2 * c, 'column 0, x:');
    expectClose(m[1], 2 * s, 'column 0, y:');
    expectClose(m[2], 0, 'column 0, z:');
    expectClose(m[4], 3 * -s, 'column 1, x:');
    expectClose(m[5], 3 * c, 'column 1, y:');
    expectClose(m[8], 0, 'column 2, x:');
    expectClose(m[9], 0, 'column 2, y:');
    expectClose(m[10], 4, 'column 2, z:');
    // The distinguishing invariant: a scaled rotation has determinant x*y*z.
    expectClose(mat4.determinant(m), 2 * 3 * 4, 'determinant:');
  });

  test('leaves the translation where it was', () => {
    // The regression in one line: the row-scaling form also multiplied
    // m[12..14], walking a node at (5, 6, 7) out to (10, 12, 14).
    const m = mat4.fromTranslation(mat4.create(), 5, 6, 7);
    mat4.scale(m, 2, 3, 4);
    expectClose(m[12], 5, 'x:');
    expectClose(m[13], 6, 'y:');
    expectClose(m[14], 7, 'z:');
  });

  test('agrees with an explicit post-multiply by a scale matrix', () => {
    const r = rng(20260929);
    for (let t = 0; t < 500; t++) {
      const m = wellConditioned(r);
      const x = r() * 4 - 2;
      const y = r() * 4 - 2;
      const z = r() * 4 - 2;
      const byHand = mat4.mul(mat4.create(), m, mat4.fromScale(mat4.create(), x, y, z));
      mat4.scale(m, x, y, z);
      expectMat4Close(m, byHand, `post-multiply by (${x}, ${y}, ${z}):`);
    }
  });

  test('leaves the homogeneous row alone', () => {
    const m = mat4.fromTranslation(mat4.create(), 1, 2, 3);
    mat4.scale(m, 5, 6, 7);
    expect(m[3]).toBe(0);
    expect(m[7]).toBe(0);
    expect(m[11]).toBe(0);
    expect(m[15]).toBe(1);
  });
});

describe('quat.fromEulerXYZ composition (regression)', () => {
  test('is exactly qz * qy * qx for random triples', () => {
    // The shipped form had the sign of the `cx*sy*sz` term in x and the
    // `sx*cy*sz` term in y flipped. It still produced a *unit* quaternion, and
    // it still produced exactly the right answer whenever y or z was zero, so
    // a single-axis test cannot see it — only the composition can.
    const r = rng(4242);
    for (let t = 0; t < 2000; t++) {
      const x = r() * 12 - 6;
      const y = r() * 12 - 6;
      const z = r() * 12 - 6;
      const got = quat.fromEulerXYZ(quat.create(), x, y, z);
      const want = quat.mul(
        quat.create(),
        quat.mul(
          quat.create(),
          quat.setAxisAngle(quat.create(), vec3.create(0, 0, 1), z),
          quat.setAxisAngle(quat.create(), vec3.create(0, 1, 0), y),
        ),
        quat.setAxisAngle(quat.create(), vec3.create(1, 0, 0), x),
      );
      for (let i = 0; i < 4; i++) {
        expectClose(got[i], want[i], `euler(${x}, ${y}, ${z}) component ${i}:`);
      }
    }
  });

  test('is the matrix product Rz * Ry * Rx', () => {
    const r = rng(31337);
    for (let t = 0; t < 500; t++) {
      const x = r() * 12 - 6;
      const y = r() * 12 - 6;
      const z = r() * 12 - 6;
      const got = quat.toMat4(mat4.create(), quat.fromEulerXYZ(quat.create(), x, y, z));
      const want = mat4.mul(
        mat4.create(),
        mat4.mul(mat4.create(), mat4.fromRotationZ(mat4.create(), z), mat4.fromRotationY(mat4.create(), y)),
        mat4.fromRotationX(mat4.create(), x),
      );
      expectMat4Close(got, want, `euler(${x}, ${y}, ${z}):`);
    }
  });

  test('is exact on each single axis, not just X', () => {
    // The pre-existing test covered X only. Y and Z are what the sign error
    // was hiding behind: with the other two angles zero the two flipped terms
    // both vanish and the answer is right either way.
    const angle = 30 * DEG;
    const axes: readonly (readonly [number, number, number, Float32Array])[] = [
      [angle, 0, 0, vec3.create(1, 0, 0)],
      [0, angle, 0, vec3.create(0, 1, 0)],
      [0, 0, angle, vec3.create(0, 0, 1)],
    ];
    for (const [x, y, z, axis] of axes) {
      const byAxis = quat.setAxisAngle(quat.create(), axis, angle);
      const got = quat.fromEulerXYZ(quat.create(), x, y, z);
      for (let k = 0; k < 4; k++) {
        expectClose(got[k], byAxis[k], `axis (${x}, ${y}, ${z}) component ${k}:`);
      }
    }
  });

  test('stays unit length', () => {
    const r = rng(777);
    for (let t = 0; t < 1000; t++) {
      const q = quat.fromEulerXYZ(quat.create(), r() * 12 - 6, r() * 12 - 6, r() * 12 - 6);
      expectClose(quat.length(q), 1, '|q|:');
    }
  });
});

describe('vec3.min / vec3.max NaN handling (regression)', () => {
  test('drops a NaN in the first operand and propagates one in the second', () => {
    // The old doc claimed "NaN in either input propagates, per IEEE-754",
    // which was true of neither the code nor the standard. The comparison is
    // `a[i] < b[i] ? a[i] : b[i]`, so a NaN in `a` loses the comparison and
    // falls through to `b`, while a NaN in `b` wins the fallback. Pinned here
    // because a caller genuinely cannot treat the two as interchangeable.
    const nan = vec3.create(Number.NaN, Number.NaN, Number.NaN);
    const plain = vec3.create(1, 2, 3);
    expectVec3Close(vec3.min(vec3.create(), nan, plain), [1, 2, 3], 'min, NaN in a:');
    expectVec3Close(vec3.max(vec3.create(), nan, plain), [1, 2, 3], 'max, NaN in a:');
    // `expectVec3Close` cannot express this — `NaN <= EPS` is false — so the
    // propagating case is asserted as a NaN check.
    for (const [label, out] of [
      ['min, NaN in b', vec3.min(vec3.create(), plain, nan)],
      ['max, NaN in b', vec3.max(vec3.create(), plain, nan)],
    ] as const) {
      for (let i = 0; i < 3; i++) {
        if (!Number.isNaN(out[i]!)) throw new Error(`${label} component ${i}: expected NaN, got ${out[i]}`);
      }
    }
  });

  test('still orders ordinary values', () => {
    expectVec3Close(vec3.min(vec3.create(), vec3.create(3, -1, 2), vec3.create(1, 4, 0)), [1, -1, 0]);
    expectVec3Close(vec3.max(vec3.create(), vec3.create(3, -1, 2), vec3.create(1, 4, 0)), [3, 4, 2]);
    expectVec3Close(vec3.min(vec3.create(), vec3.create(2, 2, 2), vec3.create(1, 1, 1)), [1, 1, 1]);
    expectVec3Close(vec3.max(vec3.create(), vec3.create(2, 2, 2), vec3.create(1, 1, 1)), [2, 2, 2]);
  });
});

describe('sphere.transform radius (regression)', () => {
  test('takes one square root of the longest squared column, bit for bit', () => {
    // `sqrt` is monotonic and correctly rounded, so `max(sqrt(a), sqrt(b),
    // sqrt(c))` and `sqrt(max(a, b, c))` are the same double. `Object.is` is
    // used deliberately: this claims exactness, not closeness.
    const r = rng(5150);
    const slots = [0, 1, 2, 4, 5, 6, 8, 9, 10] as const;
    for (let t = 0; t < 20000; t++) {
      const m = mat4.create();
      for (const s of slots) m[s] = r() * 200 - 100;
      const threeSq = Math.max(
        Math.sqrt(m[0]! * m[0]! + m[1]! * m[1]! + m[2]! * m[2]!),
        Math.sqrt(m[4]! * m[4]! + m[5]! * m[5]! + m[6]! * m[6]!),
        Math.sqrt(m[8]! * m[8]! + m[9]! * m[9]! + m[10]! * m[10]!),
      );
      const q0 = m[0]! * m[0]! + m[1]! * m[1]! + m[2]! * m[2]!;
      const q1 = m[4]! * m[4]! + m[5]! * m[5]! + m[6]! * m[6]!;
      const q2 = m[8]! * m[8]! + m[9]! * m[9]! + m[10]! * m[10]!;
      const oneSq = Math.sqrt(q0 > q1 ? (q0 > q2 ? q0 : q2) : q1 > q2 ? q1 : q2);
      expect(Object.is(threeSq, oneSq)).toBe(true);
    }
  });

  test('stays conservative under negative and zero scale', () => {
    const unit = sphere.create(vec3.create(0, 0, 0), 1);
    const cases: readonly (readonly [number, number, number])[] = [
      [2, 1, 1],
      [-3, 1, 1],
      [1, -1, -1],
      [0, 1, 1],
    ];
    for (const [x, y, z] of cases) {
      const out = sphere.transform(sphere.create(vec3.create(), 0), unit, mat4.fromScale(mat4.create(), x, y, z));
      expect(out.radius).toBeGreaterThanOrEqual(Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) - 1e-6);
    }
  });
});

describe('mat4.mulAffine', () => {
  test('matches mat4.mul exactly on affine operands', () => {
    // The precondition is what makes this safe, so the test states it: both
    // operands must carry the (0, 0, 0, 1) bottom row.
    const r = rng(9090);
    for (let t = 0; t < 1000; t++) {
      const a = wellConditioned(r);
      const b = wellConditioned(r);
      const general = mat4.mul(mat4.create(), a, b);
      const fast = mat4.mulAffine(mat4.create(), a, b);
      for (let i = 0; i < 16; i++) {
        if (general[i] !== fast[i]) {
          throw new Error(`element ${i}: mul gave ${general[i]}, mulAffine gave ${fast[i]}`);
        }
      }
    }
  });

  test('agrees with mat4.mul when an axis is scaled to exactly zero', () => {
    // A zero axis is the degenerate case that makes an "is it affine" guard
    // look like it needs a determinant test. It does not: the row is still
    // (0, 0, 0, 1), and `x * 0` is still `x * 0`.
    const r = rng(1212);
    for (let t = 0; t < 500; t++) {
      const a = wellConditioned(r);
      const b = wellConditioned(r);
      const axis = (r() * 3) | 0;
      mat4.scale(b, axis === 0 ? 0 : 1, axis === 1 ? 0 : 1, axis === 2 ? 0 : 1);
      const general = mat4.mul(mat4.create(), a, b);
      const fast = mat4.mulAffine(mat4.create(), a, b);
      for (let i = 0; i < 16; i++) {
        if (general[i] !== fast[i]) throw new Error(`element ${i}: ${general[i]} vs ${fast[i]}`);
      }
    }
  });

  test('writes the affine bottom row', () => {
    const r = rng(3434);
    const out = mat4.mulAffine(mat4.create(), wellConditioned(r), wellConditioned(r));
    expect(out[3]).toBe(0);
    expect(out[7]).toBe(0);
    expect(out[11]).toBe(0);
    expect(out[15]).toBe(1);
  });

  test('allows out to alias either operand', () => {
    const r = rng(5656);
    const a = wellConditioned(r);
    const b = wellConditioned(r);
    const want = mat4.mul(mat4.create(), a, b);
    const o1 = new Float32Array(a);
    mat4.mulAffine(o1, o1, b);
    const o2 = new Float32Array(b);
    mat4.mulAffine(o2, a, o2);
    for (let i = 0; i < 16; i++) {
      expect(o1[i]).toBe(want[i]);
      expect(o2[i]).toBe(want[i]);
    }
  });

  test('is not interchangeable with mul on a projection', () => {
    // Pins the precondition from the other side: a projective matrix is what
    // `mul` exists for, and the two must never be confused.
    const proj = mat4.perspective(mat4.create(), 60 * DEG, 1.5, 0.1, 100);
    const m = wellConditioned(rng(99));
    const general = mat4.mul(mat4.create(), proj, m);
    const fast = mat4.mulAffine(mat4.create(), proj, m);
    expect(general[11]).not.toBe(fast[11]);
  });
});

describe('mat4.normalMatrix', () => {
  const scratch = (): Float32Array => new Float32Array(12);

  test('agrees bit for bit with the shipped adjugate on random transforms', () => {
    // No tolerance. The arithmetic is character-for-character the same, so any
    // difference at all would be a bug, not a rounding question.
    const r = rng(606);
    for (let t = 0; t < 2000; t++) {
      const m = wellConditioned(r);
      const mine = mat4.normalMatrix(scratch(), m);
      const theirs = referenceNormalMatrix(scratch(), m);
      for (let i = 0; i < 12; i++) {
        if (mine[i] !== theirs[i]) {
          throw new Error(`element ${i}: got ${mine[i]}, reference ${theirs[i]}`);
        }
      }
    }
  });

  test('agrees bit for bit on uniform scale, non-uniform scale, shear and a perspective row', () => {
    const base = quat.toMat4(
      mat4.create(),
      quat.setAxisAngle(quat.create(), vec3.create(0.3, 0.5, 0.81), 1.1),
    );
    const cases: Float32Array[] = [];
    for (const s of [[1, 1, 1], [2, 2, 2], [-1, -1, -1], [1, 2, 3], [1e-3, 1e-3, 1e-3], [1e3, 1e3, 1e3]]) {
      cases.push(mat4.scale(new Float32Array(base), s[0]!, s[1]!, s[2]!));
    }
    const sheared = new Float32Array(base);
    sheared[1] = 0.25;
    sheared[9] = -0.5;
    cases.push(sheared);
    const projected = new Float32Array(base);
    projected[3] = 0.1;
    projected[7] = -0.2;
    projected[11] = 0.3;
    cases.push(projected);
    const zeroed = new Float32Array(projected);
    zeroed[3] = 0;
    zeroed[7] = 0;
    zeroed[11] = 0;
    cases.push(zeroed);
    for (const m of cases) {
      const mine = mat4.normalMatrix(scratch(), m);
      const theirs = referenceNormalMatrix(scratch(), m);
      for (let i = 0; i < 12; i++) expect(mine[i]).toBe(theirs[i]);
    }
  });

  test('ignores the perspective row entirely', () => {
    const base = quat.toMat4(
      mat4.create(),
      quat.setAxisAngle(quat.create(), vec3.create(0.3, 0.5, 0.81), 1.1),
    );
    const projected = new Float32Array(base);
    projected[3] = 0.1;
    projected[7] = -0.2;
    projected[11] = 0.3;
    projected[15] = 0.4;
    const a = mat4.normalMatrix(scratch(), base);
    const b = mat4.normalMatrix(scratch(), projected);
    for (let i = 0; i < 12; i++) expect(b[i]).toBe(a[i]);
  });

  test('writes the WGSL mat3x3 layout: three 16-byte columns, never a tight 3x3', () => {
    // The layout this function exists to get right. Column c starts at 4c and
    // occupies 4c..4c+2, with 4c+3 as vec3 padding.
    const n = mat4.normalMatrix(scratch(), mat4.fromScale(mat4.create(), 2, 4, 8));
    expectClose(n[0], 0.5, 'column 0, x:');
    expectClose(n[1], 0, 'column 0, y:');
    expectClose(n[2], 0, 'column 0, z:');
    expectClose(n[4], 0, 'column 1, x:');
    expectClose(n[5], 0.25, 'column 1, y:');
    expectClose(n[6], 0, 'column 1, z:');
    expectClose(n[8], 0, 'column 2, x:');
    expectClose(n[9], 0, 'column 2, y:');
    expectClose(n[10], 0.125, 'column 2, z:');
    for (const pad of [3, 7, 11]) expect(n[pad]).toBe(0);
  });

  test('leaves the vec3 padding zero on every path, forever', () => {
    // The contract that lets the fast path skip three stores: the padding is
    // never written with anything but zero, so these three slots can only ever
    // hold zero. A regression here is the shipped normal-matrix bug returning.
    const r = rng(808);
    const n = scratch();
    const cases: Float32Array[] = [
      mat4.identity(mat4.create()),
      mat4.fromScale(mat4.create(), 0, 1, 1),
      mat4.fromScale(mat4.create(), 0, 0, 0),
      wellConditioned(r),
      mat4.fromScale(mat4.create(), Number.NaN, 1, 1),
      mat4.fromScale(mat4.create(), Infinity, 1, 1),
    ];
    for (let t = 0; t < 500; t++) {
      for (const m of cases) {
        mat4.normalMatrix(n, m);
        for (const pad of [3, 7, 11]) {
          if (n[pad] !== 0) throw new Error(`padding slot ${pad} became ${n[pad]}`);
        }
      }
    }
  });

  test('writes identity for a singular or non-finite basis', () => {
    const cases: readonly (readonly [string, Float32Array])[] = [
      ['zero scale on X', mat4.fromScale(mat4.create(), 0, 1, 1)],
      ['zero scale on Y', mat4.fromScale(mat4.create(), 1, 0, 1)],
      ['zero scale on Z', mat4.fromScale(mat4.create(), 1, 1, 0)],
      ['all-zero basis', mat4.fromScale(mat4.create(), 0, 0, 0)],
      ['NaN in the basis', mat4.fromScale(mat4.create(), Number.NaN, 1, 1)],
      ['Infinity in the basis', mat4.fromScale(mat4.create(), Infinity, 1, 1)],
    ];
    for (const [label, m] of cases) {
      const n = mat4.normalMatrix(scratch(), m);
      for (let i = 0; i < 12; i++) expect(n[i], label).toBe(IDENTITY_NORMAL[i]);
    }
  });

  test('is the inverse transpose of the upper 3x3', () => {
    // An independent formulation — extract the 3x3, `mat4.invert` it,
    // transpose — agreeing to a stated tolerance. Both are exact in real
    // arithmetic, so the gap is pure float rounding: with no basis column
    // shorter than 0.5 the condition number stays under about 10, so each is
    // wrong by at most cond * eps, about 1.2e-6. 1e-5 leaves two orders of
    // margin over that, and is still far tighter than any normal can tell.
    const r = rng(9091);
    for (let t = 0; t < 500; t++) {
      const m = wellConditioned(r);
      const tight = mat4.identity(mat4.create());
      for (let c = 0; c < 3; c++) for (let row = 0; row < 3; row++) tight[c * 4 + row] = m[c * 4 + row]!;
      const inv = mat4.invert(mat4.create(), tight);
      const n = mat4.normalMatrix(scratch(), m);
      for (let c = 0; c < 3; c++) {
        for (let row = 0; row < 3; row++) {
          expectClose(n[c * 4 + row]!, inv[row * 4 + c]!, `element (row ${row}, col ${c}):`);
        }
      }
    }
  });

  test('keeps a normal perpendicular to the surface it came from', () => {
    // The property the whole thing exists for. Under a non-uniform scale the
    // model matrix alone does not preserve perpendicularity; the normal matrix
    // is what restores it. A wrong normal matrix can hide inside any single
    // matrix comparison and is unmissable here.
    //
    // The tangent is made perpendicular to the normal in *object* space first.
    // Projecting it out is not optional: a tangent that is only perpendicular
    // in world space is circular, and one that is not perpendicular in object
    // space has no reason to be perpendicular after the transform either.
    const r = rng(1919);
    let worst = 0;
    for (let t = 0; t < 500; t++) {
      const m = wellConditioned(r);
      const n = mat4.normalMatrix(scratch(), m);
      const nObj = vec3.normalize(vec3.create(), vec3.create(1, 1, 1));
      const raw = new Float32Array(3);
      for (let row = 0; row < 3; row++) {
        raw[row] = n[row]! * nObj[0]! + n[4 + row]! * nObj[1]! + n[8 + row]! * nObj[2]!;
      }
      const nWorld = vec3.normalize(vec3.create(), raw);
      for (const tangent of [[1, -1, 0], [1, 2, -3], [2, 1, 1]] as const) {
        const tv = vec3.create(tangent[0], tangent[1], tangent[2]);
        const along = vec3.dot(nObj, tv);
        const tObj = vec3.normalize(
          vec3.create(),
          vec3.set(vec3.create(), tv[0]! - nObj[0]! * along, tv[1]! - nObj[1]! * along, tv[2]! - nObj[2]! * along),
        );
        const w = mat4.transformDirection(vec3.create(), tObj, m);
        worst = Math.max(worst, Math.abs(vec3.dot(nWorld, w)));
      }
    }
    // One f32 rounding on each of two normalised 3-vectors, so a handful of
    // ulps. 1e-5 is the same bar as the rest of this file and three orders
    // below anything a normal can visibly care about.
    expect(worst).toBeLessThan(1e-5);
  });

  test('is the rotation itself for a rotation, and that over s squared for a uniform scale', () => {
    // The mathematical fact behind the fast path that was measured and
    // rejected. It is worth pinning: it is what makes the fast path *possible*
    // and it is why its precondition — exact orthonormality — cannot be tested
    // in floating point without a tolerance, and a tolerance there is a
    // silently-wrong-normal hazard.
    //
    // For a rotation R, R^-1 is R^T, so (R^-1)^T is R: the normal matrix is
    // the rotation *itself*, not its transpose. For a uniform scale s the
    // basis is sR, and (sR)^-T is R/s, which is the matrix itself over s^2.
    const q = quat.setAxisAngle(
      quat.create(),
      vec3.normalize(vec3.create(), vec3.create(0.3, 0.5, 0.81)),
      1.1,
    );
    const r = quat.toMat4(mat4.create(), q);
    const slots = [0, 1, 2, 4, 5, 6, 8, 9, 10] as const;
    const n = mat4.normalMatrix(scratch(), r);
    for (const i of slots) expectClose(n[i]!, r[i]!, `rotation, element ${i}:`);

    const s = 2.5;
    const scaled = mat4.scale(new Float32Array(r), s, s, s);
    const ns = mat4.normalMatrix(scratch(), scaled);
    for (const i of slots) {
      expectClose(ns[i]!, scaled[i]! / (s * s), `uniform scale, element ${i}:`);
    }
  });
});
