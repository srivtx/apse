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
