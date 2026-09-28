/**
 * Instancing tests.
 *
 * There is no GPU here, and for this feature that is not a compromise — the
 * risky part of instancing is not the `drawIndexed` call, it is the data:
 * whether a matrix is column-major, whether `fromTRS` composes in the order the
 * shader assumes, whether growing an instance array loses the transforms that
 * were already in it, and whether the layout hands the pipeline the byte offsets
 * the CPU wrote. All of that is testable here, and all of it is wrong in ways
 * that render a plausible-looking frame.
 *
 * The central assertion is *two paths, one truth*: `fromTRS` and `fromMatrices`
 * are built by different code, and the test feeds the second one matrices
 * composed by hand out of `mat4` and requires the results to agree to 1e-6. If
 * the ergonomic path ever drifts from the explicit one, that is where it shows.
 */

import { describe, expect, test } from 'bun:test';
import {
  GpuInstances,
  InstanceData,
  INSTANCE_ATTRIBUTES,
  INSTANCE_ATTRIBUTES_COLORED,
  INSTANCE_COLOR,
  instancedLayout,
  layout,
  POSITION_LAYOUT,
  STANDARD_ATTRIBUTES,
  STANDARD_LAYOUT,
  TRANSFORM_STRIDE,
  uploadInstances,
  type GpuInstancesOptions,
} from '../src/geometry/index.ts';
import {
  generateScaffold,
  validateGeneratedWGSL,
} from '../src/material/scaffold.ts';
import { instancedMaterialSpec } from '../src/material/instanced.ts';
import { AseError } from '../src/core/error.ts';
import { fromQuat, fromScale, fromTranslation, mul } from '../src/math/mat4.ts';

// ---------------------------------------------------------------------------
// A device just large enough
// ---------------------------------------------------------------------------

/**
 * The minimal `GPUDevice` `uploadInstances` touches: `limits.maxBufferSize`,
 * `createBuffer`, and `queue.writeBuffer`. It records what it was handed so the
 * upload path is asserted rather than assumed — the buffer usage flags and the
 * single `writeBuffer` are the two things a wrong implementation gets wrong
 * quietly.
 */
interface FakeBuffer {
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  destroyed: boolean;
  readonly bytes: number[];
}

interface FakeDevice {
  readonly device: GPUDevice;
  readonly buffers: FakeBuffer[];
  readonly writes: { readonly label: string; readonly offset: number; readonly elements: number; readonly floats: Float32Array }[];
}

const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

function fakeDevice(maxBufferSize = 256 * 1024 * 1024): FakeDevice {
  // Bun has no WebGPU globals, and `uploadInstances` reads the usage constants
  // the way a browser does. Installing the spec's bit values is enough, and it
  // is also a tripwire: a wrong bit here would show up as a wrong `usage` below.
  (globalThis as unknown as { GPUBufferUsage: unknown }).GPUBufferUsage = GPU_BUFFER_USAGE;

  const buffers: FakeBuffer[] = [];
  const writes: FakeDevice['writes'] = [];
  const device = {
    limits: { maxBufferSize },
    createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
      const b: FakeBuffer = {
        label: desc.label ?? '',
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        bytes: [],
      };
      buffers.push(b);
      return {
        get label() { return b.label; },
        get size() { return b.size; },
        destroy() { b.destroyed = true; },
      } as unknown as GPUBuffer;
    },
    queue: {
      writeBuffer(
        _buffer: GPUBuffer,
        bufferOffset: number,
        data: Float32Array,
        dataOffset: number,
        size: number,
      ): void {
        // Copied, exactly as a real queue does — a test that held a reference to
        // the scratch array would see later uploads mutate it.
        const floats = new Float32Array(size);
        floats.set(data.subarray(dataOffset, dataOffset + size));
        writes.push({
          label: buffers[buffers.length - 1].label,
          offset: bufferOffset,
          elements: size,
          floats,
        });
      },
    },
  } as unknown as GPUDevice;
  return { device, buffers, writes };
}

function identityMatrices(count: number, tint = 0): Float32Array {
  const out = new Float32Array(count * TRANSFORM_STRIDE);
  for (let i = 0; i < count; i++) {
    const o = i * TRANSFORM_STRIDE;
    out[o] = 1;
    out[o + 5] = 1;
    out[o + 10] = 1;
    out[o + 15] = 1;
    out[o + 12] = i + tint;
  }
  return out;
}

function throwCode(fn: () => void): AseError {
  try {
    fn();
  } catch (e) {
    if (e instanceof AseError) return e;
    throw e;
  }
  throw new Error('expected an AseError, nothing was thrown');
}

// ---------------------------------------------------------------------------
// InstanceData
// ---------------------------------------------------------------------------

describe('InstanceData.fromMatrices', () => {
  test('count, stride, and 16 floats per transform for 1, 2, and 1000 entries', () => {
    for (const n of [1, 2, 1000]) {
      const data = InstanceData.fromMatrices(identityMatrices(n), { name: `n${n}` });
      expect(data.count).toBe(n);
      expect(data.transformStride).toBe(TRANSFORM_STRIDE);
      expect(data.transforms.length).toBe(n * TRANSFORM_STRIDE);
      expect(data.name).toBe(`n${n}`);
    }
  });

  test('preserves column-major order, verbatim', () => {
    // 1 at [0,0], 1 at [1,1] etc: a row-major reader would see this as
    // transposed, which is the whole reason the brief says column-major twice.
    const data = InstanceData.fromMatrices(identityMatrices(2));
    const first = Array.from(data.transforms.subarray(0, 16));
    expect(first).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    // The translation is the last column, m[12..14] — the layout rule from
    // src/math/mat4.ts, and the one instance 1's offset shift proves.
    expect(data.transforms[12]).toBe(0);
    expect(data.transforms[TRANSFORM_STRIDE + 12]).toBe(1);
  });

  test('takes a Float32Array by reference rather than copying it', () => {
    const source = identityMatrices(2);
    const data = InstanceData.fromMatrices(source);
    expect(data.transforms).toBe(source);
    source[12] = 99;
    expect(data.transforms[12]).toBe(99);
  });

  test('converts a plain number array', () => {
    const data = InstanceData.fromMatrices([...identityMatrices(1)]);
    expect(data.transforms).toBeInstanceOf(Float32Array);
    expect(data.count).toBe(1);
  });

  test('rejects a run that is not a whole number of transforms', () => {
    const e = throwCode(() => InstanceData.fromMatrices(new Float32Array(20)));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('20');
    expect(e.message).toContain('16');
  });

  test('rejects an array of matrices, naming the flat form', () => {
    const nested = [identityMatrices(1), identityMatrices(1)] as unknown as number[];
    const e = throwCode(() => InstanceData.fromMatrices(nested));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('array of arrays');
  });
});

describe('InstanceData.fromTRS', () => {
  test('composes T · R · S identically to a hand-built fromMatrices', () => {
    // 8 instances: a non-uniform scale, a real quaternion, and a translation.
    const n = 8;
    const translations = new Float32Array(n * 3);
    const scales = new Float32Array(n * 3);
    const rotations = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      translations[i * 3] = i * 1.5;
      translations[i * 3 + 1] = -i * 0.25;
      translations[i * 3 + 2] = i * 2;
      scales[i * 3] = 1 + i * 0.1;
      scales[i * 3 + 1] = 2;
      scales[i * 3 + 2] = 0.5 + i * 0.05;
      // A unit quaternion, not an Euler triple: (0, sin(a/2), 0, cos(a/2)).
      const a = i * 0.37;
      rotations[i * 4 + 1] = Math.sin(a / 2);
      rotations[i * 4 + 3] = Math.cos(a / 2);
    }

    const trs = InstanceData.fromTRS(translations, scales, rotations);
    expect(trs.count).toBe(n);
    expect(trs.transformStride).toBe(TRANSFORM_STRIDE);

    // The independent path: mat4 composition, fed to fromMatrices.
    const t = new Float32Array(16);
    const r = new Float32Array(16);
    const s = new Float32Array(16);
    const rs = new Float32Array(16);
    const trsMat = new Float32Array(16);
    const explicit = new Float32Array(n * TRANSFORM_STRIDE);
    for (let i = 0; i < n; i++) {
      fromTranslation(t, translations[i * 3], translations[i * 3 + 1], translations[i * 3 + 2]);
      fromQuat(r, rotations.subarray(i * 4, i * 4 + 4) as Float32Array);
      fromScale(s, scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]);
      mul(rs, r, s);
      mul(trsMat, t, rs);
      explicit.set(trsMat, i * TRANSFORM_STRIDE);
    }
    const manual = InstanceData.fromMatrices(explicit);

    for (let k = 0; k < n * TRANSFORM_STRIDE; k++) {
      expect(trs.transforms[k]).toBeCloseTo(manual.transforms[k], 6);
    }
  });

  test('a uniform scale number applies to all three axes', () => {
    const trs = InstanceData.fromTRS(new Float32Array([1, 2, 3]), 2);
    expect(trs.transforms[12]).toBe(1);
    expect(trs.transforms[13]).toBe(2);
    expect(trs.transforms[14]).toBe(3);
    expect(trs.transforms[0]).toBeCloseTo(2, 6);
    expect(trs.transforms[5]).toBeCloseTo(2, 6);
    expect(trs.transforms[10]).toBeCloseTo(2, 6);
    // A scale must not touch the homogeneous column: m[15] stays 1 or every
    // vertex divides by the scale and collapses onto the camera plane.
    expect(trs.transforms[15]).toBe(1);
  });

  test('the scale is applied before the rotation, not to the result', () => {
    // 90° about Y with a *non-uniform* scale, which is the only way the two
    // orderings differ: a uniform scale commutes with a rotation.
    const q = new Float32Array([0, Math.SQRT1_2, 0, Math.SQRT1_2]);
    const scaleTriplet = new Float32Array([2, 3, 4]);
    const trs = InstanceData.fromTRS(new Float32Array([0, 0, 0]), scaleTriplet, q);
    const manual = new Float32Array(16);
    fromTranslation(manual, 0, 0, 0);
    const r = new Float32Array(16);
    fromQuat(r, q);
    const s = new Float32Array(16);
    fromScale(s, 2, 3, 4);
    const rs = new Float32Array(16);
    mul(rs, r, s);
    const want = new Float32Array(16);
    mul(want, manual, rs);
    for (let k = 0; k < 16; k++) expect(trs.transforms[k]).toBeCloseTo(want[k], 6);
    // And not the other order, which would scale the world axes instead of the
    // instance's own — a guard against the test passing because both paths
    // happened to agree.
    const wrong = new Float32Array(16);
    mul(wrong, s, r);
    mul(wrong, manual, wrong);
    expect(trs.transforms[2]).not.toBeCloseTo(wrong[2], 6);
  });

  test('without rotations every instance is a scale+translation', () => {
    const trs = InstanceData.fromTRS(new Float32Array([0, 0, 0, 5, 0, 0]), 1);
    for (let i = 0; i < 2; i++) {
      const o = i * TRANSFORM_STRIDE;
      expect(trs.transforms[o + 0]).toBe(1);
      expect(trs.transforms[o + 5]).toBe(1);
      expect(trs.transforms[o + 10]).toBe(1);
      expect(trs.transforms[o + 15]).toBe(1);
      expect(trs.transforms[o + 12]).toBe(i * 5);
    }
  });

  test('rejects a translation run that is not a multiple of 3', () => {
    const e = throwCode(() => InstanceData.fromTRS(new Float32Array(7)));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('3 floats');
  });

  test('rejects a scale run that disagrees with the translation count', () => {
    const e = throwCode(() => InstanceData.fromTRS(new Float32Array(6), new Float32Array(3)));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('scale');
  });

  test('rejects a partial rotation run', () => {
    const e = throwCode(() => InstanceData.fromTRS(new Float32Array(6), undefined, new Float32Array(4)));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('quaternion');
    expect(e.message).toContain('2 instances');
  });

  test('builds 5000 instances from 15000 numbers', () => {
    const data = InstanceData.fromTRS(new Float32Array(15000));
    expect(data.count).toBe(5000);
    expect(data.transforms.length).toBe(80000);
  });
});

describe('InstanceData colours', () => {
  test('no colours means a null array and the uncoloured layout', () => {
    const data = InstanceData.fromMatrices(identityMatrices(3));
    expect(data.colors).toBeNull();
    expect(data.layout).toBe(INSTANCE_ATTRIBUTES);
    expect(Object.keys(data.layout)).toEqual([
      'instanceTransform0', 'instanceTransform1', 'instanceTransform2', 'instanceTransform3',
    ]);
  });

  test('colours add one attribute and the coloured layout', () => {
    const data = InstanceData.fromMatrices(identityMatrices(2), { colors: new Float32Array(8) });
    expect(data.colors).not.toBeNull();
    expect(data.colors!.length).toBe(8);
    expect(data.layout).toBe(INSTANCE_ATTRIBUTES_COLORED);
    expect(data.layout[INSTANCE_COLOR]).toBe('float32x4');
  });

  test('rejects a colour run of the wrong length', () => {
    const e = throwCode(() => InstanceData.fromMatrices(identityMatrices(2), { colors: new Float32Array(4) }));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    // The message carries both numbers: what was given, and what the count needs.
    expect(e.message).toContain('4 colour components');
    expect(e.message).toContain('8');
  });
});

describe('InstanceData.setTransform / setColor', () => {
  test('setTransform writes one full stride', () => {
    const data = InstanceData.fromMatrices(identityMatrices(3));
    const m = identityMatrices(1);
    m[12] = 42;
    data.setTransform(1, m);
    expect(data.transforms[TRANSFORM_STRIDE + 12]).toBe(42);
    expect(data.transforms[12]).toBe(0);
    expect(data.transforms[TRANSFORM_STRIDE * 2 + 12]).toBe(2);
  });

  test('setColor writes rgba and defaults alpha to 1', () => {
    const data = InstanceData.fromMatrices(identityMatrices(2), { colors: new Float32Array(8) });
    data.setColor(0, 0.1, 0.2, 0.3, 0.4);
    data.setColor(1, 1, 1, 1);
    expect(Array.from(data.colors!)).toEqual(
      Array.from(new Float32Array([0.1, 0.2, 0.3, 0.4, 1, 1, 1, 1])),
    );
  });

  test('an out-of-range index names the index and the count', () => {
    const data = InstanceData.fromMatrices(identityMatrices(3), { colors: new Float32Array(12) });
    for (const [fn, what] of [
      [() => data.setTransform(3, identityMatrices(1)), 'setTransform(3)'],
      [() => data.setColor(3, 1, 1, 1), 'setColor(3)'],
    ] as const) {
      const e = throwCode(fn);
      expect(e.code).toBe('INTERNAL_INVARIANT');
      expect(e.message).toContain('3');
      expect(e.message).toContain('holds 3 instances');
      expect(e.message).toContain(what.split('(')[0]);
    }
  });

  test('a negative or fractional index is refused too', () => {
    const data = InstanceData.fromMatrices(identityMatrices(2));
    expect(throwCode(() => data.setTransform(-1, identityMatrices(1))).code).toBe('INTERNAL_INVARIANT');
    expect(throwCode(() => data.setTransform(1.5, identityMatrices(1))).code).toBe('INTERNAL_INVARIANT');
  });

  test('a short matrix is refused rather than half-copied', () => {
    const data = InstanceData.fromMatrices(identityMatrices(1));
    const e = throwCode(() => data.setTransform(0, new Float32Array(12)));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('12');
  });

  test('setColor on data with no colours is refused, not dropped', () => {
    const data = InstanceData.fromMatrices(identityMatrices(2));
    const e = throwCode(() => data.setColor(0, 1, 0, 0));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain(INSTANCE_COLOR);
  });
});

describe('InstanceData.reserve', () => {
  test('growth preserves every transform and colour already written', () => {
    const data = InstanceData.fromMatrices(identityMatrices(3), { colors: new Float32Array(12) });
    data.setColor(1, 0.5, 0.25, 0.125, 0.75);
    const before = Array.from(data.transforms);
    const colorsBefore = Array.from(data.colors!);

    data.reserve(10);

    expect(data.count).toBe(10);
    expect(data.transforms.length).toBe(10 * TRANSFORM_STRIDE);
    // The bug this test exists for: reallocating without copying leaves zeros
    // where the transforms were, and every one of those instances collapses.
    expect(Array.from(data.transforms.subarray(0, 3 * TRANSFORM_STRIDE))).toEqual(before);
    expect(Array.from(data.colors!.subarray(0, 12))).toEqual(colorsBefore);
  });

  test('new instances start at identity and white, not zero', () => {
    const data = InstanceData.fromMatrices(identityMatrices(1), { colors: new Float32Array(4) });
    data.reserve(2);
    const o = TRANSFORM_STRIDE;
    expect(Array.from(data.transforms.subarray(o, o + TRANSFORM_STRIDE)))
      .toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    expect(Array.from(data.colors!.subarray(4))).toEqual([1, 1, 1, 1]);
  });

  test('growth on data with no colours grows only the transforms', () => {
    const data = InstanceData.fromMatrices(identityMatrices(1));
    data.reserve(4);
    expect(data.count).toBe(4);
    expect(data.colors).toBeNull();
    expect(data.layout).toBe(INSTANCE_ATTRIBUTES);
  });

  test('shrinking is refused, with the count in the message', () => {
    const data = InstanceData.fromMatrices(identityMatrices(8));
    const e = throwCode(() => data.reserve(4));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('already holds 8');
    expect(data.count).toBe(8);
  });

  test('reserving the same count is a no-op', () => {
    const data = InstanceData.fromMatrices(identityMatrices(3));
    const before = data.transforms;
    data.reserve(3);
    expect(data.count).toBe(3);
    expect(data.transforms).toBe(before);
  });

  test('a negative or fractional count is refused', () => {
    const data = InstanceData.fromMatrices(identityMatrices(2));
    expect(throwCode(() => data.reserve(-1)).code).toBe('INTERNAL_INVARIANT');
    expect(throwCode(() => data.reserve(2.5)).code).toBe('INTERNAL_INVARIANT');
  });

  test('a from-empty reserve is the spawn path', () => {
    const data = InstanceData.fromMatrices(new Float32Array(0));
    expect(data.count).toBe(0);
    data.reserve(2);
    data.setTransform(0, identityMatrices(1));
    expect(data.count).toBe(2);
    expect(data.transforms[15]).toBe(1);
  });
});

describe('InstanceData transform stride', () => {
  test('a stride below 16 or unaligned is refused', () => {
    expect(throwCode(() => new InstanceData({ transforms: new Float32Array(8), transformStride: 8 })).code)
      .toBe('INTERNAL_INVARIANT');
    expect(throwCode(() => new InstanceData({ transforms: new Float32Array(20), transformStride: 18 })).code)
      .toBe('INTERNAL_INVARIANT');
  });

  test('a 20-float stride is accepted by the data and divides the count', () => {
    const data = new InstanceData({ transforms: new Float32Array(40), transformStride: 20 });
    expect(data.count).toBe(2);
    expect(data.transformStride).toBe(20);
  });
});

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

describe('VertexLayout instance attributes', () => {
  test('an existing layout is byte-for-byte what it was', () => {
    // The regression guard for every additive change in layout.ts.
    expect(STANDARD_LAYOUT.gpuLayout()).toEqual({
      arrayStride: 32,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x2' },
      ],
    });
    expect(STANDARD_LAYOUT.stride).toBe(32);
    expect(STANDARD_LAYOUT.key).toBe('position:float32x3|normal:float32x3|uv:float32x2');
    expect(STANDARD_LAYOUT.attributeCount).toBe(3);
    expect(STANDARD_LAYOUT.instanceAttributeCount).toBe(0);
    expect(STANDARD_LAYOUT.instanceStride).toBe(0);
    expect(STANDARD_LAYOUT.instanced).toBe(false);
    expect(STANDARD_LAYOUT.instanceAttributes).toEqual([]);
    expect(STANDARD_LAYOUT.gpuLayouts()).toHaveLength(1);
    expect(STANDARD_LAYOUT.gpuLayouts()[0]).toEqual(STANDARD_LAYOUT.gpuLayout());
    expect(STANDARD_LAYOUT.wgslStruct('VertexIn')).toBe([
      'struct VertexIn {',
      '  @location(0) position : vec3<f32>,',
      '  @location(1) normal : vec3<f32>,',
      '  @location(2) uv : vec2<f32>,',
      '};',
    ].join('\n'));
  });

  test('POSITION_LAYOUT and a 2-arg layout() call are unchanged', () => {
    expect(POSITION_LAYOUT.stride).toBe(12);
    expect(POSITION_LAYOUT.key).toBe('position:float32x3');
    expect(POSITION_LAYOUT.gpuLayouts()).toHaveLength(1);
    const two = layout({ position: 'float32x3' }, 64);
    expect(two.stride).toBe(12);
    expect(two.instanceStride).toBe(0);
    expect(two.gpuLayouts()).toHaveLength(1);
  });

  test('the instance half has a stride, a second slot, and continues the locations', () => {
    const l = instancedLayout();
    expect(l.instanceStride).toBe(64);
    expect(l.instanceAttributeCount).toBe(4);
    expect(l.attributeCount).toBe(3);
    expect(l.instanced).toBe(true);
    expect(l.gpuLayouts()).toHaveLength(2);

    const slots = l.gpuLayouts();
    // Slot order is the setVertexBuffer index, so this ordering is load-bearing.
    expect(slots[0].stepMode).toBe('vertex');
    expect(slots[0].arrayStride).toBe(32);
    expect(slots[1].stepMode).toBe('instance');
    expect(slots[1].arrayStride).toBe(64);
    expect(slots[1].attributes).toEqual([
      { shaderLocation: 3, offset: 0, format: 'float32x4' },
      { shaderLocation: 4, offset: 16, format: 'float32x4' },
      { shaderLocation: 5, offset: 32, format: 'float32x4' },
      { shaderLocation: 6, offset: 48, format: 'float32x4' },
    ]);
  });

  test('gpuLayouts is cached and frozen', () => {
    const l = instancedLayout();
    expect(l.gpuLayouts()).toBe(l.gpuLayouts());
    expect(Object.isFrozen(l.gpuLayouts())).toBe(true);
  });

  test('the colour attribute is location 7, making 8 of the 16', () => {
    const l = instancedLayout({ color: true });
    expect(l.instanceStride).toBe(80);
    expect(l.instanceAttributeCount).toBe(5);
    expect(l.instanceAttribute(INSTANCE_COLOR)?.location).toBe(7);
    expect(l.instanceAttribute(INSTANCE_COLOR)?.offset).toBe(64);
    // The budget the brief asked for: 3 vertex + 4 transform + 1 colour.
    expect(l.attributeCount + l.instanceAttributeCount).toBe(8);
  });

  test('instanceWgslStruct emits four vec4s at locations 3..6 for a mat4x4f', () => {
    expect(instancedLayout().instanceWgslStruct()).toBe([
      'struct InstanceIn {',
      '  @location(3) instanceTransform0 : vec4<f32>,',
      '  @location(4) instanceTransform1 : vec4<f32>,',
      '  @location(5) instanceTransform2 : vec4<f32>,',
      '  @location(6) instanceTransform3 : vec4<f32>,',
      '};',
    ].join('\n'));
  });

  test('the instance struct is the tail of VertexIn, at the same locations', () => {
    const l = instancedLayout({ color: true });
    const vertex = l.wgslStruct('VertexIn');
    const instance = l.instanceWgslStruct('InstanceIn');
    const tail = instance.split('\n').slice(1, -1);
    for (const line of tail) {
      expect(vertex).toContain(line);
    }
    // A named struct for the same name must not return the wrong one.
    expect(l.wgslStruct('VertexIn')).toBe(vertex);
    expect(l.instanceWgslStruct('VertexIn')).not.toBe(vertex);
    expect(l.wgslStruct('VertexIn')).toBe(vertex);
  });

  test('a non-instanced layout has an empty instance struct', () => {
    expect(STANDARD_LAYOUT.instanceWgslStruct()).toBe('struct InstanceIn {\n};');
  });

  test('the key differs from the same layout without instancing', () => {
    const plain = layout(STANDARD_ATTRIBUTES);
    const inst = instancedLayout();
    expect(plain.key).toBe('position:float32x3|normal:float32x3|uv:float32x2');
    expect(inst.key).toBe(
      'position:float32x3|normal:float32x3|uv:float32x2'
      + '|inst:instanceTransform0:float32x4|instanceTransform1:float32x4'
      + '|instanceTransform2:float32x4|instanceTransform3:float32x4',
    );
    expect(inst.key).not.toBe(plain.key);
    expect(instancedLayout({ color: true }).key).not.toBe(inst.key);
  });

  test('two builds of the same instanced layout share a key', () => {
    expect(instancedLayout().key).toBe(instancedLayout().key);
  });

  test('a name used on both halves is refused as a duplicate struct field', () => {
    const e = throwCode(() => layout({ position: 'float32x3', uv: 'float32x2' }, 2048, { uv: 'float32x4' }));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain('twice');
  });

  test('exceeding 16 locations across both halves throws VARYING_LOCATION_OVERFLOW', () => {
    const wide: Record<string, string> = {};
    for (let i = 0; i < 13; i++) wide[`a${i}`] = 'float32x4';
    const e = throwCode(() => layout(wide as never, 2048, INSTANCE_ATTRIBUTES));
    expect(e.code).toBe('VARYING_LOCATION_OVERFLOW');
    // The message has to say how many and by what, or it is a number to guess at.
    expect(e.message).toContain('13');
    expect(e.message).toContain('4');
    expect(e.why).toContain('maxVertexAttributes');
  });

  test('an over-wide instance stride throws ATTRIBUTE_LAYOUT_OVERFLOW', () => {
    const e = throwCode(() => instancedLayout({ maxStride: 32 }));
    expect(e.code).toBe('ATTRIBUTE_LAYOUT_OVERFLOW');
    expect(e.message).toContain('64');
  });
});

describe('VertexLayout.assertCompatible with instances', () => {
  test('a mesh layout has no instance half and is still compatible', () => {
    // The Material.updateMesh(mesh) contract: a mesh is not the source of
    // instance data, so the instance half must not be compared against it. Only
    // the vertex half has to agree.
    expect(() => instancedLayout().assertCompatible(STANDARD_LAYOUT, 'mesh')).not.toThrow();
    expect(() => instancedLayout({ color: true }).assertCompatible(STANDARD_LAYOUT, 'mesh')).not.toThrow();
    expect(() => instancedLayout({ vertexAttributes: { position: 'float32x3' } })
      .assertCompatible(POSITION_LAYOUT, 'mesh')).not.toThrow();
  });

  test('two layouts that both declare instances must agree exactly', () => {
    instancedLayout().assertCompatible(instancedLayout(), 'the instance data');
    const renamed = layout(
      STANDARD_ATTRIBUTES,
      2048,
      { instanceTransform0: 'float32x4', instanceTransform1: 'float32x4', instanceTransform2: 'float32x4', instanceScale: 'float32x4' },
    );
    const missing = throwCode(() => instancedLayout().assertCompatible(renamed, 'the instance data'));
    expect(missing.code).toBe('ATTRIBUTE_MISSING');
    expect(missing.message).toContain('instance attribute "instanceTransform3"');
  });

  test('a format disagreement on an instance attribute is a LAYOUT_MISMATCH', () => {
    const halfPrecision = layout(STANDARD_ATTRIBUTES, 2048, {
      instanceTransform0: 'unorm8x4',
      instanceTransform1: 'float32x4',
      instanceTransform2: 'float32x4',
      instanceTransform3: 'float32x4',
    });
    const e = throwCode(() => instancedLayout().assertCompatible(halfPrecision, 'the instance data'));
    expect(e.code).toBe('LAYOUT_MISMATCH');
    expect(e.message).toContain('instanceTransform0');
  });

  test('a coloured layout and an uncoloured one are incompatible', () => {
    // The other direction of the same check: the data offers an attribute the
    // material does not read.
    const e = throwCode(() => instancedLayout().assertCompatible(instancedLayout({ color: true }), 'the instance data'));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain(`supplies instance attribute "${INSTANCE_COLOR}"`);
  });

  test('the vertex half is still compared when both sides declare instances', () => {
    const e = throwCode(() => instancedLayout().assertCompatible(
      instancedLayout({ vertexAttributes: { position: 'float32x3' } }),
      'the mesh',
    ));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain('normal');
  });
});

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

describe('uploadInstances', () => {
  test('one VERTEX|COPY_DST buffer and one writeBuffer', () => {
    const { device, buffers, writes } = fakeDevice();
    const data = InstanceData.fromTRS(new Float32Array([1, 2, 3, 4, 5, 6]));
    const gpu = uploadInstances(device, data);

    expect(buffers).toHaveLength(1);
    expect(buffers[0].usage).toBe(GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST);
    expect(writes).toHaveLength(1);
    expect(writes[0].offset).toBe(0);
    expect(writes[0].elements).toBe(2 * TRANSFORM_STRIDE);
    expect(gpu.count).toBe(2);
    expect(gpu.buffer).toBeDefined();
    expect(gpu.byteLength).toBe(2 * TRANSFORM_STRIDE * 4);
    expect(gpu.instanceStride).toBe(64);
  });

  test('the transform-only upload is the caller\'s array, with no repacking', () => {
    const { device, writes } = fakeDevice();
    const source = identityMatrices(2);
    uploadInstances(device, InstanceData.fromMatrices(source));
    // Four consecutive vec4s, so a flat 16-float run is already in buffer order.
    // Anything else would mean a copy, and a copy is 320 KB a frame at 5000.
    expect(Array.from(writes[0].floats)).toEqual(Array.from(source));
  });

  test('colours are interleaved into an 80-byte record', () => {
    const { device, writes } = fakeDevice();
    const data = InstanceData.fromMatrices(identityMatrices(2), { colors: new Float32Array([1, 0, 0, 1, 0, 0, 1, 1]) });
    const gpu = uploadInstances(device, data);
    expect(gpu.instanceStride).toBe(80);
    expect(gpu.byteLength).toBe(160);
    expect(writes[0].elements).toBe(2 * 20);
    // Instance 0: 16 transform floats then its 4 colour floats.
    expect(Array.from(writes[0].floats.subarray(16, 20))).toEqual([1, 0, 0, 1]);
    expect(Array.from(writes[0].floats.subarray(36, 40))).toEqual([0, 0, 1, 1]);
  });

  test('the layout is the combined one, and matches the data', () => {
    const { device } = fakeDevice();
    const data = InstanceData.fromMatrices(identityMatrices(3), { colors: new Float32Array(12) });
    const gpu = uploadInstances(device, data);
    expect(gpu.layout.instanceStride).toBe(80);
    expect(gpu.layout.instanceAttributeCount).toBe(5);
    expect(gpu.layout.gpuLayouts()).toHaveLength(2);
    instancedLayout({ color: true }).assertCompatible(gpu.layout, 'the instance data');
  });

  test('a vertex half that does not match the mesh is a named mismatch', () => {
    const { device } = fakeDevice();
    const gpu = uploadInstances(device, InstanceData.fromMatrices(identityMatrices(1)), {
      vertexAttributes: { position: 'float32x3' },
    } as GpuInstancesOptions);
    const e = throwCode(() => instancedLayout().assertCompatible(gpu.layout, 'the instance data'));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
  });

  test('a padded transform stride cannot be uploaded, and says why', () => {
    const { device, buffers } = fakeDevice();
    const data = new InstanceData({ transforms: new Float32Array(40), transformStride: 20 });
    const e = throwCode(() => uploadInstances(device, data));
    expect(e.code).toBe('INTERNAL_INVARIANT');
    expect(e.message).toContain('cannot be uploaded');
    expect(buffers).toHaveLength(0);
  });

  test('an instance set larger than maxBufferSize is refused', () => {
    const { device, buffers } = fakeDevice(64);
    const e = throwCode(() => uploadInstances(device, InstanceData.fromTRS(new Float32Array(3 * 8))));
    expect(e.code).toBe('MESH_DATA_TOO_LARGE');
    expect(buffers).toHaveLength(0);
  });

  test('an empty instance set still allocates a valid buffer', () => {
    const { device, buffers, writes } = fakeDevice();
    const gpu = uploadInstances(device, InstanceData.fromMatrices(new Float32Array(0)));
    expect(gpu.count).toBe(0);
    expect(buffers[0].size).toBeGreaterThan(0);
    expect(writes[0].elements).toBe(0);
  });

  test('dispose destroys the buffer exactly once', () => {
    const { device, buffers } = fakeDevice();
    const gpu = uploadInstances(device, InstanceData.fromTRS(new Float32Array([0, 0, 0])));
    expect(buffers[0].destroyed).toBe(false);
    gpu.dispose();
    expect(buffers[0].destroyed).toBe(true);
    expect(gpu.disposed).toBe(true);
    gpu.dispose();
    expect(buffers[0].destroyed).toBe(true);
  });

  test('the resource is reference counted like every other apse resource', () => {
    const { device } = fakeDevice();
    const gpu = uploadInstances(device, InstanceData.fromTRS(new Float32Array([0, 0, 0])));
    expect(gpu.refCount).toBe(1);
    gpu.ref();
    expect(gpu.refCount).toBe(2);
    gpu.unref();
    expect(gpu.disposed).toBe(false);
    gpu.unref();
    expect(gpu.disposed).toBe(true);
  });

  test('GpuInstances is a Resource and InstanceData is too', () => {
    const { device } = fakeDevice();
    const gpu = uploadInstances(device, InstanceData.fromTRS(new Float32Array([0, 0, 0])));
    expect(gpu).toBeInstanceOf(GpuInstances);
    const data = InstanceData.fromTRS(new Float32Array([0, 0, 0]));
    data.dispose();
    expect(data.disposed).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// The material, generated with no device in sight
// ---------------------------------------------------------------------------

describe('instancedMaterialSpec', () => {
  test('generates a valid program through the normal spec path', () => {
    const generated = generateScaffold(instancedMaterialSpec());
    expect(validateGeneratedWGSL(generated.code)).toEqual({ ok: true });
    // The instance columns are in VertexIn, at the locations the layout assigned.
    expect(generated.code).toContain('@location(3) instanceTransform0 : vec4<f32>,');
    expect(generated.code).toContain('@location(6) instanceTransform3 : vec4<f32>,');
    // And read through the prelude, so the body validator never sees a field
    // it does not know about.
    expect(generated.code).toContain('fn instanceModel(v : VertexIn) -> mat4x4f {');
    expect(generated.code).toContain('mat4x4f(v.instanceTransform0, v.instanceTransform1, v.instanceTransform2, v.instanceTransform3)');
    expect(generated.code).toContain('* obj.model * instanceModel(in) * vec4f(in.position, 1.0)');
    // No storage buffer anywhere: maxStorageBuffersInVertexStage is 0 in compat.
    expect(generated.code).not.toContain('var<storage');
  });

  test('the instance half is never a storage buffer read', () => {
    const generated = generateScaffold(instancedMaterialSpec({ instanceColor: true }));
    expect(generated.code).not.toContain('storage');
    expect(generated.code).toContain('@location(7) instanceColor : vec4<f32>,');
    expect(generated.code).toContain('fn instanceTint(v : VertexIn) -> vec4f {');
    expect(generated.code).toContain('out.color = instanceTint(in);');
  });

  test('the uncoloured program and the coloured one are different pipelines', () => {
    const plain = generateScaffold(instancedMaterialSpec());
    const colored = generateScaffold(instancedMaterialSpec({ instanceColor: true }));
    expect(plain.pipelineKey).not.toBe(colored.pipelineKey);
    expect(plain.pipelineStateKey).not.toBe(colored.pipelineStateKey);
    // And both are deterministic, which is what makes those keys cache keys.
    expect(generateScaffold(instancedMaterialSpec()).pipelineKey).toBe(plain.pipelineKey);
  });

  test('the material layout agrees with the data it will be drawn with', () => {
    const layoutFor = InstanceData.fromTRS(new Float32Array(3)).layout;
    const spec = instancedMaterialSpec();
    // Same names, same formats, same order — the two must not drift apart, or
    // the upload writes offsets the pipeline does not read.
    expect(spec.layout!.instanceAttributes.map((a) => `${a.name}:${a.format}`))
      .toEqual(Object.entries(layoutFor).map(([k, v2]) => `${k}:${v2}`));

    const coloredData = InstanceData.fromTRS(new Float32Array(9), 1, undefined, { colors: new Float32Array(12) });
    const coloredSpec = instancedMaterialSpec({ instanceColor: true });
    expect(coloredSpec.layout!.instanceAttributes.map((a) => `${a.name}:${a.format}`))
      .toEqual(Object.entries(coloredData.layout).map(([k, v2]) => `${k}:${v2}`));
  });

  test('a layout whose instance attributes are not the known ones is refused', () => {
    const custom = layout(STANDARD_ATTRIBUTES, 2048, { m0: 'float32x4', m1: 'float32x4', m2: 'float32x4', m3: 'float32x4' });
    const e = throwCode(() => instancedMaterialSpec({ layout: custom }));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain('instanceTransform0');
  });

  test('a layout with no instance half at all is refused', () => {
    const e = throwCode(() => instancedMaterialSpec({ layout: STANDARD_LAYOUT }));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain('none');
  });

  test('a layout with the right names but the wrong format is refused', () => {
    // The names matching is not enough: a packed transform would compile and then
    // read four bytes per column where the shader multiplies a vec4.
    const packed = layout(STANDARD_ATTRIBUTES, 2048, {
      instanceTransform0: 'unorm8x4',
      instanceTransform1: 'unorm8x4',
      instanceTransform2: 'unorm8x4',
      instanceTransform3: 'unorm8x4',
    });
    const e = throwCode(() => instancedMaterialSpec({ layout: packed }));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain('unorm8x4');
  });

  test('a vertex layout with no position is refused by name', () => {
    const noPosition = layout({ uv: 'float32x2' }, 2048, INSTANCE_ATTRIBUTES);
    const e = throwCode(() => instancedMaterialSpec({ layout: noPosition }));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain('position');
  });

  test('it needs no uv varying, so a position-only vertex half works', () => {
    // Unlit and untextured: an inter-stage varying would spend a location and
    // bandwidth on a value nothing reads, and would demand a uv the mesh may not
    // have.
    const plain = generateScaffold(instancedMaterialSpec({
      layout: instancedLayout({ vertexAttributes: { position: 'float32x3' } }),
    }));
    expect(plain.resolved.userVaryings).toEqual([]);
    expect(plain.resolved.layout.gpuLayouts()[0].attributes).toEqual([
      { shaderLocation: 0, offset: 0, format: 'float32x3' },
    ]);
    // The instance half is unchanged, so the mesh buffer is now location 0..0
    // and the instance columns are still 1..4.
    // `attributes` is typed as an Iterable by @webgpu/types, not as an array,
    // so anything walking it needs Array.from — the renderer will too.
    expect(Array.from(plain.resolved.layout.gpuLayouts()[1].attributes, (a) => a.shaderLocation))
      .toEqual([1, 2, 3, 4]);
  });

  test('an uncoloured material cannot be given a coloured layout', () => {
    const e = throwCode(() => instancedMaterialSpec({ layout: instancedLayout({ color: true }) }));
    expect(e.code).toBe('ATTRIBUTE_MISSING');
    expect(e.message).toContain(INSTANCE_COLOR);
  });

  test('the transparent variant changes the blend and the phase, nothing else', () => {
    const opaque = generateScaffold(instancedMaterialSpec());
    const clear = generateScaffold(instancedMaterialSpec({ transparent: true }));
    expect(clear.resolved.phase).toBe('transparent');
    expect(clear.resolved.blend).not.toBeNull();
    expect(opaque.resolved.blend).toBeNull();
    expect(clear.resolved.layout.key).toBe(opaque.resolved.layout.key);
  });
});
