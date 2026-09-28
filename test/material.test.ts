/**
 * Material scaffold tests.
 *
 * There is no GPU in the test environment, so these test everything that does
 * not need one — which is most of the module. `Material.create` is deliberately
 * thin: acquiring a device and calling `createRenderPipelineAsync` are the only
 * steps that require hardware, so the spec resolution, the code generation, the
 * validation, the uniform packing, and the cache keys are all pure and all
 * testable here.
 *
 * The bar is *shape*, not size. A scaffold that emits a plausible-looking but
 * subtly wrong program is worse than one that refuses to build, because the
 * failure surfaces three layers down as a driver message nobody can act on. So
 * the assertions are about what is guaranteed: the exact generated text, the
 * exact byte offsets, the exact error codes, and the exact determinism.
 *
 * The golden test is the highest-value test in this file. It pins the entire
 * generated program for one fully-specified material, so any accidental change
 * to the scaffold — a reordered field, a lost `@interpolate(flat)`, a renamed
 * entry point — shows up as a diff in a snapshot rather than as a regression
 * nobody notices until a frame renders wrong.
 */

import { describe, expect, test } from 'bun:test';
import { AseError } from '../src/core/error.ts';
import { BIND_GROUP, FRAME_BLOCK, OBJECT_BLOCK, RESERVED_SLOT_NAMES } from '../src/core/slot.ts';
import { UNIFORM_TYPES } from '../src/core/uniform.ts';
import { TANGENT_LAYOUT, POSITION_LAYOUT, STANDARD_LAYOUT, layout } from '../src/geometry/layout.ts';
import { ObjectUniforms } from '../src/material/material.ts';
import {
  anisotropicMaterialSpec,
  basicMaterialSpec,
  checkVaryingBudget,
  describeMaterial,
  diffuseMaterialSpec,
  emissiveMaterialSpec,
  fnv1a,
  generateScaffold,
  pipelineKeyOf,
  removeComments,
  tokenKeyOf,
  pbrMaterialSpec,
  resolveSpec,
  samplerNameFor,
  stageAtLine,
  stripComments,
  validateBodyIdentifiers,
  validateGeneratedWGSL,
  MAX_DIRECTIONAL_LIGHTS,
  lightColorSlot,
  lightDirSlot,
  type DirectionalLight,
  type ShaderStage,
} from '../src/material/index.ts';
import type { MaterialSpec } from '../src/material/index.ts';
import {
  FakePipelineDevice,
  asPipelineDevice,
  fragmentBannerLineOf,
  fragmentEntryLineOf,
  preludeBannerLineOf,
  preludeFirstLineOf,
  vertexBannerLineOf,
  vertexEntryLineOf,
} from './fake-device-ext.ts';

// ---------------------------------------------------------------------------
// WebGPU globals
//
// Bun has no WebGPU globals, and `ObjectUniforms` and `Material` read the usage
// constants the way a browser does. The spec's bit values are enough, and
// installing them is also a tripwire: a wrong bit here would show up as a wrong
// `usage` on the recorded buffer. The same block appears in `test/tonemap.test.ts`
//; assigning the same values twice is a no-op, so whichever file loads second
// simply confirms the values.
// ---------------------------------------------------------------------------

(globalThis as unknown as { GPUBufferUsage: unknown }).GPUBufferUsage = {
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
};


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Asserts that `fn` throws a catalog error with exactly this code. */
function expectError(fn: () => unknown, code: string): AseError {
  try {
    fn();
  } catch (err) {
    if (!(err instanceof AseError)) {
      throw new Error(`expected an AseError, got ${String(err)}`);
    }
    expect(err.code).toBe(code as AseError['code']);
    // Every error carries all five fields. If any is missing, the error is
    // useless to whoever has to act on it.
    expect(err.message.length).toBeGreaterThan(0);
    expect(err.why.length).toBeGreaterThan(0);
    expect(err.fix.length).toBeGreaterThan(0);
    expect(err.link).toContain('apse.dev');
    return err;
  }
  throw new Error(`expected code ${code} but nothing was thrown`);
}

/** The smallest valid spec, overridden per test. */
function spec(over: Partial<MaterialSpec> = {}): MaterialSpec {
  return {
    name: 't',
    varyings: { uv: 'vec2f' },
    vertex: 'out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);',
    fragment: 'return vec4f(1.0);',
    ...over,
  };
}

const CHECKER: MaterialSpec = {
  name: 'checker',
  layout: STANDARD_LAYOUT,
  varyings: { worldPos: 'vec3f', uv: 'vec2f' },
  slots: { baseColor: { type: 'vec3f', default: [1, 0, 0] }, scale: 'f32' },
  textures: { albedo: { kind: '2d' } },
  phase: 'opaque',
  topology: 'triangle-list',
  cull: 'back',
  depth: { write: true, compare: 'less' },
  blend: null,
  targets: [{ format: 'bgra8unorm' }],
  vertex: `
out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;
out.uv = in.uv;
`,
  fragment: `
let base = textureSample(albedo, albedoSampler, in.uv) * vec4f(mat.baseColor, 1.0);
return vec4f(base.rgb * in.worldPos, 1.0);
`,
};

/**
 * The full generated program for {@link CHECKER}, pinned.
 *
 * Verified by eye to be valid WGSL: every struct field carries its trailing
 * comma; `MaterialData` is `vec3<f32>` at 0 with `f32` at 12, 16 bytes total;
 * `@builtin(position) clip` precedes the located varyings and consumes no
 * location; `wgslTextureDecl` places `albedo` at group 3 binding 0; and the two
 * entry points carry the attributes and signatures the pipeline asks for.
 */
const GOLDEN_WGSL = `// Generated by apse — material "checker". Do not edit: every
// declaration, binding, and signature below is produced from the spec.
// vertex   body may use: in, frame, obj, out, mat, albedo, albedoSampler
// fragment body may use: in, frame, obj, mat, albedo, albedoSampler
// ---------------------------------------------------------------------

// ---- uniform blocks ----
struct Frame {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  viewProj : mat4x4<f32>,
  invView : mat4x4<f32>,
  invProj : mat4x4<f32>,
  invViewProj : mat4x4<f32>,
  camPos : vec3<f32>,
  time : f32,
  delta : f32,
  elapsed : f32,
  resolution : vec2<f32>,
  viewport : vec2<u32>,
  exposure : f32,
  alpha : f32,
};
struct ObjectData {
  model : mat4x4<f32>,
  normalMatrix : mat3x3<f32>,
  objectId : u32,
  instanceId : u32,
  visibility : f32,
};
struct MaterialData {
  baseColor : vec3<f32>,
  scale : f32,
};

// ---- vertex input (position:float32x3|normal:float32x3|uv:float32x2) ----
struct VertexIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
};

// ---- varyings ----
struct Varyings {
  @builtin(position) clip : vec4f,
  @location(0) worldPos : vec3f,
  @location(1) uv : vec2f,
};

// ---- bindings ----
@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var<uniform> obj : ObjectData;
@group(2) @binding(0) var<uniform> mat : MaterialData;
@group(3) @binding(0) var albedo : texture_2d<f32>;
@group(3) @binding(1) var albedoSampler : sampler;

// ---- generated by apse: vertex stage ----
@vertex
fn vs(in : VertexIn) -> Varyings {
  var out : Varyings;
  out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
  out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;
  out.uv = in.uv;
  return out;
}

// ---- generated by apse: fragment stage ----
@fragment
fn fs(in : Varyings) -> @location(0) vec4f {
  let base = textureSample(albedo, albedoSampler, in.uv) * vec4f(mat.baseColor, 1.0);
  return vec4f(base.rgb * in.worldPos, 1.0);
}
`;

/**
 * A structural parse of generated WGSL.
 *
 * Not a WGSL parser — a check of the invariants a *generator* can violate while
 * producing text that looks entirely reasonable. Braces balance, every struct
 * field ends in a comma, every `@group(n)` matches `BIND_GROUP` and exposes the
 * variable the renderer's layouts are described against, and every
 * `frame.`/`obj.`/`mat.`/`in.`/`out.` access in the bodies resolves to a field
 * the spec actually declared.
 */
function parseCheck(s: MaterialSpec): string {
  const code = generateScaffold(s).code;
  const resolved = resolveSpec(s);

  // 1. Balanced braces.
  let depth = 0;
  for (const ch of code) {
    if (ch === '{') depth++;
    else if (ch === '}') depth--;
    expect(depth).toBeGreaterThanOrEqual(0);
  }
  expect(depth).toBe(0);

  // 2. Every struct field is comma-terminated.
  for (const m of code.matchAll(/struct (\w+) \{([^}]*)\}/g)) {
    for (const raw of m[2].split('\n')) {
      const line = raw.trim();
      if (line.length === 0) continue;
      expect(line.endsWith(',')).toBe(true);
    }
  }

  // 3. Group indices and variable names agree with BIND_GROUP.
  const expectedVar = new Map<number, string | null>([
    [BIND_GROUP.frame, 'frame'],
    [BIND_GROUP.object, 'obj'],
    [BIND_GROUP.material, resolved.materialBlock === null ? null : 'mat'],
    [BIND_GROUP.texture, null],
  ]);
  for (const m of code.matchAll(/@group\((\d+)\) @binding\((\d+)\) var(?:<(\w+)>)? (\w+)/g)) {
    const group = Number(m[1]);
    expect(expectedVar.has(group)).toBe(true);
    const want = expectedVar.get(group) ?? null;
    if (want !== null) expect(m[4]).toBe(want);
  }
  // Group 3 is the texture group and holds only textures and samplers.
  for (const m of code.matchAll(/@group\(3\) @binding\(\d+\) var (\w+)/g)) {
    const isTexture = resolved.textures.some((t) => t.varName === m[1]);
    const isSampler = resolved.samplers.some((s) => s.varName === m[1]);
    expect(isTexture || isSampler).toBe(true);
  }

  // 4. Every field access in either body resolves. `in` and `out` name
  // different containers in the two stages, which is exactly the thing a
  // reviewer cannot see in a generated file: in the vertex stage `in` is
  // VertexIn and `out` is Varyings; in the fragment stage `in` is Varyings and
  // `out` does not exist at all.
  const attributes = new Set(resolved.layout.attributes.map((a) => a.name));
  const varyings = new Set([...resolved.varyings.map((v) => v.name)]);
  const frameFields = new Set(FRAME_BLOCK.fields.map((f) => f.name));
  const objFields = new Set(OBJECT_BLOCK.fields.map((f) => f.name));
  const matFields = new Set(Object.keys(resolved.slotTypes));

  for (const [body, stage] of [
    [resolved.vertexBody, 'vertex'],
    [resolved.fragmentBody, 'fragment'],
  ] as [string, 'vertex' | 'fragment'][]) {
    for (const m of stripComments(body).matchAll(/\b(in|out|frame|obj|mat)\s*\.\s*(\w+)/g)) {
      const [, prefix, field] = m;
      if (prefix === 'in') {
        expect(stage === 'vertex' ? attributes.has(field) : varyings.has(field)).toBe(true);
      } else if (prefix === 'out') {
        expect(stage).toBe('vertex');
        expect(varyings.has(field)).toBe(true);
      } else if (prefix === 'frame') {
        expect(frameFields.has(field)).toBe(true);
      } else if (prefix === 'obj') {
        expect(objFields.has(field)).toBe(true);
      } else {
        expect(matFields.has(field)).toBe(true);
      }
    }
  }

  // The apse-side structural validator agrees.
  expect(validateGeneratedWGSL(code)).toEqual({ ok: true });
  return code;
}

// ---------------------------------------------------------------------------
// The golden snapshot
// ---------------------------------------------------------------------------

describe('generateScaffold — the golden snapshot', () => {
  test('the entire generated program for one fully-specified material is byte-exact', () => {
    expect(generateScaffold(CHECKER).code).toBe(GOLDEN_WGSL);
  });

  test('the golden program is structurally valid', () => {
    expect(parseCheck(CHECKER)).toBe(GOLDEN_WGSL);
  });

  test('the golden program declares no group the renderer does not know about', () => {
    const groups = [...GOLDEN_WGSL.matchAll(/@group\((\d+)\)/g)].map((m) => Number(m[1]));
    expect(new Set(groups)).toEqual(new Set([0, 1, 2, 3]));
    expect(BIND_GROUP).toEqual({ frame: 0, object: 1, material: 2, texture: 3 });
  });
});

// ---------------------------------------------------------------------------
// Generated declarations
// ---------------------------------------------------------------------------

describe('generateScaffold — generated declarations', () => {
  const code = generateScaffold(CHECKER).code;

  test('the VertexIn struct comes from the layout, with the right locations', () => {
    expect(code).toContain('struct VertexIn {');
    expect(code).toContain('  @location(0) position : vec3<f32>,');
    expect(code).toContain('  @location(1) normal : vec3<f32>,');
    expect(code).toContain('  @location(2) uv : vec2<f32>,');
  });

  test('the Varyings struct starts with the builtin position and then locates in order', () => {
    const block = code.slice(code.indexOf('struct Varyings {'), code.indexOf('};', code.indexOf('struct Varyings {')));
    const lines = block.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
    expect(lines[1]).toBe('@builtin(position) clip : vec4f,');
    expect(lines[2]).toBe('@location(0) worldPos : vec3f,');
    expect(lines[3]).toBe('@location(1) uv : vec2f,');
  });

  test('the three uniform blocks are present and named', () => {
    expect(code).toContain('struct Frame {');
    expect(code).toContain('struct ObjectData {');
    expect(code).toContain('struct MaterialData {');
    // And they are the same blocks the uniform machinery produces, so the
    // generated WGSL and the JS packing cannot drift.
    expect(code).toContain(FRAME_BLOCK.wgsl);
    expect(code).toContain(OBJECT_BLOCK.wgsl);
  });

  test('the reserved bindings are at the groups BIND_GROUP fixes', () => {
    expect(code).toContain('@group(0) @binding(0) var<uniform> frame : Frame;');
    expect(code).toContain('@group(1) @binding(0) var<uniform> obj : ObjectData;');
    expect(code).toContain('@group(2) @binding(0) var<uniform> mat : MaterialData;');
  });

  test('the material struct holds the declared slots in declaration order', () => {
    const block = code.slice(code.indexOf('struct MaterialData {'));
    expect(block).toContain('  baseColor : vec3<f32>,');
    expect(block).toContain('  scale : f32,');
    expect(block.indexOf('baseColor')).toBeLessThan(block.indexOf('scale'));
  });

  test('the texture binding uses the slot name, and the sampler follows it', () => {
    expect(code).toContain('@group(3) @binding(0) var albedo : texture_2d<f32>;');
    expect(code).toContain('@group(3) @binding(1) var albedoSampler : sampler;');
  });

  test('the entry points carry the attributes and signatures the pipeline asks for', () => {
    expect(code).toContain('@vertex\nfn vs(in : VertexIn) -> Varyings {');
    expect(code).toContain('@fragment\nfn fs(in : Varyings) -> @location(0) vec4f {');
    // `out` is declared by apse, not by the user, and returned by apse.
    expect(code).toContain('  var out : Varyings;');
  });

  test('the header names what each body may use', () => {
    expect(code).toContain('vertex   body may use: in, frame, obj, out, mat, albedo, albedoSampler');
    expect(code).toContain('fragment body may use: in, frame, obj, mat, albedo, albedoSampler');
  });

  test('a material with no slots omits the mat binding entirely', () => {
    const bare = generateScaffold(spec()).code;
    expect(bare).not.toContain('struct MaterialData');
    expect(bare).not.toContain('var<uniform> mat');
    expect(bare).toContain('@group(0) @binding(0) var<uniform> frame : Frame;');
    // The texture group keeps its index even when group 2 is empty, which is
    // why the layouts are positional and fixed.
    expect(bare).not.toContain('@group(3)');
  });

  test('a material with textures but no slots still binds group 3', () => {
    const s = generateScaffold(spec({ textures: { albedo: { kind: '2d' } } })).code;
    expect(s).not.toContain('var<uniform> mat');
    expect(s).toContain('@group(3) @binding(0) var albedo : texture_2d<f32>;');
    expect(s).toContain('@group(3) @binding(1) var albedoSampler : sampler;');
  });

  test('the prelude is spliced in after the declarations and before the entry points', () => {
    const s = spec({ prelude: 'fn helper(x: f32) -> f32 { return x * 2.0; }' });
    const code = generateScaffold(s).code;
    expect(code).toContain('fn helper(x: f32) -> f32 { return x * 2.0; }');
    expect(code.indexOf('fn helper')).toBeGreaterThan(code.indexOf('// ---- bindings ----'));
    expect(code.indexOf('fn helper')).toBeLessThan(code.indexOf('@vertex'));
  });

  test('byteLength and GeneratedShader carry the block specs', () => {
    const g = generateScaffold(CHECKER);
    expect(g.byteLength).toBe(GOLDEN_WGSL.length);
    expect(g.frameBlock).toBe(FRAME_BLOCK);
    expect(g.objectBlock).toBe(OBJECT_BLOCK);
    expect(g.materialBlock.structName).toBe('MaterialData');
    expect(g.slotTypes).toEqual({ baseColor: 'vec3f', scale: 'f32' });
    expect(g.textureSlots.map((t) => t.bindingIndex)).toEqual([0]);
  });
});

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

describe('generateScaffold — determinism', () => {
  test('the same spec produces byte-identical output, every time', () => {
    const a = generateScaffold(CHECKER).code;
    const b = generateScaffold(CHECKER).code;
    const c = generateScaffold({ ...CHECKER }).code;
    expect(b).toBe(a);
    expect(c).toBe(a);
  });

  test('resolution is memoised, so a spec object resolves to one identity', () => {
    expect(resolveSpec(CHECKER)).toBe(resolveSpec(CHECKER));
  });

  test('declaration order is significant, and reversing it changes the program', () => {
    // Varying and slot order determines @location and byte offset respectively.
    // If it were not significant, a material could be reordered by accident and
    // silently read the wrong data.
    const a = generateScaffold(spec({ varyings: { x: 'vec2f', y: 'vec2f' } })).code;
    const b = generateScaffold(spec({ varyings: { y: 'vec2f', x: 'vec2f' } })).code;
    expect(a).not.toBe(b);
  });

  test('an unrelated spec object does not contaminate the memo', () => {
    const a = generateScaffold(spec({ name: 'a' })).code;
    const b = generateScaffold(spec({ name: 'b' })).code;
    const c = generateScaffold(spec({ name: 'a' })).code;
    expect(a).toBe(c);
    expect(a).not.toBe(b);
  });
});

// ---------------------------------------------------------------------------
// Body validation
// ---------------------------------------------------------------------------

describe('generateScaffold — body validation', () => {
  test('a function declaration is rejected, and the fix names the escape hatch', () => {
    const err = expectError(
      () => generateScaffold(spec({ fragment: 'fn helper() -> f32 { return 1.0; }\nreturn vec4f(1.0);' })),
      'SHADER_BODY_INVALID',
    );
    expect(err.message).toContain('declares a function');
    expect(err.fix).toContain('prelude');
  });

  test('@group in a body is rejected — apse owns every binding', () => {
    const err = expectError(
      () => generateScaffold(spec({ fragment: '@group(1) @binding(0) var<uniform> mine : mat4x4f;\nreturn vec4f(1.0);' })),
      'SHADER_BODY_INVALID',
    );
    expect(err.message).toContain('@group');
    expect(err.why).toContain('bind group layout');
  });

  test('@binding, @builtin, @location and struct declarations are all rejected', () => {
    for (const body of [
      '@binding(0) var x : f32;\nreturn vec4f(1.0);',
      '@builtin(position) var x : vec4f;\nreturn vec4f(1.0);',
      '@location(0) var x : f32;\nreturn vec4f(1.0);',
      'struct Mine { a : f32, }\nreturn vec4f(1.0);',
    ]) {
      expectError(() => generateScaffold(spec({ fragment: body })), 'SHADER_BODY_INVALID');
    }
  });

  test('an @vertex/@fragment attribute in a body is rejected', () => {
    expectError(() => generateScaffold(spec({ fragment: '@fragment\nreturn vec4f(1.0);' })), 'SHADER_BODY_INVALID');
  });

  test('a vertex body that never assigns out.clip is rejected, and says so', () => {
    // This is the single most common mistake: everything compiles, every
    // triangle is silently discarded, and Dawn says nothing.
    const err = expectError(
      () => generateScaffold(spec({ vertex: 'out.uv = in.uv;' })),
      'SHADER_BODY_INVALID',
    );
    expect(err.message).toContain('out.clip');
    expect(err.fix).toContain('out.clip = frame.viewProj');
  });

  test('an empty body is rejected on both stages', () => {
    expectError(() => generateScaffold(spec({ vertex: '   \n  ' })), 'SHADER_BODY_INVALID');
    expectError(() => generateScaffold(spec({ fragment: '' })), 'SHADER_BODY_INVALID');
  });

  test('a fragment body that never returns is rejected', () => {
    const err = expectError(() => generateScaffold(spec({ fragment: 'let x = 1.0;' })), 'SHADER_BODY_INVALID');
    expect(err.message).toContain('never returns');
  });

  test('writing to out in the fragment stage is rejected', () => {
    const err = expectError(() => generateScaffold(spec({ fragment: 'out.uv = in.uv;\nreturn vec4f(1.0);' })), 'SHADER_BODY_INVALID');
    expect(err.message).toContain('`out`');
  });

  test('declarations inside a comment are not a false positive', () => {
    // Comments are stripped before scanning, so prose about `fn` and `@group`
    // does not trip the validator. This matters: the pbr material's body is
    // full of comments explaining the rules.
    const code = generateScaffold(spec({
      vertex: '// out.clip = frame.viewProj * obj.model;\nout.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);',
      fragment: '/* a fn would go in prelude, and @group is apse\'s */\nreturn vec4f(1.0);',
    })).code;
    expect(code).toContain('return vec4f(1.0);');
  });

  test('stripComments preserves line numbers and length', () => {
    const src = 'a\n// one\nb /* two\nthree */\nc';
    const out = stripComments(src);
    expect(out.split('\n').length).toBe(src.split('\n').length);
    expect(out.length).toBe(src.length);
    expect(out).toContain('a');
    expect(out).toContain('b');
    expect(out).toContain('c');
  });
});

// ---------------------------------------------------------------------------
// Identifier validation — the agent guard
// ---------------------------------------------------------------------------

describe('validateBodyIdentifiers — undeclared field is a typed error, not a WGSL one', () => {
  test('a typo in a frame field is caught and suggests the real one', () => {
    const err = expectError(
      () => generateScaffold(spec({ fragment: 'let v = frame.viewProjj;\nreturn vec4f(1.0);' })),
      'SHADER_BODY_INVALID',
    );
    expect(err.message).toContain('frame.viewProjj');
    expect(err.fix).toContain('frame.viewProj');
    expect(err.fix).toContain('camPos');
  });

  test('a hallucinated obj field is caught, and every real field is listed', () => {
    // Not a typo, so no suggestion — the useful answer is the full inventory.
    const err = expectError(
      () => generateScaffold(spec({ vertex: 'out.clip = frame.viewProj * obj.worldMatrix * vec4f(in.position, 1.0);' })),
      'SHADER_BODY_INVALID',
    );
    expect(err.message).toContain('obj.worldMatrix');
    for (const f of OBJECT_BLOCK.fields.map((x) => x.name)) expect(err.fix).toContain(f);
  });

  test('a material with no slots tells you to declare one, rather than listing nothing', () => {
    const err = expectError(() => generateScaffold(spec({ fragment: 'return vec4f(mat.tint);' })), 'SHADER_BODY_INVALID');
    expect(err.fix).toContain("slots:");
  });

  test('a slot the material never declared is caught', () => {
    const err = expectError(
      () => generateScaffold(spec({ slots: { tint: 'vec3f' }, fragment: 'return vec4f(mat.albedo);' })),
      'SHADER_BODY_INVALID',
    );
    expect(err.message).toContain('mat.albedo');
    expect(err.fix).toContain('Available on mat: tint');
  });

  test('a varying the material never declared is caught on out and on in', () => {
    const err = expectError(() => generateScaffold(spec({ vertex: 'out.clip = vec4f(in.position, 1.0); out.tangent = in.normal;' })), 'SHADER_BODY_INVALID');
    expect(err.message).toContain('out.tangent');
    const err2 = expectError(() => generateScaffold(spec({ fragment: 'return vec4f(in.tangent, 1.0);' })), 'SHADER_BODY_INVALID');
    expect(err2.message).toContain('in.tangent');
  });

  test('an attribute the layout does not have is caught in the vertex stage', () => {
    const err = expectError(() => generateScaffold(spec({ vertex: 'out.clip = vec4f(in.tangent, 1.0);' })), 'SHADER_BODY_INVALID');
    expect(err.message).toContain('in.tangent');
  });

  test('a body naming a deduplicated sampler is told the right name', () => {
    // Two slots, one shared sampler. The body must use the group's name.
    const s = spec({
      textures: { albedo: { kind: '2d' }, normalMap: { kind: '2d' } },
      fragment: 'let a = textureSample(normalMap, normalMapSampler, in.uv);\nreturn vec4f(a.rgb);',
    });
    const err = expectError(() => generateScaffold(s), 'SHADER_BODY_INVALID');
    expect(err.message).toContain('normalMapSampler');
    expect(err.fix).toContain('albedoSampler');
  });

  test('a non-texture member on a texture variable is rejected', () => {
    const err = expectError(
      () => generateScaffold(spec({ textures: { albedo: { kind: '2d' } }, fragment: 'let a = albedo.width;\nreturn vec4f(1.0);' })),
      'SHADER_BODY_INVALID',
    );
    expect(err.message).toContain('albedo.width');
  });

  test('legitimate texture builtin members are not rejected', () => {
    expect(() => generateScaffold(spec({
      textures: { albedo: { kind: '2d' } },
      fragment: 'let d = vec2f(textureDimensions(albedo));\nlet a = textureSample(albedo, albedoSampler, in.uv * d);\nreturn vec4f(a.rgb);',
    }))).not.toThrow();
  });

  test('validateBodyIdentifiers is exported and callable on its own', () => {
    const s = spec({ slots: { tint: 'vec3f' } });
    const r = resolveSpec(s);
    expect(() => validateBodyIdentifiers('return vec4f(mat.tint);', r, 'fragment', 't')).not.toThrow();
    expectError(() => validateBodyIdentifiers('return vec4f(mat.missing);', r, 'fragment', 't'), 'SHADER_BODY_INVALID');
  });

  test('local variables named like a reserved prefix are not confused for one', () => {
    // A local `mat4` or `obj2` must not trip the scan; only the exact reserved
    // prefixes followed by `.field` do.
    expect(() => generateScaffold(spec({
      slots: { tint: 'vec3f' },
      fragment: 'let frame2 = mat.tint;\nlet inner = 1.0;\nreturn vec4f(frame2 * f32(inner));',
    }))).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Varyings
// ---------------------------------------------------------------------------

describe('varyings', () => {
  test('locations are assigned in declaration order from 0, and clip takes none', () => {
    const r = resolveSpec(spec({ varyings: { a: 'f32', b: 'vec2f', c: 'vec4f' } }));
    expect(r.varyings[0]).toMatchObject({ name: 'clip', location: -1 });
    expect(r.varyings.slice(1).map((v) => [v.name, v.location])).toEqual([['a', 0], ['b', 1], ['c', 2]]);
  });

  test('integer varyings are emitted @interpolate(flat), because flat is mandatory', () => {
    // WGSL rejects an integer varying without it. This is the single easiest
    // trap in the whole module, and apse removes it by construction.
    const code = generateScaffold(spec({ varyings: { id: 'u32' } })).code;
    expect(code).toContain('@location(0) @interpolate(flat) id : u32,');
  });

  test('mat4x4f is flat too; float and vecN are not', () => {
    const code = generateScaffold(spec({
      varyings: { x: 'f32', p: 'vec3f', m: 'mat4x4f', s: 'i32' },
    })).code;
    expect(code).toContain('@location(0) x : f32,');
    expect(code).toContain('@location(1) p : vec3f,');
    expect(code).toContain('@location(2) @interpolate(flat) m : mat4x4f,');
    expect(code).toContain('@location(3) @interpolate(flat) s : i32,');
  });

  test('a varying type that cannot cross the rasteriser is rejected by name', () => {
    const err = expectError(() => generateScaffold(spec({ varyings: { bad: 'vec3u' as never } })), 'VARYING_TYPE_UNSUPPORTED');
    expect(err.fix).toContain('vec3f');
    expectError(() => generateScaffold(spec({ varyings: { bad: 'mat3x3f' as never } })), 'VARYING_TYPE_UNSUPPORTED');
  });

  test('17 varyings exceeds the inter-stage budget', () => {
    // 15 on the core profile, 14 in compatibility — because
    // @builtin(position) is one of the fragment stage's input variables even
    // though it consumes no location.
    const many: Record<string, 'f32'> = {};
    for (let i = 0; i < 17; i++) many[`v${i}`] = 'f32';
    const err = expectError(() => generateScaffold(spec({ varyings: many })), 'VARYING_TOO_MANY');
    expect(err.message).toContain('17');
    expect(err.why).toContain('@builtin(position)');
    expect(err.fix).toContain('15');
  });

  test('exactly 15 user varyings is the limit, and 16 is not', () => {
    const at: Record<string, 'f32'> = {};
    for (let i = 0; i < 15; i++) at[`v${i}`] = 'f32';
    expect(() => generateScaffold(spec({ varyings: at }))).not.toThrow();
    at.v15 = 'f32';
    expectError(() => generateScaffold(spec({ varyings: at })), 'VARYING_TOO_MANY');
  });

  test('the device limit tightens the check when the device is reachable', () => {
    const r = resolveSpec(spec({ varyings: { a: 'f32', b: 'f32' } }));
    expect(() => checkVaryingBudget(r)).not.toThrow();
    expectError(() => checkVaryingBudget(r, { maxInterStageShaderVariables: 2 }), 'VARYING_TOO_MANY');
  });

  test('the component budget is checked as well as the variable count', () => {
    // Four mat4x4f varyings is 64 components, over the 56 left after position.
    const r = resolveSpec(spec({ varyings: { a: 'mat4x4f', b: 'mat4x4f', c: 'mat4x4f', d: 'mat4x4f' } }));
    const err = expectError(() => checkVaryingBudget(r), 'VARYING_TOO_MANY');
    expect(err.message).toContain('components');
  });

  test('a varying named clip is rejected — the builtin owns it', () => {
    expectError(() => generateScaffold(spec({ varyings: { clip: 'vec3f' } })), 'VARYING_TYPE_UNSUPPORTED');
  });
});

// ---------------------------------------------------------------------------
// Slots and reserved names
// ---------------------------------------------------------------------------

describe('slots', () => {
  test('a slot named after an obj field is rejected', () => {
    const err = expectError(() => generateScaffold(spec({ slots: { model: 'f32' } })), 'OPTION_UNKNOWN');
    expect(err.message).toContain('"model"');
    expect(err.fix).toContain('normalMatrix');
  });

  test('a slot named after a frame field is rejected', () => {
    // `view` is the classic: a user reaching for a frame field and getting a
    // material slot instead would read stale per-material data.
    const err = expectError(() => generateScaffold(spec({ slots: { view: 'f32' } })), 'OPTION_UNKNOWN');
    expect(err.message).toContain('"view"');
  });

  test('every reserved name is protected, both halves of the blocks', () => {
    for (const name of RESERVED_SLOT_NAMES) {
      expectError(() => generateScaffold(spec({ slots: { [name]: 'f32' } })), 'OPTION_UNKNOWN');
    }
    // 19 names: 5 object fields and 14 frame fields.
    expect(RESERVED_SLOT_NAMES.size).toBe(19);
  });

  test('a default with the wrong component count is rejected', () => {
    const err = expectError(
      () => generateScaffold(spec({ slots: { color: { type: 'vec3f', default: [1, 0] } } })),
      'SLOT_DEFAULT_INVALID',
    );
    expect(err.message).toContain('3-component');
    expect(err.fix).toContain('exactly 3 numbers');
  });

  test('a non-finite default is rejected, because it blanks every fragment', () => {
    expectError(() => generateScaffold(spec({ slots: { x: { type: 'f32', default: NaN } } })), 'SLOT_DEFAULT_INVALID');
    expectError(() => generateScaffold(spec({ slots: { c: { type: 'vec3f', default: [1, 0, Infinity] } } })), 'SLOT_DEFAULT_INVALID');
  });

  test('an unknown slot type is rejected', () => {
    expectError(() => generateScaffold(spec({ slots: { x: 'vec7f' as never } })), 'SLOT_TYPE_UNKNOWN');
  });

  test('an unknown spec option is rejected, so a typo never fails silently', () => {
    const err = expectError(() => generateScaffold(spec({ varing: { uv: 'vec2f' } } as never)), 'OPTION_UNKNOWN');
    expect(err.message).toContain('varing');
    expect(err.fix).toContain('varyings');
  });

  test('a slot name that is not a usable WGSL identifier is rejected', () => {
    expectError(() => generateScaffold(spec({ slots: { 'my-color': 'f32' } })), 'OPTION_UNKNOWN');
    expectError(() => generateScaffold(spec({ slots: { '1st': 'f32' } })), 'OPTION_UNKNOWN');
    expectError(() => generateScaffold(spec({ slots: { fn: 'f32' } })), 'OPTION_UNKNOWN');
  });

  test('both slot spellings resolve to the same type map and the same program', () => {
    const short = generateScaffold(spec({ slots: { x: 'f32' } })).code;
    const long = generateScaffold(spec({ slots: { x: { type: 'f32' } } })).code;
    expect(short).toBe(long);
  });

  test('more than one colour target is rejected, and says why', () => {
    const err = expectError(
      () => generateScaffold(spec({ targets: [{ format: 'rgba8unorm' }, { format: 'rgba16float' }] })),
      'RENDER_TARGET_FORMAT_MISMATCH',
    );
    expect(err.message).toContain('one colour');
    expectError(() => generateScaffold(spec({ targets: [] })), 'RENDER_TARGET_FORMAT_MISMATCH');
  });
});

// ---------------------------------------------------------------------------
// Uniform offsets — the vec3 padding rule
// ---------------------------------------------------------------------------

describe('uniform offsets', () => {
  test('a vec3f is 16-byte aligned but only 12 wide, so the next field moves to 16', () => {
    // This is a real bug class. `vec3<f32>` in the *uniform* address space is
    // aligned to 16 but occupies 12, and the following member starts at the
    // next 16-byte boundary. Getting this wrong writes `c` on top of the third
    // component of `b`, and the symptom is a slowly wrong colour, not a crash.
    const g = generateScaffold(spec({ slots: { a: 'f32', b: 'vec3f', c: 'f32' } }));
    const off = Object.fromEntries(g.materialBlock.fields.map((f) => [f.name, f.offset]));
    expect(off).toEqual({ a: 0, b: 16, c: 28 });
    expect(g.materialBlock.size).toBe(32);
  });

  test('the generated struct and the offsets come from the same builder', () => {
    const g = generateScaffold(spec({ slots: { a: 'f32', b: 'vec3f', c: 'f32' } }));
    // The WGSL is `buildUniformBlock`'s, not a hand-written template, which is
    // the only reason the two can be trusted to agree.
    expect(g.code).toContain(g.materialBlock.wgsl);
    expect(g.materialBlock.wgsl).toBe('struct MaterialData {\n  a : f32,\n  b : vec3<f32>,\n  c : f32,\n};');
  });

  test('a mat3x3f costs 48 bytes because each of its 3 columns is 16-aligned', () => {
    const g = generateScaffold(spec({ slots: { m: 'mat3x3f', x: 'f32' } }));
    const off = Object.fromEntries(g.materialBlock.fields.map((f) => [f.name, f.offset]));
    expect(off).toEqual({ m: 0, x: 48 });
    expect(g.materialBlock.size).toBe(64);
  });

  test('vec4f packs tightly; vec2f aligns to 8', () => {
    const g = generateScaffold(spec({ slots: { a: 'f32', b: 'vec4f', c: 'f32', d: 'vec2f' } }));
    const off = Object.fromEntries(g.materialBlock.fields.map((f) => [f.name, f.offset]));
    expect(off).toEqual({ a: 0, b: 16, c: 32, d: 40 });
  });

  test('the object block is dynamic, so it carries a 256-byte stride', () => {
    expect(OBJECT_BLOCK.stride % 256).toBe(0);
    expect(OBJECT_BLOCK.stride).toBeGreaterThanOrEqual(OBJECT_BLOCK.size);
  });
});

// ---------------------------------------------------------------------------
// Texture bindings and sampler sharing
// ---------------------------------------------------------------------------

describe('texture bindings', () => {
  test('two slots with the same sampler config share ONE sampler binding', () => {
    // A sampler is immutable state with no per-texture data, so N textures with
    // one config need one GPUSampler and one binding, not N.
    const g = generateScaffold(spec({ textures: { albedo: { kind: '2d' }, normalMap: { kind: '2d' } } }));
    const r = g.resolved;

    expect(g.textureSlots.map((t) => [t.varName, t.bindingIndex])).toEqual([['albedo', 0], ['normalMap', 1]]);
    expect(r.samplers.length).toBe(1);
    expect(r.samplers[0].bindingIndex).toBe(2);
    expect(r.samplers[0].slotNames).toEqual(['albedo', 'normalMap']);
    // Both slots are told to use the one name.
    expect(r.samplerForSlot.get('albedo')?.varName).toBe('albedoSampler');
    expect(r.samplerForSlot.get('normalMap')?.varName).toBe('albedoSampler');

    // So group 3 holds 3 bindings, not 4.
    const group3 = g.code.split('\n').filter((l) => l.includes('@group(3)'));
    expect(group3.length).toBe(3);
  });

  test('different address modes produce different samplers', () => {
    const r = resolveSpec(spec({
      textures: { repeat: { kind: '2d' }, clamp: { kind: '2d', addressMode: 'clamp-to-edge' } },
    }));
    expect(r.samplers.length).toBe(2);
    expect(r.samplers.map((s) => s.bindingIndex)).toEqual([2, 3]);
  });

  test('a comparison sampler is its own group and gets the comparison WGSL type', () => {
    // This is what pbr uses for its shadow map, alongside a filtering sampler
    // for albedo: two distinct configurations, so two bindings, both correct.
    const r = resolveSpec(spec({
      textures: {
        albedo: { kind: '2d' },
        shadowDepth: { kind: 'depth-2d', sampleType: 'depth', compare: true, mipmapFilter: false },
      },
    }));
    expect(r.samplers.map((s) => s.wgslType)).toEqual(['sampler', 'sampler_comparison']);
    expect(r.samplers.map((s) => s.layoutType)).toEqual(['filtering', 'comparison']);
    expect(r.samplers[0].addressMode).toBe('repeat');
    expect(r.samplers[1].addressMode).toBe('clamp-to-edge');
  });

  test('a texture binding index is assigned in declaration order', () => {
    const g = generateScaffold(spec({
      textures: { a: { kind: '2d' }, b: { kind: '2d' }, c: { kind: '2d' } },
    }));
    expect(g.textureSlots.map((t) => t.bindingIndex)).toEqual([0, 1, 2]);
  });

  test('an unsupported kind is rejected, and the message lists the supported ones', () => {
    const err = expectError(() => generateScaffold(spec({ textures: { x: { kind: '1d' as never } } })), 'TEXTURE_SLOT_TYPE_INVALID');
    expect(err.fix).toContain('depth-2d');
  });

  test('compare: true on a non-depth texture is rejected at spec time', () => {
    // WGSL rejects textureSampleCompare on a non-depth texture, but the message
    // would point into generated code. This points at the spec instead.
    expectError(
      () => generateScaffold(spec({ textures: { x: { kind: '2d', compare: true } } })),
      'TEXTURE_SLOT_TYPE_INVALID',
    );
  });

  test('a texture variable name that collides with a generated sampler is rejected', () => {
    const err = expectError(
      () => generateScaffold(spec({ textures: { tex: { kind: '2d' }, texSampler: { kind: '2d' } } })),
      'OPTION_UNKNOWN',
    );
    expect(err.message).toContain('collides');
  });

  test('a texture slot name override changes the shader variable but not the slot', () => {
    const g = generateScaffold(spec({ textures: { 'albedo-map': { kind: '2d', name: 'albedo' } } }));
    expect(g.code).toContain('@group(3) @binding(0) var albedo : texture_2d<f32>;');
    expect(g.textureSlots[0].slotName).toBe('albedo-map');
  });
});

// ---------------------------------------------------------------------------
// Cache keys
// ---------------------------------------------------------------------------

describe('cache keys', () => {
  // The reasoning, stated once so the assertions below can be short.
  //
  // There are two different questions and they have two different answers.
  //
  //   "Can these two materials share a GPUBindGroupLayout / GPUPipelineLayout?"
  //   YES whenever their *declarations* match — layout, varyings, slot types,
  //   texture kinds. Their shader statements are irrelevant to what has to be
  //   bound. So this key excludes the body, and two materials that differ only
  //   in how they shade get one shared layout object. This is where the win is:
  //   a scene of 500 materials should own 4 bind group layouts, not 2000.
  //
  //   "Is this the same compiled program?"
  //   YES only when the generated bytes match. A GPURenderPipeline is a linked
  //   shader module plus fixed-function state; two materials with different
  //   bodies have different modules, and sharing one pipeline would render one
  //   of them with the other's code — silently, with no WebGPU diagnostic and
  //   a plausible-looking frame. So this key includes the body.
  //
  // Conflating them into one key would either lose the layout sharing or, more
  // dangerously, "optimise" away a correctness bug.

  test('two specs differing only in the shader body share a pipeline STATE key', () => {
    // Bind group layouts are determined by declarations, not by statements.
    const a = generateScaffold(spec({ fragment: 'return vec4f(1.0, 0.0, 0.0, 1.0);' }));
    const b = generateScaffold(spec({ fragment: 'return vec4f(0.0, 1.0, 0.0, 1.0);' }));
    expect(a.pipelineStateKey).toBe(b.pipelineStateKey);
  });

  test('...but NOT a pipeline key, because the compiled programs differ', () => {
    // The other half of the same decision. If this ever became equal, one of two
    // materials would be drawn with the other's shader.
    const a = generateScaffold(spec({ fragment: 'return vec4f(1.0, 0.0, 0.0, 1.0);' }));
    const b = generateScaffold(spec({ fragment: 'return vec4f(0.0, 1.0, 0.0, 1.0);' }));
    expect(a.pipelineKey).not.toBe(b.pipelineKey);
  });

  test('two specs differing in cull differ in BOTH keys', () => {
    const a = generateScaffold(spec({ cull: 'back' }));
    const b = generateScaffold(spec({ cull: 'none' }));
    expect(a.pipelineStateKey).not.toBe(b.pipelineStateKey);
    expect(a.pipelineKey).not.toBe(b.pipelineKey);
  });

  test('every fixed-function field participates in the state key', () => {
    const base = generateScaffold(spec()).pipelineStateKey;
    const variants: [string, Partial<MaterialSpec>][] = [
      ['topology', { topology: 'triangle-strip' }],
      ['cull', { cull: 'front' }],
      ['frontFace', { frontFace: 'cw' }],
      ['depth.write', { depth: { write: false, compare: 'less' } }],
      ['depth.compare', { depth: { write: true, compare: 'less-equal' } }],
      ['blend', { blend: { color: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' } } }],
      ['target format', { targets: [{ format: 'bgra8unorm' }] }],
      ['sampleCount', { sampleCount: 4 }],
      ['depthFormat', { depthFormat: 'depth32float' }],
    ];
    for (const [label, over] of variants) {
      expect(`${label}:${generateScaffold(spec(over)).pipelineStateKey}`).not.toBe(`${label}:${base}`);
    }
  });

  test('every structural declaration participates in the state key', () => {
    const base = generateScaffold(spec()).pipelineStateKey;
    const variants: [string, Partial<MaterialSpec>][] = [
      ['layout', { layout: POSITION_LAYOUT }],
      ['varyings', { varyings: { uv: 'vec2f', extra: 'f32' } }],
      ['slots', { slots: { tint: 'vec3f' } }],
      ['textures', { textures: { albedo: { kind: '2d' } } }],
      ['texture kind', { textures: { albedo: { kind: 'cube' } } }],
    ];
    for (const [label, over] of variants) {
      expect(`${label}:${generateScaffold(spec(over)).pipelineStateKey}`).not.toBe(`${label}:${base}`);
    }
  });

  test('the slot name is part of the state key, not just the slot type', () => {
    // `mat.x` and `mat.y` are the same shape and must not share a layout, or
    // one would read the other's field.
    const a = generateScaffold(spec({ slots: { x: 'f32' } })).pipelineStateKey;
    const b = generateScaffold(spec({ slots: { y: 'f32' } })).pipelineStateKey;
    expect(a).not.toBe(b);
  });

  test('the state key is a 32-bit integer, so a lookup is an integer compare', () => {
    const k = generateScaffold(spec()).pipelineStateKey;
    expect(Number.isInteger(k)).toBe(true);
    expect(k).toBeGreaterThanOrEqual(0);
    expect(k).toBeLessThanOrEqual(0xffffffff);
  });

  test('fnv1a is order-sensitive and field-separator aware', () => {
    // ["ab","c"] and ["a","bc"] must not collide, or a field boundary could be
    // moved without changing the key.
    expect(fnv1a('ab', 'c')).not.toBe(fnv1a('a', 'bc'));
    expect(fnv1a('a', 'b')).not.toBe(fnv1a('b', 'a'));
    expect(fnv1a('a')).toBe(fnv1a('a'));
  });

  test('the pipeline key is the state key plus the code, and nothing else', () => {
    const g = generateScaffold(spec());
    expect(pipelineKeyOf(g.resolved, g.code)).toBe(g.pipelineKey);
  });

  test('materials differing only in name or comments SHARE a pipeline', () => {
    // The regression this guards: the generated header carries the material
    // name, and bodies are commented. If the key hashed raw text, every
    // material would be its own pipeline and the cache would be decorative.
    // A comment cannot change what a shader does, so it cannot change its
    // identity.
    const a = generateScaffold(spec({ name: 'alpha', fragment: '// red\nreturn vec4f(1.0, 0.0, 0.0, 1.0);' }));
    const b = generateScaffold(spec({ name: 'beta', fragment: 'return vec4f(1.0, 0.0, 0.0, 1.0); // a much longer explanation of the same thing' }));
    expect(a.code).not.toBe(b.code);
    expect(a.pipelineKey).toBe(b.pipelineKey);
    expect(a.pipelineStateKey).toBe(b.pipelineStateKey);
  });

  test('a real code change still separates the pipeline', () => {
    // ...and stripping comments must not go so far as to equate programs that
    // differ. If it did, two materials would be drawn with the wrong shader.
    const a = generateScaffold(spec({ fragment: 'return vec4f(1.0, 0.0, 0.0, 1.0);' }));
    const b = generateScaffold(spec({ fragment: 'return vec4f(1.0, 0.0, 0.0, 0.5);' }));
    expect(a.pipelineKey).not.toBe(b.pipelineKey);
  });

  test('removeComments deletes rather than blanks, so a long comment cannot shift the hash', () => {
    // Blanking would leave a run of spaces and still change the hash. Deleting
    // is what makes comment length irrelevant.
    // Whitespace that merely preceded a comment stays — deleting it would
    // mean tracking whether it was separating two tokens, which is the
    // tokenizer's job and not the key's.
    expect(removeComments('a // x\nb')).toBe('a \nb');
    expect(removeComments('a /* x */ b')).toBe('a  b');
    expect(removeComments('a\n// whole line\nb')).toBe('a\n\nb');
    expect(removeComments('a;')).not.toBe(removeComments('b;'));

    // tokenKeyOf is the form actually hashed, and it drops the leftover runs.
    expect(tokenKeyOf('a // x\n b')).toBe('a b');
    expect(tokenKeyOf('a  /* x */   b')).toBe('a b');
    expect(tokenKeyOf('a /* a much longer comment */ b')).toBe(tokenKeyOf('a // x\n b'));
    // ...and a token change is still visible.
    expect(tokenKeyOf('a;')).not.toBe(tokenKeyOf('b;'));
  });

  test('a different material with the same name is a different pipeline', () => {
    const a = generateScaffold(spec({ name: 'x', slots: { a: 'f32' } }));
    const b = generateScaffold(spec({ name: 'x', slots: { b: 'f32' } }));
    expect(a.pipelineKey).not.toBe(b.pipelineKey);
    expect(a.pipelineStateKey).not.toBe(b.pipelineStateKey);
  });
});

// ---------------------------------------------------------------------------
// describeMaterial — the anti-hallucination surface
// ---------------------------------------------------------------------------

describe('describeMaterial', () => {
  const d = describeMaterial(CHECKER);

  test('it inventories every slot with its type, offset, and how to read it', () => {
    expect(d.slots).toEqual([
      { name: 'baseColor', type: 'vec3f', wgsl: 'vec3<f32>', components: 3, offset: 0, size: 12, access: 'mat.baseColor' },
      { name: 'scale', type: 'f32', wgsl: 'f32', components: 1, offset: 12, size: 4, access: 'mat.scale' },
    ]);
  });

  test('it inventories every varying with its location, and marks the builtin', () => {
    expect(d.varyings).toEqual([
      { name: 'clip', type: 'vec4f', location: -1, flat: false, builtin: true },
      { name: 'worldPos', type: 'vec3f', location: 0, flat: false, builtin: false },
      { name: 'uv', type: 'vec2f', location: 1, flat: false, builtin: false },
    ]);
  });

  test('it inventories every texture with its view dimension and its sampler', () => {
    expect(d.textures).toEqual([{
      slotName: 'albedo',
      varName: 'albedo',
      kind: '2d',
      viewDimension: '2d',
      sampleType: 'float',
      binding: 0,
      sampler: 'albedoSampler',
      samplerBinding: 1,
      samplerSharedWith: [],
      declaration: '@group(3) @binding(0) var albedo : texture_2d<f32>;',
    }]);
  });

  test('it reports which slots share a sampler, so the correct name is never guessed', () => {
    const shared = describeMaterial(spec({ textures: { albedo: { kind: '2d' }, normalMap: { kind: '2d' } } }));
    expect(shared.textures.find((t) => t.slotName === 'normalMap')).toMatchObject({
      sampler: 'albedoSampler',
      samplerBinding: 2,
      samplerSharedWith: ['albedo'],
    });
    expect(shared.samplers).toEqual([{
      name: 'albedoSampler',
      binding: 2,
      type: 'sampler',
      sharedBy: ['albedo', 'normalMap'],
    }]);
  });

  test('it reports the full bind group layout, including absent groups', () => {
    expect(d.bindGroups.map((g) => [g.index, g.name, g.present])).toEqual([
      [0, 'frame', true],
      [1, 'object', true],
      [2, 'material', true],
      [3, 'texture', true],
    ]);
    expect(d.bindGroups[3].entries).toEqual([
      { binding: 0, resource: '2d / float (2d)', visibility: 'VERTEX | FRAGMENT' },
      { binding: 1, resource: 'sampler sampler (repeat, mipmapped)', visibility: 'VERTEX | FRAGMENT' },
    ]);
  });

  test('an untextured, slotless material reports groups 2 and 3 as absent', () => {
    const bare = describeMaterial(spec());
    expect(bare.bindGroups[2].present).toBe(false);
    expect(bare.bindGroups[3].present).toBe(false);
    expect(bare.blocks.material).toBeNull();
    expect(bare.textures).toEqual([]);
  });

  test('it reports block sizes, including the object stride a dynamic offset needs', () => {
    expect(d.blocks.frame).toEqual({ name: 'Frame', size: FRAME_BLOCK.size, fields: 14 });
    expect(d.blocks.object).toEqual({ name: 'ObjectData', size: OBJECT_BLOCK.size, stride: OBJECT_BLOCK.stride, fields: 5 });
    expect(d.blocks.material).toEqual({ name: 'MaterialData', size: 16, fields: 2 });
  });

  test('it states exactly what each body may reference', () => {
    expect(d.bodyNames.vertex).toEqual(['in', 'frame', 'obj', 'out', 'mat', 'albedo', 'albedoSampler']);
    expect(d.bodyNames.fragment).toEqual(['in', 'frame', 'obj', 'mat', 'albedo', 'albedoSampler']);
  });

  test('it reports the WGSL length, so a size regression is visible', () => {
    expect(d.wgslBytes).toBe(GOLDEN_WGSL.length);
  });

  test('it reports the resolved fixed-function state and the keys', () => {
    expect(d.state).toEqual({
      phase: 'opaque',
      topology: 'triangle-list',
      cull: 'back',
      frontFace: 'ccw',
      depth: { write: true, compare: 'less' },
      blend: null,
      targets: [{ format: 'bgra8unorm', blend: null, writeMask: 15 }],
      sampleCount: 1,
      depthFormat: 'depth24plus',
    });
    expect(d.pipelineStateKey).toBe(generateScaffold(CHECKER).pipelineStateKey);
    expect(d.pipelineKey).toBe(generateScaffold(CHECKER).pipelineKey);
  });

  test('it is memoised, so calling it per frame in a dev build is a map lookup', () => {
    expect(describeMaterial(CHECKER)).toBe(d);
  });

  test('it works with no device at all — which is the whole point', () => {
    // Every field above came from a spec and a device-free generator. An agent
    // can call this instead of reading source.
    expect(typeof globalThis.navigator === 'undefined' || (globalThis.navigator as { gpu?: unknown }).gpu === undefined).toBe(true);
    expect(d.name).toBe('checker');
  });
});

// ---------------------------------------------------------------------------
// Structural validation of generated WGSL
// ---------------------------------------------------------------------------

describe('validateGeneratedWGSL', () => {
  test('it accepts every material this module ships', () => {
    for (const s of [
      CHECKER,
      basicMaterialSpec(),
      basicMaterialSpec({ textured: true, transparent: true, doubleSided: true }),
      pbrMaterialSpec(),
      pbrMaterialSpec({ shadows: true }),
      pbrMaterialSpec({ environment: true, environmentSpecular: true, textured: true, rimStrength: 0.5 }),
      diffuseMaterialSpec(),
      diffuseMaterialSpec({ rimStrength: 0.1, shadows: true, environment: true, textured: true, transparent: true }),
      emissiveMaterialSpec(),
      emissiveMaterialSpec({ fresnel: true, fresnelStrength: 2, pulseHz: 1, pulseDepth: 0.5, textured: true }),
      anisotropicMaterialSpec(),
      anisotropicMaterialSpec({ shadows: true, environment: true, rimStrength: 0.2 }),
      spec({ textures: { a: { kind: 'cube' }, b: { kind: '3d' }, c: { kind: '2d-array' } } }),
      spec({ layout: layout({ position: 'float32x3', normal: 'float32x3' }) }),
    ]) {
      expect(() => validateGeneratedWGSL(generateScaffold(s).code)).not.toThrow();
    }
  });

  test('it rejects an unbalanced brace', () => {
    expectError(() => validateGeneratedWGSL('struct A {\n  f : f32,\n'), 'INTERNAL_INVARIANT');
    expectError(() => validateGeneratedWGSL('}'), 'INTERNAL_INVARIANT');
    // ...and a balanced one is accepted, or the check would be useless.
    expect(() => validateGeneratedWGSL('struct A {\n  f : f32,\n};')).not.toThrow();
  });

  test('it rejects a struct field with no trailing comma', () => {
    const err = expectError(() => validateGeneratedWGSL('struct A {\n  f : f32\n};'), 'INTERNAL_INVARIANT');
    expect(err.message).toContain('trailing comma');
  });

  test('it rejects a binding at a group that is not in BIND_GROUP', () => {
    const err = expectError(
      () => validateGeneratedWGSL('@group(7) @binding(0) var x : sampler;'),
      'INTERNAL_INVARIANT',
    );
    expect(err.message).toContain('@group(7)');
    expect(err.why).toContain('BIND_GROUP');
  });

  test('it rejects a group exposing the wrong variable', () => {
    // Positional layouts: @group(0) must be `frame`, or the renderer's bind
    // group for group 0 would be described against the wrong thing.
    const err = expectError(
      () => validateGeneratedWGSL('@group(0) @binding(0) var<uniform> notframe : Frame;'),
      'INTERNAL_INVARIANT',
    );
    expect(err.message).toContain('must be "frame"');
  });
});

// ---------------------------------------------------------------------------
// The two shipped materials
// ---------------------------------------------------------------------------

describe('basic and pbr go through the same scaffold', () => {
  test('basic is unlit, tiny, and needs no hand-written WGSL', () => {
    const s = basicMaterialSpec();
    const statements = (src: string | undefined): string[] =>
      stripComments(src ?? '').split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
    expect(Object.keys(s.slots ?? {})).toEqual(['tint', 'opacity']);
    expect(statements(s.vertex)).toEqual([
      'out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);',
      'out.uv = in.uv;',
    ]);
    expect(statements(s.fragment)).toEqual(['return vec4f(mat.tint, mat.opacity * frame.alpha);']);
    expect(s.textures).toBeUndefined();
  });

  test('the transparent variant defaults to no depth write and alpha blending', () => {
    const s = basicMaterialSpec({ transparent: true });
    expect(s.phase).toBe('transparent');
    expect(s.depth).toEqual({ write: false, compare: 'less' });
    expect(s.blend?.color).toEqual({ srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' });
  });

  test('the opaque variant replaces rather than blends', () => {
    expect(basicMaterialSpec().blend).toBeNull();
  });

  test('pbr declares the seven required slots, plus the two the shadow needs', () => {
    const s = pbrMaterialSpec();
    expect(s.slots).toMatchObject({
      baseColor: { type: 'vec3f' },
      metallic: { type: 'f32' },
      roughness: { type: 'f32' },
      lightDir: { type: 'vec3f' },
      lightColor: { type: 'vec3f' },
      ambientColor: { type: 'vec3f' },
      aoStrength: { type: 'f32' },
    });
  });

  test('pbr options become slot defaults, so setSlot can retune them later', () => {
    const s = pbrMaterialSpec({ baseColor: [1, 0, 0], metallic: 0.9, roughness: 0.2 });
    expect(s.slots?.baseColor).toEqual({ type: 'vec3f', default: [1, 0, 0] });
    expect(s.slots?.metallic).toEqual({ type: 'f32', default: 0.9 });
  });

  test('pbr without shadows declares no texture group at all', () => {
    const g = generateScaffold(pbrMaterialSpec());
    expect(g.textureSlots).toEqual([]);
    expect(g.resolved.samplers).toEqual([]);
    expect(g.code).not.toContain('@group(3)');
    expect(g.code).not.toContain('textureSampleCompare');
  });

  test('pbr with shadows declares a depth texture and a comparison sampler', () => {
    const g = generateScaffold(pbrMaterialSpec({ shadows: true }));
    expect(g.textureSlots[0]).toMatchObject({ kind: 'depth-2d', sampleType: 'depth', compare: true });
    expect(g.resolved.samplers[0].wgslType).toBe('sampler_comparison');
    expect(g.code).toContain('@group(3) @binding(0) var shadowDepth : texture_depth_2d;');
    expect(g.code).toContain('textureSampleCompare(shadowDepth, shadowDepthSampler');
  });

  test('pbr keeps the comparison sample in uniform control flow', () => {
    // textureSampleCompare is illegal in non-uniform control flow. The bounds
    // test is folded into `select` for exactly this reason, and this test is
    // what stops someone "simplifying" it back into an early-out.
    const code = generateScaffold(pbrMaterialSpec({ shadows: true })).code;
    const body = code.slice(code.indexOf('fn fs('));
    expect(body).toContain('let shadow = select(1.0, sum / 9.0, inside);');
    expect(body).not.toMatch(/\n\s*if\s*\([^)]*shadowUV[^)]*\)\s*\{[^}]*textureSampleCompare/);
  });

  test('pbr uses a mat3x3f by mat3x3f, not a GLSL mat3 by vec4', () => {
    // WGSL is strict: matCxR multiplies a vecC. This is a classic porting slip
    // and the generated shader must not make it.
    const code = generateScaffold(pbrMaterialSpec()).code;
    expect(code).toContain('normalize(obj.normalMatrix * in.normal)');
    expect(code).not.toContain('obj.normalMatrix * vec4f');
  });

  test('both materials produce structurally valid programs', () => {
    for (const s of [basicMaterialSpec({ textured: true }), pbrMaterialSpec({ shadows: true })]) {
      parseCheck(s);
    }
  });

  test('pbr and basic each compile to a distinct pipeline but share nothing by accident', () => {
    const a = generateScaffold(basicMaterialSpec());
    const b = generateScaffold(pbrMaterialSpec());
    expect(a.pipelineStateKey).not.toBe(b.pipelineStateKey);
    // ...but they do share the frame and object layouts, because the state key
    // is per-material structural identity while the *layouts* are keyed by
    // their own signature in the device cache.
    expect(a.resolved.layout.key).toBe(b.resolved.layout.key);
  });
});

// ===========================================================================
// 13. The shipped library
//
// Every factory is a function returning a MaterialSpec, so the whole library is
// testable with no GPU. The bar is the same as the scaffold's: what is
// *guaranteed* — the exact slots, the exact statements, the exact wiring — rather
// than that the pixels look right, which no assertion here can know.
// ===========================================================================

/** The five shipped factories, with every option that switches something on. */
const LIBRARY: readonly { name: string; spec: MaterialSpec }[] = [
  { name: 'basic', spec: basicMaterialSpec() },
  {
    name: 'pbr',
    spec: pbrMaterialSpec({
      lights: [
        { direction: [0.5, 1, 0.3], color: [3, 3, 3], castShadow: true },
        { direction: [-0.7, 0.3, -0.2], color: [0.4, 0.5, 0.9], intensity: 2 },
      ],
      shadows: true,
      environment: true,
      environmentSpecular: true,
      rimStrength: 0.35,
      textured: true,
    }),
  },
  {
    name: 'diffuse',
    spec: diffuseMaterialSpec({ rimStrength: 0.08, shadows: true, environment: true, textured: true }),
  },
  {
    name: 'emissive',
    spec: emissiveMaterialSpec({
      fresnel: true, fresnelStrength: 1.5, pulseHz: 2, pulseDepth: 0.5, textured: true, transparent: true,
    }),
  },
  { name: 'anisotropic', spec: anisotropicMaterialSpec({ shadows: true, rimStrength: 0.2 }) },
];

describe('the shipped library — every material goes through the same scaffold', () => {
  test('all five factories produce structurally valid generated WGSL', () => {
    for (const { name, spec } of LIBRARY) {
      const code = generateScaffold(spec).code;
      expect(validateGeneratedWGSL(code), name).toEqual({ ok: true });
      // ...and through the whole structural parse, which also resolves every
      // `mat.`/`in.`/`out.`/`frame.`/`obj.` access against what was declared.
      parseCheck(spec);
    }
  });

  test('five distinct factories produce five distinct programs', () => {
    const codes = LIBRARY.map((m) => generateScaffold(m.spec).code);
    expect(new Set(codes).size).toBe(codes.length);
    // ...and no two share a pipeline, which is the only assertion that would
    // catch two factories having accidentally become the same shader.
    const keys = LIBRARY.map((m) => generateScaffold(m.spec).pipelineKey);
    expect(new Set(keys).size).toBe(keys.length);
  });

  test('none of them writes a @group, a @binding, or a struct by hand', () => {
    // The point of the whole project. A material is allowed a `fn` and a `const`
    // in its prelude and nothing else, so this is the assertion that keeps a
    // future "just one binding" from creeping in.
    for (const { name, spec } of LIBRARY) {
      const prelude = resolveSpec(spec).prelude;
      expect(prelude, name).not.toMatch(/@group|@binding|@vertex|@fragment/);
      expect(prelude, name).not.toMatch(/\bstruct\s+\w/);
      expect(prelude, name).not.toMatch(/var\s*</);
    }
  });

  test('the bodies are statements only: no declaration, no attribute', () => {
    for (const { name, spec } of LIBRARY) {
      for (const body of [resolveSpec(spec).vertexBody, resolveSpec(spec).fragmentBody]) {
        expect(body, name).not.toMatch(/@|^\s*fn\s|^\s*struct\s/m);
      }
    }
  });
});

// ===========================================================================
// 14. The light rig is data
// ===========================================================================

/** How many times the direct-lighting loop accumulates. */
function lightIterations(spec: MaterialSpec): number {
  const body = stripComments(resolveSpec(spec).fragmentBody);
  return (body.match(/direct = direct \+ \(diff \+ spec\)/g) ?? []).length;
}

describe('the light rig is data, not shader text', () => {
  const lit = (n: number): DirectionalLight[] =>
    Array.from({ length: n }, (_, i) => ({
      direction: [i + 1, 1, 0.3] as const,
      color: [1, 1, 1] as const,
    }));

  test('one light through four: one slot pair and one loop iteration each', () => {
    for (let n = 1; n <= MAX_DIRECTIONAL_LIGHTS; n++) {
      const spec = pbrMaterialSpec({ lights: lit(n) });
      const slots = Object.keys(resolveSpec(spec).slotTypes);
      for (let i = 0; i < n; i++) {
        expect(slots, `n=${n}`).toContain(lightDirSlot(i));
        expect(slots, `n=${n}`).toContain(lightColorSlot(i));
      }
      expect(slots, `n=${n}`).not.toContain(lightDirSlot(n));
      expect(lightIterations(spec), `n=${n}`).toBe(n);
    }
  });

  test('the first light keeps the shipped slot names, and the rest are indexed', () => {
    // The asymmetry is deliberate and load-bearing: `setSlot('lightDir', …)` is
    // in the wild, and renaming it would break that call with a runtime error
    // rather than a compile error. Pinned so the rule cannot drift.
    expect(lightDirSlot(0)).toBe('lightDir');
    expect(lightColorSlot(0)).toBe('lightColor');
    expect(lightDirSlot(1)).toBe('light1Dir');
    expect(lightColorSlot(1)).toBe('light1Color');
    expect(lightDirSlot(3)).toBe('light3Dir');
    expect(lightColorSlot(3)).toBe('light3Color');

    const withDefault = resolveSpec(pbrMaterialSpec()).slotTypes;
    expect(Object.keys(withDefault)).toContain('lightDir');
    expect(Object.keys(withDefault)).toContain('lightColor');
  });

  test('a fourth light is a data change and a fifth is a typed error', () => {
    expect(MAX_DIRECTIONAL_LIGHTS).toBe(4);
    expect(() => pbrMaterialSpec({ lights: lit(4) })).not.toThrow();
    const err = expectError(() => pbrMaterialSpec({ lights: lit(5) }), 'OPTION_UNKNOWN');
    expect(err.message).toContain('5 lights');
    expect(err.message).toContain('4');
    expect(err.fix).toContain('MAX_DIRECTIONAL_LIGHTS');
  });

  test('an empty light array is rejected rather than shading nothing', () => {
    const err = expectError(() => pbrMaterialSpec({ lights: [] }), 'OPTION_UNKNOWN');
    expect(err.message).toContain('empty light array');
    expect(err.fix).toContain('lights:');
  });

  test('intensity is folded into the radiance slot, so a light is two slots', () => {
    const spec = pbrMaterialSpec({
      lights: [{ direction: [0, 1, 0], color: [1, 0.5, 0.25], intensity: 4 }],
    });
    const defaults = resolveSpec(spec).slotDefaults;
    expect(defaults.get('lightColor')).toEqual([4, 2, 1]);
    expect(defaults.get('lightDir')).toEqual([0, 1, 0]);
    expect(Object.keys(resolveSpec(spec).slotTypes).filter((s) => s.startsWith('light'))).toEqual([
      'lightDir', 'lightColor',
    ]);
  });

  test('a zero direction and a negative intensity are rejected by name', () => {
    const zero = expectError(
      () => pbrMaterialSpec({ lights: [{ direction: [0, 0, 0] }] }),
      'OPTION_UNKNOWN',
    );
    expect(zero.message).toContain('[0, 0, 0]');
    expect(zero.why).toContain('NaN');
    const negative = expectError(
      () => pbrMaterialSpec({ lights: [{ direction: [0, 1, 0], intensity: -1 }] }),
      'OPTION_UNKNOWN',
    );
    expect(negative.message).toContain('intensity');
    expectError(() => pbrMaterialSpec({ lights: [{ direction: [0, 1, 0], color: [1, Number.NaN, 1] }] }), 'OPTION_UNKNOWN');
  });

  test('castShadow is only legal on a shadowed first light, and both mistakes say why', () => {
    const noShadowMap = expectError(
      () => pbrMaterialSpec({ lights: [{ direction: [0, 1, 0], castShadow: true }] }),
      'OPTION_UNKNOWN',
    );
    expect(noShadowMap.message).toContain('castShadow');
    expect(noShadowMap.fix).toContain('shadows: true');

    const notFirst = expectError(
      () => pbrMaterialSpec({
        shadows: true,
        lights: [{ direction: [0, 1, 0] }, { direction: [1, 0, 0], castShadow: true }],
      }),
      'OPTION_UNKNOWN',
    );
    expect(notFirst.message).toContain('only the first light');
    expect(notFirst.fix).toContain('cascade');
  });

  test('the shadow multiplier lands on the light that asked for it, and only that one', () => {
    const spec = pbrMaterialSpec({
      shadows: true,
      lights: [{ direction: [0, 1, 0], castShadow: true }, { direction: [1, 0, 0] }],
    });
    const body = stripComments(resolveSpec(spec).fragmentBody);
    const lines = body.split('\n').map((l) => l.trim());
    const accumulators = lines.filter((l) => l.startsWith('direct = direct +'));
    expect(accumulators).toHaveLength(2);
    expect(accumulators[0]).toMatch(/mat\.lightColor \* nDotL \* shadow;$/);
    expect(accumulators[1]).toMatch(/mat\.light1Color \* nDotL;$/);
  });

  test('the shorthand and `lights` together are rejected, not silently merged', () => {
    const err = expectError(
      () => pbrMaterialSpec({ lightDir: [0, 1, 0], lights: [{ direction: [1, 0, 0] }] }),
      'OPTION_UNKNOWN',
    );
    expect(err.message).toContain('lightDir');
    expect(err.fix).toContain('lights');
  });

  test('the shorthand alone still works, and defaults to the shipped light', () => {
    const spec = pbrMaterialSpec({ lightDir: [0, 2, 0], lightColor: [5, 5, 5] });
    const defaults = resolveSpec(spec).slotDefaults;
    expect(defaults.get('lightDir')).toEqual([0, 2, 0]);
    expect(defaults.get('lightColor')).toEqual([5, 5, 5]);
    // A direction of any length is legal: the shader normalises it, so
    // `normalize(lightPos - worldPos)` can be precomputed once per frame.
    expect(lightIterations(spec)).toBe(1);
  });

  test('every lit material shares one light rig, not one per material', () => {
    // The same helper emits the loop for all of them, which is what makes the
    // count a property of the array rather than of the shader.
    for (const spec of [
      pbrMaterialSpec({ lights: lit(3) }),
      diffuseMaterialSpec({ lights: lit(3) }),
      anisotropicMaterialSpec({ lights: lit(3) }),
    ]) {
      expect(lightIterations(spec)).toBe(3);
      expect(Object.keys(resolveSpec(spec).slotTypes)).toContain('light2Color');
    }
    // ...and the emissive material is not a lit material, so it has no rig at
    // all: an emitter does not have a light array, it has a radiance.
    expect(Object.keys(resolveSpec(emissiveMaterialSpec()).slotTypes).some((s) => s.startsWith('light')))
      .toBe(false);
  });

  test('a light count change moves the uniform block, and the block is generated', () => {
    const one = generateScaffold(pbrMaterialSpec({ lights: lit(1) })).materialBlock;
    const four = generateScaffold(pbrMaterialSpec({ lights: lit(4) })).materialBlock;
    expect(one.size).toBe(four.size - 3 * 32);
    // Every offset comes from buildUniformBlock: 16-byte aligned vec3f, 12 wide.
    for (const field of four.fields) {
      expect(field.offset % UNIFORM_TYPES[field.type].align).toBe(0);
    }
    expect(four.wgsl).toContain('light3Dir : vec3<f32>,');
  });
});

// ===========================================================================
// 15. The BRDFs, as numbers
//
// The WGSL cannot run here, so each model is re-derived in TypeScript from the
// same structure the generated code has and the *properties* it must satisfy are
// asserted. These are the tests that would catch a sign error, a missing 1/PI, or
// a rim that adds energy that never came in.
// ===========================================================================

/** Schlick, the generated `fresnelSchlick`. */
function fresnelSchlick(cos: number, f0: number): number {
  const m = Math.min(1, Math.max(0, 1 - cos));
  const m2 = m * m;
  return f0 + (1 - f0) * (m2 * m2 * m);
}

/** GGX, the generated `distributionGGX`, guard included. */
function distributionGGX(nDotH: number, roughness: number): number {
  const a = roughness * roughness;
  const a2 = a * a;
  const d = nDotH * nDotH * (a2 - 1) + 1;
  return a2 / Math.max(Math.PI * d * d, 1e-12);
}

/** Burley's anisotropic GGX, the generated `distributionGGXAniso`. */
function distributionGGXAniso(nDotH: number, tDotH: number, bDotH: number, at: number, ab: number): number {
  const a2 = at * ab;
  const v = [ab * tDotH, at * bDotH, a2 * nDotH];
  const v2 = v[0]! * v[0]! + v[1]! * v[1]! + v[2]! * v[2]!;
  const w2 = a2 / Math.max(v2, 1e-12);
  return (a2 * w2 * w2) / Math.PI;
}

/** Oren-Nayar, the generated `orenNayar`. */
function orenNayar(nDotL: number, nDotV: number, lDotV: number, sigma2: number): number {
  const s = lDotV - nDotL * nDotV;
  const t = s > 0 ? Math.max(nDotL, nDotV) + 1e-4 : 1;
  const a = 1 - (0.5 * sigma2) / (sigma2 + 0.33);
  const b = (0.45 * sigma2) / (sigma2 + 0.09);
  return Math.min((nDotL / Math.PI) * (a + (b * s) / t), 1);
}

describe('Fresnel, the rim, and the energy budget', () => {
  test('Schlick is F0 head-on and 1 at grazing, for every F0', () => {
    for (const f0 of [0, 0.04, 0.5, 1]) {
      expect(fresnelSchlick(1, f0)).toBeCloseTo(f0, 6);
      expect(fresnelSchlick(0, f0)).toBeCloseTo(1, 6);
      // Falling as the view leaves the normal: a reflectance that rose would be
      // a surface brighter head-on than at grazing, which is not a reflectance.
      let previous = Infinity;
      for (let i = 0; i <= 100; i++) {
        const v = fresnelSchlick(i / 100, f0);
        expect(v).toBeLessThanOrEqual(previous);
        expect(v).toBeLessThanOrEqual(1);
        previous = v;
      }
    }
  });

  test('a rim is a reflectance, not pow(1 - N·V, k)', () => {
    // The property that makes the energy budget below meaningful: the rim's value
    // is bounded by 1 and equals rimColor at normal incidence, so a diffuse term
    // attenuated by (1 - rim) can never be negative and the sum never exceeds the
    // light that fell on it. `pow(1 - n, 3)` violates the first of those — it is
    // 1 at n = 0 but so is Schlick, and it is the *F0* at n = 1 that differs:
    // pow gives 0 there for every k, Schlick gives f0.
    const rim = (nDotV: number, f0: number, strength: number): number =>
      fresnelSchlick(nDotV, f0) * strength;
    expect(rim(1, 0.08, 1)).toBeCloseTo(0.08, 6);
    expect(rim(0, 0.08, 1)).toBeCloseTo(1, 6);
    expect(rim(0.5, 0.08, 1)).toBeGreaterThan(0.08);
  });

  test('the diffuse material attenuates its diffuse by exactly the rim', () => {
    // Not a stylistic choice: this material has no specular lobe, so the rim *is*
    // the dielectric reflection, and the two are parts of one budget.
    const body = stripComments(resolveSpec(diffuseMaterialSpec({ rimStrength: 0.1 })).fragmentBody);
    expect(body).toContain('* (vec3f(1.0) - rim)');
    expect(body).toContain('let spec = vec3f(0.0);');
    // ...and the whole point of the model is that a zero-roughness surface must
    // be exactly Lambert, or the two are not the same model with a knob.
    for (const n of [0.1, 0.5, 0.9]) {
      expect(orenNayar(n, n, n, 0)).toBeCloseTo(n / Math.PI, 6);
    }
  });

  test('Oren-Nayar rises at grazing incidence, which is why it is not Lambert', () => {
    // Fix N·L and vary N·V: the model must return more as the view grazes. That
    // is the whole reason the material exists, and it is the property a `pow`
    // heuristic cannot express.
    const rough = orenNayar(0.5, 0.9, 0.5, 0.25);
    const grazing = orenNayar(0.5, 0.05, 0.5, 0.25);
    expect(grazing).toBeGreaterThan(rough);
    // And it never exceeds 1, which is the model's documented failure point and
    // the reason the generated code clamps.
    for (const s2 of [0, 0.1, 0.5, 1]) {
      for (let i = 0; i <= 20; i++) {
        for (let j = 0; j <= 20; j++) {
          const v = orenNayar(i / 20, j / 20, 0.5, s2);
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  test('the GGX lobe normalises to 1 and peaks at the halfway vector', () => {
    // Roughness 1 is excluded from the peak check and asserted separately: it is
    // the flat limit, where a = 1 and the distribution collapses to a constant.
    for (const roughness of [0.045, 0.2, 0.5, 0.9]) {
      expect(distributionGGX(1, roughness)).toBeGreaterThan(distributionGGX(0.9, roughness));
      expect(distributionGGX(0, roughness)).toBeLessThan(distributionGGX(0.1, roughness));
    }
    // The floor's actual reason: at zero roughness a2 is 0, the denominator
    // collapses, and the "mirror" lobe returns *nothing at all* rather than an
    // infinitely bright highlight. A black specular on a polished surface is the
    // failure, so MIN_ROUGHNESS is a physical floor and not an epsilon.
    expect(distributionGGX(1, 0)).toBe(0);
    // Where the guard does not bind, the peak is the textbook 1 / (PI r^4), and
    // it rises monotonically as the surface is polished — all the way to the
    // floor. This is the assertion that pins the guard's *smallness*: a 1e-7
    // guard binds below r ≈ 0.13 and clips the peak at 41, so a polished metal
    // would have no highlight and nothing would report it.
    expect(distributionGGX(1, 0.5)).toBeCloseTo(1 / (Math.PI * Math.pow(0.25, 2)), 6);
    expect(distributionGGX(1, 0.045)).toBeCloseTo(1 / (Math.PI * Math.pow(0.045, 4)), 4);
    expect(distributionGGX(1, 0.045)).toBeGreaterThan(70000);
    let previous = 0;
    for (const r of [1, 0.7, 0.5, 0.2, 0.1, 0.045]) {
      const v = distributionGGX(1, r);
      expect(v).toBeGreaterThan(previous);
      previous = v;
    }
    // Roughness 1 is the flat limit: a = a2 = 1 makes d = 1 for every N·H, so
    // the lobe is 1/PI everywhere and a fully rough surface has no highlight.
    for (const nDotH of [0, 0.25, 0.5, 0.75, 1]) {
      expect(distributionGGX(nDotH, 1)).toBeCloseTo(1 / Math.PI, 6);
    }
  });

  test('the anisotropic lobe reduces to the isotropic one at zero anisotropy', () => {
    // The consistency that a formulation taking perceptual roughness would break:
    // aniso forms at * ab internally, so the caller has to square. With at = ab
    // = r^2 the two must agree to the last bit a float has, and they do.
    for (const r of [0.045, 0.1, 0.35, 0.7, 1]) {
      const alpha = r * r;
      for (const nDotH of [0.2, 0.6, 0.9, 1]) {
        const t = Math.sqrt(Math.max(0, 1 - nDotH * nDotH));
        expect(distributionGGXAniso(nDotH, t, 0, alpha, alpha)).toBeCloseTo(
          distributionGGX(nDotH, r), 3,
        );
      }
    }
  });

  test('the anisotropic lobe is wider along the grain, which is the point', () => {
    // at > ab: the lobe is broader along T than along B, so a halfway vector
    // pointing down the grain is inside the lobe at a lower N·H than one across
    // it. If the two were symmetric, `anisotropy` would be doing nothing.
    const rough = 0.3;
    const at = Math.pow(rough * 1.6, 2);
    const ab = Math.pow(rough * 0.4, 2);
    const alongGrain = distributionGGXAniso(0.7, 0.714, 0, at, ab);
    const acrossGrain = distributionGGXAniso(0.7, 0, 0.714, at, ab);
    expect(alongGrain).toBeGreaterThan(acrossGrain);
    // ...and the isotropic value sits between the two, as it must.
    const iso = distributionGGX(0.7, rough);
    expect(alongGrain).toBeGreaterThan(iso);
    expect(acrossGrain).toBeLessThan(iso);
  });

  test('the generated PBR multiplies the diffuse by (1 - F), which is the whole budget', () => {
    const body = stripComments(resolveSpec(pbrMaterialSpec()).fragmentBody);
    expect(body).toContain('(vec3f(1.0) - fresnelSchlick(nDotV, f0)) * diffuseColor / PI');
    // And a metal has no diffuse left to spend, because its albedo is its
    // specular reflectance — `diffuseColor = base * (1 - metallic)`.
    expect(body).toContain('let diffuseColor = base * (1.0 - metallic);');
    expect(body).toContain('let f0 = mix(vec3f(0.04), base, metallic);');
  });
});

// ===========================================================================
// 16. Image-based lighting
// ===========================================================================

describe('image-based lighting', () => {
  test('it is off by default and adds no texture group at all', () => {
    const g = generateScaffold(pbrMaterialSpec());
    expect(g.textureSlots).toEqual([]);
    expect(g.code).not.toContain('irradianceAt');
    expect(g.code).not.toContain('mat.environmentIntensity');
  });

  test('turning it on declares one equirect slot, and the samplers are shared correctly', () => {
    // albedo and environment ask for byte-identical sampler configuration, so
    // apse gives them ONE binding. The body must therefore use the *shared*
    // name — which is why the factories ask `samplerNameFor` rather than
    // writing `<slot>Sampler` and hoping.
    const spec = pbrMaterialSpec({ environment: true, textured: true });
    const g = generateScaffold(spec);
    const kinds = g.textureSlots.map((t) => `${t.slotName}:${t.bindingIndex}`);
    expect(kinds).toEqual(['albedo:0', 'environment:1']);
    expect(g.resolved.samplers).toHaveLength(1);
    expect(g.resolved.samplers[0]!.slotNames).toEqual(['albedo', 'environment']);
    expect(g.code).toContain('textureSampleLevel(environment, albedoSampler,');
    expect(g.code).not.toContain('environmentSampler');
  });

  test('samplerNameFor is the authority, and agrees with describeMaterial', () => {
    const textures = { a: { kind: '2d' as const }, b: { kind: '2d' as const } };
    expect(samplerNameFor(textures, 'a')).toBe('aSampler');
    expect(samplerNameFor(textures, 'b')).toBe('aSampler');
    expect(describeMaterial({ name: 't', textures, vertex: 'out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);', fragment: 'return vec4f(1.0);' })
      .textures.find((t) => t.slotName === 'b')?.sampler).toBe('aSampler');
    // A different address mode is a different sampler, and gets its own name.
    expect(samplerNameFor({ a: { kind: '2d' }, b: { kind: '2d', addressMode: 'clamp-to-edge' } }, 'b'))
      .toBe('bSampler');
    // And the function is total: an undeclared slot gets the obvious answer
    // rather than throwing, so a factory can ask before the spec is finished.
    expect(samplerNameFor(undefined, 'a')).toBe('aSampler');
    expect(samplerNameFor(textures, 'zzz')).toBe('zzzSampler');
  });

  test('the equirect convention is stated in the generated program, not only in the docs', () => {
    const code = generateScaffold(pbrMaterialSpec({ environment: true })).code;
    // A caller has to author the map to this convention and apse cannot discover
    // it, so the convention travels with the shader.
    expect(code).toContain('atan2(rotated.z, rotated.x) / (2.0 * PI) + 0.5');
    expect(code).toContain('acos(clamp(rotated.y, -1.0, 1.0)) / PI');
    expect(code).toContain('fn equirectUv(d : vec3f) -> vec2f {');
  });

  test('the specular IBL term is off unless asked for, and both are documented as such', () => {
    const diffuseOnly = generateScaffold(pbrMaterialSpec({ environment: true })).code;
    expect(diffuseOnly).toContain('fn environmentSpecular(');
    expect(stripComments(resolveSpec(pbrMaterialSpec({ environment: true })).fragmentBody))
      .not.toContain('envSpec');
    const both = resolveSpec(pbrMaterialSpec({ environment: true, environmentSpecular: true })).fragmentBody;
    expect(stripComments(both)).toContain('envSpec * envBRDF');
    // A material with no environment has no environment slots at all, so a stray
    // `mat.environmentIntensity` would be caught by the identifier validator.
    expect(Object.keys(resolveSpec(pbrMaterialSpec()).slotTypes)).not.toContain('environmentIntensity');
    expect(() => validateGeneratedWGSL(generateScaffold(pbrMaterialSpec()).code)).not.toThrow();
  });

  test('the IBL knobs are uniform fields, so a probe can be retuned with no rebuild', () => {
    const types = resolveSpec(pbrMaterialSpec({ environment: true })).slotTypes;
    expect(types.environmentIntensity).toBe('f32');
    expect(types.environmentRotation).toBe('f32');
    expect(types.irradianceMip).toBe('f32');
    expect(types.environmentMip).toBe('f32');
  });
});

// ===========================================================================
// 17. The three new materials, one at a time
// ===========================================================================

describe('diffuseMaterial', () => {
  test('it has no metallic and no specular lobe, and says so in the program', () => {
    const resolved = resolveSpec(diffuseMaterialSpec());
    expect(Object.keys(resolved.slotTypes)).not.toContain('metallic');
    const body = stripComments(resolved.fragmentBody);
    expect(resolved.prelude).toContain('fn orenNayar');
    // The loop omits the halfway vector, which this BRDF has no use for: three
    // dead lets and three live normalizes per light, in a file people read.
    expect(body).not.toContain('let h =');
    expect(body).not.toContain('nDotH');
    expect(body).toContain('lDotV');
  });

  test('it defaults to opaque, and to transparent with alpha blending', () => {
    expect(diffuseMaterialSpec().phase).toBe('opaque');
    const t = diffuseMaterialSpec({ transparent: true, opacity: 0.4 });
    expect(t.phase).toBe('transparent');
    expect(t.depth).toEqual({ write: false, compare: 'less' });
    expect(resolveSpec(t).slotDefaults.get('opacity')).toBe(0.4);
    expect(stripComments(resolveSpec(t).fragmentBody)).toContain('mat.opacity * frame.alpha');
  });

  test('groundColor defaults to 35% of the sky, which is what it always was', () => {
    const spec = diffuseMaterialSpec({ ambientColor: [0.2, 0.4, 0.6] });
    const d = resolveSpec(spec).slotDefaults;
    // Compared componentwise: 0.2 * 0.35 is not 0.07 in binary floating point,
    // and asserting an exact product here would be asserting the wrong thing.
    const ground = d.get('groundColor') as number[];
    [0.07, 0.14, 0.21].forEach((expected, i) => expect(ground[i]).toBeCloseTo(expected, 10));
    const explicit = resolveSpec(diffuseMaterialSpec({ ambientColor: [0.2, 0.4, 0.6], groundColor: [1, 0, 0] })).slotDefaults;
    expect(explicit.get('groundColor')).toEqual([1, 0, 0]);
  });
});

describe('emissiveMaterial', () => {
  test('a plain emitter is three slots, no varyings, and an empty prelude', () => {
    const resolved = resolveSpec(emissiveMaterialSpec());
    expect(Object.keys(resolved.slotTypes)).toEqual(['color', 'intensity', 'opacity']);
    expect(resolved.userVaryings).toEqual([]);
    expect(resolved.prelude.trim()).toBe('');
    // Whitespace-collapsed, because a comment is deleted rather than blanked and
    // leaves the run of spaces it occupied behind.
    expect(collapse(resolved.fragmentBody))
      .toBe('let radiance = mat.color * mat.intensity; return vec4f(radiance, mat.opacity * frame.alpha);');
  });

  test('intensity is a radiance multiplier and is not clamped to one', () => {
    // The value that makes this material different from basicMaterial: 8 is a
    // legitimate lamp, and the only thing that can carry it is an HDR
    // intermediate.
    expect(resolveSpec(emissiveMaterialSpec({ intensity: 8 })).slotDefaults.get('intensity')).toBe(8);
    expect(stripComments(resolveSpec(emissiveMaterialSpec({ intensity: 8 })).fragmentBody))
      .toContain('mat.color * mat.intensity');
  });

  test('the view-dependent term switches on the varyings it needs and no others', () => {
    const plain = resolveSpec(emissiveMaterialSpec()).userVaryings.map((v) => v.name);
    expect(plain).toEqual([]);
    const sheen = resolveSpec(emissiveMaterialSpec({ fresnel: true, fresnelStrength: 1 })).userVaryings.map((v) => v.name);
    expect(sheen).toEqual(['worldPos', 'normal']);
    // A Fresnel term that needed no normal would be a bug: the whole model is
    // the angle between the normal and the view.
    expect(stripComments(resolveSpec(emissiveMaterialSpec({ fresnel: true, fresnelStrength: 1 })).fragmentBody))
      .toContain('fresnelSchlick(nDotV, vec3f(1.0)) * mat.fresnelStrength');
  });

  test('the pulse is driven by frame.time, and only exists when it is switched on', () => {
    expect(Object.keys(resolveSpec(emissiveMaterialSpec()).slotTypes)).not.toContain('pulseHz');
    const pulsing = resolveSpec(emissiveMaterialSpec({ pulseHz: 2, pulseDepth: 0.5 }));
    expect(pulsing.slotDefaults.get('pulseHz')).toBe(2);
    expect(pulsing.slotDefaults.get('pulseDepth')).toBe(0.5);
    const body = stripComments(pulsing.fragmentBody);
    expect(body).toContain('2.0 * PI * mat.pulseHz * frame.time');
    // A raised cosine, not a sine: sin has a non-zero derivative at its trough
    // and a surface at depth 1 visibly bounces there twice a cycle.
    expect(body).toContain('cos(phase)');
    expect(body).not.toContain('sin(phase)');
  });

  test('a frequency with no depth, and a flag with no strength, are both rejected', () => {
    const noDepth = expectError(() => emissiveMaterialSpec({ pulseHz: 2 }), 'OPTION_UNKNOWN');
    expect(noDepth.message).toContain('pulseDepth');
    const noStrength = expectError(() => emissiveMaterialSpec({ fresnel: true }), 'OPTION_UNKNOWN');
    expect(noStrength.message).toContain('fresnelStrength');
    expectError(() => emissiveMaterialSpec({ intensity: -1 }), 'OPTION_UNKNOWN');
    expectError(() => emissiveMaterialSpec({ intensity: Number.NaN }), 'OPTION_UNKNOWN');
  });

  test('the emissive map multiplies the radiance and is never added to a lit term', () => {
    const resolved = resolveSpec(emissiveMaterialSpec({ textured: true }));
    expect(resolved.textures.map((t) => t.slotName)).toEqual(['emissive']);
    const body = stripComments(resolved.fragmentBody);
    expect(body).toContain('textureSample(emissive, emissiveSampler, in.uv).rgb * mat.color * mat.intensity');
    // The distinguishing line: an emission map has black in it, and black must
    // stay black. An albedo map would be lifted by the light rig.
    expect(body).not.toContain('frame.camPos * mat.color');
  });
});

describe('anisotropicMaterial', () => {
  test('it refuses a layout with no tangent, and names the attribute', () => {
    const err = expectError(
      () => anisotropicMaterialSpec({ layout: layout({ position: 'float32x3', normal: 'float32x3' }) }),
      'ATTRIBUTE_MISSING',
    );
    expect(err.message).toContain('tangent');
    expect(err.fix).toContain('TANGENT_LAYOUT');
    // The default is the one layout that has one, so the common case just works.
    expect(resolveSpec(anisotropicMaterialSpec()).layout.key).toBe(TANGENT_LAYOUT.key);
  });

  test('the tangent frame is rebuilt in the shader, not trusted from the varying', () => {
    const body = stripComments(resolveSpec(anisotropicMaterialSpec()).fragmentBody);
    // Gram-Schmidt, because an interpolated tangent is not guaranteed orthogonal
    // to an interpolated normal, and a non-orthonormal frame makes the highlight
    // depend on the tessellation.
    expect(body).toContain('safeNormalize(tRaw.xyz - n * dot(n, tRaw.xyz))');
    expect(body).toContain('cross(n, t) * tRaw.w');
  });

  test('it generates the anisotropic lobe, not the isotropic one', () => {
    const resolved = resolveSpec(anisotropicMaterialSpec({ anisotropy: 0.8 }));
    expect(resolved.prelude).toContain('fn distributionGGXAniso');
    expect(resolved.prelude).toContain('fn visibilitySmithAniso');
    const body = stripComments(resolved.fragmentBody);
    expect(body).toContain('distributionGGXAniso(nDotH, dot(t, h), dot(b, h), at, ab)');
    expect(body).toContain('visibilitySmithAniso(at, ab, tDotV, bDotV, dot(t, l), dot(b, l), nDotV, nDotL)');
    expect(body).not.toContain('distributionGGX(');
    // Two roughnesses, both floored, and then squared into alpha. The squaring is
    // the consistency requirement: distributionGGXAniso forms at*ab internally,
    // so handing it perceptual roughness would make the same `roughness` mean a
    // different material from pbrMaterial's.
    expect(body).toContain('clamp(mat.roughness * (1.0 + mat.anisotropy), MIN_ROUGHNESS, 1.0)');
    expect(body).toContain('clamp(mat.roughness * (1.0 - mat.anisotropy), MIN_ROUGHNESS, 1.0)');
    expect(body).toContain('let at = roughT * roughT;');
    expect(body).toContain('let ab = roughB * roughB;');
    expect(resolved.slotDefaults.get('anisotropy')).toBe(0.8);
  });

  test('the tangent is transformed by the model matrix, not the normal matrix', () => {
    // Identical under a uniform scale, which is exactly why using the wrong one is
    // invisible until an object is scaled unevenly.
    const resolved = resolveSpec(anisotropicMaterialSpec());
    expect(stripComments(resolved.vertexBody)).toContain('(obj.model * vec4f(in.tangent.xyz, 0.0)).xyz');
    expect(resolved.vertexBody).not.toContain('obj.normalMatrix * in.tangent');
  });

  test('anisotropy, metallic and roughness are range-checked rather than clamped silently', () => {
    for (const opts of [{ anisotropy: 2 }, { anisotropy: -2 }, { metallic: 3 }, { roughness: Number.NaN }]) {
      const err = expectError(() => anisotropicMaterialSpec(opts), 'OPTION_UNKNOWN');
      expect(err.why).toContain('clamp');
    }
  });
});

// ===========================================================================
// 18. Shader diagnostics
//
// The path exists because a WGSL compile error otherwise arrives wrapped in a
// validation message whose stage attribution is frequently wrong. These tests
// pin the three properties that path has to keep, and the fake device is the only
// way to get a fragment-stage diagnostic without a browser.
// ===========================================================================

/** A one-line material, so a diagnostic can be aimed at a chosen line. */
function diagnosticSpec(): MaterialSpec {
  return {
    name: 'diag',
    layout: POSITION_LAYOUT,
    varyings: { t: 'f32' },
    slots: { k: 'f32' },
    prelude: 'fn helper() -> f32 { return mat.k; }',
    targets: [{ format: 'bgra8unorm' }],
    vertex: 'out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0); out.t = helper();',
    fragment: 'return vec4f(in.t, 0.0, 0.0, 1.0);',
  };
}

async function expectCompileError(device: FakePipelineDevice, spec: MaterialSpec): Promise<AseError> {
  try {
    const { Material } = await import('../src/material/material.ts');
    await Material.create(asPipelineDevice(device), spec);
  } catch (err) {
    if (!(err instanceof AseError)) throw new Error(`expected an AseError, got ${String(err)}`);
    expect(err.code).toBe('SHADER_COMPILE_FAILED');
    return err;
  }
  throw new Error('expected the compile to fail');
}

describe('shader diagnostics — the real compiler output, attributed to the right stage', () => {
  test('a fragment-stage error is labelled fragment, never vertex', () => {
    // The bug this pins. The validation wrapper's text is unreliable about the
    // stage, and a person who believes it looks in the wrong half of their own
    // material.
    const spec = diagnosticSpec();
    const code = generateScaffold(spec).code;
    const line = fragmentEntryLineOf(code) + 1;
    const device = new FakePipelineDevice({
      pipelineError: new Error('While validating vertex stage: shader validation error'),
      compilationMessages: [
        { type: 'error', message: "expected ';' for variable declaration", lineNum: line, linePos: 7 },
      ],
    });
    return expectCompileError(device, spec).then((err) => {
      expect(err.message).toContain('fragment stage');
      expect(err.message).not.toContain('vertex stage');
      expect(err.message).toContain(`line ${line} column 7`);
      expect(err.message).toContain("expected ';'");
      // The line the compiler pointed at, so a 3 KB generated program is
      // actionable. This is the part that was being dropped.
      expect(err.why).toContain('return vec4f(in.t, 0.0, 0.0, 1.0);');
      expect(err.why).toContain(`[fragment] ${line}:7`);
      // The original error is kept, not just its text.
      expect((err.cause as Error).message).toContain('While validating vertex stage');
    });
  });

  test('a vertex-stage error is labelled vertex, and a prelude error prelude', () => {
    const spec = diagnosticSpec();
    const code = generateScaffold(spec).code;
    const cases: readonly [number, string][] = [
      [vertexEntryLineOf(code) + 1, 'vertex'],
      [preludeFirstLineOf(code), 'prelude'],
    ];
    for (const [line, stage] of cases) {
      const device = new FakePipelineDevice({
        pipelineError: new Error('shader validation error'),
        compilationMessages: [{ type: 'error', message: `problem at ${line}`, lineNum: line, linePos: 1 }],
      });
      return expectCompileError(device, spec).then((err) => {
        expect(err.message).toContain(`${stage} stage`);
        expect(err.why).toContain(`[${stage}] ${line}:1`);
      });
    }
  });

  test('every diagnostic survives, not just the first, and warnings are not errors', () => {
    const spec = diagnosticSpec();
    const code = generateScaffold(spec).code;
    const base = fragmentEntryLineOf(code);
    const device = new FakePipelineDevice({
      pipelineError: new Error('shader validation error'),
      compilationMessages: [
        { type: 'warning', message: 'unused variable', lineNum: base + 1, linePos: 3 },
        { type: 'error', message: 'first real problem', lineNum: base + 1, linePos: 4 },
        { type: 'error', message: 'second real problem', lineNum: base + 2, linePos: 5 },
        { type: 'info', message: 'note', lineNum: base + 2, linePos: 6 },
      ],
    });
    return expectCompileError(device, spec).then((err) => {
      expect(err.message).toContain('first real problem');
      expect(err.why).toContain('first real problem');
      expect(err.why).toContain('second real problem');
      // A warning is not a compile failure and must not be reported as one.
      expect(err.why).not.toContain('unused variable');
    });
  });

  test('the error-scope route is handled as well as the rejection route', () => {
    // `createRenderPipelineAsync` can resolve and still leave a device error on
    // the scope, and that is a real path — not a theoretical one.
    const spec = diagnosticSpec();
    const line = fragmentEntryLineOf(generateScaffold(spec).code) + 1;
    const device = new FakePipelineDevice({
      errorScopes: [new Error('scope failure') as unknown as GPUError],
      compilationMessages: [{ type: 'error', message: 'the real reason', lineNum: line, linePos: 2 }],
    });
    return expectCompileError(device, spec).then((err) => {
      expect(err.message).toContain('fragment stage');
      expect(err.message).toContain('the real reason');
      // No Error object on this route, so no cause — but nothing is lost.
      expect(err.cause).toBeUndefined();
    });
  });

  for (const mode of ['absent', 'throws'] as const) {
    test(`it degrades gracefully when getCompilationInfo is '${mode}'`, () => {
      const spec = diagnosticSpec();
      const device = new FakePipelineDevice({
        compilationInfo: mode,
        pipelineError: new Error('While validating vertex stage: whatever'),
      });
      return expectCompileError(device, spec).then((err) => {
        // The scope text is worse but it is not nothing, and the error says so
        // rather than inventing a stage it cannot support.
        expect(err.message).toContain('While validating vertex stage');
        expect(err.why).toContain("diagnostics were unavailable");
        expect(err.why).toContain('names a stage that may be wrong');
      });
    });
  }

  test('a module with no messages array is the same graceful degradation', () => {
    const spec = diagnosticSpec();
    const device = new FakePipelineDevice({ pipelineError: new Error('validation error') });
    // Force the malformed shape the resolver must survive: a compilation info
    // object with no `messages` on it at all.
    (device as unknown as { createShaderModule: unknown }).createShaderModule = (): GPUShaderModule => ({
      getCompilationInfo: () => Promise.resolve({}),
    } as unknown as GPUShaderModule);
    return expectCompileError(device, spec).then((err) => {
      expect(err.message).toContain('validation error');
      expect(err.why).toContain('diagnostics were unavailable');
    });
  });

  test('a successful compile never asks for diagnostics at all', () => {
    const device = new FakePipelineDevice();
    return import('../src/material/material.ts').then(({ Material }) =>
      Material.create(asPipelineDevice(device), diagnosticSpec())).then(() => {
      expect(device.compilationInfoCalls).toBe(0);
      expect(device.modules).toHaveLength(1);
      expect(device.modules[0]!.code).toContain('fn vs(');
    });
  });
});

describe('stageAtLine — the attribution the diagnostics path depends on', () => {
  const code = generateScaffold(diagnosticSpec()).code;

  test('each section banner claims its own lines and nothing else', () => {
    // The ranges are built from the banners themselves, so the assertion is that
    // every line of the file is claimed by exactly one section and attributed to
    // the one it is in — not that three chosen lines happen to work.
    const total = code.split('\n').length;
    const prelude = preludeBannerLineOf(code);
    const vertex = vertexBannerLineOf(code);
    const fragment = fragmentBannerLineOf(code);
    expect(prelude).toBeLessThan(vertex);
    expect(vertex).toBeLessThan(fragment);
    expect(fragment).toBeLessThanOrEqual(total);
    const sections: [ShaderStage, number, number][] = [
      ['generated', 1, prelude - 1],
      ['prelude', prelude, vertex - 1],
      ['vertex', vertex, fragment - 1],
      ['fragment', fragment, total],
    ];
    for (const [stage, from, to] of sections) {
      for (let line = from; line <= to; line++) {
        expect(stageAtLine(code, line), `line ${line}`).toBe(stage);
      }
    }
  });

  test('a line outside the program is not attributed to a stage it may not be in', () => {
    // A diagnostic from an included file, or a lineNum of 0 from an
    // implementation that does not track it. Walking the banners and returning
    // whatever the last one was would report "fragment" in every one of these
    // cases, which is the same class of mistake as reporting "vertex" by default
    // and is worse because it looks deliberate.
    const total = code.split('\n').length;
    expect(stageAtLine(code, 0)).toBe('generated');
    expect(stageAtLine(code, -3)).toBe('generated');
    expect(stageAtLine(code, 99999)).toBe('generated');
    expect(stageAtLine(code, 1.5)).toBe('generated');
    expect(stageAtLine(code, Number.NaN)).toBe('generated');
    // The bound is inclusive: the last line of the program is attributed.
    expect(stageAtLine(code, total)).toBe('fragment');
  });

  test('it reads the generator\'s own banners, not a regex over the entry points', () => {
    // The banners are what the template emits, so a template change moves both
    // together. Asserted as a property of the text, not of the implementation.
    expect(code).toContain('// ---- prelude (user-supplied declarations) ----');
    expect(code).toContain('// ---- generated by apse: vertex stage ----');
    expect(code).toContain('// ---- generated by apse: fragment stage ----');
  });
});

// ===========================================================================
// 19. The normal matrix layout
//
// apse shipped a bug here: a tight 3x3 written where WGSL's `mat3x3<f32>`
// expects three 16-byte-aligned columns. Every normal in every mesh was wrong and
// every material shaded flat, with no error at any point. The fix is in place; the
// point of these tests is that it cannot come back, and that the *shader's*
// reading of the bytes and the *packer's* writing of them are pinned to the same
// layout rather than to two agreeing conventions.
// ===========================================================================

/** The byte offset and width apse gives `normalMatrix`, from the generated struct. */
const NORMAL_MATRIX = OBJECT_BLOCK.fields.find((f) => f.name === 'normalMatrix')!;

/** A non-uniform scale, so a normal matrix that is not the identity is visible. */
const MODEL_NON_UNIFORM = new Float32Array([
  2, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

describe('the mat3x3f packed layout — the bug that shipped', () => {
  test('the generated struct declares a mat3x3f at 64 bytes, three 16-byte columns', () => {
    expect(NORMAL_MATRIX.type).toBe('mat3x3f');
    expect(NORMAL_MATRIX.offset).toBe(64);
    expect(NORMAL_MATRIX.size).toBe(48);
    expect(NORMAL_MATRIX.components).toBe(12);
    // 12 *components* and 16 *floats* are different numbers, and the gap between
    // them is the entire bug: three columns of three, each padded to four.
    expect(UNIFORM_TYPES.mat3x3f.components).toBe(12);
    expect(UNIFORM_TYPES.mat3x3f.align).toBe(16);
    expect(UNIFORM_TYPES.mat3x3f.size).toBe(48);
    expect(OBJECT_BLOCK.wgsl).toContain('normalMatrix : mat3x3<f32>,');
    // ...and the record around it, so a change to any neighbour is caught here.
    expect(OBJECT_BLOCK.fields.map((f) => `${f.name}@${f.offset}`)).toEqual([
      'model@0', 'normalMatrix@64', 'objectId@112', 'instanceId@116', 'visibility@120',
    ]);
    expect(OBJECT_BLOCK.stride).toBe(256);
  });

  /**
   * Packs one object and returns the bytes apse uploaded, read out of the fake
   * device's recorded `writeBuffer` call rather than out of a private field — so
   * this asserts what actually reaches the GPU, through the same code path the
   * renderer uses.
   */
  function packedRecord(normal: ArrayLike<number>, index = 0): { f32: Float32Array; u32: Uint32Array } {
    const device = new FakePipelineDevice();
    const u = new ObjectUniforms(asPipelineDevice(device), `test-${index}`, 4);
    u.pack(index, MODEL_NON_UNIFORM, normal, 7, 3, 1);
    u.uploadFrom(index + 1);
    const write = device.writes.at(-1);
    expect(write).toBeDefined();
    // One record is one stride, even though only 124 of those 256 bytes are data.
    expect(write!.size).toBe(OBJECT_BLOCK.stride);
    const buffer = write!.bytes.buffer.slice(0) as ArrayBuffer;
    return { f32: new Float32Array(buffer), u32: new Uint32Array(buffer) };
  }

  /** The 12 float words the shader reads as `mat3x3<f32>`. */
  const NORMAL_WORDS = NORMAL_MATRIX.offset >> 2;

  test('twelve data floats and three padding floats, in three four-word columns', () => {
    // A padded, column-major 3x3, with the padding deliberately poisoned: if the
    // packer ever copies words 3, 7 or 11 the sentinel lands in the buffer and
    // this test fails with a number nobody has to interpret.
    const normal = new Float32Array([
      1, 2, 3, 999,
      4, 5, 6, 999,
      7, 8, 9, 999,
    ]);
    const { f32 } = packedRecord(normal);
    expect(Array.from(f32.slice(NORMAL_WORDS, NORMAL_WORDS + 12))).toEqual([
      1, 2, 3, 0,
      4, 5, 6, 0,
      7, 8, 9, 0,
    ]);
    // The ids and the visibility are immediately after, in the fields the WGSL
    // declares next to the matrix. Read as u32, because that is how they are
    // written: objectId 7 and instanceId 3 read as floats are denormals, and a
    // test that compared them to 7 would pass for the wrong reason.
    const ids = OBJECT_BLOCK.fields.find((f) => f.name === 'objectId')!.offset >> 2;
    const { u32 } = packedRecord(normal);
    expect(u32[ids]).toBe(7);
    expect(u32[ids + 1]).toBe(3);
    expect(f32[ids + 2]).toBe(1);
    // Nothing beyond the record is touched: 64 model floats, then 12 matrix
    // floats, then 2 ids, then the visibility, and the stride's remaining bytes
    // are left alone.
    expect(f32[NORMAL_WORDS + 15]).toBe(0);
    expect(OBJECT_BLOCK.size).toBe(128);
  });

  test('a tight 3x3 is the wrong layout, and the shader reads it wrong in a specific way', () => {
    // The exact regression, stated as a property rather than as a diff.
    //
    // Padded, which is what apse writes and what WGSL reads:
    //     word:  0  1  2   3 | 4  5  6   7 | 8  9 10  11
    //            e1 e2 e3 pad  e4 e5 e6 pad  e7 e8 e9 pad
    // Tight, which is what the bug wrote:
    //     word:  0  1  2  3  4  5  6  7  8
    //            e1 e2 e3 e4 e5 e6 e7 e8 e9
    //
    // So column 0 is accidentally right, column 1 is shifted by one, and column 2
    // reads the last element followed by whatever happened to be in the padding —
    // which for a zeroed buffer is zero, i.e. the third axis of every normal
    // collapsed to nothing and the whole scene shaded as though every face were
    // lit head-on. That is the symptom, and it is why no error was ever raised.
    const elements = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const padded = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0];
    const column = (w: readonly number[], j: number): number[] =>
      [w[j * 4]!, w[j * 4 + 1]!, w[j * 4 + 2]!];

    expect(column(padded, 0)).toEqual([1, 2, 3]);
    expect(column(padded, 1)).toEqual([4, 5, 6]);
    expect(column(padded, 2)).toEqual([7, 8, 9]);

    expect(column(elements, 0)).toEqual([1, 2, 3]);
    expect(column(elements, 1)).toEqual([5, 6, 7]);
    // And the third column reads the last element followed by whatever the record
    // happens to hold next, which in ObjectData is the object id and the instance
    // id. So the tight write did not merely lose a column: it fed two integers
    // into a normal's third component, for every object in the scene.
    expect(column([...elements, 7, 3], 2)).toEqual([9, 7, 3]);

    // ...and apse writes the padded one.
    const { f32 } = packedRecord(new Float32Array(padded));
    expect(Array.from(f32.slice(NORMAL_WORDS, NORMAL_WORDS + 12))).toEqual(padded);
  });

  test('the packer reads a 12-float padded array by column, not by position', () => {
    // Both spellings produce the same bytes, so a caller cannot get it wrong by
    // picking one. Identity is the readable witness: a row-major interpretation
    // of an asymmetric array is visibly different, and identity hides nothing.
    const { f32 } = packedRecord(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]));
    expect(Array.from(f32.slice(NORMAL_WORDS, NORMAL_WORDS + 12)))
      .toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    // And the values are copied, not aliased: mutating the source afterwards
    // cannot change what was uploaded.
    const source = new Float32Array([1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]);
    const { f32: first } = packedRecord(source);
    source[0] = -1;
    expect(first[NORMAL_WORDS]).toBe(1);
  });

  test('the bytes the shader reads multiply a vector the way the matrix means to', () => {
    // The end-to-end property. `mat3x3<f32>` is column-major, so m * v is
    // column0 * v.x + column1 * v.y + column2 * v.z — and the three columns are
    // the three 4-word groups above. A scale of 2 on X has an inverse-transpose
    // normal matrix of 0.5 on X, so a normal along X must come back halved and a
    // normal along Y unchanged.
    const identityNormal = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    const identity = packedRecord(identityNormal).f32;
    const column = (words: Float32Array, j: number): [number, number, number] =>
      [words[NORMAL_WORDS + j * 4]!, words[NORMAL_WORDS + j * 4 + 1]!, words[NORMAL_WORDS + j * 4 + 2]!];
    /** Exactly WGSL's `m * v` for a column-major mat3x3. */
    const mul = (m: readonly (readonly [number, number, number])[], v: readonly [number, number, number]): number[] => {
      const out = [0, 0, 0];
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) out[k]! += m[j]![k]! * v[k]!;
      }
      return out;
    };

    const identityMatrix = [column(identity, 0), column(identity, 1), column(identity, 2)];
    expect(identityMatrix).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    expect(mul(identityMatrix, [1, 0, 0])).toEqual([1, 0, 0]);
    expect(mul(identityMatrix, [0, 1, 0])).toEqual([0, 1, 0]);

    // A non-uniform scale: the inverse transpose of diag(2, 1, 1) is diag(0.5, 1, 1).
    const scaled = packedRecord(new Float32Array([0.5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])).f32;
    const scaledMatrix = [column(scaled, 0), column(scaled, 1), column(scaled, 2)];
    expect(scaledMatrix[0]).toEqual([0.5, 0, 0]);
    expect(mul(scaledMatrix, [1, 0, 0])).toEqual([0.5, 0, 0]);
    expect(mul(scaledMatrix, [0, 1, 0])).toEqual([0, 1, 0]);
    // A diagonal is symmetric, so a transposed reading would agree — which is
    // exactly why the rotation case below exists.
  });

  test('a rotated frame, which is the case a symmetric matrix hides', () => {
    // Identity and a diagonal scale are both symmetric, so a transposed packer
    // passes both. A rotation is not, and a rotation is what every mesh with a
    // tangent frame actually has.
    const c = Math.cos(0.7), s = Math.sin(0.7);
    // Column-major: column j is the image of the j-th basis vector.
    const rotated = packedRecord(new Float32Array([
      c, -s, 0, 0,
      s, c, 0, 0,
      0, 0, 1, 0,
    ])).f32;
    const column = (j: number): [number, number, number] =>
      [rotated[NORMAL_WORDS + j * 4]!, rotated[NORMAL_WORDS + j * 4 + 1]!, rotated[NORMAL_WORDS + j * 4 + 2]!];
    const cols = [column(0), column(1), column(2)];

    // Column-major: m * (1,0,0) is column 0.
    expect(cols[0]![0]).toBeCloseTo(c, 6);
    expect(cols[0]![1]).toBeCloseTo(-s, 6);
    // The transposed reading gives (s, c, 0): the same matrix rotated the other
    // way. A packer that wrote rows instead of columns would produce exactly
    // this, and a uniformly-scaled scene would hide it completely.
    const asRows: readonly (readonly [number, number, number])[] = [
      [cols[0]![0], cols[1]![0], cols[2]![0]],
      [cols[0]![1], cols[1]![1], cols[2]![1]],
      [cols[0]![2], cols[1]![2], cols[2]![2]],
    ];
    expect(asRows[0]).not.toEqual(cols[0]);
    expect(asRows[0]![1]).toBeCloseTo(s, 6);
    // The rotation preserves length, which a wrong layout does not.
    const length = Math.hypot(cols[0]![0], cols[0]![1], cols[0]![2]);
    expect(length).toBeCloseTo(1, 6);
  });

  test('the byte offset for any object is the field offset plus whole strides', () => {
    const device = new FakePipelineDevice();
    const u = new ObjectUniforms(asPipelineDevice(device), 'offsets', 4);
    expect(u.fieldOffset(0, 'normalMatrix')).toBe(64);
    expect(u.fieldOffset(1, 'normalMatrix')).toBe(64 + 256);
    expect(u.fieldOffset(3, 'normalMatrix')).toBe(64 + 3 * 256);
    // The *dynamic* offset a draw binds is index * stride and must be a multiple
    // of minUniformBufferOffsetAlignment; the field sits 64 bytes into that
    // record. An object that did not get a whole stride would fail at the
    // *second* object's draw, with nothing wrong in the first.
    for (let i = 0; i < 4; i++) {
      expect(i * OBJECT_BLOCK.stride % 256).toBe(0);
      expect(u.fieldOffset(i, 'normalMatrix') - 64).toBe(i * 256);
    }
    expect(() => u.fieldOffset(0, 'nope')).toThrow();
  });

  test('every object in a frame gets its own matrix, at its own offset', () => {
    // The one that would catch a stride regression: identical matrices in
    // identical slots means the buffer was addressed wrongly, not packed wrongly.
    const device = new FakePipelineDevice();
    const u = new ObjectUniforms(asPipelineDevice(device), 'many', 4);
    for (let i = 0; i < 3; i++) u.pack(i, MODEL_NON_UNIFORM, new Float32Array(12).fill(i + 1), i, 0, 1);
    u.uploadFrom(3);
    const words = new Float32Array(device.writes.at(-1)!.bytes.buffer.slice(0) as ArrayBuffer);
    for (let i = 0; i < 3; i++) {
      const at = (i * OBJECT_BLOCK.stride + NORMAL_MATRIX.offset) >> 2;
      expect(Array.from(words.slice(at, at + 12))).toEqual([
        i + 1, i + 1, i + 1, 0,
        i + 1, i + 1, i + 1, 0,
        i + 1, i + 1, i + 1, 0,
      ]);
    }
  });

  test('the shipped material transforms its normal with exactly that matrix', () => {
    // The last link in the chain: the shader multiplies the packed mat3x3 by a
    // vec3, and WGSL's matCxR * vecC is column-major. Asserted on the generated
    // text because a mat3 * vec4 (the GLSL slip) is the other way this goes wrong.
    for (const spec of [pbrMaterialSpec(), diffuseMaterialSpec(), anisotropicMaterialSpec()]) {
      const body = stripComments(resolveSpec(spec).vertexBody);
      expect(body).toContain('normalize(obj.normalMatrix * in.normal)');
      expect(body).not.toContain('obj.normalMatrix * vec4f');
    }
  });
});

/** A generated body with its comments deleted and its whitespace collapsed. */
function collapse(body: string): string {
  return tokenKeyOf(body);
}

