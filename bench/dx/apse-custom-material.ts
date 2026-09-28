/**
 * bench/dx — the apse side of the authoring-cost comparison.
 *
 * The material under test is a two-light Lambert + Blinn-Phong surface with an
 * albedo texture, a per-material tint, and a camera-relative fresnel rim. The
 * same shading model, with the same numbers, is written in GLSL in
 * `three-custom-material.js`; the two images are compared pixel by pixel.
 *
 * Two rules govern this file, and both exist so the line counts in
 * `bench/dx/README.md` are mechanical rather than asserted:
 *
 *   1. Everything inside a `// @count:<section>` … `// @count:end` block is the
 *      material code, and `run.ts` counts those lines by reading this file.
 *      Everything outside them is scene setup or test scaffolding and is
 *      reported separately.
 *   2. `PARAMS` and `ALBEDO_BYTES` live here and are imported by the three.js
 *      file, so both shaders read bit-identical inputs. If the two libraries
 *      disagree it is the shader plumbing, not the constants.
 *
 * One line in here is a workaround for a real defect, and it is called out
 * where it appears: see `out.normalW` in DX_VERTEX.
 */

import { Material, generateScaffold, describeMaterial } from '../../src/index.ts';
import type { MaterialSpec } from '../../src/index.ts';

// ---------------------------------------------------------------------------
// Shared inputs. Imported by the three.js version; see the note above.
// ---------------------------------------------------------------------------

export const PARAMS = {
  fov: 45,
  near: 0.1,
  far: 100,
  eye: [0, 0, 3.2] as const,
  target: [0, 0, 0] as const,
  tint: [0.85, 0.42, 0.16] as const,
  lights: [
    { dir: [0.55, 0.72, 0.42] as const, color: [1.30, 1.22, 1.05] as const },
    { dir: [-0.62, 0.18, 0.35] as const, color: [0.22, 0.34, 0.60] as const },
  ],
  shininess: 24,
  specColor: [1.0, 0.95, 0.85] as const,
  rim: { power: 3, color: [0.15, 0.55, 0.95] as const },
  /** 1/PI, spelled the same on both sides so the two shaders agree bit for bit. */
  lambert: 0.3183098861837907,
} as const;

/** 4x4 RGBA8, 64 bytes. Asymmetric, so a flipped V axis is visible. */
export const ALBEDO_SIZE = 4;
export const ALBEDO_BYTES = new Uint8Array([
  255, 40, 30, 255,   40, 255, 60, 255,   30, 90, 255, 255,  255, 200, 20, 255,
   10, 20, 30, 255,  200, 210, 220, 255,  130, 60, 240, 255,   90, 200, 150, 255,
  240, 240, 240, 255,  20, 30, 40, 255,  160, 90, 20, 255,  255, 255, 40, 255,
   70, 20, 120, 255,  120, 220, 90, 255,  15, 90, 160, 255,  200, 160, 200, 255,
]);

// ---------------------------------------------------------------------------
// The material. This is the whole of the apse side of the comparison.
// ---------------------------------------------------------------------------

// @count:declarations
/** The spec. Nine slots, three varyings, one texture. */
export const DX_DECLARATIONS = {
  name: 'dx-lambert-blinn',
  varyings: { worldPos: 'vec3f', normalW: 'vec3f', uv: 'vec2f' },
  slots: {
    tint: { type: 'vec3f', default: PARAMS.tint },
    lightDir0: { type: 'vec3f', default: PARAMS.lights[0].dir },
    lightColor0: { type: 'vec3f', default: PARAMS.lights[0].color },
    lightDir1: { type: 'vec3f', default: PARAMS.lights[1].dir },
    lightColor1: { type: 'vec3f', default: PARAMS.lights[1].color },
    specColor: { type: 'vec3f', default: PARAMS.specColor },
    rimColor: { type: 'vec3f', default: PARAMS.rim.color },
    shininess: { type: 'f32', default: PARAMS.shininess },
    rimPower: { type: 'f32', default: PARAMS.rim.power },
  },
  textures: { albedoMap: { kind: '2d' } },
  phase: 'opaque',
  topology: 'triangle-list',
  cull: 'back',
  frontFace: 'ccw',
  depth: { write: true, compare: 'less' },
  sampleCount: 1,
} as const;
// @count:end

// @count:prelude
/**
 * The prelude: the only place a material may declare a function.
 *
 * These 12 lines are cost the three.js version does not pay, because GLSL lets
 * a `ShaderMaterial` declare a helper inline. Counted honestly on both sides.
 */
export const DX_PRELUDE = `
// One light's contribution: Lambert diffuse + Blinn-Phong specular.
fn dxAccumulate(
  n : vec3f, v : vec3f, dir : vec3f, color : vec3f,
  albedo : vec3f, spec : vec3f, shininess : f32,
) -> vec3f {
  let l = normalize(dir);
  let irradiance = color * max(dot(n, l), 0.0);
  let halfDir = normalize(l + v);
  let nDotH = max(dot(n, halfDir), 0.0);
  let norm = (shininess + 2.0) / 8.0;
  return irradiance * (albedo * ${PARAMS.lambert} + norm * pow(nDotH, shininess) * spec);
}`;
// @count:end

// @count:texture
/**
 * The albedo texture. apse has no asset loaders, so the caller supplies a
 * `GPUTextureView` and this function is the whole of the "here is a texture"
 * story: create the texture, stage the rows, write them, hand back a view.
 */
export function createAlbedoView(device: GPUDevice): GPUTextureView {
  const texture = device.createTexture({
    label: 'dx:albedo',
    size: [ALBEDO_SIZE, ALBEDO_SIZE, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  // `writeTexture` also demands a 256-byte row pitch, so a 4x4 texture cannot be
  // written directly and needs a staging copy. An asset loader absorbs this;
  // apse has none, so it is the author's problem.
  const rowBytes = ALBEDO_SIZE * 4;
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const padded = new Uint8Array(bytesPerRow * ALBEDO_SIZE);
  for (let y = 0; y < ALBEDO_SIZE; y++) {
    padded.set(ALBEDO_BYTES.subarray(y * rowBytes, (y + 1) * rowBytes), y * bytesPerRow);
  }
  device.queue.writeTexture(
    { texture }, padded,
    { bytesPerRow, rowsPerImage: ALBEDO_SIZE },
    { width: ALBEDO_SIZE, height: ALBEDO_SIZE },
  );
  return texture.createView();
}
// @count:end

// @count:vertex
export const DX_VERTEX = `
out.clip     = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;
// obj.normalMatrix, not obj.model, would be the obvious choice here — and it is
// wrong in this build. Renderer packs a 9-float normal matrix while
// ObjectUniforms.pack reads a mat3x3f as three 4-float-padded columns, so the
// third column's y and z are read past the end of the array and arrive as NaN.
// Measured: see bench/dx/README.md. For a rotation-only transform, taking the
// upper 3x3 of obj.model and renormalising is exact, so that is what is used.
out.normalW  = normalize((obj.model * vec4f(in.normal, 0.0)).xyz);
out.uv       = in.uv;
`;
// @count:end

// @count:fragment
export const DX_FRAGMENT = `
let n    = normalize(in.normalW);
let v    = normalize(frame.camPos - in.worldPos);
let base = mat.tint * textureSample(albedoMap, albedoMapSampler, in.uv).rgb;
let lit  = dxAccumulate(n, v, mat.lightDir0, mat.lightColor0, base, mat.specColor, mat.shininess)
         + dxAccumulate(n, v, mat.lightDir1, mat.lightColor1, base, mat.specColor, mat.shininess);
let rim  = mat.rimColor * pow(1.0 - max(dot(n, v), 0.0), mat.rimPower);
return vec4f(lit + rim, 1.0);
`;
// @count:end

// @count:assemble
/**
 * Assembles the spec, compiles it, and binds the texture.
 *
 * `Material.create` is async because pipeline compilation must not block the
 * main thread; see the comment in `src/material/material.ts`.
 */
export const DX_SPEC: MaterialSpec = {
  ...DX_DECLARATIONS, prelude: DX_PRELUDE, vertex: DX_VERTEX, fragment: DX_FRAGMENT,
};

export async function createDxMaterial(
  device: GPUDevice,
  texture?: GPUTextureView,
): Promise<Material> {
  const material = await Material.create(device, DX_SPEC);
  if (texture !== undefined) material.setTexture('albedoMap', texture);
  return material;
}
// @count:end

// ---------------------------------------------------------------------------
// Scaffolding below this line is test harness, not material code.
// ---------------------------------------------------------------------------

/** The full generated program, for the report. */
export function dxGeneratedWGSL(): string {
  return generateScaffold(DX_SPEC).code;
}

export function dxDescribe(): unknown {
  return describeMaterial(DX_SPEC);
}

/**
 * Silent-failure probe 1: a varying that is declared but never assigned.
 *
 * apse's `validateBody` only insists that the vertex body assigns `out.clip`.
 * Every other varying it emits into a function-scope `var out : Varyings;`,
 * and WGSL zero-initialises that, so a forgotten assignment is a silent
 * `vec3f(0)` at the far end of the interpolator. No error, no warning, and a
 * frame that is merely wrong.
 */
export function probeUnassignedVaryingSpec(): MaterialSpec {
  return {
    name: 'dx-unassigned-varying',
    varyings: { worldPos: 'vec3f', normalW: 'vec3f', uv: 'vec2f' },
    slots: { tint: { type: 'vec3f', default: PARAMS.tint } },
    textures: { albedoMap: { kind: '2d' } },
    // out.clip, out.normalW and out.uv are written. out.worldPos is not.
    vertex: `
out.clip    = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.normalW = normalize((obj.model * vec4f(in.normal, 0.0)).xyz);
out.uv      = in.uv;
`,
    fragment: `
let n = normalize(in.normalW);
let v = normalize(frame.camPos - in.worldPos);
return vec4f(mat.tint * (0.5 + 0.5 * max(dot(n, v), 0.0)), 1.0);
`,
  };
}

/** The same material with `out.worldPos` actually written, for the contrast. */
export function probeWrittenVaryingSpec(): MaterialSpec {
  const spec = probeUnassignedVaryingSpec();
  return {
    ...spec,
    name: 'dx-worldpos-written',
    vertex: spec.vertex + '\nout.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;',
  };
}

/**
 * Silent-failure probe 2: a typo inside `prelude`.
 *
 * The body validator never sees the prelude, so a mistyped helper name is a Tint
 * error whose line and column point into the middle of a generated program.
 * Recorded verbatim in the report.
 */
export async function probePreludeTypo(device: GPUDevice): Promise<string> {
  try {
    await Material.create(device, {
      name: 'dx-prelude-typo',
      slots: { tint: { type: 'vec3f', default: PARAMS.tint } },
      prelude: 'fn dxLift(c : vec3f) -> vec3f { return c * 1.2; }',
      vertex: 'out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);',
      fragment: 'return vec4f(dxLifft(mat.tint), 1.0);', // one letter short
    });
    return 'NO ERROR';
  } catch (e) {
    return describeError(e);
  }
}

/**
 * Loud-failure probe: a body identifier that does not exist.
 * The control for the prelude probe, and the claim apse makes in its README.
 */
export function probeBodyTypo(): string {
  try {
    generateScaffold({
      name: 'dx-typo',
      slots: { tint: { type: 'vec3f', default: [1, 1, 1] } },
      vertex: 'out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);',
      fragment: 'return vec4f(frame.viewProjj.xyz * mat.tint, 1.0);',
    });
    return 'NO ERROR';
  } catch (e) {
    return describeError(e);
  }
}

/**
 * Varying-type probe: assign a `vec2f` into a `vec3f` varying.
 *
 * apse shares one `Varyings` struct between the two stages, so a *declared*
 * mismatch is unrepresentable — there is nowhere to write one type and read
 * another. What is left is a body that lies about a value's type, and that is
 * only caught by the WGSL compiler. This compiles it, so the message the
 * developer actually sees is recorded rather than assumed.
 */
export async function probeVaryingType(device: GPUDevice): Promise<string> {
  try {
    await Material.create(device, {
      name: 'dx-varying-type',
      varyings: { bad: 'vec3f' },
      slots: { tint: { type: 'vec3f', default: [1, 1, 1] } },
      vertex: `
out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.bad  = in.uv;
`,
      fragment: 'return vec4f(mat.tint * in.bad, 1.0);',
    });
    return 'NO ERROR';
  } catch (e) {
    return describeError(e);
  }
}

/**
 * Probe for the `obj.normalMatrix` defect, as a real material rather than an
 * assertion. The mesh has an identity model matrix, so the correct normal is
 * the attribute unchanged and the correct `obj.normalMatrix` is the identity.
 * Both are rendered and compared.
 */
export function probeNormalMatrixSpecs(): { attribute: MaterialSpec; objNormalMatrix: MaterialSpec } {
  const base = {
    varyings: { normalW: 'vec3f' } as const,
    vertexPrefix: 'out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);',
    fragment: 'return vec4f(in.normalW, 1.0);',
  };
  return {
    attribute: {
      name: 'dx-nrm-attribute',
      varyings: { normalW: 'vec3f' },
      vertex: `${base.vertexPrefix}\nout.normalW = in.normal;`,
      fragment: base.fragment,
    },
    objNormalMatrix: {
      name: 'dx-nrm-objmatrix',
      varyings: { normalW: 'vec3f' },
      vertex: `${base.vertexPrefix}\nout.normalW = obj.normalMatrix * in.normal;`,
      fragment: base.fragment,
    },
  };
}

/** Flattens an AseError (or anything else) into the text a developer sees. */
export function describeError(e: unknown): string {
  const any = e as Record<string, unknown>;
  if (any && typeof any === 'object' && typeof any['code'] === 'string') {
    return [
      `AseError code=${any['code']}`,
      `message: ${String(any['message'])}`,
      `why: ${String(any['why'])}`,
      `fix: ${String(any['fix'])}`,
    ].join('\n');
  }
  return e instanceof Error ? `${e.name}: ${e.message}` : String(e);
}
