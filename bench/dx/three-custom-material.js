/**
 * bench/dx — the three.js side of the authoring-cost comparison.
 *
 * Three separate materials, because "how hard is a custom material in three.js"
 * has three different answers depending on which door you come in:
 *
 *   V1  `ShaderMaterial` with hand-written GLSL. The path three.js's own
 *       ShaderMaterial docs describe. This is the one that is pixel-compared
 *       against the apse version.
 *   V2  `MeshPhongMaterial` + `onBeforeCompile` + `#include` surgery. The path
 *       the claim under test actually describes — undocumented chunk names, a
 *       mandatory `customProgramCacheKey`, and reading your own uniform back out
 *       of `material.userData.shader.uniforms`. It reuses three.js's own
 *       Blinn-Phong lighting instead of writing it, which is its whole appeal
 *       and also why it cannot be pixel-compared.
 *   V3  `MeshStandardNodeMaterial` + TSL under `WebGPURenderer`. The competitor
 *       apse's own JSDoc points at, so it gets measured rather than dismissed.
 *
 * The same rule as the apse file applies: everything inside a
 * `// @count:<section>` … `// @count:end` block is material code and is counted
 * mechanically by `run.ts`, which reads this file. Probes and scene setup live
 * outside those blocks and are reported separately.
 *
 * `PARAMS` and `ALBEDO_BYTES` are imported from the apse file so both shaders
 * read bit-identical inputs. The comparison is between the two shader paths,
 * not between two sets of constants.
 */

import { PARAMS, ALBEDO_BYTES, ALBEDO_SIZE } from './apse-custom-material.ts';

export { PARAMS, ALBEDO_BYTES, ALBEDO_SIZE };

// ---------------------------------------------------------------------------
// V1 — ShaderMaterial, hand-written GLSL. The pixel-compared version.
// ---------------------------------------------------------------------------

// @count:declarations
/**
 * The three varyings, declared once and pasted into both stages.
 *
 * `position`, `normal`, `uv`, `modelMatrix`, `viewMatrix`, `projectionMatrix`,
 * `normalMatrix` and `cameraPosition` are **not** here: `WebGLProgram` emits all
 * of them into a ~200-line generated prefix for every non-raw material. The
 * claim that a three.js custom material must hand-declare the matrix stack is
 * true only of `RawShaderMaterial`, and this file is the working counter-example.
 */
export const DX_VARYINGS_GLSL = `
varying vec3 vWorldPos;
varying vec3 vNormalW;
varying vec2 vUv;
`;
// @count:end

// @count:uniformDeclarations
/**
 * Nine `uniform` declarations. The same nine names, written twice, with nothing deriving one list
 * from the other and no diagnostic when they disagree.
 */
export const DX_UNIFORMS_GLSL = `
uniform vec3  tint;
uniform vec3  lightDir0;
uniform vec3  lightColor0;
uniform vec3  lightDir1;
uniform vec3  lightColor1;
uniform vec3  specColor;
uniform vec3  rimColor;
uniform float shininess;
uniform float rimPower;
uniform sampler2D albedoMap;
`;
// @count:end

// @count:texture
/** The albedo texture. DataTexture is three.js's zero-asset-loader path. */
export function createAlbedoTexture(THREE) {
  const tex = new THREE.DataTexture(
    ALBEDO_BYTES, ALBEDO_SIZE, ALBEDO_SIZE, THREE.RGBAFormat, THREE.UnsignedByteType,
  );
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}
// @count:end

// @count:vertex
export const DX_VERTEX_GLSL = `
void main() {
  vec4 world = modelMatrix * vec4( position, 1.0 );
  vWorldPos = world.xyz;
  vNormalW = normalize( mat3( modelMatrix ) * normal );
  vUv = uv;
  gl_Position = projectionMatrix * ( viewMatrix * world );
}
`;
// @count:end

// @count:fragment
export const DX_FRAGMENT_GLSL = `
vec3 dxAccumulate(
  vec3 n, vec3 v, vec3 dir, vec3 color,
  vec3 albedo, vec3 spec, float shininess
) {
  vec3 l = normalize( dir );
  vec3 irradiance = color * max( dot( n, l ), 0.0 );
  vec3 halfDir = normalize( l + v );
  float nDotH = max( dot( n, halfDir ), 0.0 );
  float norm = ( shininess + 2.0 ) / 8.0;
  return irradiance * ( albedo * ${PARAMS.lambert} + norm * pow( nDotH, shininess ) * spec );
}

void main() {
  vec3 n = normalize( vNormalW );
  vec3 v = normalize( cameraPosition - vWorldPos );
  vec3 base = tint * texture2D( albedoMap, vUv ).rgb;
  vec3 lit = dxAccumulate( n, v, lightDir0, lightColor0, base, specColor, shininess )
           + dxAccumulate( n, v, lightDir1, lightColor1, base, specColor, shininess );
  vec3 rim = rimColor * pow( 1.0 - max( dot( n, v ), 0.0 ), rimPower );
  gl_FragColor = vec4( lit + rim, 1.0 );
}
`;
// @count:end

// @count:assemble
/** V1. One call. The cache key is derived from the shader strings themselves. */
export function createShaderMaterial(THREE, albedoTexture) {
  return new THREE.ShaderMaterial({
    name: 'dx-lambert-blinn',
    uniforms: {
      tint:        { value: new THREE.Vector3(...PARAMS.tint) },
      lightDir0:   { value: new THREE.Vector3(...PARAMS.lights[0].dir) },
      lightColor0: { value: new THREE.Vector3(...PARAMS.lights[0].color) },
      lightDir1:   { value: new THREE.Vector3(...PARAMS.lights[1].dir) },
      lightColor1: { value: new THREE.Vector3(...PARAMS.lights[1].color) },
      specColor:   { value: new THREE.Vector3(...PARAMS.specColor) },
      rimColor:    { value: new THREE.Vector3(...PARAMS.rim.color) },
      shininess:   { value: PARAMS.shininess },
      rimPower:    { value: PARAMS.rim.power },
      albedoMap:   { value: albedoTexture },
    },
    vertexShader: DX_VARYINGS_GLSL + DX_VERTEX_GLSL,
    fragmentShader: DX_VARYINGS_GLSL + DX_UNIFORMS_GLSL + DX_FRAGMENT_GLSL,
    side: THREE.FrontSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });
}
// @count:end

// ---------------------------------------------------------------------------
// V2 — MeshPhongMaterial + onBeforeCompile. The path the claim describes.
// ---------------------------------------------------------------------------

// @count:phong
/**
 * V2, done properly: the `customProgramCacheKey` override is present.
 *
 * All three of the things the claim names are real, and all three are here:
 *
 *   - `#include <common>` and `#include <opaque_fragment>` are internal chunk
 *     names. Not public API, not documented, and the only index is
 *     `grep -r opaque_fragment node_modules/three/src/renderers/shaders/`.
 *   - `customProgramCacheKey` is required, because `MeshPhongMaterial` is a
 *     built-in: its cache key is `shaderID` plus ~110 fields read off a
 *     `parameters` object whose shape is internal, and `onBeforeCompile`
 *     contributes nothing to it unless this method says so.
 *   - `material.uniforms` does not exist on a built-in material, so the only
 *     handle on the uniforms the patch created is
 *     `this.userData.shader.uniforms`.
 *
 * No vertex patch is needed: `vViewPosition` and `vNormal` are already
 * interpolated by `<normal_pars_fragment>` and the Phong vertex stage, so the
 * fresnel is 2 lines in the fragment stage alone.
 */
export function createPhongPatched(THREE, albedoTexture, opts = {}) {
  const { tint = PARAMS.tint, rimPower = PARAMS.rim.power } = opts;

  const material = new THREE.MeshPhongMaterial({
    map: albedoTexture,
    shininess: PARAMS.shininess,
    specular: new THREE.Color(...PARAMS.specColor),
  });

  material.onBeforeCompile = (shader) => {
    shader.uniforms.tint = { value: new THREE.Vector3(...tint) };
    shader.uniforms.rimPower = { value: rimPower };
    shader.uniforms.rimColor = { value: new THREE.Vector3(...PARAMS.rim.color) };

    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', [
        '#include <common>',
        'uniform vec3 tint;',
        'uniform float rimPower;',
        'uniform vec3 rimColor;',
      ].join('\n'))
      .replace('#include <opaque_fragment>', [
        'vec3 dxV = normalize( vViewPosition );',
        'float dxF = pow( 1.0 - saturate( dot( normalize( vNormal ), dxV ) ), rimPower );',
        'outgoingLight += tint * rimColor * dxF;',
        '#include <opaque_fragment>',
      ].join('\n'));

    material.userData.shader = shader;
  };

  material.customProgramCacheKey = () => `dx-phong|${tint.join(',')}|${rimPower}`;

  return material;
}

/** Read your own uniform back out, the way the claim describes. */
export function readOwnUniform(material, name) {
  return material.userData.shader.uniforms[name].value;
}

/** ...and the reason that indirection exists: a built-in material has none. */
export function builtInMaterialHasUniforms(material) {
  return material.uniforms === undefined;
}

/** The same read, on the V1 path, where it is a documented public property. */
export function readOwnUniformShaderMaterial(material, name) {
  return material.uniforms[name].value;
}
// @count:end

// ---------------------------------------------------------------------------
// V3 — MeshStandardNodeMaterial + TSL, under WebGPURenderer.
// ---------------------------------------------------------------------------

// @count:nodes
/**
 * V3. The same feature set — two lights, a texture, a tint, a camera-relative
 * fresnel rim — in three.js's node idiom. No GLSL strings, no chunk names, no
 * cache key, no `userData` round trip. This is the version that threatens the
 * claim, so it is measured rather than argued about.
 */
export function createNodeMaterial(THREE, TSL, albedoTexture, opts = {}) {
  const { tint = PARAMS.tint, rimPower = PARAMS.rim.power } = opts;
  const { float, vec3, texture, normalWorld, positionWorld, cameraPosition } = TSL;

  const material = new THREE.MeshStandardNodeMaterial();
  material.lights = true;
  material.map = albedoTexture;
  material.color = new THREE.Color(1, 1, 1);
  material.roughness = 0.4;
  material.metalness = 0.0;

  const viewDir = cameraPosition.sub(positionWorld).normalize();
  const fresnel = float(1).sub(normalWorld.dot(viewDir).clamp(0, 1)).pow(rimPower);
  material.colorNode = vec3(...tint).mul(texture(albedoTexture).rgb);
  material.emissiveNode = vec3(...tint).mul(vec3(...PARAMS.rim.color)).mul(fresnel);

  return material;
}
// @count:end

// ---------------------------------------------------------------------------
// Silent-failure and error probes. Not material code; not counted.
// ---------------------------------------------------------------------------

/**
 * Probe A: the same patch as V2, with `customProgramCacheKey` left alone.
 *
 * The patch bakes a per-material gain into the source as a `#define`. Two
 * materials that are identical in every field `getParameters` can observe get
 * the same cache key, so three.js hands them the *same* `WebGLProgram`, and the
 * second is drawn with the first one's `#define`. Uniform values are per
 * material, so only the baked constant is wrong — and nothing says so.
 *
 * Returns the two materials. The caller renders B alone and then B beside A; if
 * the two images differ, the programs were wrongly shared.
 */
export function createPhongUncached(THREE, albedoTexture, { gain, tint, cacheKey }) {
  const material = new THREE.MeshPhongMaterial({
    map: albedoTexture,
    shininess: PARAMS.shininess,
    specular: new THREE.Color(...PARAMS.specColor),
  });

  material.onBeforeCompile = (shader) => {
    shader.uniforms.tint = { value: new THREE.Vector3(...tint) };
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', [
        '#include <common>',
        'uniform vec3 tint;',
        `#define DX_GAIN ${gain.toFixed(3)}`,
      ].join('\n'))
      .replace('#include <opaque_fragment>', [
        'outgoingLight = outgoingLight * DX_GAIN + tint * 0.15;',
        '#include <opaque_fragment>',
      ].join('\n'));
  };
  // The override is the fix. Leaving it off is the bug under test, and the
  // harness needs both variants to compare a broken material against a correct
  // one that differs in nothing else.
  if (cacheKey !== undefined) material.customProgramCacheKey = () => cacheKey;
  return material;
}

/**
 * Probe B: a `.replace()` whose target token is not in the source.
 *
 * Takes the real `ShaderLib.phong.fragmentShader` and substitutes a chunk name
 * that does not exist. `String.prototype.replace` on an absent substring is a
 * no-op, so the material compiles, draws, and is quietly unpatched.
 */
export function probeMissingIncludeReplacement(THREE) {
  const src = THREE.ShaderLib.phong.fragmentShader;
  const before = src.length;
  const after = src.replace('#include <opaque_output_fragment>', 'BROKEN').length;
  return {
    tokenPresent: src.includes('#include <opaque_output_fragment>'),
    changed: before !== after,
    lengthBefore: before,
    lengthAfter: after,
  };
}

/** Probe C: an `#include` name that survives into `resolveIncludes`. */
export function probeUnknownInclude(THREE) {
  return new THREE.ShaderMaterial({
    vertexShader: 'void main() { gl_Position = vec4( position, 1.0 ); }',
    fragmentShader: [
      '#include <a_chunk_that_does_not_exist>',
      'void main() { gl_FragColor = vec4( 1.0 ); }',
    ].join('\n'),
  });
}

/** Probe D: `varying vec2` out of the vertex stage, `varying vec3` into fragment. */
export function probeVaryingTypeMismatch(THREE) {
  return new THREE.ShaderMaterial({
    uniforms: { tint: { value: new THREE.Vector3(1, 1, 1) } },
    vertexShader: [
      'varying vec2 vProbe;',
      'void main() { vProbe = vec2( 1.0 ); gl_Position = vec4( position, 1.0 ); }',
    ].join('\n'),
    fragmentShader: [
      'uniform vec3 tint;',
      'varying vec3 vProbe;',
      'void main() { gl_FragColor = vec4( tint * vProbe, 1.0 ); }',
    ].join('\n'),
  });
}

/**
 * Probe E: a value-level type error — the direct analogue of apse's
 * `out.bad = in.uv`, i.e. assigning a `vec2` where a `vec3` is expected.
 */
export function probeShaderTypeError(THREE) {
  return new THREE.ShaderMaterial({
    uniforms: { tint: { value: new THREE.Vector3(1, 1, 1) } },
    vertexShader: [
      'varying vec3 vProbe;',
      'void main() { vProbe = uv; gl_Position = vec4( position, 1.0 ); }',
    ].join('\n'),
    fragmentShader: [
      'uniform vec3 tint;',
      'varying vec3 vProbe;',
      'void main() { gl_FragColor = vec4( tint * vProbe, 1.0 ); }',
    ].join('\n'),
  });
}

/**
 * Captures the exact GLSL three.js hands the driver, by shadowing
 * `gl.shaderSource` for the duration of one render. This is how the report gets
 * apse's generated WGSL and three.js's generated GLSL side by side, and it is
 * also how "how much did the library write for you" is counted rather than
 * asserted.
 */
export function captureGeneratedGLSL(renderer, render) {
  const gl = renderer.getContext();
  const original = gl.shaderSource;
  const stages = [];
  gl.shaderSource = function (shader, src) {
    stages.push(src);
    return original.call(this, shader, src);
  };
  try {
    render();
  } finally {
    gl.shaderSource = original;
  }
  return {
    vertex: stages.find((s) => s.includes('gl_Position')) ?? null,
    fragment: stages.find((s) => !s.includes('gl_Position')) ?? null,
    compileCount: stages.length,
  };
}

/**
 * Runs `render()` with `console.error` and page errors intercepted, so a
 * shader failure can be reported as text rather than crashing the harness.
 */
export function captureDiagnostics(render) {
  const errors = [];
  const original = console.error;
  console.error = (...a) => { errors.push(a.map((x) => (x && x.stack) ? x.stack : String(x)).join(' ')); };
  let thrown = null;
  try {
    render();
  } catch (e) {
    thrown = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  } finally {
    console.error = original;
  }
  return { errors, thrown };
}
