# bench/dx — is a custom material actually simpler in apse than in three.js?

```
bun run bench/dx/run.ts
```

This directory exists to make one paragraph of the root `README.md` falsifiable.
The paragraph claims:

> Writing a custom material is dramatically simpler in apse than in three.js,
> because apse generates every binding, uniform struct, varying struct and
> entry-point signature, whereas three.js requires hand-declared uniforms, string
> surgery on undocumented `#include` chunk names, a mandatory undocumented
> `customProgramCacheKey`, and reaching into `material.userData.shader.uniforms`
> to read your own uniforms.

So the same material was written twice and measured. Everything below came out of
a run of `bench/dx/run.ts` on an Apple M-series GPU, headless Chrome, 640×360,
apse on WebGPU and three.js r186 on WebGL2, **in the same browser process on the
same device**. Nothing here is estimated. Where something could not be measured,
it says so.

**Short answer: the claim is roughly half true, and the half that is false is the
half the paragraph leads with.** apse does generate everything it says it
generates, and it is measurably better at the failure modes. But the line count
is a dead heat (80 vs 83), three.js does *not* make you hand-declare
`modelViewMatrix` and `projectionMatrix`, and the node-material path that apse's
own JSDoc tells you to migrate to is **15 lines** — less than a fifth of either.

---

## 1. Do the two images actually agree?

Yes, to within one 8-bit step.

```
apse   covered 61127 px, mean luma 25.26, min 3, max 255
three  covered 61127 px, mean luma 25.26, min 3, max 255

diff over all 230400 pixels
  maxChannelDiff   1
  meanAbsDiff       0.00002
  pixels differing by > 0     5
  pixels differing by > 1     0
  worst pixel      apse [255,253,198]  three [255,253,199]  at (366, 72)
```

Five pixels out of 230 400 differ, none by more than one step, and the worst case
is a saturated specular highlight sitting on a quantisation boundary. Both
images have byte-identical coverage, mean, min and max. Both draw one call and
6016 triangles from the same 3057-vertex, 18048-index buffer.

How the comparison is made fair:

- **One geometry.** apse's `sphere()` produces a `MeshData`; the same
  `Float32Array` is uploaded to WebGPU *and* de-interleaved into a three.js
  `BufferGeometry`. No pixel difference can be blamed on the mesh.
- **One camera.** apse's `PerspectiveCamera` computes the world→view matrix;
  three.js is handed that exact matrix as `matrixWorldInverse` / `matrixWorld`,
  so `cameraPosition` and every world-space value agree bit for bit. The
  *projections* stay each library's own, because the two use different clip-space
  Z ranges and no honest comparison can share one.
- **One set of constants.** `PARAMS` and `ALBEDO_BYTES` are exported from the
  apse file and imported by the three.js file.
- **One colour space.** `renderer.outputColorSpace = LinearSRGBColorSpace` and
  the render target is `NoColorSpace`, so the sRGB OETF that three.js's generated
  prefix would otherwise apply is out of the way. Both clear colours are
  `(0.02, 0.02, 0.03)`.
- **One row order.** `readRenderTargetPixels` returns rows bottom-up and
  `Renderer.capture()` returns them top-down, so the three.js readback is
  flipped. This one is worth flagging: a vertically flipped sphere has *identical*
  coverage, mean, min and max. The first version of this harness skipped the flip
  and reported a `maxChannelDiff` of 239 with a perfect-looking summary.

Two real differences remain between the renderers and are not shader bugs:
`bytesPerRow` is 256-byte padded in both readbacks (handled), and the two
materials are each bound to a different colour target format (handled by leaving
apse's `targets` at its default, which matches `capture()`).

**Not measured:** whether the two agree on a rotated or non-uniformly-scaled
mesh. apse's `mat.normalMatrix` (now fixed: normalMatrixOf writes a padded 12-float scratch) is broken in this build (see §6), so the
comparison uses `obj.model` for the normal transform, which is exact for a
rotation-only transform and is what both shaders do here.

---

## 2. Lines of user-written material code

Counted by `run.ts` reading the `// @count:<section>` … `// @count:end` blocks in
the two files, with comments and blank lines removed. The counts are not asserted
anywhere; re-running the harness re-derives them.

Two things about the counting, so the numbers can be argued with:

- The comment stripper is not string-aware, so it also removes the WGSL comment
  inside the apse `prelude` and the GLSL comment inside the three.js fragment.
  That is one line in each, and it is the only place the two libraries' totals
  are affected by an artefact of the counter rather than by the code.
| counted section | apse | three.js V1 `ShaderMaterial` | three.js V2 `onBeforeCompile` | three.js V3 node material |
|---|---:|---:|---:|---:|
| `declarations` | 22 | 5 | — | — |
| `uniformDeclarations` | — | 12 | — | — |
| `prelude` | 12 | — | — | — |
| `texture` | 20 | 12 | — | — |
| `vertex` | 6 | 9 | — | — |
| `fragment` | 9 | 22 | — | — |
| `assemble` | 11 | 23 | — | — |
| `phong` — the whole patched material | — | — | 38 | — |
| `nodes` — the whole node material | — | — | — | 15 |
| **total** | **80** | **83** | **38** | **15** |

Every cell is a section `run.ts` read out of the source. V2 and V3 are each a
single section because each is one function; there is nothing to subdivide, and
inventing a subdivision would be the first dishonest number in this document.

Read that as: **apse 80 lines, three.js 83 lines.** There is no dramatic
difference in the amount of code, and the difference that exists runs *against*
apse on shading logic — apse spends 12 lines on a `prelude` helper that GLSL
declares inline in the middle of the fragment body it already had to write, and
20 lines supplying a texture that three.js supplies in 12.

What the line counts hide:

- **Uniforms.** apse's 22 declaration lines carry the nine values as well as the
  nine types, because a slot default *is* the uniform's initial value, and those
  nine entries also become the WGSL types, the byte offsets, the `MaterialData`
  struct and the writer. three.js needs the same nine names twice — 12 lines of
  GLSL declarations and 12 lines inside the 23-line `assemble` section — because
  GLSL has no way to derive a uniform block from an object literal. That is the
  one axis on which apse is clearly ahead, and it is worth about 13 lines.
- **Textures.** apse is behind here, and the reason is not the API. `writeTexture`
  also requires a 256-byte row pitch, so a 4×4 texture cannot be written directly
  and needs a staging copy. That is 8 lines apse charges you for and nobody else
  does, and it is the first item on a list an asset loader would absorb. apse has
  no asset loaders.
- **V2 and V3 are not comparable to each other.** 38 lines buys you a tint and a
  fresnel on top of three.js's entire Blinn-Phong light pipeline; 15 lines buys
  you the same on top of a PBR one. Neither is writing the lighting, which is
  what the 80 and 83 are.

### What each library generated

```
apse generated WGSL              3690 bytes   107 lines   (30 authored, 77 generated)
three V1 vertex                  1839 bytes    81 lines   (10 authored, 71 generated)
three V1 fragment                3159 bytes    94 lines   (37 authored, 57 generated)
three V2 vertex                 20100 bytes   654 lines
three V2 fragment               48013 bytes  1330 lines
```

For a from-scratch `ShaderMaterial`, three.js generates **128 lines** for a
**47-line** authored body; apse generates **77** for **30**. So the "apse writes
the boilerplate for you" part of the claim holds, and holds by about 1.7×, on
this material. On the `onBeforeCompile` path — the one that reuses three.js's
own lighting — three.js generates **1984 lines** to host 6 lines of authored GLSL.

The authored-line counts for three.js are approximate: a body is re-indented on
the way into the generated program, so lines are matched by trimmed content. The
byte and total-line counts are exact.

---

## 3. Distinct library concepts the author must know

Even-handed, including apse's own learning cost.

### three.js, V1 (`ShaderMaterial`)

1. **Every custom uniform is declared twice** — once as GLSL text, once in a
   `uniforms` object. Nothing derives one from the other, nothing warns when they
   disagree, and an unused declaration is stripped by the compiler.
2. **Values need a type wrapper.** `uniforms: { tint: { value: new
   THREE.Vector3(…) } }`, not `[0.85, 0.42, 0.16]`. `ShaderMaterial` accepts
   plain arrays for `uniform3fv` too, but the idiomatic form is the wrapper.
3. **A texture needs four lines of configuration** (`magFilter`, `minFilter`,
   `generateMipmaps`, `needsUpdate`) and an understanding of `flipY`, which
   defaults to `true` on `Texture` and `false` on `DataTexture`. Getting it wrong
   is a silent vertical flip.
4. **`texture2D`, not `texture`.** three.js `#define`s the GLSL3 name back, so
   the WebGL1 spelling is the one that works and the one every tutorial shows.
5. **varyings must be pasted into both stages by hand.** A `ShaderMaterial` gets
   no `varying` declarations for you.
6. **You must not include anything you did not ask for.** The generated prefix is
   ~120 lines; the *only* index of chunk names is the source tree.

That is the whole V1 list. Notably **absent**:

- ❌ **hand-declaring `modelViewMatrix` / `projectionMatrix`.** The claim says you
  have to. You do not. Measured against the generated source:

  ```
  modelMatrix       declared by three.js in: threeV1Vertex, threeV2Vertex
  viewMatrix        declared by three.js in: threeV1Vertex, threeV1Fragment, threeV2Vertex, threeV2Fragment
  projectionMatrix  declared by three.js in: threeV1Vertex, threeV2Vertex
  modelViewMatrix   declared by three.js in: threeV1Vertex, threeV2Vertex
  normalMatrix      declared by three.js in: threeV1Vertex, threeV2Vertex, threeV2Fragment
  cameraPosition    declared by three.js in: threeV1Vertex, threeV1Fragment, threeV2Vertex, threeV2Fragment
  normal attribute  declared by three.js in: threeV1Vertex
  uv attribute      declared by three.js in: threeV1Vertex
  ```

  `WebGLProgram` emits all of them, plus `position` / `normal` / `uv`, into a
  generated prefix for every non-raw material. This is only true of
  `RawShaderMaterial`, and the claim does not say so. **This part of the
  paragraph is false.**
- ❌ **`customProgramCacheKey`.** Not needed on V1. A `ShaderMaterial` has no
  `shaderID`, so `getProgramCacheKey` pushes `customVertexShaderID` and
  `customFragmentShaderID`, which come from a `WebGLShaderCache` keyed on the
  shader *source text*. Two materials with different source get different
  programs without any help. **Also false as stated.**
- ❌ **`material.userData.shader.uniforms`.** Not needed on V1: `material.uniforms`
  is a public property and holds your own values. Measured:
  `v1_readBackVia_material_uniforms → [0.85, 0.42, 0.16]`, the value that was
  set. **Also false as stated** — for this path.

All three of those *are* real on V2, which is the path the claim is really about.

### three.js, V2 (`MeshPhongMaterial` + `onBeforeCompile`)

1. `#include <common>` and `#include <opaque_fragment>` are internal chunk names,
   undocumented, indexed only by `src/renderers/shaders/ShaderChunk/`.
2. `onBeforeCompile` receives a **mutable `parameters` object whose shape is
   internal**: `parameters.uniforms`, `parameters.vertexShader`,
   `parameters.fragmentShader`, and about a hundred other fields assembled by
   `getParameters` from the scene, the geometry and the material.
3. `customProgramCacheKey` is **mandatory** here and is undocumented — it appears
   in the source as a method on `Material` and in `WebGLPrograms` as one of ~110
   `array.push(...)` calls. The base implementation returns
   `onBeforeCompile.toString()`, which is identical for every material built by
   one factory, so any per-material decision the patch makes is invisible to the
   cache.
4. Reading your own uniform back requires `material.userData.shader.uniforms`,
   because a built-in material has **no** `uniforms` property. Measured:
   `v2_builtInMaterialHasNoUniformsProperty → true`,
   `v2_readBackVia_userData_shader_uniforms → [0.9, 0.1, 0.1]`.
5. The patch is a *mutation* of a 654-line vertex and 1330-line fragment program.
   You are editing text you have never read, inside a program whose line numbers
   the error messages refer to.
6. `onBeforeCompile` is called by `WebGLRenderer` and **nowhere else** in the
   whole of three.js r186 — `grep -rn onBeforeCompile src/` matches two files,
   `Material.js` and `WebGLRenderer.js`. `WebGPURenderer` never calls it.

### apse

Beyond the documented `MaterialSpec`, the author has to know:

1. **`mat.normalMatrix` (now fixed: normalMatrixOf writes a padded 12-float scratch) is broken.** See §6. This is not a learning cost, it is a
   defect, and the material here works around it.
2. **`prelude` is the only place a `fn` may go.** A body that declares a
   function, a struct, or anything with an `@attribute` is rejected. Measured, and
   the message is good: `SHADER_BODY_INVALID … it declares a function`.
3. **A body must assign `out.clip` and must `return` a `vec4f`.** Both enforced
   before compilation, both with a good message.
4. **An unassigned varying is silently zero.** `validateBody` checks `out.clip`
   and nothing else. See §5.
5. **There is no asset loader.** You write `createTexture` + `writeTexture` (with
   a 256-byte row pitch) + `createView`, then `material.setTexture`. The error
   path for a missing texture is good: `TEXTURE_SLOT_MISSING` names the slots
   that *are* assigned.
6. **Slot writes are `material.setSlot(name, value)`,** which validates the name
   and the component count. There is no `userData` indirection and no second
   spelling.
7. **The generated program is available**, via `material.wgsl` or
   `generateScaffold(spec).code`, and the spec is introspectable via
   `describeMaterial(spec)`. This is a real advantage: the whole `describe()`
   inventory — slot offsets, varying locations, sampler names, bind-group
   contents, the body-visible identifier list — is a documented function.

**Not measured:** how long it takes a competent developer who has read
`ARCHITECTURE.md` to write either version. Line count is a proxy, and a bad one.
The failure-mode table in §5 is the better proxy for "how hard is this to get
right", and it points the other way from the claim.

---

## 4. Time to first drawn frame

Five samples per run, each on a **fresh document** — a shader program compiles
once, so timing in a page that has already drawn measures a cache hit. Both
libraries in the same browser process. Three runs of the harness:

| | run 1 median | run 2 median | run 3 median |
|---|---:|---:|---:|
| apse, `await Material.create` → first readback | 7.7 ms | 8.0 ms | 7.8 ms |
| three.js, `new ShaderMaterial` → first readback | 2.8 ms | 2.8 ms | 2.9 ms |

Ranges within a run: apse 7.0–9.1 ms, three.js 2.7–3.8 ms. Per-sample breakdown
from one run:

```
apse compile 4.3 + frame 4.3 | three first render 2.7
apse compile 4.1 + frame 3.1 | three first render 2.8
apse compile 4.5 + frame 4.4 | three first render 3.0
apse compile 4.7 + frame 3.5 | three first render 2.8
```

Caveats, because they matter:

- apse's `createRenderPipelineAsync` really is async, so most of its 4–5 ms is
  off the main thread. three.js's `linkProgram` is synchronous and blocks. A
  wall-clock number flatters the synchronous library here, and the *frame* half
  (2.4–5.6 ms) is the part actually on the main thread. The honest summary is
  "apse is about 2.8× slower end to end, and the difference is mostly a
  readback that is not comparable" — see the next two points.
- Both numbers end at a complete pixel readback, and the two readbacks are not
  the same work. apse's is `mapAsync` on a 256-byte-padded buffer; three.js's is
  a synchronous `readPixels`. I have not normalised them, so the 2.8× is an
  upper bound on the difference in material cost and a lower bound on nothing.
- Sample 1 is the coldest for both; later samples in the same browser process may
  benefit from a warm Chrome shader cache. On sample 1 alone the two are within
  2.3×, and in one run they were within 1.5×.
- These are warm-ish numbers, not the 2–5 second cold-cache stall
  `src/material/material.ts` warns about. That stall is real but I did not
  measure it, and I am not going to quote a number I did not take.
- `Renderer.create` and `new WebGLRenderer()` are excluded. Both are excluded for
  both.

---

## 5. Things that fail silently

### three.js

**A. Missing `customProgramCacheKey` — silent, and it moves real pixels.**
Two materials, same patch, `DX_GAIN 1.0` and `DX_GAIN 0.2` baked in as
`#define`s, neither overriding the key. A is placed nearer the camera, and
three.js sorts opaque objects front to back, so A compiles the shared program.
B is a byte-identical material that *does* override the key, and serves as the
correct reference. Same two objects, same positions; the only difference is the
override.

```
B without customProgramCacheKey  vs  B with it
  pixels compared            46316
  maxChannelDiff             204
  meanAbsDiff                26.79
  differing by > 16          11403
  worst pixel   with A [255,255,255]   with key [51,90,51]
  programs compiled for the probe        2
```

No exception, no console message, no warning. 12 412 of B's 46 316 pixels are
wrong. This is the single worst failure mode measured in this harness, and it is
the one the claim names.

**B. `.replace()` on a chunk name that is not there — silent.**
`ShaderLib.phong.fragmentShader` is 2096 bytes.
`src.replace('#include <opaque_output_fragment>', 'BROKEN')` returns 2096 bytes.
`tokenPresent: false, changed: false`. No diagnostic, and the material compiles
and draws as if the patch had been applied. The claim is right about this one.

**C. A `#include` that survives into `resolveIncludes` is *not* silent.**
The claim implies it is. It throws:

```
Error: THREE.WebGLProgram: Can not resolve #include <a_chunk_that_does_not_exist>
```

**D. A varying type mismatch between stages — loud, but not a throw.**
`varying vec2` out, `varying vec3` in:

```
THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false
Program Info Log: Types of varying 'vProbe' differ between VERTEX and FRAGMENT shaders.
FRAGMENT varying vProbe does not match any VERTEX varying
```

It is a `console.error` from `WebGLProgram`, not an exception, and the frame is
entirely the clear colour (`coveredPixels: 0`). Loud in a devtools console, silent
in a headless run, and it costs you the object.

### apse

**A. A declared varying that the body never assigns — silent.** The one apse has
that is arguably worse than three.js, because it is *defined* rather than
undefined: WGSL zero-initialises the generated `var out : Varyings`, so a
forgotten assignment arrives at the far end of the interpolator as `vec3f(0)`.
`validateBody` only insists on `out.clip`.

```
broken (out.worldPos never written)  vs  written
  rendered            yes / yes
  threw               no  / no
  warning or log      none / none
  maxChannelDiff      34
  meanAbsDiff         4.47
  pixels differing    63108 of 64108 covered
```

**B. `prelude` typos get the worst message in this whole exercise.** The same
error apse is proudest of elsewhere. A one-letter typo in a helper name:

```
AseError code=SHADER_COMPILE_FAILED
why: … Compiler said: [Invalid ShaderModule "apse:shader:dx-prelude-typo"]
     is invalid due to a previous error.
     - While validating vertex stage …
fix: Read the message above — it names the offending line …
```

It does not name the offending line. The line it points at is inside a generated
`fn vs`. The real diagnostic exists and is not passed on. Asking the device
directly for the same WGSL:

```
getCompilationInfo() → error 66:16  unresolved call target 'dxLifft'
```

and the browser's own uncaptured-error path, reached only by bypassing apse's
error scope, eventually reports:

```
[GPUValidationError] Error while parsing WGSL: :66:16 error: unresolved call target 'dxLifft'
  return vec4f(dxLifft(mat.tint), 1.0);
               ^^^^^^^
```

So the information is there and apse does not surface it. This is the exact
failure the scaffold's own header comment claims to have eliminated — "a WGSL
compile error whose line number points into *generated* code, a message that names
a line the user cannot see". `prelude` is where it still happens, and it is worse
than three.js, which prints the offending source line verbatim:

```
ERROR: 0:70: 'assign' : cannot convert from 'in highp 2-component vector of float'
                   to 'out highp 3-component vector of float'
 69: varying vec3 vProbe;
> 70: void main() { vProbe = uv; gl_Position = vec4( position, 1.0 ); }
```

**C. A body identifier typo — genuinely good, and the README's example is real.**
Measured, verbatim:

```
AseError code=SHADER_BODY_INVALID
message: The fragment body of material "dx-typo" is invalid: `frame.viewProjj` is not a field.
fix: Did you mean `frame.viewProj`? Available on frame: view, proj, viewProj,
     invView, invProj, invViewProj, camPos, time, delta, elapsed, resolution,
     viewport, exposure, alpha.
```

This is better than anything three.js does, and it is the strongest thing apse
has going. It is also the only part of the claim that survives unqualified.

**D. A varying *value* of the wrong type — caught, but by the compiler, and with
the unhelpful message from (B).** apse shares one `Varyings` struct between the
two stages, so a *declared* type mismatch is unrepresentable — there is nowhere
to write one type and read another. `out.bad = in.uv;` with `bad : vec3f` is the
only way to ask the question, and it produces the same
`is invalid due to a previous error` as (B).

### Scorecard

| | three.js | apse |
|---|---|---|
| silent, and the object's appearance is wrong | 2 (missing cache key, `.replace` no-op) | 1 (unassigned varying) |
| silent, and it is the library's own data | 0 | **1 (`mat.normalMatrix` (now fixed: normalMatrixOf writes a padded 12-float scratch), §6)** |
| throws with the offending token named | 1 (unknown `#include`) | 0 |
| `console.error` with the offending line quoted | 2 (varying mismatch, type error) | 0 |
| typed error naming the field and listing the alternatives | 0 | **1 (body identifier)** |
| typed error that does *not* name the fault | — | 2 (prelude typo, varying value type) |

---

## 6. A defect found while measuring: `mat.normalMatrix` (now fixed: normalMatrixOf writes a padded 12-float scratch) is wrong

Not part of the brief, but it made the comparison impossible until it was found,
and it is silent, so it belongs here.

`Renderer` packs object data like this:

```ts
// src/render/renderer.ts
const _normal = new Float32Array(9);           // line 140
normalMatrixOf(item.model, _normal);           // line 594
```

and `normalMatrixOf` writes a **tightly packed** 3×3:

```ts
out[0] = A * det;              out[1] = B * det;              out[2] = C * det;
out[3] = …;                    out[4] = …;                    out[5] = …;
out[6] = …;                    out[7] = …;                    out[8] = …;
```

`ObjectUniforms.pack` reads a `mat3x3f` the way WGSL lays it out — three
16-byte-aligned columns of four floats, so column *j* starts at `j * 4`:

```ts
for (let j = 0; j < 3; j++) {
  const col = j * 4;
  f32[at + col]     = normalMatrix[col];
  f32[at + col + 1] = normalMatrix[col + 1];
  f32[at + col + 2] = normalMatrix[col + 2];
}
```

So the producer writes indices 0–8 and the consumer reads 0,1,2 / 4,5,6 / 8,9,10.
Indices 3 and 7 are padding the producer never emits, and 9 and 10 are past the
end of a `Float32Array(9)` — `undefined`, which becomes `NaN` when written into
the `Float32Array` mirror.

Measured, with a mesh whose model matrix is the identity, where the correct
answer for both materials is the vertex normal unchanged:

```
from the vertex attribute   maxLuma 147.33   varied
from obj.normalMatrix       maxLuma 255      saturated everywhere
diff  maxChannelDiff 255, meanAbsDiff 65.53, 64108 of 64108 covered pixels differ
rendered: yes.  threw: no.  warning: none.
```

apse's own `pbrMaterial` uses `mat.normalMatrix` (now fixed: normalMatrixOf writes a padded 12-float scratch), and it renders the same sphere
as a single flat colour — `covered mean luma == maxLuma == 55`. The existing
`bench/index.html` readback check does not catch this, because it asks "does any
pixel differ from the clear colour", and a flat lit sphere does.

The workaround used in `bench/dx/apse-custom-material.ts` is
`normalize((obj.model * vec4f(in.normal, 0.0)).xyz)`, which is exact for a
rotation-only transform and makes the apse and three.js shaders the same
expression. I did not fix `src/`, which is out of scope for this task.

**Not measured:** the effect under non-uniform scale, where `obj.model`'s upper 3×3
would be wrong in a way the identity test cannot reveal.

---

## 7. The node-material competitor

apse's own `src/material/scaffold.ts` header says: *"A custom material has to
hand-declare `modelViewMatrix`, `projectionMatrix`, `modelMatrix`,
`normalMatrix`, `cameraPosition`, and `#ifdef` blocks for fog, clipping, morph
targets, and skinning"*, and `src/materials/Material.js` in three.js says: *"The
recommended approach when customizing materials is to use `WebGPURenderer` with
the new Node Material system and TSL."* Both of those point at
`MeshStandardNodeMaterial`, so it was measured rather than dismissed.

```js
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
```

**15 lines.** No GLSL strings, no chunk names, no `customProgramCacheKey`, no
`userData`, no `prelude`, no hand-declared bindings. It has the two-light
lighting, the texture, the tint and the camera-relative fresnel rim, and the
lights come from the scene for free. Measured in its own document (Chrome drops
the first WebGPU instance when a second device is created in the same document,
so it cannot share the apse page):

```
three revision        186
backend               WebGPUBackend   (isWebGPU: true)
material type         MeshStandardNodeMaterial
colorNode             set
emissiveNode          set
lights                true
pixels covered        64108   (coverage 0.2782)   maxLuma 91
first render          36–38 ms
```

The offscreen-render-plus-readback number is deliberately omitted: it came out at
107.3 ms on one run and 11.3 ms on another, which is not a measurement.

It draws: 64 108 non-background pixels, which is exactly the coverage apse's
materials get for the same sphere at the same camera. `info.render.triangles`
reported 0 and is not reported here, because the counters are reset by the
readback render and are not trustworthy at the point they were sampled.

It is **not** pixel-comparable, and the reason is structural rather than a
convenience: the node path is a PBR material, so it accumulates Cook-Torrance
GGX through three.js's light system, where the apse and `ShaderMaterial` versions
in §1 hand-write Lambert + Blinn-Phong. Comparing them pixel-wise would measure
two different lighting models. The comparison this section supports is authoring
cost, and on authoring cost the node path is **15 lines against 80 and 83.**

Two things about it that are not in its favour, and that I did not measure:

- It could only be run on a **second page**. That is a real cost for a library
  you want to use alongside something else, and it is a consequence of how Chrome
  handles multiple WebGPU devices, not of three.js.
- three.js's own documentation recommends `WebGLRenderer` for WebGL2
  applications and labels `WebGPURenderer` WIP. `bench/index.html` already makes
  that choice deliberately. A comparison that quietly picked the node path for
  three.js would be a straw man dressed up as a victory; a comparison that
  quietly picked `WebGLRenderer` would be measuring across backends. So: apse is
  WebGPU and `ShaderMaterial` is WebGL2, and that is stated rather than hidden.

---

## 8. The honest verdict

**The claim is overstated, and overstated in the wrong direction.** Broken into
its four assertions:

| assertion | verdict |
|---|---|
| apse generates every binding, uniform struct, varying struct and entry-point signature | **True.** 77 generated lines for 30 authored, from a 22-line spec. And the generated program is a pure function of the spec, which the pipeline cache relies on. |
| three.js requires hand-declared uniforms | **False for the `ShaderMaterial` path** (one list, and `material.uniforms` is public), **true for the `onBeforeCompile` path** (a second list, plus a mutable internal `parameters` object). |
| string surgery on undocumented `#include` chunk names | **True on the `onBeforeCompile` path** — and there it is 1984 generated lines hosting 6 authored ones. **Not applicable** to a from-scratch `ShaderMaterial`, which needs no chunks. |
| a mandatory undocumented `customProgramCacheKey` | **False on the `ShaderMaterial` path** (the shader source is hashed), **mandatory and undocumented on the `onBeforeCompile` path**, and omitting it is the worst silent failure measured here: 12 412 pixels wrong, no diagnostic. |
| reaching into `material.userData.shader.uniforms` to read your own uniforms | **True, but only on the `onBeforeCompile` path** (`material.uniforms` is `undefined` on a built-in). On a `ShaderMaterial` the value is a public property. |

The "dramatically simpler" part is not supported. On lines it is a dead heat —
**80 against 83** — and apse is *behind* on shading logic (21 lines of helper plus
body against 22) and on texture supply (20 against 12). What apse genuinely wins
on is the failure surface: a body identifier typo produces a typed error that
names the field and lists the alternatives, which is the best diagnostic in this
whole exercise and which three.js has no equivalent of. And the node-material path
that apse's own JSDoc points at is 15 lines, which is a stronger argument for
"write your material in three.js's node system" than anything in this paragraph.

**And the case against apse is not a matter of taste.** apse has one silent
failure three.js does not have, in its own generated data rather than in the
author's shader: `mat.normalMatrix` (now fixed: normalMatrixOf writes a padded 12-float scratch) arrives on the GPU as `(1,0,0), (1,0,0),
(1,NaN,NaN)` for an identity transform, which makes apse's own PBR material
render flat. It also has two error paths that produce
`is invalid due to a previous error` when the real diagnostic is available and
cheap to obtain via `getCompilationInfo`. Those belong in the README next to the
`ERROR_CATALOG` paragraph, which currently implies the failure surface is
uniformly good.

### Suggested rewrite of the README paragraph

> A custom material is about the same amount of code in both libraries — measured
> at 80 lines in apse against 83 for an equivalent three.js `ShaderMaterial`,
> rendering the same image to within one 8-bit step. The difference is in the
> failure surface, not the line count: apse turns a mistyped `frame.viewProjj`
> into a typed error that names the field and lists what exists, and three.js
> cannot do that at all. If you are patching a built-in material instead,
> `onBeforeCompile` is materially worse: it needs two internal chunk names, a
> mandatory undocumented `customProgramCacheKey` — omit it and 12 412 pixels
> render wrong with no diagnostic — and `material.userData.shader.uniforms` to
> read a uniform back. On the node-material path, which is what three.js's own
> docs recommend, the same material is 15 lines and none of the above applies.

---

## Appendix A — the apse material, in full

`bench/dx/apse-custom-material.ts`, counted sections only. 80 lines.

```ts
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
```

Note `out.normalW` uses `obj.model`, not `mat.normalMatrix` (now fixed: normalMatrixOf writes a padded 12-float scratch). That is §6, and it is
a defect in apse, not a style choice.

## Appendix B — the three.js `ShaderMaterial` version, in full

`bench/dx/three-custom-material.js`, counted sections only. 83 lines.

```js
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
```

## Appendix C — the `onBeforeCompile` version, in full

The path the claim is really about. 38 lines.

```js
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
```

Three of the paragraph's four assertions are visible in 12 lines of this: two
internal chunk names, a mandatory undocumented `customProgramCacheKey`, and
`material.userData.shader.uniforms`. The fourth — hand-declared
`modelViewMatrix` / `projectionMatrix` — is not here, because it is not true.

---

## Reproducing

```
bun run bench/dx/run.ts          # the whole thing; writes nothing to disk
DX_VERBOSE=1 bun run bench/dx/run.ts   # forward the page's console
```

The harness serves the repository root, transpiles `bench/dx/*.ts` on the way
out, and drives headless Chrome. It writes no files and asserts nothing: every
number above is printed, so a disagreement is a disagreement you can see.
