# Diagnostic quality: apse vs three.js 0.186.1

**The question.** apse has a 42-entry typed error catalog where every entry carries a
`why` and a `fix`. three.js 0.186.1 has `console.error`, a handful of `THREE.*` messages,
and a lot of silence. The claim under test is that apse's catalog is a real advantage.

**The method.** Nine failure modes, triggered for real in both libraries, in one headless
Chrome, in one page, on one GPU. For each: the exception and its fields, every console line,
every WebGPU validation error, and the pixels that came out. Not a reading of the source —
a run.

```
bun run bench/diag/run.ts          # all nine
bun run bench/diag/run.ts 4 6      # only those
```

Environment: macOS, Chrome (headless, real GPU), `apple metal-3 · compatibility`,
apse 0.0.1, three.js r186, WebGL2 `OpenGL ES 3.0 Chromium`.
apes' canvas 640×360 backing store; three.js clear colour `0x050508` to match apse's
`(5,5,8)`, so "drew black" and "drew nothing" stay distinguishable in both.

Two harness facts that make or break these numbers, both copied from `bench/index.html`:
the canvas is sized in CSS **and** has `clientWidth`/`clientHeight` defined (otherwise
`CanvasSizer` yields a 1×1 backing store and the GPU is never touched), and
`onValidationError` is wired to an array (a WebGPU validation error does not throw, it
invalidates the object, and the visible result is a blank frame). One apse device for the
whole run — Chrome drops the first WebGPU instance when a second is created, so scenario 8
runs last.

---

## 1. The table

Classification is of **what the developer sees**, not of what the library intends.
`apse-build` = the failure is unrepresentable, so no message is needed.
`could not trigger` is stated with the reason.

| # | Failure mode | apse | three.js 0.186.1 |
|---|---|---|---|
| 1 | Shader compile error in a custom material | **misleading** — typed `SHADER_COMPILE_FAILED`, but the quoted "compiler said" text names the *vertex* stage for a fragment error and contains no line or column, and the `fix` asserts the opposite | **actionable** — `console.error` with the GLSL line and a 12-line window of the source around it; no throw, black frame |
| 2 | Uniform name typo | **actionable** — `SHADER_BODY_INVALID` with a did-you-mean and the full field list, before any compiler runs. Silent for a *bare* undeclared identifier | **silent** for a misspelled `uniforms` key (renders black, nothing anywhere); **actionable** for a typo inside the GLSL |
| 3 | Varying type mismatch between stages | **apse-build** — one `varyings` declaration generates the struct for both stages, so there is nothing to disagree with. `VARYING_TOO_MANY` is a genuinely excellent typed error | **actionable** — two `console.error`s that name the mismatch exactly, though one reports a stale `Shader Error 1282` and neither anchors a source line |
| 4 | Program cache | **actionable by construction** — the shader body is in the pipeline key. Silent if you mutate a spec object after first use | **silent** — the classic unpatched/patched collision is **fixed** in 0.186; the closure-factory form is not, and the green material renders red |
| 5 | Wrong colour attachment format | **actionable** — `RENDER_TARGET_FORMAT_MISMATCH` naming both formats and the exact call to make; the raw consequence is separately verified | **silent** — 5 of 6 format combinations draw a call and produce a black frame, no message; three.js never calls `checkFramebufferStatus` |
| 6 | Reading pixels back from the canvas | **equal** — the documented hazard did **not** reproduce; `capture()` is correct and names the padded stride | **equal** — same; plus a raw `TypeError` on a bad MRT index and a silent no-op on an out-of-range rect |
| 7 | Disposing a shared resource | **actionable, opt-in** — `refCount` and `disposed` are observable, but `scene.add()` takes **no** reference, so the mistake the module claims to make impossible is still possible | **silent** — removing frees nothing, `material.dispose()` keeps the texture, a shared material disposed while still in use renders **black** in the other scene, no error |
| 8 | Using a renderer after `dispose()` | **misleading** — `RENDERER_ALREADY_DISPOSED` is unreachable; you get `DEVICE_LOST` with a `fix` that tells you to ignore your own teardown | **silent** — `render()` after `dispose()` works fine, re-registers the geometry, and draws the right picture |
| 9 | Wrong `minUniformBufferOffsetAlignment` stride | **actionable, unreachable** — 256 is a compile-time constant; a raw misaligned offset becomes `GPU_VALIDATION_FAILED`. The stride *is* a parameter of the public `collectDrawItems` and 128 produces `[0, 128, 256]` silently | **not applicable** — the WebGL2 material path has no UBO and no dynamic offset, so the invariant does not exist |

### Count

| | |
|---|---|
| apse better | **5** — 2, 4, 5, 7, 9 |
| apse worse | **2** — 1, 8 |
| equal | **2** — 3 (apse wins by unrepresentability, three.js wins on message text for the reachable case), 6 |

Two further numbers that matter more than the tally:

- **4 of apse's 42 codes are never raised anywhere in `src/`.** `MATERIAL_DISPOSED`,
  `MESH_DISPOSED`, `VARYING_MISMATCH`, `SHADER_NO_ENTRYPOINT`. The catalog advertises a
  failure surface that four of its members do not cover.
- **38 of 140 `fail()` call sites — 55 sites — are `INTERNAL_INVARIANT`**, whose catalog
  entry reads *"This always indicates a bug in apse, not in your code."* The most-raised
  code is the one that says the fault is not yours. (The call sites supply their own
  `why`/`fix`, so the misleading catalog text does not reach the developer — but the
  *code* and its *link* do, and a handler branching on `code` sees a user bug labelled as
  a library bug. See scenario 8.)

---

## 2. Evidence, verbatim

### 2.1 The compiler's real diagnostic, and the error that replaces it

apse never calls `getCompilationInfo()` — `grep -rn getCompilationInfo src/ test/` returns
nothing. The information exists. It arrives as an **uncaptured** WebGPU error:

```
[GPUValidationError] Error while parsing WGSL: :67:3 error: expected ';' for variable
  declaration return vec4f(mat.tint * l, 1.0); ^^^^^^ - While calling [Device
  "apse.device"].CreateShaderModule([ShaderModuleDescriptor ""apse:s1"]])
```

The thrown error, which is what a developer catches, carries this instead:

```
AseError  code  SHADER_COMPILE_FAILED
message   Material "typo-demo" did not compile.
why       ... Compiler said: [Invalid ShaderModule "apse:shader:typo-demo"] is invalid due
          to a previous error.
            - While validating vertex stage ([Invalid ShaderModule
              "apse:shader:typo-demo"], entryPoint: "vs").
            - While validating vertex state.
fix       Read the message above — it names the offending line, which is inside your body,
          not in generated code. Call describeMaterial() for the fields that exist, and
          pass `scaffold: true` to print the full 1779-byte program.
link      https://apse.dev/errors/shader-compile-failed
```

The typo was in the **fragment** body. The message names the **vertex** stage. The `fix`
says the message names the offending line; it does not. This is *misleading* under the
task's own definition — it names something, and the thing it names is not what you did
wrong. The identical cascade appears for a fragment-only typo, a type error, and a bare
undeclared identifier, so it is what you get for **every** WGSL compile error, not an
unlucky one.

The generated program, so the position is checkable. `|` fences survive the printer's
whitespace collapsing:

```
62|// ---- generated by apse: fragment stage ----
63|@fragment
64|fn fs(in : Varyings) -> @location(0) vec4f {
65|  let n = normalize(in.nrm);
66|  let l = max(dot(n, normalize(vec3f(0.3, 0.9, 0.4))), 0.0)
67|  return vec4f(mat.tint * l, 1.0);
68|}
69|
```

The compiler says `67:3`. That is one line *after* the mistake — Tint reports where it
discovered the missing `;`, not where it belongs. Worth knowing, because a developer who
trusts the line number will edit line 67 and see the same error.

**three.js, same bug, same page:**

```
error  THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false
       Material Name: Material  Type: MeshStandardMaterial
       Program Info Log: Fragment shader is not compiled.
       FRAGMENT ERROR: 0:1566: 'diffuseColor' : syntax error
       1561: varying vec3 vClipPosition;
       1562: uniform vec4 clippingPlanes[ 0 ];
       1563: #endif
       1564: void main() {
      >1565: float fogAmount = 0.25
       1566: diffuseColor.rgb = mix( diffuseColor.rgb, vec3( 0.5 ), fogAmount );
       1567: vec4 diffuseColor = vec4( diffuse, opacity );
       1568: #if 0
```

No exception. `programs: 1`, `drawCalls: 1`, and the readback is `centre: [5, 5, 8, 255]` —
byte-identical to the clear colour, i.e. nothing drew. `diagnostics.runnable === false` is
available on the program object, but only if you know to look.

Two things to be fair about. The **line number is useless** — 1566 of a source the
developer has never seen; the source they hold is 131 lines. The **window is the value**,
and it does contain both of the user's lines with `>` on the faulty one. And the message is
itself slightly wrong: `'diffuseColor' : syntax error` names the token *after* the missing
semicolon. Both libraries blame the following line; three.js at least shows you the line
before it.

### 2.2 The guard that works, and its boundary

A prefixed typo, caught before any compiler is involved:

```
AseError  code  SHADER_BODY_INVALID
message   The fragment body of material "uniform-typo-frame" is invalid:
          `frame.viewProt` is not a field.
fix       Did you mean `frame.viewProj`? Available on frame: view, proj, viewProj, invView,
          invProj, invViewProj, camPos, time, delta, elapsed, resolution, viewport, exposure,
          alpha.
```

A slot typo: `` `mat.tinnt` is not a field. `` → `Did you mean `mat.tint`? Available on mat: tint.`
A runtime slot write: `OPTION_UNKNOWN: Material "basic" has no slot "tinnt". — fix: Use one of: tint, opacity. call describeMaterial() for the full inventory including offsets.`
(That last one has a lowercase `call` after a full stop. Copy nit.)

**The boundary.** The identifier scan only sees `in.`, `out.`, `frame.`, `obj.`, `mat.`,
and declared texture variables. A *bare* undeclared identifier sails past it:

```
{ name: 'uniform-typo-bare', slots: { tint: 'vec3f' },
  fragment: 'let c = albedoTex * mat.tint;\nreturn vec4f(c, 1.0);' }
→ SHADER_COMPILE_FAILED   caughtBeforeCompile: false
```

And then it gets the cascade message from §2.1. So the class of mistake the guard is
advertised for is covered, and the very similar class one token away is not.

**three.js, same page, two variants.** A misspelled `uniforms` key:

```
uniforms: { uColour: { value: new THREE.Color(1,0,0) } }   // GLSL declares uColor
→ console:  (nothing printed)
→ centre:   [ 0, 0, 0, 255 ]      // drew, uniformly black
→ program uniform list: ["modelViewMatrix", "projectionMatrix", "uColor"]
```

Perfectly silent. three.js *knows* `uColor` exists — it is in the program's uniform list —
and never says that your map does not contain it. With the key spelled correctly the
centre is `[255, 0, 0, 255]`. With the typo inside the GLSL instead:

```
error  THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false
       FRAGMENT ERROR: 0:59: 'uColor' : undeclared identifier
       ERROR: 0:59: 'constructor' : not enough data provided for construction
       58: uniform vec3 uColur;
      >59: void main() { gl_FragColor = vec4( uColor, 1.0 ); }
```

Names the line, names the identifier, shows the source. Also emits a second, spurious
error about `'constructor'` that has nothing to do with the user's code.

### 2.3 Varyings: unrepresentable, versus well-diagnosed

apse's `varyings` declaration generates the struct for both stages from one source:

```wgsl
struct Varyings {
  @builtin(position) clip : vec4f,
  @location(0) nrm : vec3f,
};
```

There is no second place to disagree with, so `VARYING_MISMATCH` cannot fire. The catalog
says it exists. `grep -rn "fail('VARYING_MISMATCH'" src/` returns nothing.

The limit that *can* be reached is excellent, and correctly reads the device rather than a
hardcoded default:

```
AseError  code  VARYING_TOO_MANY
message   Material "too-many-varyings" declares 20 varyings, but the fragment stage allows
          15 input variables in total and `@builtin(position)` takes one of them.
fix       Declare at most 14 varyings. Pack related values into one vec4, or pass
          high-frequency data through a flat storage buffer indexed by the varying that is
          already there.
```

15 and 14 come from `device.limits.maxInterStageShaderVariables` on this compatibility
device, not from the core default of 16.

Forcing a type error anyway (assigning a `vec4f` to a `vec3f` varying) produces the §2.1
cascade. The test intended to check an unsupported varying type compiled successfully, so it
tests nothing; `vec2f` is a legal varying type and the test was wrong, not the library.

**three.js, both reachable variants:**

```
error  THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false
       Program Info Log: Types of varying 'vN' differ between VERTEX and FRAGMENT shaders.
       FRAGMENT varying vN does not match any VERTEX varying

error  THREE.WebGLProgram: Shader Error 1282 - VALIDATE_STATUS false
       Program Info Log: FRAGMENT varying vNeverWritten does not match any VERTEX varying
```

These are **better messages than anything apse emits for the same class of problem**: they
name the varying, name the direction of the mismatch, and distinguish "wrong type" from
"no matching output". The defects are that the second reports `Shader Error 1282` — a stale
`gl.getError()` from the previous failure, since no GL error occurred here — and neither
anchors a source line, because there is no `ERROR: 0:N` to anchor on. Both leave the object
silently undrawn: centre `[255, 0, 255, 255]`, the background, with `drawCalls: 1`.

### 2.4 The program cache

**The classic form is fixed in 0.186.1, and this is a point in three.js's favour.**
`Material.customProgramCacheKey()` (node_modules/three/src/materials/Material.js:544)
returns `this.onBeforeCompile.toString()`, so the patch's source text is in the cache key:

```
classicForm.cacheKeysEqual    false
  patchedKey  "function (shader) { shader.fragmentShader = shader.fragmentShader.replace(…"
  plainKey    "onBeforeCompile( /* shaderobject, renderer */ ) {}"
  patchedCentre  [152, 0, 0, 255]     plainCentre  [152, 152, 152, 255]
  programsAfterPatched 6 → programsAfterPlain 7        collapseObserved: false
```

Two programs, two different pictures. If any comparison document claims three.js 0.186
shares a program between a patched and an unpatched material, that claim is out of date.

**The form that is still broken, and is silent.** The key is the patch function's *source
text*, so two closures from one factory are indistinguishable however different their
captured values:

```js
function makeTint(r, g, b) {
  return function (shader) {
    shader.fragmentShader = shader.fragmentShader.replace(
      'vec4 diffuseColor = vec4( diffuse, opacity );',
      `vec4 diffuseColor = vec4( diffuse * vec3( ${r}, ${g}, ${b} ), opacity );`);
  };
}
mRed.onBeforeCompile   = makeTint(1, 0, 0);
mGreen.onBeforeCompile = makeTint(0, 1, 0);
```

```
redKeyEqualsGreenKey    true          differentFunctionObjects  true
redCentre   [152,   0,   0, 255]
greenCentre [152,   0,   0, 255]      <-- the green material renders red
newProgramsForGreen     0
sameProgramObject       true
console: (nothing printed)   validation: (none)
```

`WebGLRenderer.js:2196` computes the cache key, and `:2248` calls `onBeforeCompile` —
the patched source is never an input to the key. Nothing is printed, nothing throws, and
the frame looks entirely plausible. This is the hours-long one.

**apse, same scenario.** The body is in the key, so two materials that shade differently
cannot share a pipeline:

```
redKey 2227091249   greenKey 3194149123   keysDiffer true   samePipelineObject false
redCentre   [255, 0, 0, 255]      greenCentre [0, 255, 0, 255]
```

`pipelineKeyOf` hashes the pipeline state *plus* `tokenKeyOf(code)` — the whole generated
program with comments stripped and whitespace collapsed — so sharing is given up exactly
where correctness requires it and nowhere else. This is a real design win, and it needs no
escape hatch from the user.

**The 32-bit hash.** A brute-force search over generated programs found no collision in
1,048,576 programs (~37 s; the search is time-budgeted, so a slower machine reports fewer).
Report it as what it is: a measured absence, not a proof, and not a characterisation of the
margin. For scale, 4,096 materials over 2³² keys is a ~0.2 %
birthday probability. Cheap insurance would be a 53-bit or 64-bit key, or a collision check
on the token string.

**apse's own version of the same bug, opt-in.** `resolveSpec` memoises on the spec object in
a `WeakMap`, so mutating a spec after its first use is silently ignored:

```js
const spec = { name: 'mutated-spec', vertex: /*…*/, fragment: 'return vec4f(1.0,0.0,0.0,1.0);' };
const first  = await Material.create(device, spec);
spec.fragment = 'return vec4f(0.0,1.0,0.0,1.0);';     // the object now says green
const second = await Material.create(device, spec);
→ firstWgslHasGreen false   secondWgslHasGreen false
  samePipelineObject true   firstCentre [255,0,0]   secondCentre [255,0,0]
```

The spec object literally contains the green body and the material built from it is still
red. Less dangerous than three.js's variant because it requires mutating a spec in place,
which nobody does deliberately — but it is the same shape of defect: the identity used for
caching is not the identity of the thing being cached.

### 2.5 The attachment format, and proof of the consequence

```
AseError  code  RENDER_TARGET_FORMAT_MISMATCH
message   Material "pbr" was compiled for colour format "bgra8unorm" but is being drawn
          into a target of format "rgba16float".
why       WebGPU bakes the colour attachment format into a render pipeline. A pipeline
          cannot be used with a pass whose colour format differs, and the mismatch
          invalidates the whole command buffer — not just this draw — so nothing appears
          on screen and no exception is thrown.
fix       Pass the target's format when creating the material:
          pbrMaterial(device, { targetFormat: "rgba16float" }). The canvas default is
          navigator.gpu.getPreferredCanvasFormat(), which is bgra8unorm on desktop.
```

That `why` makes a falsifiable claim, so it was tested on raw WebGPU — an `rgba8unorm`
pipeline drawn into a `bgra8unorm` pass, then the texture read back:

```
[GPUValidationError] Attachment state of [RenderPipeline "diag:raw-pipeline-rgba8"] is not
  compatible with [RenderPassEncoder (unlabeled)]. [RenderPassEncoder (unlabeled)] expects
  an attachment state of { colorTargets: [0={format:TextureFormat::BGRA8Unorm}],
  sampleCount: 1 }. [RenderPipeline "diag:raw-pipeline-rgba8"] has an attachment state of
  { colorTargets: [0={format:TextureFormat::RGBA8Unorm}], sampleCount: 1 }.
[GPUValidationError] [Invalid CommandBuffer from CommandEncoder "diag:raw-encoder"] is
  invalid due to a previous error.
readback first pixel: [0, 0, 0, 0]      clear was (0,0,64); red would mean the draw landed
```

**The claim is verified: the clear did not even land.** The `why` is not a plausible story,
it is what happens.

**three.js, six format combinations, one process:**

| render target texture | framebuffer status | draws | lit pixels read back | said |
|---|---|---|---|---|
| RGBA8 / UnsignedByte (control) | COMPLETE | 1 | 540 | — |
| RGBA8UI / UnsignedInt | COMPLETE | 1 | **0** | **nothing** |
| RGBA32F / FloatType | COMPLETE | 1 | **0** | **nothing** |
| RGBA16F / HalfFloatType | COMPLETE | 1 | **0** | **nothing** |
| R8 / RedFormat | COMPLETE | 1 | 150 | nothing |
| RGB8 / RGBFormat | COMPLETE | 1 | **0** | one `readRenderTargetPixels` error |

Five of six draw a call and produce a black frame with no message. `gl.getError()` returned
`1282` (`INVALID_OPERATION`) after three of the binds and three.js swallowed it.
`grep -rn checkFramebufferStatus node_modules/three/src/` returns **0 hits** — the library
never asks whether its framebuffer is usable. That the check exists is not in doubt: a
hand-built `RGBA8UI` colour attachment reports `INCOMPLETE_ATTACHMENT`.

One point in three.js's favour, the only diagnostic in that table: asking for a readback of
a non-RGBA target does say something —

```
error  THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or
       implementation defined format.
```

— but only if you ask, and it names the readback rather than the mistake.

### 2.6 Reading pixels back: the documented hazard did not reproduce

Honest negative result. The claim in `capture()`'s docstring and in `AGENTS.md` is that a
WebGPU canvas has no `preserveDrawingBuffer` and reading it after present gives black. In
headless Chrome on macOS, the canvas was readable in **every** case, for **both** libraries:

| read | apse | three.js (`preserveDrawingBuffer: false`) |
|---|---|---|
| same task, immediately after `render()` | `[242, 89, 31, 255]` | `[152, 152, 152, 255]` |
| after `await queue.onSubmittedWorkDone()` | `[242, 89, 31, 255]` | `[152, 152, 152, 255]` |
| after `await setTimeout(0)` | `[242, 89, 31, 255]` | `[152, 152, 152, 255]` |
| after `await requestAnimationFrame()` | `[242, 89, 31, 255]` | `[152, 152, 152, 255]` |
| `toDataURL('image/png')`, decoded | `[242, 89, 31, 255]` | `[152, 152, 152, 255]` |

All five agree with the supported path, so the test does not distinguish them. **Caveat:
headless has no compositor present step, so this does not exercise the hazard the docstring
describes.** Treat it as "not reproduced here", not as refuted — and note that it took five
probes to establish even that, which is itself the cost.

Where the two differ is in what they *offer*:

- `capture()` returns `{ data, width, height, bytesPerRow, format }` and the type says
  `bytesPerRow` is **not** `width * 4`. The padding is real:
  `bytesPerRow (2564) is not a multiple of 256` at width 641, where the padded stride is
  2816 — 252 bytes of shear per row if you index at `width * 4`.
- `readRenderTargetPixels` throws a raw `TypeError: Cannot read properties of undefined
  (reading 'format')` from inside three.js when given an out-of-range MRT index, and
  silently leaves the caller's buffer untouched with no message when given an
  out-of-range read rectangle.

### 2.7 Disposal

**apse, step by step:**

```
baseline              mesh refCount 1   material refCount 1
afterTwoNodesInScene  mesh refCount 1   material refCount 1   <-- scene.add() refs nothing
afterExplicitRef      mesh refCount 2   material refCount 2
afterSceneRemove      mesh refCount 2   material refCount 2   meshDisposed false
afterUnrefToZero      mesh refCount 0   meshDisposed true
drawWithDisposedMesh  threw false   console (nothing)
                      validation: [Buffer "box:vertex"] used in submit while destroyed.
                      pixels: coverage 1, maxLuma 0, centre [0, 0, 0, 0]
```

The refcount is real, observable, and correct. It is also **opt-in and outside the scene
graph**: `MeshNode`'s constructor stores `options.mesh` and `options.material` without
taking a reference, and `Scene.add` is `root.add(node)`. `src/core/resource.ts` says
"removing an object from a scene frees nothing … and makes both mistakes impossible". The
first half is true of both libraries. The second half is not: in apse, drawing a disposed
mesh produced **no exception and no console output** — one raw Dawn line, and a readback that
was entirely empty (`maxLuma 0`, alpha 0).

**three.js, same page, `renderer.info` counters:**

```
before                    geometries 0  textures 0  programs 0
afterFirstRender           geometries 1  textures 1  programs 1
afterSceneRemove           geometries 1  textures 1  programs 1   <-- removing frees nothing
afterMaterialDispose       geometries 1  textures 1  programs 0   <-- the texture survives
afterGeometryDispose       geometries 1  textures 1  programs 0   <-- and the counter does not drop
afterReuseOfDisposedMat    geometries 2  textures 2  programs 1   <-- silent resurrection
reuseCentre                [18, 18, 25, 255]                      <-- and nearly black
```

The claims hold: removing frees nothing, `material.dispose()` does not free its texture.
`info.memory.geometries` also failed to decrement after `geometry.dispose()` — the counter
appears to lag until the next render, so you cannot use it to confirm a free.

The expensive one is sharing:

```js
// scene 1 and scene 2 both use sharedMat. Dispose it from scene 1's teardown.
sharedMat.dispose();
three.renderer.render(scene2, camera);
→ otherSceneThrew    null
→ otherSceneCentre   [0, 0, 0, 255]
```

**A material disposed while another scene still uses it turns black there. No exception, no
console output, no counter change.** And three.js exposes no liveness information at all to
check with: `geometryHasDisposeFlag: false`, `materialHasDisposeFlag: false`,
`textureHasDisposeFlag: false`, `refCountProperty: null`. There is nothing to assert on.

### 2.8 After `dispose()`

**apse.** A disposed material, then written to:

```
AseError  code  INTERNAL_INVARIANT
message   Material "basic" was used after it was disposed.
why       The GPU buffers backing it are gone, so the resulting draw is undefined.
fix       Hold a reference with `.ref()` for as long as a second object still uses this
          resource.
link      https://apse.dev/errors/internal-invariant
```

The message and the fix are right. The **code is wrong**, and the code is the part the
contract says to branch on: `INTERNAL_INVARIANT` is catalogued as *"This always indicates a
bug in apse, not in your code."* This is unambiguously the user's bug. `MATERIAL_DISPOSED`
exists in the catalog, is documented, and is never raised. A handler branching on `code`
files this under "apse is broken".

A disposed material or mesh still **renders**: `disposedMaterial.draw: "rendered with no
complaint"`, and the disposed mesh produced one raw Dawn line and an empty frame.

And the renderer itself:

```
AseError  code  DEVICE_LOST
message   This apse device can no longer be used: destroy() was called on it.
fix       Handle `onDeviceLost`, then call `device.recover()` and rebuild every GPU
          resource. … Check `reason` and ignore "destroyed", which is your own teardown.
```

`RENDERER_ALREADY_DISPOSED` exists, is documented, and is **unreachable**: `render()` calls
`#assertLive()` at line 279, which reaches `device.destroy()`'s `#lost` flag, and throws
`DEVICE_LOST` before reaching the `#disposed` check at line 280. The developer gets told to
handle a device-loss event and then told, in the same message, to ignore this one because it
is their own teardown. The right answer — "create a new Renderer" — is in the catalog
`fix` for a code that cannot be reached.

**three.js.** A second WebGL context, so the shared one survives:

```
disposedRenderer  threw false   pixels coverage 0.1284   centre [152, 152, 152, 255]
                 memoryAfter geometries 2 textures 1   programsAfter 1
disposedMaterialAndGeometry
                 threw false   centre [0, 152, 0, 255]   programs 1 → 1
```

`renderer.dispose()` does not lose the context and does not mark the renderer dead.
`render()` afterwards **works**, silently re-registers the geometry, and draws the correct
picture. Disposing a material and a geometry and rendering with them again also works, and
again silently. There is no signal at all — and note that here, "nothing happened" is the
best possible outcome, which is exactly what makes it untestable by inspection.

`dispose()` also does not reset `material.version` (still 0), so any code watching that as
a change signal is watching nothing.

### 2.9 The 256-byte stride

apse's stride is a compile-time constant, and it is checked:

```
declaredStrideBytes 256   deviceMinAlignment 256   strideIsAMultiple true
objectUniformsStride 256  objectUniformsSize 128   offsetFor(0,1,2) → [0, 256, 512]
```

128 bytes of data padded to 256, exactly as `device.ts` documents. The renderer always uses
`OBJECT_UNIFORM_STRIDE`, so you cannot get a bad offset through it.

**The one reachable hole.** `Scene.collectDrawItems(out, camera, objectUniformStride = 256)`
takes the stride as a public parameter, and a wrong one produces a misaligned offset with no
error at all:

```
scene.collectDrawItems(items, camera, 128)  →  offsets [0, 128, 256]
                                               aligned  [true, false, true]
```

Object 1 sits at byte 128. Nothing complains at the point of the mistake. Forcing that
offset onto a real bind group:

```
[GPUValidationError] Dynamic Offset[0] (128) is not 256 byte aligned. - While encoding
  [RenderPassEncoder (unlabeled)].SetBindGroup(0, [BindGroup (unlabeled)], 1, ...).
[GPUValidationError] [Invalid CommandBuffer …] is invalid due to a previous error.
readback centre [0, 0, 0, 0]        (aligned offsets gave [0, 0, 0, 255] — the clear landed)
```

and inside `withErrorScope`:

```
AseError  code  GPU_VALIDATION_FAILED
message   apse:object bind group at a 128-byte dynamic offset raised a WebGPU "validation"
          error. Driver said: Dynamic Offset[0] (128) is not 256 byte aligned. …
fix       Read the raw driver text above: it names the exact descriptor field that was
          wrong. `Dynamic Offset[0] (128) is not 256 byte aligned.
          - While encoding [RenderPassEncoder (unlabeled)].SetBindGroup(0, …).
          - While finishing [CommandEncoder "diag:offset-scoped"].`
```

Typed, yes. But the `fix` is the raw message pasted into backticks a second time, newlines
and all, and it does not say the one thing that would help: *the dynamic offset must be a
multiple of `minUniformBufferOffsetAlignment`*. It also repeats itself — the driver text is
already in `message`. `GPU_VALIDATION_FAILED` is the one code in the catalog whose `fix` is
generated by string concatenation rather than written, and it shows.

**three.js: not applicable.** The WebGL2 material path has no uniform buffer object and no
dynamic offset. Measured: `UNIFORM_BUFFER_OFFSET_ALIGNMENT 16`,
`MAX_UNIFORM_BUFFER_BINDINGS 32`, and the material's uniform strategy is
`per-program WebGLUniforms list of 11 entries uploaded with gl.uniform* on every draw; no
uniform buffer object and no dynamic offset`. The invariant does not exist to violate.
`THREE.UniformsGroup` is the only UBO path and it is not reachable from a material.

---

## 3. Verdict

**"Typed errors that teach" is a real but narrow advantage, and the README's version of the
claim is too strong.**

What survives contact:

- Where a code fires, the `why`/`fix` pair is usually better than anything three.js emits,
  and in four cases (`SHADER_BODY_INVALID`, `VARYING_TOO_MANY`, `RENDER_TARGET_FORMAT_MISMATCH`,
  `OPTION_UNKNOWN` on a slot name) it is dramatically better — a did-you-mean and a full
  field list, before any compiler is involved, is a class of fix-time that no amount of
  `console.error` reaches.
- Two failures are **unrepresentable** rather than diagnosed: a cross-stage varying
  mismatch (§2.3) and a pipeline-cache collision between different shader bodies (§2.4).
  Not having an error because the mistake cannot be expressed is worth more than having a
  good one.
- One of apse's claims about itself is falsifiable and **verified**: the
  `RENDER_TARGET_FORMAT_MISMATCH` `why` says the command buffer is invalidated; the raw
  readback confirms the clear did not even land.

What does not survive:

- The **most expensive failures are still apse's silent ones**, and they are the same
  failures three.js has. A disposed material renders (§2.8). A disposed mesh renders, with
  one raw Dawn line reachable only through `onValidationError` and an empty frame
  (§2.7). The whole-command-buffer invalidation of §2.5 is silent by construction —
  `onValidationError` is the only channel, it must be wired deliberately, and
  `isDevelopmentMode()` gates the console fallback so a production build says nothing.
- **The catalog advertises a surface four of its members do not cover**, and the code that
  *is* most used is the one that says the fault is not yours. A user bug
  (`Material used after dispose`) is reported as `INTERNAL_INVARIANT`, documented as *"always
  a bug in apse"*. For a contract whose whole pitch is "branch on `code`, never on
  `message`", that is the single most damaging defect found.
- **The flagship message is the weakest one.** `SHADER_COMPILE_FAILED` quotes a cascade that
  names the wrong stage and no line, and its `fix` asserts that the quoted text names the
  offending line. It does not. The real diagnostic is available through
  `getCompilationInfo()` and is thrown away. A developer who trusts the `fix` will read a
  message with no position in it, and be told to look for one.
- **The one class of failure apse claims to have eliminated is still there.** Sharing and
  disposal is exactly where the hours go, and apse's mechanism is real but opt-in and
  outside the scene graph. `scene.add()` takes no reference, so "removing an object from a
  scene frees nothing" is true of apse exactly as it is of three.js.

### The uncovered failure space — what to build next

Roughly two thirds of the measured failure space is untouched, and the shape of it is
consistent: **everywhere the mistake is made in a descriptor apse did not write, and
everywhere the mistake is a *lifecycle* mistake rather than a *value* mistake.**

Ranked by cost per unit of work:

1. **Wire `getCompilationInfo()` into `SHADER_COMPILE_FAILED`.** Smallest change, largest
   effect, and it makes an existing `fix` true. Pop the error scope, read
   `module.getCompilationInfo()`, and put `lineNum:linePos`, the source line, and the
   `^^^^` caret span into the `why`. The information is already proven to exist (§2.1).
   This one change converts apse's worst message into its best.
2. **Raise `MATERIAL_DISPOSED` and `MESH_DISPOSED`, and delete them from `INTERNAL_INVARIANT`.**
   The catalog has them; `assertLive` does not use them. Then a user's bug stops being
   catalogued as a library bug, which is the defect the whole contract turns on.
3. **Make the scene graph take references.** `MeshNode`'s constructor calls `mesh.ref()`
   and `material.ref()`; `removeFromParent` unrefs. `refCount` and `disposed` already exist
   and are already correct. This is the difference between "apse reference-counts" being
   true and being available, and it closes §2.7 — the failure that costs the most hours in
   both libraries.
4. **Close the `collectDrawItems` stride hole** (§2.9). Either drop the parameter, or
   `fail('OPTION_UNKNOWN', …)` when it is not a multiple of the device's
   `minUniformBufferOffsetAlignment`. Today it produces a bad offset silently, and the only
   symptom appears two frames later at a `setBindGroup` the caller never wrote.
5. **Make `RENDERER_ALREADY_DISPOSED` reachable**, and make `DEVICE_LOST` distinguish an
   explicit teardown from a real loss in the `fix`. Right now the app's own `dispose()`
   produces an error whose remedy is to ignore it.
6. **Give `GPU_VALIDATION_FAILED` a written `fix`.** It is the only generated one, and
   concatenating the raw driver text into a `fix` that already appears in `message` is worse
   than leaving it out. A short per-operation hint — for a dynamic offset, *"the offset must
   be a multiple of `minUniformBufferOffsetAlignment`"* — is the pattern the catalog
   already demonstrates everywhere else.
7. **Widen the 32-bit pipeline key to 53+ bits.** No collision in 1,048,576 generated programs, but the
   margin is uncharacterised and a 64-bit key costs nothing measurable.

### What to change in the README

- **Drop or narrow "42 codes".** 38 are reachable; 4 are documented and never raised. The
  honest phrasing is "42 codes, 38 of them reachable today" — which is a *stronger* claim,
  because it is the one a reader can check.
- **Drop any claim that apse's `SHADER_COMPILE_FAILED` points at your line.** It does not
  today. Say what it *does*: a typed code at the exact call that compiled the shader, with
  the driver's text attached. Then fix it and change the sentence.
- **Soften "makes both mistakes impossible" in `resource.ts` and `ARCHITECTURE.md`.** The
  mechanism exists and is observable; the scene graph does not use it. That sentence is
  checkable, and it is currently wrong.
- **Keep the `RENDER_TARGET_FORMAT_MISMATCH` claim** — it is the one that held up under
  test, and the raw readback in §2.5 is the proof worth quoting.
- **Quote §2.4's three.js result if you make the program-cache comparison.** The classic
  form is fixed in 0.186 and claiming otherwise would be wrong; the closure-factory form is
  live and silent, and that is the one to contrast against.
