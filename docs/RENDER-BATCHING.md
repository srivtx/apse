# Can apse beat three.js per draw call?

**Status: research and design. No `src/**` is edited by this document's author.**
One real bug is reported in [§7](#7-a-real-bug-not-fixed). Two claims in the
brief this document set out to test turned out to be false, and both are retracted
here rather than carried forward — see [§1](#1-the-premise-is-wrong).

Every number is labelled **[measured here]**, **[measured elsewhere]**, **[inferred]**
or **[unknown]**. Claims about the WebGPU spec or browser behaviour carry a URL.
Where I could not verify something, it says so.

---

## 1. The premise is wrong

The brief's central claim is that three.js reaches *bind 1, draw* per mesh, that
WebGPU cannot, and that apse's floor is therefore *bind 2, draw* — and that this
structural difference is the 0.87 vs 0.66 µs/draw gap.

**All three of those numbers are wrong.** I instrumented three.js r186's WebGL
context and counted every GL call for 500 distinct meshes:

```
500 DISTINCT geometries (a new VAO per mesh):
  total 1501   per draw 3.00   drawElements 500  bindVertexArray 500  uniformMatrix4fv 500
500 meshes SHARING one geometry (one VAO reused):
  total 1001   per draw 2.00   drawElements 500                  0     uniformMatrix4fv 500
```

**[measured here]** three.js r0.186.1, `MeshStandardMaterial`, 500 meshes, one
shared material, headless Chrome, same machine as the project's harness. Three
consecutive frames each, identical to the digit. Probe:
`three-census.html` (harness described in [§8](#8-reproducing-my-measurements)).

So three.js issues **3.00 GL calls per draw** for distinct meshes, not 1. The VAO
is one of the three. The other two are `drawElements` — which apse also issues —
and a per-draw `uniformMatrix4fv` for the object transform, which apse does
**not** issue per draw.

Against that, apse's measured 4.00 calls per draw:

| | calls/draw | breakdown |
|---|---:|---|
| three.js, distinct meshes **[measured here]** | **3.00** | `bindVertexArray` + `uniformMatrix4fv` + `drawElements` |
| three.js, shared geometry **[measured here]** | **2.00** | `uniformMatrix4fv` + `drawElements` |
| apse, 500 distinct meshes **[measured elsewhere]** | **4.00** | `setBindGroup` + `setVertexBuffer` + `setIndexBuffer` + `drawIndexed` |

**The gap runs the other way.** apse issues one more call per draw than three.js
does, and the extra one is not the vertex buffer — it is `setBindGroup`. The
"three.js gets to 1 call" claim in `bench/diag/perf/FINDING.md:103` ("three.js's
WebGLRenderer gets to roughly 4 too") is wrong in the same direction but by a
smaller margin, and the earlier claim in that file's first half — that the fix was
merging two bind groups — was already retracted there.

### What the VAO actually buys, measured

The control case isolates it. Holding the scene fixed and changing only whether
the 500 meshes share one `BufferGeometry`:

- distinct geometry → `bindVertexArray` fires 500×, total goes 1001 → 1501
- shared geometry → `bindVertexArray` fires 0×, total stays 1001

**[measured here]** So the VAO is worth **exactly one GL call per draw** — the
`bindVertexArray` — and *only* when geometries differ. three.js's own advantage
here is smaller than the brief assumed, and it is spent on a call apse makes
anyway for a different reason.

This does not contradict the Babylon.js team's published measurement that
`bindVertexArray` is "2.5x time faster than the combined `setIndexBuffer` +
`setVertexBuffer` calls" ([gpuweb#1640](https://github.com/gpuweb/gpuweb/discussions/1640)).
Both can be true: a VAO bind may be *cheaper* than two buffer binds while still
being an *extra* call that a WebGPU renderer has no way to avoid. The brief
conflated "cheaper per call" with "fewer calls". They are different questions and
the answer differs for each.

---

## 2. Direct answer: **only in cases X**

**No**, apse cannot beat three.js per draw call on a scene of N distinct meshes,
and the reason is now known rather than suspected: it is not a missing WebGPU
primitive. It is that apse spends a call per draw on `setBindGroup` to carry a
64-byte matrix, and three.js spends its equivalent call on a `uniformMatrix4fv`
that is cheap.

**But the framing "apse is 1.32× slower per draw" is the wrong target**, because:

1. apse's per-draw path is the path the project tells users not to use. Instanced
   CPU is **flat at 0.012 ms from 1,000 to 100,000 instances** and apse is
   **70× ahead at 100,000 objects** **[measured elsewhere** — `README.md:169-180`**.
   Optimising the per-draw path cannot improve the number the project leads with.
2. The measured cost of the thing to remove is small. `setBindGroup` with a
   dynamic offset is **0.35–0.55 µs** marginal in my harness **[measured here]**,
   and `setVertexBuffer` + `setIndexBuffer` together are **0.12–0.39 µs**
   **[measured here]**. There is no 0.21 µs/draw sitting in the vertex-buffer
   binds to be recovered.
3. The one mechanism that would remove it — `setImmediates` — is new, is **not
   in every browser**, and I measured that it *does* work in compatibility mode
   ([§3](#3-immediates-the-one-real-lever-and-it-is-new)).

---

## 3. Every mechanism that could reduce per-draw call count

The device is `compatibility` mode **[measured here]**, and apse's strategic bet
is compatibility mode, so anything core-only is not an answer.

| Mechanism | Exists today? | Works in compat? | Reduces calls/draw? | Source |
|---|---|---|---|---|
| **`setImmediates`** (push constants) | **Yes**, Chrome 149+ | **Yes — measured working** | **No.** Replaces 1 call with 1 | [chrome 149-150](https://developer.chrome.com/blog/new-in-webgpu-149-150), [spec §14.2](https://gpuweb.github.io/gpuweb/#programmable-passes-immediate-data) |
| **Render bundles** | Yes, since Chrome 106 | Yes — **measured working** | 4 → 1 call, **but see caveat** | [spec §18](https://gpuweb.github.io/gpuweb/#bundles) |
| **`multiDrawIndexedIndirect`** | **No** — draft PR, open since 2021 | N/A | Would be 4 → 0 per draw | [gpuweb#2315](https://github.com/gpuweb/gpuweb/pull/2315) |
| **`drawIndexedIndirect`** | Yes, core | Yes | **No** — still 1 draw per call | [spec](https://gpuweb.github.io/gpuweb/#render-pass-encoder-draw-indexed-indirect) |
| `indirect-first-instance` | Yes, optional feature | **No** — not requested by apse | No | [spec §25.10](https://gpuweb.github.io/gpuweb/#indirect-first-instance) |
| **Bindless / dynamic binding arrays** | **No** — proposal stage | N/A | Unknown | [gpuweb#5379](https://github.com/gpuweb/gpuweb/issues/5379) |
| `WEBGL_multi_draw` (what three.js uses) | **WebGL-only extension** | N/A | n/a | [WebGLRenderer.js:1317](https://github.com/mrdoob/three.js/blob/r186/src/renderers/WebGLRenderer.js#L1317) |

### 3.1 Immediates — the one real lever, and it is new

`setImmediates(rangeOffset, data, dataOffset, dataSize)` writes small data
straight to the shader, with no buffer and no bind group
([spec §14.2](https://gpuweb.github.io/gpuweb/#programmable-passes-immediate-data)).
Chrome shipped it in 149-150 and the blog's example is *precisely* apse's
situation: "a unique object ID or a 3D transformation matrix for hundreds of
objects" ([chrome](https://developer.chrome.com/blog/new-in-webgpu-149-150)).

**I tested it functionally, not just for presence** — four differently-tinted
quads in one pass, each with its own immediate, then a pixel readback:

```
featureLevel "compatibility"   maxImmediateSize 64
wgsl has immediate_address_space true   setImmediates present true
requested [[255,0,0],[0,255,0],[0,0,255],[255,255,0]]
observed  [[255,0,0,255],[0,255,0,255],[0,0,255,255],[255,255,0,255]]
distinct 4 -> immediates usable in compat: true    errors []
```

**[measured here]** It works in compatibility mode on this device, with zero
validation errors. A first attempt at this test returned four identical bands;
the cause was **my geometry** (I fed pixel offsets in as clip-space `x`), not the
feature. The corrected test is what is quoted above.

And it fits apse exactly: `OBJECT_FIELDS.model` is a `mat4x4f`
([`src/core/slot.ts:90`](../../src/core/slot.ts)) = **64 bytes**, and
`maxImmediateSize` is **64** **[measured here]**. The model matrix is exactly the
one thing that fits.

**Why it is still not a fix, stated honestly:**

- It does not reduce the call count. It replaces `setBindGroup` with
  `setImmediates`: 1 call becomes 1 call. It removes the *uniform buffer read*
  and the bind-group bookkeeping, not the call.
- It is not universally available. Chrome 149+ only, at time of writing. Safari's
  position is **[unknown]** — I did not test Safari, and I am not going to assert
  one. apse's compatibility-mode bet is precisely about devices that are
  *behind*, so a feature shipping in the newest Chrome is a poor foundation for
  the shipped default path.
- The WG resolved that immediates are a **core** feature, exposed in compat too
  ([WG minutes 2026-01-14](https://github.com/gpuweb/gpuweb/wiki/GPU-Web-2026%E2%80%9001%E2%80%9014)),
  which is consistent with what I measured. But "will be everywhere eventually"
  is not "is everywhere now".
- It would need `normalMatrix`, `objectId`, `instanceId`, `visibility` too
  (`src/core/slot.ts:92-98`), which do not fit in 64 bytes alongside the matrix.

**Verdict:** worth an *optional* fast path behind
`navigator.gpu.wgslLanguageFeatures.has('immediate_address_space')` — the
detection the Chrome blog itself recommends. Not worth making the default.

### 3.2 Render bundles — the 4 → 1 lever, and its real catch

Bundles exist and **work in compatibility mode** **[measured here]**: I created one
on this device, and `executeBundles` ran without a validation error.

Measured cost, from the same harness:

```
executeBundles, 1 call/draw, vs 4 inline calls:  -0.31 us/draw
executeBundles, 1 call/draw, vs 1 inline call:   +0.55 us/draw
```

**[measured here]** So a bundle is **not** a 4-call-for-1-call win. It is roughly
**0.31 µs/draw cheaper than the four inline calls, and 0.55 µs/draw *more
expensive* than a single `drawIndexed`**. A bundle is a container, not a fast
path; the commands inside it still cost what they cost.

The catch Babylon hit and apse would hit too: a bundle is keyed to
(pipeline, bind groups, vertex buffers, format), so you need **one bundle per
distinct mesh per pass** and you must detect every invalidation
([gpuweb#1640](https://github.com/gpuweb/gpuweb/discussions/1640),
[Babylon NCM docs](https://doc.babylonjs.com/setup/support/webGPU/webGPUOptimization/webGPUNonCompatibilityMode)).
With 500 distinct meshes that is 500 bundles to create, cache, and invalidate —
and apse's objects move every frame, so the per-object transform is a dynamic
offset, which a bundle **cannot** carry (bundle state is fixed at record time;
`executeBundles` does not inherit the pass's bind groups,
[spec §17.2.4](https://gpuweb.github.io/gpuweb/#render-pass-encoder-executebundles)).

**Verdict: not usable for the per-draw path as it stands.** It is a bundle-per-mesh
cache with a per-draw correctness cliff, for a 0.31 µs/draw saving.

### 3.3 Multi-draw indirect — does not exist

`multiDrawIndexedIndirect` is a **draft PR open since November 2021**
([gpuweb#2315](https://github.com/gpuweb/gpuweb/pull/2315)), superseding a closed
PR (#1949) and tracking issue (#1354). Chrome has an **experimental** version
behind `chrome://flags/#enable-unsafe-webgpu` under the non-standard feature name
`"chromium-experimental-multi-draw-indirect"`
([Chrome 131 blog](https://developer.chrome.com/blog/new-in-webgpu-131)).

I confirmed it is absent on this device: `multiDrawIndexedIndirect in
GPURenderPassEncoder.prototype` → **false** **[measured here]**.

Two further blockers even if it shipped: **Metal has no multi-draw at all** and
would need a per-draw loop or an ICB workaround
([#2315 thread](https://github.com/gpuweb/gpuweb/pull/2315)), and apse's
strategic bet is compatibility mode, i.e. **OpenGL ES 3.1** targets where
`multiDrawIndirect` is not available at all
([#1949](https://github.com/gpuweb/gpuweb/pull/1949), 63% of Android devices).

**Verdict: not an answer.** It does not exist, it is not in compat, and the
backend for apse's primary target lacks the hardware feature.

---

## 4. Is it the vertex-buffer binds or the draw?

**[measured here]** Directly: the marginal cost of each command, in a harness
whose variants differ by exactly one command and render identical geometry
(64 draws of 3 vertices into a 64×64 offscreen target, real submits, queue drained
every 4 frames, variants interleaved, `performance.now()` quantum 0.1 ms).

Two independent runs:

| marginal, vs 1-call baseline | run 1 | run 2 |
|---|---:|---:|
| `setVertexBuffer` (2 calls/draw) | 0.117 µs | 0.195 µs |
| `setIndexBuffer` (2 calls/draw) | 0.195 µs | 0.156 µs |
| `setBindGroup`, **dynamic offset** | **0.430 µs** | **0.547 µs** |
| `setBindGroup`, **no dynamic offsets** | **0.078 µs** | **0.000 µs** |
| all three (apse's shape) | 0.742 µs | 1.094 µs |
| `executeBundles`, 1 call/draw | 0.547 µs | 0.781 µs |
| `drawIndexed` alone (baseline) | 1.875 µs | 2.266 µs |

**The two vertex-buffer binds are not the problem.** Together they are
0.12–0.39 µs. Even eliminating both entirely — which requires merging every mesh
into one buffer — cannot recover a 0.21 µs/draw gap on its own.

**`setBindGroup` with a dynamic offset is.** At 0.43–0.55 µs it is **2–4× the
cost of a vertex-buffer bind**, and it is the single most expensive call apse
issues per draw. Note the contrast: the *same* `setBindGroup` **without** dynamic
offsets costs 0.00–0.08 µs. **The dynamic offset is the cost**, not the bind.

That is a sharper and more actionable finding than "the vertex-buffer binds are
the floor", and it points at a different line of code.

**Two harness failures worth recording, because both produced plausible-looking
nonsense.** My first two versions measured the wrong thing entirely:

- v1 submitted inside the timed loop and reported `draw_only` **slower** than
  `full_distinct` — impossible, and the tell. It was measuring queue
  backpressure. `bench/diag/perf/README.md` already lists this exact trap.
- v2/v3 never submitted and showed a **100× spread** between min and median for
  identical work, plus validation errors on a path that should have been valid.

Only v4/v5 (real submits, drained queue, interleaved) gave the tight min/p50/max
above. **A per-draw microbenchmark that does not submit is not measuring encode.**

---

## 5. What three.js does per draw that apse does not

From reading `node_modules/three` r0.186.1, with the per-draw GL census from §1
as the check on it.

### 5.1 It does *more* per draw, not less

three.js's per-draw path issues **3 GL calls**, apse's issues **4**. three.js is
still 0.66 µs/draw against apse's 0.87 **[measured elsewhere]**. So call count is
not the explanation, and anyone optimising toward "fewer calls than three.js" is
chasing the wrong variable.

### 5.2 The thing apse is genuinely missing: per-object data costs three.js a cheap call

`renderBufferDirect` calls `setProgram` on **every draw**
([WebGLRenderer.js:1202](https://github.com/mrdoob/three.js/blob/r186/src/renderers/WebGLRenderer.js#L1202)),
and `setProgram` unconditionally sets the three per-object matrices:

```js
p_uniforms.setValue( _gl, 'modelViewMatrix', object.modelViewMatrix );
p_uniforms.setValue( _gl, 'normalMatrix',    object.normalMatrix );
p_uniforms.setValue( _gl, 'modelMatrix',     object.matrixWorld );
```
([WebGLRenderer.js:2810-2812](https://github.com/mrdoob/three.js/blob/r186/src/renderers/WebGLRenderer.js#L2810))

Each is a `gl.uniformMatrix4fv`, guarded only by a value cache
([WebGLUniforms.js:329](https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgl/WebGLUniforms.js#L329))
that necessarily misses when the object moved. My census counted **500 of them**
for 500 draws.

**This is the actual asymmetry, and it is the opposite of the VAO story.** In
WebGL, per-object state travels as ordinary uniform writes: cheap, no descriptor
bookkeeping, no binding model. In WebGPU, uniform data lives behind a bind group,
and a *different* object means a *different dynamic offset* — and I measured that
costs 0.43–0.55 µs **[measured here]**.

apse already took the one available step: it merged frame+object into a single
buffer and bind group so the per-draw cost is one `setBindGroup` rather than two
([`src/core/slot.ts:116-130`](../../src/core/slot.ts), already retracted once in
`FINDING.md`). That step was correct and is done. **The remaining cost is
inherent to WebGPU's binding model, not to apse's implementation of it.**

### 5.3 Everything else in three.js's per-draw path is *guarded JS*, and apse's is too

`state.setMaterial` runs per draw ([WebGLRenderer.js:1204](https://github.com/mrdoob/three.js/blob/r186/src/renderers/WebGLRenderer.js#L1204))
and touches cull face, blending, depth, stencil, polygon offset — but
[WebGLState.js:754-790](https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgl/WebGLState.js#L754)
compares every value against a cached current-state and only calls GL on change.
**Zero GL calls when the material is unchanged.** apse's `lastPipeline` /
`lastMaterialBG` / `lastTextureBG` guards
([`src/render/renderer.ts:1054-1094`](../../src/render/renderer.ts)) are the
same pattern, and my census confirms apse's material/texture groups fire 2 and 1
times per frame, not 500.

**So there is no unguarded per-draw GL work in three.js that apse has failed to
replicate.** The state-culling discipline is already equivalent. I looked for a
missing trick and did not find one; I am not going to invent one.

### 5.4 VAO caching, precisely

`WebGLBindingStates` keys a VAO by `geometry.id × (instanced ? object.id : 0) ×
program.id × wireframe`
([WebGLBindingStates.js:70-115](https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgl/WebGLBindingStates.js#L70)),
and binds it only when the state object changes
([WebGLBindingStates.js:19-24](https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgl/WebGLBindingStates.js#L19)).
Attribute updates go through `needsUpdate` +
`setupVertexAttributes`, so steady-state per-draw attribute work is zero
([WebGLBindingStates.js:26-48](https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgl/WebGLBindingStates.js#L26)).

**Per geometry, not per object** — a plain `Mesh` sharing a `BufferGeometry`
shares its VAO. That is the mechanism behind my 2.00 vs 3.00 measurement. A VAO
is not a per-mesh magic object; it is a per-(geometry, program) cache, and its
benefit only appears when geometry repeats.

---

## 6. Strategy: honest in both directions

### The case that per-draw performance genuinely matters

Not "optimise it because 1.32× is a bad number". Real classes:

1. **Imported CAD / BIM.** Thousands of unique meshes, each a distinct part, each
   with its own transform. Instancing cannot help: they are *different* geometry.
   A STEP or IFC viewer is the canonical case.
2. **Hand-modelled scenes.** A 2,000-object character or architectural scene where
   every object is unique. 2,000 draws is 1.7 ms of CPU at apse's rate — enough to
   miss 60 fps with everything else.
3. **Data visualisation with unique geometry per point.** A terrain, a point
   cloud, a scientific mesh, a 3D scatter where each datum has its own vertices.
4. **Procedural generation with a unique seed per object.** Anything where
   "same mesh, different transform" is false by construction.
5. **Frustum-culled large worlds.** The cost is per *visible* object, so culling
   is the first lever and per-draw is the second.

For all five, the answer is **the same**, and it is already in the tree: merge
the static geometry ([§7](#7-what-would-it-take-to-beat-threejs-per-draw)).

### The case against, which is stronger

1. **apse already wins 70× on the path it recommends.** Instanced CPU flat at
   0.012 ms from 1,000 to 100,000 **[measured elsewhere]**. No per-draw
   optimisation moves that number.
2. **The per-draw path is where apse is worst and the docs say so.** The README
   already states: "if your scene is genuinely 1,000 distinct meshes, three.js is
   faster today and this README will say so" ([README.md:225](../../README.md)).
   That sentence is *correct* and the brief's premise — that the gap is a missing
   WebGPU primitive — is not.
3. **The recoverable amount is small and now measured.** 0.12–0.39 µs from the
   vertex binds, ~0.43–0.55 µs from the dynamic-offset bind. Closing *all* of it
   lands apse near parity, not ahead. And parity is not a differentiator when the
   competitor is 3 calls/draw and you are 4.
4. **Adoption is decided elsewhere.** Reach (compatibility mode), bundle size
   (71 KB gzip), correctness, and the instanced path. A 0.2 µs/draw improvement
   on the discouraged path changes none of them.

### Recommendation

**Do not restructure the encode loop for per-draw speed.** The measured headroom
does not justify the blast radius, and the structural explanation the brief
offered is not the real one.

**Do, in priority order:**

1. **Make the batcher the recommended path for unique-geometry scenes.** It exists
   and is correct. It is not reachable from the scene graph and not documented as
   the answer ([§7](#7-what-would-it-take-to-beat-threejs-per-draw)). That is
   cheap, real, and worth more than any micro-optimisation.
2. **Add an optional immediates path** behind feature detection, for the
   browser versions that have it. One call either way, but it removes a uniform
   buffer read. **Explicitly optional** — it cannot be the compatibility-mode
   default.
3. **Correct the record in `FINDING.md`.** The "three.js gets to roughly 4 too"
   claim is 3.00 by measurement, and the VAO-is-the-floor framing is wrong. Two
   claims have already been retracted from this project; a third accurate
   retraction is worth more than a fourth wrong number.
4. **Consider per-object data in a vertex buffer instead of a dynamic-offset
   uniform.** **[inferred]** A per-draw 64-byte `setImmediates` or a
   `stepMode: 'instance'` vertex stream both avoid the dynamic-offset bind that I
   measured at 0.43–0.55 µs. This is the one *architectural* idea with measured
   support, and it is a much smaller change than the bind-group merge that was
   already tried and retracted.

**One caveat I will not hide:** item 4 would make every draw carry a vertex-stream
bind. If that bind costs what `setVertexBuffer` costs (0.12–0.20 µs measured), it
trades a 0.43–0.55 µs bind for a 0.12–0.20 µs one. That looks like a win, but I
have **not measured it** and it must be measured before it is claimed.

---

## 7. What would it take to beat three.js per draw

**The batcher already exists.** `mergeMeshes` → `uploadBatch` →
`GpuBatchedMesh` ([`src/geometry/batch.ts`](../../src/geometry/batch.ts)), with
`BatchedRange` sub-ranges and a `sub(i)` view. It is genuinely good: layout and
topology are validated, indices are rebased, baked normals go through the inverse
transpose and are **renormalised**, bounds are conservative, and a whole batch is
one `drawIndexed` with one `setVertexBuffer` and one `setIndexBuffer` — **3 calls
per *batch***, not per mesh. That beats three.js's 3.00 calls/draw by a factor of
N, because three.js has no equivalent path without `WEBGL_multi_draw`, which WebGPU
lacks.

**Is it reachable from the scene graph?** Partly, and this is the gap. `MeshNode`
takes a `DrawableGeometry`, and `GpuBatchedMesh` is one — so
`new MeshNode({ mesh: gpuBatch, material })` draws the whole batch. **[verified by
inspection]**. But:

- **The per-source path is broken.** See below.
- **There is no `BatchNode`.** No scene-graph object models "a group of meshes
  that can be merged", so there is nothing to mark static, nothing to re-batch, and
  nothing for a user to reach for by name.
- **The 16-bit ceiling bites early.** `mergeMeshes` fails above 65,535 total
  vertices ([`batch.ts:196-202`](../../src/geometry/batch.ts)) and its own error
  text says "merge fewer meshes per batch". 2,000 hand-modelled CAD parts blow
  through that in a handful of merges. The error is at least honest about it.

**A real bug — reported, not fixed.** `GpuBatchedMesh.sub(i)` returns a
`DrawableGeometryRange` carrying `firstIndex` and `baseVertex`
([`batch.ts:334-348`](../../src/geometry/batch.ts)). The encode loop issues:

```ts
enc.drawIndexed(geometry.indexCount, instances, 0, 0, first);
```
([`renderer.ts:1145`](../../src/render/renderer.ts)) — **literal zeros** in the
`firstIndex` and `baseVertex` positions. Grepping every `geometry.*` read in the
encode loop returns `indexBuffer`, `vertexBuffer`, `indexCount`, `instanceBuffer`
— **`firstIndex` and `baseVertex` are never read anywhere under `src/render`**.

So `sub(i)` reports the right `indexCount` and then draws from offset 0 of the
merged index buffer. `sub(1)` renders **source 0's geometry**, silently, with no
validation error — one source's shape appearing where another belongs. The unit
test at `test/geometry.test.ts:1930` asserts the *values* on the object are
right, which is why it passes.

**[inferred from code inspection, not executed]** I attempted a runtime
confirmation through the real `Renderer` and the project's own `FakeFrameDevice`
and could not get the fake adapter to mint a device inside my harness; I am not
going to claim a runtime measurement I did not obtain. The static evidence is
unambiguous — the two fields are declared, returned, documented as the draw
arguments, and read by nothing — but the severity claim ("silently wrong
geometry") rests on that reading.

**Severity: the documented per-source draw path is non-functional, and it fails
in the dangerous direction** — wrong geometry, not a crash. My recommendation is
that this is the *highest-priority* item on this branch, above every performance
question here.

**To make the batcher the recommended path, in order:**

1. **Fix `sub(i)`** — read `firstIndex`/`baseVertex` in the encode loop. Without
   this the feature is half-built and its second half silently wrong.
2. **Add a `BatchNode`** — a scene-graph node holding sources, a
   `merge()`/`upload()` lifecycle, and a `batching: 'static'` marker so the
   collect pass can skip its subtree.
3. **A batched draw-list path** — one `DrawItem` for the batch, per-range culling
   via the `boundingSphere` each range already carries. The data exists; the
   traversal does not use it.
4. **Relax the 16-bit limit to auto-chunk** at the 65,535 boundary rather than
   erroring, since the fix is mechanical and the error costs users the feature.
5. **Document it as the answer to "I have N unique meshes"** — in the README
   next to the per-draw table, with the numbers.

Items 1 and 5 are the ones that change adoption. Item 1 is a bug fix.

---

## 8. Reproducing my measurements

Everything in this document is in one directory, outside the repo, with no
dependency on apse's `src`:

```
vao-probe/
  probe5.html + run5.ts   per-command marginals. Real submits, queue drained
                          every 4 frames, variants interleaved, offscreen 64x64.
  probe7.html + run7.ts   immediates in compat mode, with pixel readback.
  three-census.html       three.js GL call census (served from /tmp/apse-vao so
  + run-three.ts          /node_modules/three resolves).
```

`run3.ts` → `probe{N}.html` is the shared runner; the first two probe versions are
kept because **their failure is part of the evidence** (queue backpressure, and
unsent command buffers). `bench/**` is a sibling agent's; I reported harnesses
rather than editing theirs, as instructed.

**The measurement I'd most like to see repeated by someone else** is the
three.js GL census in §1, because it contradicts a claim in the project's own
`FINDING.md`. It is a one-file page and it takes about twenty seconds.

---

## 9. What I could not determine

- **Whether the 0.21 µs/draw gap is fully explained by the dynamic-offset bind.**
  My `setBindGroup` marginal (0.43–0.55 µs) is *larger* than apse's entire
  per-draw cost (0.87 µs), so it cannot simply be subtracted. The harnesses are
  not the same: mine is a 3-vertex trivial pipeline at 64×64; apse's is PBR at
  1280×720. **The split of the 0.87 µs across its four calls remains
  unestablished**, and I am not going to publish a decomposition I did not
  measure. Doing it properly needs per-call instrumentation inside apse's own
  encode loop, which is `src/**` and therefore not mine to touch.
- **Safari's and Firefox's position on `setImmediates`.** **[unknown]**. Only
  Chrome was tested. This matters a lot for a recommendation, and I decline to
  guess it.
- **Whether immediates or a vertex-stream object transform actually wins.** The
  mechanism is clear; the number is not measured.
- **Multi-draw indirect on Metal/GL ES hardware.** Would need the feature to
  exist first.
- **The 100,000-object row.** Already flagged in the repo as measured once, not
  established.
- **Whether my microbench margins transfer to the real per-draw path.** The
  ordering (dynamic-offset bind ≫ vertex binds) is robust across two runs and
  across variants, but the absolute µs are floors on a trivial pipeline.
