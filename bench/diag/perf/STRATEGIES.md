# Per-draw call-count strategies: what exists, what works in compatibility mode

**Companion to [`docs/RENDER-BATCHING.md`](../../../docs/RENDER-BATCHING.md).**
That document is the argument; this one is the table, the harness, and the
per-command numbers, in the place where the project's other perf evidence lives.

This file is owned by the `perf-vao` research agent. It is **strategy
documentation and measurement provenance**, not a patch. Nothing in `src/**` was
modified. One bug was found and is reported, not fixed — see
[§5](#5-bug-report-sub-i-draws-the-wrong-source).

Evidence labels used throughout: **[measured here]**, **[measured elsewhere]**,
**[inferred]**, **[unknown]**.

---

## 1. The headline, because it contradicts a claim in this directory

`FINDING.md:103` says: *"three.js's WebGLRenderer gets to roughly 4 too."*

**three.js gets to 3.00, not 4.** Measured by wrapping every entry point of the
WebGL context and counting one frame of 500 distinct meshes:

| scene | total GL calls | per draw | `drawElements` | `bindVertexArray` | `uniformMatrix4fv` |
|---|---:|---:|---:|---:|---:|
| 500 **distinct** geometries | 1501 | **3.00** | 500 | 500 | 500 |
| 500 meshes, **shared** geometry | 1001 | **2.00** | 500 | 0 | 500 |

**[measured here]** three.js r0.186.1, `MeshStandardMaterial`, one shared
material, headless Chrome, same machine as this directory's harness. Three
consecutive frames per case, identical to the digit.

apse's own census, 500 distinct meshes, from `FINDING.md:86-93`: **4.00**.

| | calls/draw |
|---|---:|
| three.js, distinct meshes **[measured here]** | 3.00 |
| apse, distinct meshes **[measured elsewhere]** | 4.00 |

The premise that apse is structurally worse because WebGPU has no VAO does not
hold at the call-count level. apse issues **one more** call per draw than three.js,
and the extra one is `setBindGroup`, not a vertex buffer. Per-command costs are in
[§3](#3-per-command-marginals-measured-here).

**The honest reading of the gap:** it is not a missing primitive. three.js
carries per-object data with a cheap `uniformMatrix4fv`; apse carries it with a
bind group and a dynamic offset, which I measured at 0.43–0.55 µs. That is a real
WebGPU-vs-WebGL difference, and it is the opposite shape to the VAO story.

---

## 2. Strategy table

Device: `apple metal-3`, **`compatibility`** feature level **[measured here]**.
apse's strategic bet is compatibility mode, so a mechanism that is core-only is
not an answer, however good it looks on a laptop.

| # | strategy | exists today | works in compat | calls/draw | verdict |
|---|---|---|---|---|---|
| 1 | **Merge geometry** (`mergeMeshes` → `uploadBatch`) | **yes, in apse** | yes | **3 per batch** | **the answer.** Beats three.js by ~N× |
| 2 | **Instancing** (`uploadInstances`) | **yes, in apse** | yes | **1 total** | already the recommended path; 70× ahead at 100k |
| 3 | `setImmediates` (push constants) | yes, Chrome 149+ | **yes — measured** | 1 → 1 | optional path; removes a uniform read, not a call |
| 4 | Render bundles | yes, since Chrome 106 | **yes — measured** | 4 → 1, but only **0.31 µs/draw** cheaper | not viable; see §4 |
| 5 | `multiDrawIndexedIndirect` | **no** — [draft PR #2315](https://github.com/gpuweb/gpuweb/pull/2315), open since 2021 | n/a | would be 4 → 0 | not an answer |
| 6 | `drawIndexedIndirect` | yes, core | yes | 4 → 3 | no call saving; still one draw per call |
| 7 | `indirect-first-instance` | yes, optional | **not requested** | none | orthogonal |
| 8 | Bindless / dynamic binding arrays | **no** — [proposal #5379](https://github.com/gpuweb/gpuweb/issues/5379) | n/a | unknown | too early |
| 9 | `WEBGL_multi_draw` | **WebGL only** | n/a | n/a | three.js uses it ([WebGLRenderer.js:1317](https://github.com/mrdoob/three.js/blob/r186/src/renderers/WebGLRenderer.js#L1317)); WebGPU has no equivalent |

### Why #5 is not an answer even if it shipped

Chrome has an **experimental** `multiDrawIndexedIndirect` behind
`chrome://flags/#enable-unsafe-webgpu`, feature name
`"chromium-experimental-multi-draw-indirect"`
([Chrome 131 blog](https://developer.chrome.com/blog/new-in-webgpu-131)).
Confirmed absent on this device: `'multiDrawIndexedIndirect' in
GPURenderPassEncoder.prototype` → **false** **[measured here]**.

Two further blockers: **Metal has no multi-draw** and needs a per-draw loop or ICB
workaround ([#2315](https://github.com/gpuweb/gpuweb/pull/2315)); and apse's
primary compatibility target is **OpenGL ES 3.1**, where `multiDrawIndirect` is
absent on 37% of Android devices
([#1949](https://github.com/gpuweb/gpuweb/pull/1949)). Given the compatibility
bet, a feature that is unshipped, non-standard, and absent from the target
backend's hardware is not a foundation.

---

## 3. Per-command marginals [measured here]

Is it the vertex-buffer binds or the draw? Measured directly.

**Harness.** Each variant renders **identical geometry** — 64 draws of 3
vertices into a 64×64 offscreen target — and differs by exactly one command. Real
`queue.submit`, queue drained every 4 frames, variants **interleaved** so drift
hits all of them equally, `performance.now()` quantum 0.1 ms (reported), 2,560
draws per timed unit, 14 interleaved units per variant, p50 of 14. Zero
validation errors during the measured phase.

| marginal cost, vs the 1-call baseline | run 1 | run 2 |
|---|---:|---:|
| `setVertexBuffer` (2 calls/draw) | 0.117 µs | 0.195 µs |
| `setIndexBuffer` (2 calls/draw) | 0.195 µs | 0.156 µs |
| **`setBindGroup`, dynamic offset** | **0.430 µs** | **0.547 µs** |
| `setBindGroup`, **no** dynamic offsets | 0.078 µs | 0.000 µs |
| all three (apse's shape) | 0.742 µs | 1.094 µs |
| `executeBundles` (1 call/draw) | 0.547 µs | 0.781 µs |
| `drawIndexed` alone (baseline) | 1.875 µs | 2.266 µs |

**Three findings, in order of usefulness:**

1. **The vertex-buffer binds are not the story.** `setVertexBuffer` +
   `setIndexBuffer` together are 0.12–0.39 µs. Merging every mesh into one buffer
   cannot recover a 0.21 µs/draw gap by itself.
2. **`setBindGroup` with a dynamic offset is the most expensive call apse issues**
   at 0.43–0.55 µs — 2–4× a vertex-buffer bind.
3. **The dynamic offset is the cost, not the bind.** The identical call *without*
   dynamic offsets is 0.00–0.08 µs. Same bind group, same pipeline, same buffers;
   only the offset differs.

**These are floors, not apse's real per-draw cost.** This is a 3-vertex trivial
pipeline at 64×64; apse's 0.87 µs/draw is PBR at 1280×720 with four bind groups.
**The decomposition of apse's 0.87 µs across its four calls is not established** —
see [§6](#6-what-is-not-established).

### Two harness failures worth keeping

Both produced plausible-looking nonsense, and both are the traps
`bench/diag/perf/README.md` already warns about:

- **v1: submitted inside the timed loop.** Reported `draw_only` **slower** than
  `full_distinct` — arithmetically impossible for a superset of calls, and the
  tell that it was measuring queue backpressure.
- **v2/v3: never submitted.** 100× spread between min and median for identical
  work, plus validation errors on a path that should have been valid. The
  command buffer accumulated 32k draws and Dawn's deferred work landed inside the
  timed region unpredictably.

**A per-draw microbenchmark that does not submit is not measuring encode.** Only
real submits with a drained queue produced the tight distributions above. The
failed versions are kept in the probe directory deliberately.

---

## 4. Why render bundles are not the answer, measured

Bundles work in compatibility mode **[measured here]** — I created one on this
device and `executeBundles` ran clean. But the cost does not follow from "4 calls
become 1":

```
executeBundles, 1 call/draw, vs 4 inline calls:  -0.31 µs/draw
executeBundles, 1 call/draw, vs 1 inline call:   +0.55 µs/draw
```

**A bundle is 0.31 µs/draw cheaper than four inline calls, and 0.55 µs/draw more
expensive than a single `drawIndexed`.** A bundle is a container, not a fast path:
the commands inside it cost what they cost. Babylon reached the same wall from the
other direction — one bundle per (mesh, pass, format), with an invalidation
problem, and the resulting "fast path" is not a general win
([gpuweb#1640](https://github.com/gpuweb/gpuweb/discussions/1640)).

The fatal detail for apse specifically: a bundle's bind groups are **fixed at
record time**, and `executeBundles` does not inherit the pass's bind groups
([spec §17.2.4](https://gpuweb.github.io/gpuweb/#render-pass-encoder-executebundles)).
apse's per-object transform is a **dynamic offset**, which a bundle cannot carry.
500 distinct meshes ⇒ 500 bundles, all invalid whenever any object moves, to save
0.31 µs/draw. Not viable.

---

## 5. Bug report: `sub(i)` draws the wrong source

**Found while assessing `mergeMeshes` for strategy #1. Not fixed — `src/**` is
read-only for this agent.**

`GpuBatchedMesh.sub(i)` returns a `DrawableGeometryRange` carrying `firstIndex`
and `baseVertex`, documented as exactly the third and fourth arguments of
`drawIndexed` ([`src/geometry/batch.ts:66-70, 334-348`](../../../src/geometry/batch.ts)).

The encode loop issues:

```ts
enc.drawIndexed(geometry.indexCount, instances, 0, 0, first);
```
([`src/render/renderer.ts:1145`](../../../src/render/renderer.ts)) — **literal
zeros** in the `firstIndex` and `baseVertex` positions.

Every `geometry.*` read in the encode loop: `indexBuffer`, `vertexBuffer`,
`indexCount`, `instanceBuffer`. **`firstIndex` and `baseVertex` are read nowhere
under `src/render`** — verified by grep over `src/`.

**Consequence.** `sub(i)` reports the correct `indexCount` and then draws from
offset 0 of the merged index buffer. `sub(1)` renders **source 0's geometry**.
Silently — no validation error, no exception, one source's shape where another
belongs. The module's own header warns that "a rebasing bug is not a visible
artefact in isolation — it is another source's geometry appearing in the wrong
place" ([`batch.ts:35-37`](../../../src/geometry/batch.ts)); this is that bug, in
the draw path rather than the bake path.

`test/geometry.test.ts:1930` asserts the *values on the object* are correct, which
is why it passes. Nothing asserts they reach the draw.

**Status of this claim: [inferred] from code inspection.** I attempted runtime
confirmation through the real `Renderer` with the project's own
`FakeFrameDevice` and could not get the fake adapter to mint a device inside my
harness. The static evidence is unambiguous, but I did not execute a confirming
frame, and I am not claiming one.

**Why it is the top item on this branch:** the batcher is the recommended path for
every real unique-geometry workload (§2, strategy #1), and its per-source
half — the half that makes an interactive CAD or character scene workable — is
non-functional in the worst way. This outranks every performance question in this
file.

**Fix, for whoever owns `src/`:** read `firstIndex` and `baseVertex` from the
geometry in the `drawIndexed` call, defaulting to 0 for a plain `GpuMesh` (which
already reports 0 for both — `test/geometry.test.ts:1953-1955`). The type
`DrawableGeometryRange` already extends `DrawableGeometry` with exactly these two
fields, so this is a two-line change plus a test that asserts the values reach
`FakeFramePass.draws[].first` and `.base`.

---

## 6. What is not established

- **The split of apse's 0.87 µs/draw across its four calls.** My `setBindGroup`
  marginal (0.43–0.55 µs) is *larger* than apse's entire per-draw cost, so it
  cannot simply be subtracted. Different harness, different pipeline. **No
  decomposition is published here**, because none was measured. Doing it needs
  per-call instrumentation inside apse's own encode loop — `src/**`, not mine.
- **Whether immediates or a vertex-stream object transform would actually win.**
  Mechanism is clear; the number is not measured. A per-draw 64-byte
  `setImmediates` (`maxImmediateSize` is 64, and `OBJECT_FIELDS.model` is a
  `mat4x4f` = 64 bytes — an exact fit **[measured here]**) would trade a
  0.43–0.55 µs dynamic-offset bind for a ~0.12–0.20 µs call. **Plausible, not
  proven**, and it still costs one call either way.
- **Safari's and Firefox's support for `setImmediates`.** **[unknown]**, only
  Chrome tested. Given the compatibility-mode bet is about *older* devices, a
  feature in the newest Chrome is a poor default and must stay behind feature
  detection.
- **Whether these µs transfer to the real per-draw path.** The *ordering*
  (dynamic-offset bind ≫ vertex binds) is robust across two runs and across
  variants. The absolutes are floors on a trivial pipeline.

---

## 7. Recommendation, in one paragraph

Ship **batching as the documented answer to "I have N unique meshes"** — it
exists, it is correct, it collapses 4 calls/draw into 3 calls/**batch**, and it
reuses code already in the tree. Fix `sub(i)` first: it is a real bug that fails
silently, and it disables the feature's most valuable half. Add immediates as an
**optional** path behind `wgslLanguageFeatures.has('immediate_address_space')` —
never as the compatibility default. Do **not** restructure the encode loop for
per-draw speed: the recoverable amount is 0.12–0.39 µs from the vertex binds and
the project already wins 70× on the path it recommends. And correct
`FINDING.md:103` — three.js is 3.00 calls/draw, not "roughly 4", and that claim is
load-bearing for a retracted fix.

---

## 8. Provenance

Written by the `perf-vao` research agent. `bench/**` belongs to a sibling agent;
harnesses are reported here rather than added to their files, as instructed.

```
vao-probe/            (outside the repo, no dependency on apse src)
  probe5.html  run5.ts    per-command marginals
  probe7.html  run7.ts    immediates in compat mode, pixel readback
  three-census.html
  run-three.ts            three.js GL census, served from /tmp/apse-vao
                          so /node_modules/three resolves
  probe.html .. probe4.html   the two superseded versions, kept because
                          their failure mode is part of the evidence
```

The measurement most worth an independent repeat is the **three.js GL census** in
§1: one page, about twenty seconds, and it contradicts a claim in this directory.
