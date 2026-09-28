# apse

A WebGPU renderer that ships in 71 KB gzip, and a README that tells you which of its
own claims were measured — including the ones that came out worse than advertised.

[![CI](https://github.com/srivtx/apse/actions/workflows/ci.yml/badge.svg)](https://github.com/srivtx/apse/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/apse.svg)](https://www.npmjs.com/package/apse)
[![License: MIT](https://img.shields.io/npm/l/apse.svg)](https://github.com/srivtx/apse/blob/main/LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-required-8A2BE2.svg)](https://gpuweb.github.io/gpuweb/)

```bash
bun add apse     # or: npm i apse
```

## Quick start

```js
import {
  Renderer, PerspectiveCamera, Scene, MeshNode,
  box, upload, pbrMaterial, quat,
} from 'apse';

const renderer = await Renderer.create(document.querySelector('canvas'), {
  budget: { cpu: 2, drawCalls: 200 },
});

const scene  = new Scene({ name: 'lit-cube' });
const camera = new PerspectiveCamera({ fov: 45, near: 0.1, far: 100, aspect: 1 });
camera.lookAt([3.2, 2.4, 4.4], [0, 0, 0], [0, 1, 0]);
renderer.camera = camera;

const cube = upload(renderer.device.device, box({ width: 1.4 }));

// Read this line. Tone mapping is on by default, so the scene renders into an
// rgba16float intermediate rather than the canvas, and every material in that
// scene has to be compiled for that format. Omit it and you get
// RENDER_TARGET_FORMAT_MISMATCH, which is a much better error than the one
// WebGPU would have given you — but it is still an error.
const material = await pbrMaterial(renderer.device.device, {
  targetFormat: renderer.sceneFormat,
});

const node = new MeshNode({ name: 'cube', mesh: cube, material });
node.setPosition(0, 0.8, 0);
scene.add(node);

renderer.start((dt) => {
  // `rotation` is a live view into the node: a direct write bypasses the dirty
  // flag, so `markDirty()` is required. This is the one place you write the
  // array yourself.
  quat.fromEulerXYZ(node.rotation, 0, node.rotation[1] + dt * 0.6, 0);
  node.markDirty();
});
```

`examples/lit-cube.html` is the same thing against `dist/`, plus a stats overlay.

### `targetFormat` is not optional, and here is why

The default path is: scene → `rgba16float` intermediate → fullscreen ACES pass →
canvas. The intermediate exists because a tone map fed an already-clipped 8-bit
image is a full-screen pass that applies a curve to a clamped picture; the
highlights are gone before the curve sees them. The cost of doing it properly is
that the scene is no longer drawn into the canvas format, so **every** material
needs `targetFormat: renderer.sceneFormat`.

The two escape hatches are `toneMapping: null` and `hdr: false`. Both restore a
direct draw into the canvas format and both cost you the reason the pass exists.
`renderer.sceneFormat` tells you the truth either way — read it, don't guess.

### Instancing

N objects, one `MeshNode`, one `drawIndexed`:

```js
import { InstanceData, uploadInstances, instancedMaterial } from 'apse';

// `count` 4x4 column-major matrices, or InstanceData.fromTRS(t, r, s) for 8 each
const matrices = new Float32Array(count * 16);

const instances = uploadInstances(
  renderer.device.device,
  InstanceData.fromMatrices(matrices, { name: 'grid' }),
);
const material = await instancedMaterial(renderer.device.device, {
  targetFormat: renderer.sceneFormat,
});
const mesh = upload(renderer.device.device, box({ width: 0.6 }), { instances });

scene.add(new MeshNode({ name: 'grid', mesh, material }));
// renderer.stats.drawCalls === 1, for any count.
```

The per-instance stream is bound to vertex slot 1 with `stepMode: 'instance'`, so
it works in the compatibility profile where a vertex-stage storage buffer does not.
`firstInstance` and `instanceCount` narrow the range. Both are checked against the
buffer's real size before the draw, so a range that runs off the end is an apse
error and not a driver validation error at the first frame.

## Measured size

`bun run build` on an Apple M3, 2026-09-28. Every entry is a real bundle esbuild
produced from the real source, and `scripts/size-gate.ts` fails the build over a
ceiling on every one of them.

| entry | files | raw | **gzip** | brotli | ceiling | used |
|---|---:|---:|---:|---:|---:|---:|
| `apse` (root) | 16 | 284.21 KB | **95.87 KB** | 78.87 KB | 110.3 KB | 87% |
| `apse/render` | 12 | 206.18 KB | **69.95 KB** | 58.14 KB | 80.4 KB | 87% |
| `apse/material` | 9 | 165.01 KB | **56.04 KB** | 47.29 KB | 64.4 KB | 87% |
| `apse/geometry` | 7 | 92.62 KB | **32.64 KB** | 27.51 KB | 37.5 KB | 87% |
| `apse/scene` | 6 | 43.88 KB | **16.34 KB** | 13.71 KB | 18.8 KB | 87% |
| `apse/math` | 6 | 30.66 KB | **11.66 KB** | 9.52 KB | 13.4 KB | 87% |
| `apse/core` | 5 | 22.37 KB | **8.74 KB** | 7.42 KB | 10.1 KB | 87% |
| **tree-shaken app** | 1 | 205.91 KB | **71.23 KB** | 59.24 KB | 81.9 KB | 87% |

The tree-shaken row is the headline: a complete renderer, camera, PBR material,
scene graph and animation loop, from one import, importing eight symbols out of
179. The three.js comparison point is 133 KB gzip for a single PBR cube
(`THREE_JS_GZIP_BASELINE_KB` in `scripts/size-gate.ts`, its own published build).
**71.23 against 133 is 1.87×.**

`src/index.ts` went from 107 exported names to 179 this pass — 72 added, none
removed, of which 130 are runtime values and the rest are types. Most of what was
added already existed and was reachable from a subpath but not from the root:
`capsule`, `cone` and `roundedBox` beside the first six primitives;
`mergeMeshes` / `uploadBatch`; `computeTangents` / `withTangents`; `GpuTimer`;
`CaptureReadback`; `requireFeature` / `readCapabilities`; `ERROR_BLAME`; and the
three new materials. Reachability is the change, not novelty.

**This number used to be 2.9×, and reporting 1.87× is the honest version.** The
present pass and the timestamp layer cost 26 KB — 45.3 KB to 71.23 KB. That is
the price of the two things listed under [bugs this pass](#bugs-this-pass-caught),
and paying 26 KB to stop presenting a black screen and to stop reporting a GPU
that was never measured is a trade most people would take. A README claiming 2.9×
would survive exactly as long as it took somebody to run the build.

Tree-shaking is asserted, not inferred from a byte count. `bench/tree-shake.ts` is
a real minimal app bundled on every build, and `scripts/size-gate.ts` greps the
result for the five primitives it does not import and fails if any is present —
with a positive control (`box` must be found), so the check cannot pass vacuously.
The same script also asserts the module graph (every entry reaches a known set of
layers and a known module count) and that every `exports` subpath resolves to a
built file. 57 of 57 source modules are reachable from some entry.

## Performance

Measured on an **Apple M3, `compatibility` feature level, headless Chrome,
1280x720**, p50 of 120 samples x 3 trials, three.js r186 on WebGL2 in the same
process on the same device. `bun run bench:perf`.

### One draw per object — apse loses, and it is per-draw CPU

| scene | apse | three.js | ratio | draws | lit px / cols |
|---|---:|---:|---:|---:|---|
| 1,000 cubes | 0.900 ms | 0.700 ms | 0.78x | 1,000 | 4456 / 841 (three 4456 / 841) |
| 2,000 cubes | 1.800 ms | 1.350 ms | 0.75x | 2,000 | 5192 / 901 (three 5192 / 901) |
| 5,000 cubes | 4.200 ms | 3.000 ms | 0.71x | 5,000 | 5472 / 924 (three 5472 / 924) |
| 10,000 cubes | 8.400 ms | 6.900 ms | 0.82x | 10,000 | 5588 / 934 (three 5588 / 934) |
| 1,000 spheres | 1.100 ms | 0.800 ms | 0.73x | 1,000 | 4413 / 828 (three 4422 / 828) |

The marginal cost is **0.87 us per draw against three.js's 0.66 us**, a 1.32x
ratio that is stable from 1,000 to 100,000 draws. That single number is the whole
per-draw gap. It is a CPU cost: the frame is 85-100% per-draw encode, and
`setBindGroup` fires 1,006 times for 1,000 objects.

### Instanced — apse's CPU cost is flat, and this is the number that matters

| objects | apse per-draw | **apse instanced** | three.js per-draw | apse instanced vs three.js per-draw |
|---|---:|---:|---:|---:|
| 1,000 | 0.740 ms | **0.012 ms** (1 draw) | 0.675 ms | **6.3x** |
| 10,000 | 7.250 ms | **0.013 ms** (1 draw) | 5.750 ms | — |
| 100,000 | 86.675 ms | **0.012 ms** (1 draw) | 65.725 ms | **70x** |

**There is no crossover point, because apse wins at every instance count
measured.** One draw costs the same whether it carries 1,000 instances or
100,000, so instanced CPU time is flat at 0.012 ms while the per-draw path grows
linearly to 87 ms. The per-instance transforms arrive as vertex attributes, so
the per-draw cost is not merely amortised, it is absent.

Coverage is byte-identical between the instanced and per-draw paths at every
count — the same 1,845 lit pixels and 923 lit columns at 100,000 objects whether
drawn as one instanced call or as 100,000 node draws. That is what makes the
timing comparable rather than merely fast.

**Where apse still loses:** against three.js's *own* instanced path, 0.151-1.109 ms
against 0.038-1.084 ms. That residual is the present pass plus fixed per-frame
overhead, not per-draw cost, and it only matters when a frame is already a single
draw call.

### The present pass costs about 0.002 ms of CPU

| scene | delta CPU | delta end-to-end |
|---|---:|---:|
| 1,000 cubes | +0.020 ms (2.6%) | +0.060 ms (6.4%) |
| 5,000 cubes | +0.081 ms (2.1%) | +0.150 ms (3.4%) |
| 5,000 instanced | +0.002 ms | +0.076 ms |
| 100,000 instanced | +0.002 ms | +0.064 ms (4.9%) |

A fixed cost, not a per-object one. It dominates a one-draw frame and is
invisible on a thousand-draw frame. Coverage is identical with it on and off.

### What these numbers are not

- **There is no GPU-side millisecond anywhere in this section.** This device
  exposes no `timestamp-query`, so `stats.gpu` is `null` and every
  "end-to-end" figure is a CPU-side wait, not a timestamp query. GPU fragment
  cost is not separated from driver overhead.
- **`performance.now()` is quantised to 100 us** on this machine. The 0.012 ms
  instanced figure rests on batching 500 samples; rounding error there is about
  0.0002 ms.
- **One machine, one Chrome build, headless, compatibility.** The ratios are the
  finding; the absolute times are not portable.
- **The 100,000-object instanced row is measured once, not established.**
  `bun run bench:perf` has crashed headless Chrome at the top of that sweep with
  `Protocol error (Runtime.callFunctionOn): Target closed`. The 1,000 and 10,000
  rows reproduce. The cause is not diagnosed. See
  `bench/diag/perf/README.md`.
- **Instanced scenes were measured with static transforms.** A moving instanced
  scene re-uploads 32.8 MB/frame at 5,000 objects and that path is unverified.
- **Cubes and spheres only.** One lighting model, no transparency, no MSAA.

### The honest summary

apse is **1.3-1.4x slower per draw call** and that is not a rounding error. But
the per-draw call is the thing instancing exists to remove, and once you use it
apse's frame cost stops depending on object count. If your scene is thousands of
separate objects you are better served by one instanced draw in apse than by
1,000 draws in three.js; if your scene is genuinely 1,000 distinct meshes, three.js
is faster today and this README will say so.

## Where apse wins, and where it does not

Every number in this table has a harness behind it and a method next to it.
Bundle sizes: `bun run build` above. Heap: Chrome DevTools Protocol
`HeapProfiler` snapshots with GC forced, per-class slopes, `bench/mem/run.ts`.
Diagnostics: nine failure modes triggered for real in both libraries in one
headless Chrome on one GPU, `bench/diag/run.ts`. Authoring cost: the same material
written twice and counted, `bench/dx/run.ts`.

| axis | apse | three.js 0.186.1 | verdict |
|---|---:|---:|---|
| bundle, tree-shaken, full renderer + PBR + scene graph | **71.23 KB** gzip | 133 KB | apse, **1.87×** |
| heap per drawable scene-graph object (`MeshNode` vs `Mesh`) | 1,060 B / 21.0 objects | 1,240 B / 57.0 objects | apse, 1.17× by bytes, **2.7× by object count** |
| a failure mode that is structurally unrepresentable | 1 of 9 (a cross-stage varying mismatch) | 0 of 9 | apse |
| silent-vs-typed diagnostics, 9 scenarios | better 5 · worse 2 · equal 2 | — | apse, narrowly |
| custom material, BRDF written by hand | 80 lines | 83 (`ShaderMaterial`) | tie |
| custom material, prebuilt BRDF, set properties | 5 lines | 5 (`MeshStandardNodeMaterial`) | tie |
| materials shipped with a shading model | 5 | 8 | **three.js** |
| generated code per material | 77 WGSL lines from 30 authored | 1,984 lines from 6 authored (`onBeforeCompile`) | apse |
| draw calls for 1,000 copies of one mesh | 1 | 1 (`THREE.InstancedMesh`) | tie |
| ecosystem, loaders, animation, tooling | — | — | **three.js**, by a wide margin |

**The failures that cannot be expressed, rather than diagnosed.** You cannot compile
a material whose uniform block and its JS packer disagree, because one table
produces both. You cannot draw a material against a render target whose colour
format differs, because the renderer checks it before `setPipeline` — where
WebGPU's own answer is a validation error that invalidates the *whole command
buffer* and mentions two format enums rather than the mistake. A third, found by
the same harness: you cannot write a varying whose type differs between the vertex
and fragment stages, because one `varyings` declaration generates the struct for
both. Being unrepresentable beats being well-diagnosed, and the first row of the
table is the only place apse is categorically ahead.

**Instancing is a tie, and it is worth saying so.** three.js has had
`THREE.InstancedMesh` for years: 1,000 copies of one mesh is one draw call in both
libraries. What apse adds is that the path is in the compatibility profile — a
vertex buffer at slot 1, not a vertex-stage storage buffer — and that the
transform-only instance record is the caller's own `Float32Array` with no copy.
Neither of those has a benchmark behind it, so neither is claimed here.

**Where apse loses, without hedging:**

- **Ecosystem.** three.js has loaders, an editor, an inspector, a physics
  integration, a documentation site, and an enormous body of Stack Overflow
  answers. apse has none of these and will not have them soon.
- **LLM training data.** No model has been trained on apse's source. Anything
  non-obvious about this API, you are reading the repository, not recalling it.
- **Material count, and the gap narrowed without closing.** apse had two shipped
  materials and now ships **five** with a shading model — `basic` (unlit), `pbr`
  (Cook-Torrance GGX), `diffuse` (Oren-Nayar with an energy-conserving rim),
  `emissive` (a physically-shaped unlit emitter) and `anisotropic` (elliptical GGX
  in a tangent frame) — against three.js's eight. Of apse's five, **two are
  unlit**, so on materials with a lit BRDF it is three against eight. Three
  materials is real progress over two; it is not parity, and calling it that would
  be the same mistake this file is trying not to make.
- **No glTF loader.** `MeshData` takes interleaved data or one dense array per
  attribute. That is the whole geometry import surface. No OBJ, no texture
  decoders, no environment maps, no I/O.
- **No animation system.** No keyframes, skinning, morph targets, or blend trees.
  `renderer.start(cb)` is the whole story, and it hands you `dt` already clamped.
- **No WebGL2 fallback, and there will not be one.** A browser without
  `navigator.gpu` throws `WEBGPU_UNAVAILABLE`. The compatibility *profile* is
  supported; a compatibility *fallback* is not, because the scaffold generates
  WGSL and the alternative is a second shader language.
- **Per draw call, apse is 1.3-1.4x slower, and the number is stable.** 0.87 us
  marginal against three.js's 0.66 us, at every object count from 1,000 to
  100,000. If your scene is genuinely thousands of *distinct* meshes rather than
  thousands of instances, three.js has the faster frame today.
- **Against three.js's own instanced path apse loses too** — 0.151-1.109 ms
  against 0.038-1.084 ms. The gap is the present pass and fixed per-frame
  overhead, and it only shows up when a frame is already a single draw call.

### The three claims this README used to make that did not survive measurement

They are listed because a reader is entitled to know which parts of a README were
withdrawn, and because each withdrawal is a measurement somebody can repeat.

1. **"A custom material is dramatically simpler."** It is not. Writing the same
   two-light Lambert + Blinn-Phong material by hand: **80 lines in apse, 83 in
   three.js** — a tie, on a material that renders the same image to within one
   8-bit step. The scaffold's real win is not brevity, it is **generated code**:
   77 WGSL lines from 30 authored, against 1,984 generated from 6 on
   `onBeforeCompile`, and no `#include` names, no `customProgramCacheKey`, no
   `userData.shader.uniforms`.

   An earlier draft also claimed three.js's node-material path was "15 lines,
   which beats apse outright". **That comparison was invalid and has been
   removed.** Those 15 lines are property assignment on a
   `MeshStandardNodeMaterial` — a complete, library-provided Cook-Torrance BRDF.
   The apse side of that comparison was asked to *write the same BRDF by hand*,
   which compared a prebuilt shader against a hand-written one. Measured fairly —
   a prebuilt BRDF, properties set — both libraries are about 5 lines.

2. **"An `Object3D` costs 1,804 bytes."** It does not. Measured with GC-forced
   Chrome heap snapshots and per-class slopes: **1,216 B** for a bare `Object3D`
   shell plus everything only it owns, and **1,240 B** for a drawable `Mesh`,
   against apse's 1,060 B. The 450 B figure that came next was not reproducible by
   any method — Bun's `process.memoryUsage().heapUsed` cannot see this, and the
   proof is in `bench/mem/README.md`. The real difference is not the byte total. It
   is that three.js allocates **57.0 live heap objects per node to apse's 21.0**,
   and 908 of apse's 1,060 B — **86%** — is six `Float32Array`s that would be one
   array if the design were finished.

3. **"42 typed error codes, and they are all reachable."** Two were not:
   `VARYING_MISMATCH` and `SHADER_NO_ENTRYPOINT`. Both were *structurally*
   unrepresentable rather than merely unused — one `varyings` declaration
   generates the `Varyings` struct for both stages, so the two cannot disagree, and
   the scaffold always emits `vs` and `fs`, so a missing entry point is not a
   failure mode. Both are deleted; the guarantee comes from the type system
   instead, which is a stronger place for it. Two more were added in their place:
   `RESOURCE_DISPOSED` and `INVALID_USAGE`.

### The error catalog, stated precisely

42 codes, and every one carries a `why`, a `fix` and a `link`. Two of them were
deleted for being unreachable, so the count is the same and the surface is
smaller and truer.

Every code is also classified by **blame**, in a mapped table that cannot drift
from the code list, and exposed as `ERROR_BLAME`:

| blame | count | codes |
|---|---:|---|
| `library` | 1 | `INTERNAL_INVARIANT` |
| `caller` | 37 | everything a user can do wrong |
| `environment` | 4 | `WEBGPU_UNAVAILABLE`, `ADAPTER_UNAVAILABLE`, `DEVICE_LOST`, `DEVICE_REQUEST_FAILED` |

A handler that treats `INTERNAL_INVARIANT` as the general-purpose "something went
wrong" bucket files its users' typos under "apse is broken", which is the one
thing a typed catalog must not do. That was measurably happening: a previous audit
found 55 of 140 `fail()` call sites raising a code whose guidance reads *"This
always indicates a bug in apse, not in your code."* Re-counted against the source
today it is **71 of 203** call sites. Branching on `e.blame` instead of on a code
slug is the supported answer, and the count is the thing to watch.

**One gap, stated rather than hidden:** the blame split is only as good as the
`super()` call that names the resource. `Material` and `GpuMesh` both call
`super()` with no code, so they inherit the `INTERNAL_INVARIANT` default — drawing
a disposed material today still raises a `library`-blamed code, and `MESH_DISPOSED`
is not raised anywhere yet. The mechanism exists (`Resource.assertLive` takes the
code as a parameter, and it is tested); passing the code is a one-line change in
each subclass that has not been made. See [Limits](#limits).

## Bugs this pass caught

Named because they are the credibility story, and because a reader deciding
whether to trust a number on this page should know what else was wrong when the
number was taken. Each is fixed and each has a test that fails on the old code.

- **A capsule with all 168 triangles wound inside-out.** Every `capsule` was
  invisible from the outside and invisible from the inside, with a valid index
  buffer, a valid pipeline, and no validation error anywhere. Culling mode and
  winding order disagreeing is the classic silent-geometry failure.
- **An AABB pass with no finiteness check.** A `NaN` position propagated through
  the bounds computation and produced a *finite* bound that did not contain the
  mesh, which the frustum then rejected — silently dropping a visible object at
  cull time, with a draw list that looked correct and a scene with a hole in it.
  `MeshData` now rejects a non-finite vertex with the vertex index named.
- **`mergeMeshes` never renormalised baked normals.** Correct only for rigid
  transforms, which is exactly the case you do not test: under a 2:1:1 scale a
  normal came out at length 0.5, and a normal of length 0.5 is a shading bug that
  still produces a plausible image. Merged normals are now renormalised.
- **The tone map emitted `vec4f(in.position, 0.0)`.** That `0.0` is the clip-space
  `w` — the divisor of the perspective divide. `w = 0` collapsed the fullscreen
  triangle to a point, nothing rasterised, and the result was a black screen with
  a valid pipeline, a legal draw, no validation error, **and a unit test asserting
  the buggy literal.** The `1.0` is load-bearing and now says so in a comment
  three lines long.

That last one is the reason the invariants section in `ARCHITECTURE.md` exists,
and the reason this file publishes a size regression rather than hiding it.

## The one idea: a material is data, not a program

The usual way to customise a shader in a 3D library — three.js is the obvious
example — is to hand it a string and hope: override a chunk by its internal name,
then discover that the program cache key is a string join of a hundred fields the
docs never mention, plus a mandatory undocumented `customProgramCacheKey` escape
hatch. None of those chunk names is a stable API, and they change between
releases.

apse generates all of it. You write two statement lists and a description of what
they need:

```js
import { generateScaffold } from 'apse/material';

const shader = generateScaffold({
  name: 'fresnel',
  varyings: { normal: 'vec3f', worldPos: 'vec3f' },
  slots: {
    tint: { type: 'vec3f', default: [0.2, 0.7, 0.9] },
    rim:  { type: 'f32',   default: 2.5 },
  },
  vertex: `
out.clip     = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;
out.normal   = normalize(obj.normalMatrix * in.normal);
`,
  fragment: `
let n = normalize(in.normal);
let v = normalize(frame.camPos - in.worldPos);
return vec4f(mat.tint * pow(1.0 - max(dot(n, v), 0.0), mat.rim), 1.0);
`,
});

shader.code; // the complete WGSL program
```

That produces a program in which **every** declaration above the two entry points
was inferred from the spec — the `Frame` struct (14 fields), `ObjectData`,
`MaterialData`, the `VertexIn` struct, the `Varyings` struct, all three bind
groups, all three `@group`/`@binding` pairs, and both function signatures. You
wrote no `@group`, no `@binding`, no `struct`, and no `fn`. You cannot: a body
that declares a `fn`, a `struct`, or anything with an `@attribute` is rejected
before the shader is compiled. What a body gets to see is `in`, `out`, `frame`,
`obj`, `mat`, one variable per declared texture, and WGSL's builtin library.

`i32`, `u32` and `mat4x4f` varyings get `@interpolate(flat)` emitted
automatically, because the alternative is a compile error. `generateScaffold` is a
pure function of the spec — same spec in, byte-identical program out — which is
what lets apse cache pipelines by a hash instead of rebuilding a hundred-field key
on every miss. Pass `scaffold: true` on any material to log the full program.

All five lighting materials go through exactly this path with no bypass.
`basicMaterialSpec()`, `pbrMaterialSpec()`, `diffuseMaterialSpec()`,
`emissiveMaterialSpec()` and `anisotropicMaterialSpec()` are plain `MaterialSpec`
objects you can read, diff, and copy. `instancedMaterialSpec()` is the sixth and
`tonemapMaterialSpec()` the seventh.

`anisotropicMaterial` needs a `tangent` attribute and **no apse primitive produces
one** — use `computeTangents()` or `withTangents()`. It refuses a layout without
one, naming the attribute, rather than letting a WGSL error point into generated
code.

## Errors that teach

Every error apse throws is an `AseError` with six fields of its own — `code`,
`why`, `fix`, `link`, `blame`, `detail` — and all six are present on every error,
five as strings and `detail` as `undefined` where a code has no structured
context. There is no code path that produces a bare `Error`.

```js
import { generateScaffold, isAseError } from 'apse';

try {
  generateScaffold({
    name: 'typo',
    slots: { tint: { type: 'vec3f', default: [1, 1, 1] } },
    vertex:   `out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);`,
    fragment: `return vec4f(frame.viewProjj.xyz * mat.tint, 1.0);`,   // three j's
  });
} catch (e) {
  if (isAseError(e)) console.error(e.code, e.why, e.fix, e.blame);
}
```

`code` is a stable slug — branch on that, never on `message`. `why` is the rule
that was violated. `fix` is the one corrective action, and it names the field and
the alternative. `blame` says whose fault it is. `link` is the docs anchor for
that exact code. A sixth field, `detail`, carries structured context where a code
has any: a numeric range, an attribute name, a byte offset. `toJSON()` gives you
all of it at once.

The identifier check runs *before* the shader is compiled, so a typo names the
field you got wrong instead of surfacing as a WGSL error pointing at a line in
generated code you never wrote. That is the single strongest thing apse does, and
it is measured: the same harness that found three.js's worst silent failure
(omitting `customProgramCacheKey` renders 12,412 pixels wrong with no diagnostic)
found apse's typo path naming the field and listing every alternative.

For a WGSL *syntax* error there is less to be proud of, and the same harness said
so: the quoted text came from WebGPU's error scope, which names the *vertex* stage
for a fragment error and carries no line or column, and the `fix` asserted that the
compiler's line number was useful when it was not. **That is fixed** —
`Material.create` now calls `getCompilationInfo()`, derives the stage from the line
rather than assuming it, and puts every diagnostic in `why` with its line, column,
and the source line quoted. The 5/2/2 score above is therefore the last *measured*
result and is stale in apse's favour; re-running `bench/diag/run.ts` is what would
let the table move, and nobody has done that yet.

The failure surface is finite, enumerable, and typed: `ERROR_CODES` is all 42,
`ERROR_CATALOG` is the `why`/`fix` pair for each, `ERROR_BLAME` is the
classification — all three exported, so you can read the complete list of things
that can go wrong without reading a line of apse's source.

## Resource ownership

`GpuMesh` and `Material` both extend `Resource`, which is reference-counted. A
resource starts at `refCount === 1`, held by whoever called `upload()` or
`pbrMaterial()`. A `MeshNode` takes a reference to what it draws for as long as it
is in a graph, so sharing one mesh across a thousand nodes needs no bookkeeping and
nothing can be freed out from under a node that still holds it.

```js
const mesh = upload(renderer.device.device, box({ width: 1.4 }));
mesh.refCount;   // 1 — yours

mesh.ref();       // 2 — a second, independently-owned holder
mesh.unref();     // 1 — nothing freed
mesh.unref();     // 0 — the vertex and index buffers are released now
```

- `unref()` drops one reference. GPU memory is released at zero, and only at zero.
  Safe on an already-disposed resource.
- `dispose()` force-releases immediately, whatever the count. It is the "I know I
  am done" verb, and it is what scene teardown uses. On a shared resource it frees
  it for everyone, so reach for `unref()` unless you mean it.
- `ref()` returns `this`, so it assigns directly.
- `.disposed` is readable, not only raisable. A consumer that can ask "is this
  still alive" does not have to provoke an error to find out.
- Garbage collection is never relied on. Browser GPU object lifetimes are not
  deterministic, so nothing here waits for a finaliser. You release it.

`renderer.dispose()` releases the device, the render targets, and the shared frame
and object uniforms. It does **not** release your meshes and materials — those are
yours.

## Compatibility

WebGPU only. There is no WebGL2 fallback, and adding one is not on the roadmap. A
browser without `navigator.gpu` throws `WEBGPU_UNAVAILABLE` with the reason in the
message.

- **Works:** Chrome/Edge 113+, Safari 26+ including iOS, Firefox 141+ on Windows
  and 147+ on Apple Silicon.
- **Does not work:** Firefox on Linux, Firefox on Android, Firefox on Intel Macs.

apse requests the WebGPU **compatibility** profile by default, and asks for `core`
only when the adapter advertises `core-features-and-limits`. That is the profile
that reaches the largest device base, and it is what sets the two limits the API is
shaped around:

- **Zero storage buffers in the vertex stage.** Compatibility mode targets
  GLES 3.1-class hardware, which has no read-write buffer in a vertex shader.
  Per-object data reaches the vertex stage through one large uniform buffer
  addressed with dynamic offsets and written once per frame, and per-instance data
  through vertex slot 1, not through a storage buffer.
- **`copyTextureToBuffer` rows align to 256 bytes.** `renderer.capture()` returns
  `bytesPerRow` padded to that, so read rows at `bytesPerRow`, not `width * 4`.
  `unpadRows()` does it for you. It renders to an offscreen target and copies out
  of it, because a WebGPU canvas has no `preserveDrawingBuffer` and reading the
  swapchain after present yields an empty image.

`renderer.featureLevel` is `'core'` or `'compatibility'`. `readCapabilities(device)`
returns the profile, the features actually present, and the limits; `requireFeature`
and `hasFeature` probe a named one. Optional features are detected, never required:
`timestamp-query` is on roughly 44% of devices.

## Limits

Read this before planning around it. Everything here is absent, not rough.

- **No glTF loader, no OBJ, no texture decoders, no environment maps, no I/O.**
- **No animation system.** No keyframes, skinning, morph targets, or blend trees.
  Bounds are static, which is correct today and would not be with a skinned mesh.
- **No WebGL2 fallback.**
- **No post-processing chain.** There is a present pass and it tone maps. There is
  no bloom, no FXAA, no grading, and no chain: the scaffold rejects more than one
  `targets` entry, because the generated fragment entry point returns a single
  `vec4f`. A second channel is a second pass that reads the first target.
- **No shadow pass.** `pbrMaterial({ shadows: true })` is real — it declares a
  `depth-2d` slot, adds a 3×3 PCF loop, and needs `lightViewProj` set — but apse
  never renders the depth map. You render it yourself and hand it over. Turning
  the option on without doing that gives a material whose shadow term samples
  nothing.
- **No compute.** No compute pipeline, no storage buffer, no workgroup API.
- **Five materials with a shading model against three.js's eight**, two of apse's
  being unlit, so three lit BRDFs against eight. `pbrMaterial` has one
  directional light and no IBL. If you need two lights and a fresnel rim, you
  write the shader.
- **One colour attachment per pass.**
- **No editor, no scene serialisation, no inspector.**
- **No mipmap generation.** Nothing in apse generates mips today.
- **`MATERIAL_DISPOSED` and `MESH_DISPOSED` are catalogued but not yet raised.**
  `Material` and `GpuMesh` pass no code to `Resource`'s constructor, so a disposed
  one still reports `INTERNAL_INVARIANT` — a `library`-blamed code for a `caller`
  mistake. `GpuMesh` does not call `assertLive` at all. The mechanism and its tests
  exist; the two `super(...)` arguments do not.
- **The diagnostics scorecard has not been re-run.** The two scenarios where
  three.js's output was better — a WGSL syntax error's stage attribution, and
  `RENDERER_ALREADY_DISPOSED` being unreachable — have both since been fixed in
  `src/`. Nobody has re-measured, so the table above still reads 5/2/2 and is
  stale in apse's favour.
- **The benchmark is incomplete.** See [Performance](#performance). No frame time
  in this README is current.

## Commands

```bash
bun install
bun run verify   # typecheck, tests, build, size gate — the whole gate
bun test         # 830 tests, all headless
bun run bench    # headless Chrome, apse vs three.js
```

830 tests, 0 failing, `tsc --noEmit` clean, size gate passing. CI runs all of it,
and it is green-capable: it was red on every commit for a week because no job ran
`bun install`, so `tsc` failed on a missing type library before it had read a line
of apse's source. There is now an install step with `--frozen-lockfile`.

## Layout

```
src/core/       errors, result type, reference counting, uniform packing, bind groups
src/geometry/   vertex layouts, GPU mesh upload, instancing, batching, tangents,
                and nine primitives
src/material/   the scaffold, and the seven materials built through it
src/math/       vec3, mat4, quat, sphere, frustum
src/render/     device, targets, pipeline state, the frame loop, present, timing,
                readback
src/scene/      nodes, the scene graph, cameras
bench/          the headless benchmark, the tree-shaking app, diagnostics, memory
examples/       a runnable lit cube
scripts/        the build, and the size gate
test/           unit tests over the pure layers, plus a fake GPUDevice
```

## Contributing

```bash
bun install && bun run verify
```

`AGENTS.md` is the directory page — it routes you to the right file and names the
invariants that are easy to break. `ARCHITECTURE.md` is the *why*, including the
frame-order diagram and the failure modes each design choice was chosen against.

The one rule that matters for anything you add: **no claim without a measurement
or a test behind it, and state the method next to the number.** This README has
withdrawn three claims that did not survive their own benchmarks and reports one
size regression rather than the number it used to report. If you cannot source a
number, leave the placeholder and say so — a missing number gets filled in, and a
stale one gets trusted.

## Project

Pre-1.0. `VERSION` is `0.0.1` and the API will change. `FrameStats.gpu` becoming
`number | null` this pass is a preview of how that goes: the honest shape won, and
it cost a breaking change. The near-term roadmap is a post-processing chain, then
glTF, then the two `super(...)` calls the limits section is complaining about.
Until then it is a renderer with an unusually small bundle, an unusually small
heap, a real instancing path, and a material system that does not require you to
read anyone else's source to change it.
