# apse

A WebGPU renderer for people who want to write shaders, not fight a chunk system.

[![CI](https://github.com/srivtx/apse/actions/workflows/ci.yml/badge.svg)](https://github.com/srivtx/apse/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![webgpu](https://img.shields.io/badge/WebGPU-required-8A2BE2.svg)](https://gpuweb.github.io/gpuweb/)
[![types](https://img.shields.io/badge/types-in--repo-3178C6.svg)](src)
[![size](https://img.shields.io/badge/bundle-45%20KB%20gzip-42B26.svg)](#measured)
[![version](https://img.shields.io/badge/version-0.0.1%20pre--1.0-orange.svg)](#project)

```js
import {
  Renderer, PerspectiveCamera, Scene, MeshNode,
  box, upload, pbrMaterial, quat,
} from 'apse';

const renderer = await Renderer.create(document.querySelector('canvas'), {
  budget: { cpu: 2, drawCalls: 200 },
});

const scene   = new Scene({ name: 'lit-cube' });
const camera  = new PerspectiveCamera({ fov: 45, near: 0.1, far: 100, aspect: 1 });
camera.lookAt([3.2, 2.4, 4.4], [0, 0, 0], [0, 1, 0]);
renderer.camera = camera;

const cube     = upload(renderer.device.device, box({ width: 1.4 }));
const material = await pbrMaterial(renderer.device.device);

const node = new MeshNode({ name: 'cube', mesh: cube, material });
node.setPosition(0, 0.8, 0);
scene.add(node);

let angle = 0;
renderer.start((dt) => {
  angle += dt * 0.6;
  quat.fromEulerXYZ(node.rotation, 0, angle, 0);
  node.markDirty();   // `rotation` is a live view: a direct write is not noticed
});
```

`examples/lit-cube.html` is the same thing against `dist/`, plus a stats overlay.

## Measured

Apple M-series GPU, headless Chrome 153, 1280×720. Median CPU time per frame, `bun run bench`,
apse and three.js in the same browser process on the same device.

| scene | apse | three.js | ratio |
|---|---:|---:|---:|
| 1000 cubes | 0.80 ms | 0.80 ms | 1.0× |
| 2000 cubes | 1.50 ms | 1.40 ms | 0.9× |
| 4000 cubes | 2.90 ms | 2.70 ms | 0.9× |
| 5000 cubes | 3.80 ms | 3.50 ms | 0.9× |
| 1000 spheres (720k tris) | 0.70 ms | 0.90 ms | 1.3× |

**On CPU frame time, apse is currently at parity with three.js — and slower at high object counts.**
That is the honest result and it is worth being precise about why, because "at parity" hides two
different stories.

At one draw call per object, both libraries are bottlenecked by per-draw API submission, and the
term is the *number* of calls rather than anything either library does with them. apse issues two
WebGPU calls per object — a dynamic-offset `setBindGroup` plus the draw — where three.js issues
one WebGL call. apse's per-call cost is lower, and the two roughly cancel.

The margin is elsewhere, and the two tables below are the ones to read:

| | apse | three.js |
|---|---:|---:|
| bundle, tree-shaken: renderer + camera + PBR + scene graph + loop | **45.3 KB** gzip (38.6 KB brotli) | ~133 KB gzip |
| heap per scene-graph object | **1,060 B**, 21 heap objects | 1,216 B, 57 heap objects |
| isolate heap, 10,000-object scene | 12.65 MB | 17.38 MB |
| 1000-node static scene, second frame | 0 matrix writes, 1 node visit | every world matrix rewritten |
| one moving leaf 5 levels deep | 1 write, 5 visits | every world matrix rewritten |
| transform upload, static scene, steady state | 0 bytes | re-uploaded per object per frame |

The three.js bundle figure is from its own published build, for a single PBR cube. The static-scene
row matters most in a real application, and it is not a timing: `test/scene.test.ts` counts real
writes into real `Float32Array`s. Tree-shaking is verified in CI, not assumed: `bench/tree-shake.ts`
is a real minimal app bundled on every build, its gzip size is compared to a hard ceiling, and
`scripts/size-gate.ts` greps the resulting bundle to prove the five primitives the app does not
import are actually absent — with a positive control, so the check cannot pass vacuously. The gate
also prints the live size and the percentage of budget used, which is what keeps the number above
honest.

## Where apse wins, and where it does not

Measured, not asserted. Every number here comes from a harness in `bench/` you can run.

| axis | apse | three.js 0.186.1 | verdict |
|---|---:|---:|---|
| bundle, tree-shaken, full renderer + PBR + scene graph | **45.3 KB** gzip | ~133 KB | apse, 2.9× |
| heap per scene-graph object | 1,060 B / 21 heap objects | 1,216 B / 57 heap objects | apse, 1.15× by size, **2.7× by object count** |
| custom material, BRDF written by hand | 80 | 83 (`ShaderMaterial`) | tie |
| custom material, prebuilt BRDF, set properties | 5 | 5 (`MeshStandardNodeMaterial`) | tie |
| material types shipped with a full BRDF | 2 | 8 | **three.js** |
| generated code per material | 77 lines from 22 authored | 1,984 lines from 6 authored | neither — see below |
| silent-vs-typed diagnostics, 9 scenarios | better 5 · worse 2 · equal 2 | — | apse, narrowly |
| CPU frame time, 5000 objects | 3.80 ms | 3.50 ms | **three.js** |
| CPU frame time, 1000-node static scene, frame 2 | 0 matrix writes | every world matrix rewritten | apse |

**The three claims this README used to make that do not survive measurement:**

1. *"A custom material is dramatically simpler."* It is not. Writing the same two-light
   Lambert+Blinn-Phong material by hand: **80 lines in apse, 83 in three.js** — a tie. The
   scaffold's genuine win is not brevity, it is **generated code**: 77 WGSL lines from 22
   authored, against 1,984 generated from 6 on `onBeforeCompile`, and no `#include` names,
   no `customProgramCacheKey`, no `userData.shader.uniforms`.

   An earlier version of this file claimed three.js's node-material path was "15 lines, which
   beats apse outright". **That comparison was wrong and has been removed.** Those 15 lines
   are property assignment on `MeshStandardNodeMaterial` — a complete, library-provided
   Cook-Torrance BRDF. The apse side of that comparison was asked to *write the same BRDF by
   hand*. It compared a prebuilt shader against a hand-written one, and said nothing about
   either scaffold. Measured fairly — a prebuilt BRDF, properties set — both libraries are
   about 5 lines.

   The real gap is one level up: **apse ships 2 material types with full BRDFs, three.js
   ships 8.** apse's `pbrMaterial` has one directional light, no rim term, and no IBL. If you
   need two lights and a fresnel rim, you write the shader; in three.js you set properties.
   That is a shipped-materials gap, not a scaffold gap, and it is the cheapest one to close.

2. *"An `Object3D` costs 1,804 bytes."* It does not. Measured with a Chrome heap snapshot
   and GC forced: **1,216 B**, 13× less than claimed, against apse's 1,060 B. The real
   difference is not the byte total — it is that three.js allocates **57 live heap objects per
   node to apse's 21**, and 86% of apse's cost is six `Float32Array`s that would be one array
   if the design were finished.

3. *"42 typed error codes."* 38 are reachable. Two failures are **unrepresentable rather than
   diagnosed**, which beats a good error, but the expensive ones are still silent: a disposed
   material still renders, and `INTERNAL_INVARIANT` — catalogued as "always a bug in apse" —
   is used at 55 of 140 `fail()` sites, so a user's bug gets filed as a library bug.

**The one place apse is structurally better and it is not close:** two failures cannot be
expressed. You cannot compile a material whose uniform block and JS packer disagree, because
one table produces both. You cannot draw a material against a render target whose colour format
differs, because the renderer checks before `setPipeline` — where WebGPU's own answer is a
validation error that invalidates the *whole command buffer* and mentions two format enums.

## The one idea: a material is data, not a program

The usual way to customise a shader in a 3D library — three.js is the obvious example — is to
hand it a string and hope: override a chunk by its internal name, hand-declare `modelViewMatrix`
and `projectionMatrix` yourself, then discover that the program cache key is a string join of a
hundred fields the docs never mention, plus a mandatory undocumented `customProgramCacheKey`
escape hatch. None of those chunk names is a stable API, and they change between releases.

apse generates all of it. You write two statement lists and a description of what they need:

```js
import { generateScaffold } from 'apse/material';

const spec = {
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
};

const shader = generateScaffold(spec);
shader.code; // the complete WGSL program, below
```

That produces this (the `Frame` struct is the only thing shortened here — it is fifteen fields, and
its full field list is in the `fix` message further down):

```wgsl
// Generated by apse — material "fresnel". Do not edit: every
// declaration, binding, and signature below is produced from the spec.
// vertex   body may use: in, frame, obj, out, mat
// fragment body may use: in, frame, obj, mat
// ---------------------------------------------------------------------

// ---- uniform blocks ----
struct Frame { /* 15 generated fields: view, proj, viewProj, invView, invProj,
                  invViewProj, camPos, time, delta, elapsed, resolution,
                  viewport, exposure, alpha */ }
struct ObjectData {
  model : mat4x4<f32>,
  normalMatrix : mat3x3<f32>,
  objectId : u32,
  instanceId : u32,
  visibility : f32,
};
struct MaterialData {
  tint : vec3<f32>,
  rim : f32,
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
  @location(0) normal : vec3f,
  @location(1) worldPos : vec3f,
};

// ---- bindings ----
@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var<uniform> obj : ObjectData;
@group(2) @binding(0) var<uniform> mat : MaterialData;

// ---- generated by apse: vertex stage ----
@vertex
fn vs(in : VertexIn) -> Varyings {
  var out : Varyings;
  out.clip     = frame.viewProj * obj.model * vec4f(in.position, 1.0);
  out.worldPos = (obj.model * vec4f(in.position, 1.0)).xyz;
  out.normal   = normalize(obj.normalMatrix * in.normal);
  return out;
}

// ---- generated by apse: fragment stage ----
@fragment
fn fs(in : Varyings) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let v = normalize(frame.camPos - in.worldPos);
  return vec4f(mat.tint * pow(1.0 - max(dot(n, v), 0.0), mat.rim), 1.0);
}
```

Every declaration above the two entry points was inferred from the spec. You wrote no `@group`, no
`@binding`, no `struct`, and no `fn`. You cannot: a body that declares a `fn`, a `struct`, or
anything with an `@attribute` is rejected before the shader is compiled. What a body gets to see is
`in`, `out`, `frame`, `obj`, `mat`, one variable per declared texture, and WGSL's builtin library.
Nothing else.

`i32`, `u32`, and `mat4x4f` varyings get `@interpolate(flat)` emitted automatically, because the
alternative is a compile error. `generateScaffold` is a pure function of the spec — same spec in,
byte-identical program out — which is what lets apse cache pipelines by a hash instead of rebuilding
a hundred-field key on every miss. Pass `scaffold: true` on any material to log the full program.

The two shipped materials go through exactly this path, with no bypass. `basicMaterialSpec()` and
`pbrMaterialSpec()` are plain `MaterialSpec` objects you can read, diff, and copy, and the PBR one
is a real Cook-Torrance GGX shader with a PCF shadow containing no `@group`, no `@binding`, and no
`struct`.

## Errors that teach

Every error apse throws is an `AseError` with five fields, all of them always present. There is no
code path that produces a bare `Error`.

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
  if (isAseError(e)) console.error(e.code, e.why, e.fix);
}
```

```js
{
  name: 'AseError',
  code: 'SHADER_BODY_INVALID',
  message: 'The fragment body of material "typo" is invalid: `frame.viewProjj` is not a field.',
  why: 'frame is the generated Frame uniform block (@group(0)), and that container declares no
        field called "viewProjj". Every frame.field access in a body is checked against the spec
        before the shader is compiled, so an undeclared one is a mistake in the material spec
        rather than a WGSL error.',
  fix: 'Did you mean `frame.viewProj`? Available on frame: view, proj, viewProj, invView, invProj,
        invViewProj, camPos, time, delta, elapsed, resolution, viewport, exposure, alpha.',
  link: 'https://apse.dev/errors/shader-body-invalid',
}
```

`code` is a stable slug — branch on that, never on `message`. `why` is the rule that was violated.
`fix` is the one corrective action, and it names the field and the alternative. `link` is the docs
anchor for that exact code. A sixth field, `detail`, carries structured context where a code has
any: a numeric range, an attribute name, a byte offset. `toJSON()` gives you all of it at once.

Two properties matter more than the fields. The identifier check runs *before* the shader is
compiled, so a typo names the field you got wrong instead of surfacing as a WGSL error pointing at
a line in generated code you never wrote. And the failure surface is finite, enumerable, and typed:
`ERROR_CODES` is all 42 of them, `ERROR_CATALOG` is the `why`/`fix` pair for each, and both are
exported — so you can read the complete list of things that can go wrong without reading a line of
apse's source.

## Resource ownership

`GpuMesh` and `Material` both extend `Resource`, which is reference-counted. A resource starts at
`refCount === 1`, held by whoever called `upload()` or `pbrMaterial()`. Sharing one across a
thousand nodes needs no bookkeeping: a `MeshNode` points at its mesh and material and takes
nothing, so nothing can be freed out from under it while you still hold your reference.

```js
const mesh = upload(renderer.device.device, box({ width: 1.4 }));
mesh.refCount;   // 1 — yours

mesh.ref();       // 2 — a second, independently-owned holder
mesh.unref();     // 1 — nothing freed
mesh.unref();     // 0 — the vertex and index buffers are released now
```

- `unref()` drops one reference. GPU memory is released at zero, and only at zero. Safe to call on
  an already-disposed resource.
- `dispose()` force-releases immediately, whatever the count, and zeroes it. It is the "I know I am
  done" verb, and it is what scene teardown uses. On a shared resource it frees it for everyone, so
  reach for `unref()` unless you mean it.
- `ref()` returns `this`, so it assigns directly. Take one when a second thing with its own
  lifetime starts using the resource — a second scene that tears down independently, a cache, or a
  `ResourceScope`, which owns a set of resources and releases them in reverse acquisition order.
- Garbage collection is never relied on. Browser GPU object lifetimes are not deterministic, and a
  dropped wrapper can keep a multi-megabyte buffer alive indefinitely, so nothing here waits for a
  finaliser. You release it.

`renderer.dispose()` releases the device, the render targets, and the shared frame and object
uniforms. It does **not** release your meshes and materials — those are yours.

## Compatibility

WebGPU only. There is no WebGL2 fallback, and adding one is not on the roadmap. A browser without
`navigator.gpu` throws `WEBGPU_UNAVAILABLE` with the reason in the message.

- **Works:** Chrome/Edge 113+, Safari 26+ including iOS, Firefox 141+ on Windows and 147+ on Apple
  Silicon.
- **Does not work:** Firefox on Linux, Firefox on Android, Firefox on Intel Macs.

apse requests the WebGPU **compatibility** profile by default, and asks for `core` only when the
adapter advertises `core-features-and-limits`. That is the profile that reaches the largest device
base, and it is what sets the two limits the API is shaped around:

- **Zero storage buffers in the vertex stage.** Compatibility mode targets GLES 3.1-class
  hardware, which has no read-write buffer in a vertex shader. Per-object data therefore reaches
  the vertex stage through one large uniform buffer addressed with dynamic offsets and written once
  per frame, not through a per-object storage buffer.
- **`copyTextureToBuffer` rows align to 256 bytes.** `renderer.capture()` returns `bytesPerRow`
  padded to that, so read rows at `bytesPerRow`, not `width * 4`. It renders to an offscreen target
  and copies out of it, because a WebGPU canvas has no `preserveDrawingBuffer` and reading the
  swapchain after present yields an empty image.

`renderer.featureLevel` is `'core'` or `'compatibility'`, so you can branch on it.

## Status

What works: the renderer and frame loop, the scene graph, six primitives, the material scaffold,
the two shipped materials, reference-counted resources, frustum culling, depth and alpha blending,
forward rendering to a canvas or an offscreen target, `capture()` readback, declared per-frame
budgets, and typed errors over a finite catalog.

What does not work yet. Read this before planning around it.

- **The node-material comparison is a live threat, not a settled result.** three.js's
  `material.colorNode = ...` path is 15 lines against apse's 80. If that gap does not close,
  the scaffold is not a differentiator and the honest position is that apse's value is
  bundle size and correctness, not authoring speed. Closing it means the scaffold emitting
  structured errors the way the node system does — a typed node graph, not typed strings.
- **Instancing is implemented but not reachable from the scene graph.** `InstanceData`,
  `GpuInstances`, `instancedLayout()` and the second vertex-buffer slot all exist and are tested
  (`test/instancing.test.ts`, 72 tests), and a pipeline built for an instanced layout draws
  correctly on a real device. `GpuMesh` accepts `instances`, but `MeshNode` does not yet propagate
  it into the draw list, so `drawIndexed` is issued with the default instance count of 1. **Until
  that is wired, instancing is not a feature you can use**, and the CPU numbers above are the
  one-draw-per-object numbers. This is the single highest-value piece of outstanding work.
- **No batching.** Draws are grouped by material so `setPipeline` and three of the four bind
  groups are not re-set needlessly, and sorted opaque front-to-front / transparent
  back-to-front with an O(n) counting sort — but nothing merges geometry and nothing reduces the
  draw call count. Instancing is the lever; batching is the one after it.
- **One colour attachment per pass.** The fragment entry point returns a single `vec4f`.
  `PresentPass` handles the HDR intermediate, MSAA resolve and tone map for a single pass, and is
  structured so a chain of them is the next step rather than a rewrite — but there is no chain
  yet, and no bloom.
- **No animation system.** There is `renderer.start(cb)` and a version-based dirty graph. Curves,
  timelines, and skeletal animation are yours.
- **No asset loaders.** No glTF, no textures on disk, no environment maps. `textures` declares a
  slot and generates the binding; you supply the `GPUTextureView`.
- **No editor, no scene serialisation, no inspector.**
- **Four of the 42 error codes are unreachable** and two of them are the ones a user is most
  likely to hit (`MATERIAL_DISPOSED`, `MESH_DISPOSED`) — a disposed resource currently fails
  later, with a less useful message.
- **No GPU timing.** `stats.gpu` is a hardcoded `0`, and `timestamp-query` is feature-detected but
  never actually requested. Until it is, you cannot tell from the API whether a frame is CPU- or
  GPU-bound, which is the question that decides whether any of the CPU work above matters.
- **Tone mapping and MSAA are built but off by default.** `toneMapping: { }` opts in. The direct
  path writes linear values to a non-sRGB canvas, so lit surfaces come out dark — correct only when
  the canvas format is `*-srgb`.

## Commands

```
bun run verify   # typecheck, tests, build with the size gate
bun run bench    # headless Chrome, apse vs three.js, writes bench/results/report.json
```

## Layout

```
src/core/       errors, result type, reference counting, uniform packing, bind groups
src/geometry/   vertex layouts, GPU mesh upload, the six primitives
src/math/       vec3, mat4, quat, sphere, frustum
src/material/   the scaffold, and the two materials built through it
src/render/     device acquisition, targets, pipeline state, the frame loop
src/scene/      nodes, the scene graph, cameras
bench/          the headless benchmark, the tree-shaking app, its results
examples/       a runnable lit cube
scripts/        the build, and the size gate
test/           unit tests over the pure layers, plus a fake GPUDevice
```

## Project

Pre-1.0. `VERSION` is `0.0.1` and the API will change. The near-term roadmap is the shortest
distance to being usable for real work: instancing, then a post-processing chain, then glTF. Until
then it is a renderer with an unusually small bundle, an unusually small heap, and a material
system that does not require you to read anyone else's source to change it.
