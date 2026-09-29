# AGENTS.md

A directory page, not a manual: it routes you to the right file and names the
invariants that are easy to break. `ARCHITECTURE.md` is the *why*.

## What apse is

apse is a WebGPU renderer: a scene graph, a material system that generates WGSL
from a declarative spec, a geometry layer, and a frame loop. WebGPU-only, no
WebGL2 fallback. It is **not** an engine — no editor, no asset pipeline, no glTF
loader, no animation system. Start at `Renderer.create`, the `geometry`
primitives, and `Material.create`.

What it does have, and where each lives: **instancing** end to end
(`geometry/instanced.ts` → `MeshNode` → one `drawIndexed`), **geometry batching**
(`geometry/batch.ts`), **tangent generation** (`geometry/tangents.ts`), a
**present pass** with ACES tone mapping on by default (`render/present.ts` +
`material/tonemap.ts`), **GPU timestamps** (`render/timing.ts`), and **five
shipped materials** (`material/basic.ts`, `pbr.ts`, `diffuse.ts`, `emissive.ts`,
`anisotropic.ts`).

Current state: **830 tests, 0 failing**, `tsc --noEmit` clean, size gate passing.

## The file tree

Each line is a module and the one invariant it owns.

```
src/index.ts              public surface. Root + 6 subpath entries, no side effects.
src/core/error-catalog.ts every failure mode, 42 codes, each with `why` + `fix` + `blame`. Add here first.
src/core/error.ts         AseError + fail(). Six declared fields on every error, no bare Error.
src/core/result.ts        Result/ok/err. Wrong argument throws; missing capability returns.
src/core/resource.ts      ref-counted ownership. dispose() forces; unref() drops one ref.
src/core/uniform.ts       UNIFORM_TYPES + buildUniformBlock. WGSL layout rules live here.
src/core/slot.ts          the reserved blocks: FRAME_FIELDS, OBJECT_FIELDS, BIND_GROUP.
src/geometry/layout.ts    VertexLayout: one description → WGSL struct, GPU layout, CPU buffer.
src/geometry/mesh.ts      MeshData (pure CPU) vs GpuMesh (two GPUBuffers). Bounds + a finiteness check at construction.
src/geometry/instanced.ts InstanceData / GpuInstances. Vertex slot 1, stepMode 'instance'. No storage buffer.
src/geometry/batch.ts     mergeMeshes → BatchedMesh → uploadBatch. Bakes normals, and renormalises them.
src/geometry/tangents.ts  computeTangents / withTangents. The only tangent source in apse.
src/geometry/primitives/  box, sphere, plane, torus, cylinder, grid, capsule, cone, roundedBox. Pure functions, no device.
src/material/scaffold.ts  spec → WGSL, byte-identical. Body validation + the two cache keys.
src/material/material.ts  Material, FrameUniforms, ObjectUniforms, DeviceCache.
src/material/lighting.ts  the shared BRDF library: GGX, GGX-aniso, Oren-Nayar, shadow, ambient, environment.
src/material/basic.ts     unlit, two lines of WGSL per stage. The scaffold's smoke test.
src/material/pbr.ts       Cook-Torrance GGX, all bindings generated. The scaffold's proof.
src/material/diffuse.ts   Oren-Nayar with an energy-conserving Fresnel rim. No specular lobe, no metalness.
src/material/emissive.ts  a physically-shaped unlit emitter. A light source, not a surface.
src/material/anisotropic.ts  elliptical GGX in a rebuilt tangent frame. Needs a `tangent` attribute.
src/material/instanced.ts unlit per-instance tint. The material instanced draws use.
src/material/tonemap.ts   the present pass's fullscreen material. Where the w = 1.0 lives.
src/material/texture-slot.ts  texture kind → WGSL type + view dimension.
src/scene/node.ts         Node/MeshNode. localVersion vs subtreeDirty. updateWorldMatrices.
src/scene/graph.ts        Scene, culling, pooled DrawItems, OBJECT_UNIFORM_STRIDE = 256.
src/scene/camera.ts       Camera. Clip z is [0,1], near at 0. Right-handed, looking down −Z.
src/render/renderer.ts    the frame order, in one module. render(), start(), capture().
src/render/device.ts      createDevice, COMPAT_LIMITS, compatRequiredLimits, requireFeature, readCapabilities.
src/render/present.ts     PresentPass. The HDR intermediate, MSAA resolve, and the tone map.
src/render/timing.ts      GpuTimer. Query set → resolve → copy → mapAsync poll. Three pairs, never reused.
src/render/readback.ts    CaptureReadback, and the 256-byte row-pitch helpers.
src/render/target.ts      RenderTargetImpl. MSAA needs two colour textures; canvas needs one.
src/render/context.ts     CanvasSizer. CSS px vs backing px vs DPR, never zero.
src/render/pipeline-state.ts  fixed-function state: topology, cull, depth, blend, targets.
src/render/sort.ts        the counting sort, and its comparator, both exported.
src/render/types.ts       DrawItem / Drawable / RenderTarget / FrameTimingStats — the material↔render boundary.
src/math/                 vec3, mat4, quat, sphere, frustum. TypedArrays, column-major, no Euler.
scripts/build.ts          esbuild + tsc, then the gzip budget gate. Fails over budget.
scripts/size-gate.ts      checks the size *claim* is true, and the exports map resolves.
bench/run.ts              headless Chrome, both engines, pixel readback, validation check.
bench/index.html          the in-page harness: apse and three.js, same process, same device.
bench/tree-shake.ts       the minimal app the 71.23 KB headline is measured on.
bench/diag/               nine failure modes triggered for real, both libraries. The 5/2/2 scorecard.
bench/mem/                GC-forced CDP heap snapshots. The 1,060 B / 21 objects figure.
bench/dx/                 the same material written twice and counted. The 80-vs-83 line counts.
test/                     830 tests across 9 files, all headless.
examples/lit-cube.html    a working single-file app.
```

`src/index.ts` is a *superset* of the subpaths. Everything a subpath exports is
also on the root, plus the 130 runtime values and types listed there; the root
reaches 179 names. `src/index.ts` is still not the whole of `src/**` — check the
`src/<dir>/index.ts` file before assuming a name is public.

## Before you change X

- **Every material needs `targetFormat: renderer.sceneFormat`.** Tone mapping is
  on by default and the scene renders into an `rgba16float` intermediate, so the
  canvas format is wrong. Omit it and you get `RENDER_TARGET_FORMAT_MISMATCH`
  naming both formats. The two escape hatches, `toneMapping: null` and
  `hdr: false`, both give up the reason the present pass exists.
- **The instancing guard is `instanceBuffer !== null`, never `instanceCount > 1`.**
  A mesh uploaded with a thousand instances and asked to draw one of them has
  `instanceCount === 1` and a non-null buffer. Test the count and slot 1 is never
  bound, the pipeline still declares it, and the vertex stage reads whatever
  happens to be there: a wrong image and no error. `instanceCount > 0` is worse —
  it is a scene-level test that says nothing about what the layout declares.
- **`vec4f(in.position, 1.0)` is the fullscreen triangle's clip position, and the
  `1.0` is `w`.** It is the divisor of the perspective divide. `0.0` collapses
  the triangle to a point, nothing rasterises, and you get a black screen with a
  valid pipeline, a legal draw, no validation error, and a passing test. This
  exact bug shipped once. See "a test asserting a bug" below.
- **A test asserting a bug will keep the bug.** The tone map's unit test asserted
  the literal `vec4f(in.position, 0.0)` and passed, for as long as the bug
  existed, because the test recorded the implementation rather than the intent.
  When you fix something, check that the test that described the old behaviour is
  the thing that changed, and not just the code.
- **`DrawItem.model` is a live reference** to the node's `Float32Array(16)`,
  not a copy. Copy it and the transform freezes; fail to update it and the
  renderer packs identity. It is the only place a draw item meets the graph.
- **A disposed resource reports whatever code its subclass passed to `super()`.**
  `Material` and `GpuMesh` both call `super()` with none, so they get
  `INTERNAL_INVARIANT` — a `library`-blamed code for a `caller` mistake. Passing
  `'MATERIAL_DISPOSED'` / `'MESH_DISPOSED'` is the one-line fix and it is not
  done. `GpuMesh` does not call `assertLive` at all yet.
- **`FrameUniforms` and `ObjectUniforms` are shared per device** through
  `deviceCache(device)`, keyed by `DEFAULT_FRAME_LABEL` / `DEFAULT_OBJECT_LABEL`.
  A second instance means the renderer writes uniforms the materials' bind groups
  never read: no error, every object at identity.
- **Write the frame uniform through `FrameUniforms.set()`,** never `.block`.
  `set` is what marks it dirty; `flush` is a no-op otherwise. A zeroed
  `viewProj` makes every triangle degenerate while every draw succeeds.
- **`getMappedRange()` is detached by `unmap()`,** and `new Uint8Array(ab)` is
  a view. `.slice()` inside the mapped window or you read all zeroes.
- **`ObjectUniforms` stride is 256 bytes, not 124.** Every dynamic offset must be
  a multiple of `minUniformBufferOffsetAlignment`; halving the stride turns a
  compile-time constant into a validation error at the second object's draw.
- **A `mat3x3f` in a uniform block is three 16-byte-aligned columns of four
  floats.** `normalMatrixOf` writes a padded 12-float scratch for exactly this
  reason, with indices 3, 7 and 11 left at zero. A tight 9-float write makes the
  consumer read indices 9 and 10 past the end of the array and arrive as `NaN`.
  This shipped once too.
- **A timestamp write index may be written once per submission.** `GpuTimer`
  takes three pairs for that reason: a frame can open an opaque pass, a
  transparent pass, and the present. A pair opened on one pass cannot be closed
  on the next — that is a validation error, and a validation error on a render
  pass discards the pass.
- **`resolveQuerySet` and `copyBufferToBuffer` are encoded, not awaited, and
  `mapAsync` is never awaited.** The reading lands a frame or two later, which is
  what `stats.gpu` reports. A frame that opens no stamped pass reports `null`, not
  `0`, and `NaN` goes into the averaging ring — a `0` there would be averaged in
  as though the GPU had been timed at zero.
- **`copyTextureToBuffer` needs `bytesPerRow` to be a multiple of 256.** Read the
  result at that stride, not `width * 4`. `unpadRows()` does it for you.
- **A WebGPU canvas has no `preserveDrawingBuffer`.** Reading it after present
  gives black. Use `Renderer.capture()`.
- **Dynamic-offset arguments must be a plain `number[]`, never a typed array.**
  Blink's IDL conversion for `sequence<>` has a fast path that requires a real
  `v8::Array`; a `Uint32Array` falls off it onto the generic iterator protocol.
  One `setBindGroup` measured 1.22 us with `Uint32Array(1)` against 0.28 us with
  `number[]`, in core and in compatibility alike — and the call count is
  byte-identical either way, so no call census can detect the regression. This
  was the whole 1.32x per-draw deficit against three.js, and it survived three
  profiling passes. If you touch a hot WebGPU call, time one isolated call with
  the queue drained before you reason about how many calls there are.
- **`GpuMesh.vertexBuffer` / `.indexBuffer` are accessors that check liveness.**
  Hoist each into a local and read it once. They are method calls, not field
  reads, and at 5,000 draws an extra read is 5,000 extra liveness checks in the
  loop that is 85-100% of the frame.
- **Never pass `adapter.limits` as `requiredLimits`,** and never spread it:
  `GPUSupportedLimits` is prototype getters, so `{...adapter.limits}` is `{}`.
  `requiredLimits` also rejects unknown keys. Send the compatibility profile.
- **Compatibility mode zeroes `maxStorageBuffersInVertexStage`.** A vertex shader
  reading a storage buffer compiles on a laptop and fails on a phone. This is why
  instancing binds slot 1 instead.
- **`vec3<f32>` is 16-byte aligned but 12 bytes wide.** Take every offset from
  `buildUniformBlock` / `OBJECT_BLOCK`; never hand-compute one.
- **A material's pipeline is shared; never destroy it.** `Material.onDispose`
  frees only its own uniform buffer. `GPUBindGroup` has no `destroy()`.
- **Samplers with identical config share one binding under one name,** taken from
  the first slot in the group. `describeMaterial(spec)` is the authority.
- **Bounds need a finiteness check, not just a comparison.** A `NaN` vertex
  produces a *finite* bound that does not contain the mesh, and the frustum then
  drops a visible object with no error. `MeshData` rejects non-finite positions
  with the vertex index in the message.
- **Merged normals are renormalised, and that is not redundant.** Baking a
  normal under a non-uniform scale leaves it the wrong length. It is equivalent
  only for rigid transforms, which is the case you will not test.

## Commands

```
bun test              # 830 headless tests
bun run typecheck     # tsc --noEmit
bun run build         # bundle + size gate; exits non-zero over budget
bun run bench         # real headless Chrome; fails if nothing drew
bun run verify        # all four, in that order
```

`bun run typecheck` needs `dist/` to exist: `bench/dx/apse-custom-material.ts`
has a static type-level import of `../../dist/index.js` and `tsconfig.json`
typechecks `bench/**`, so on a fresh checkout `tsc` reports two TS2307 errors
before the build has run. CI works around it with a `build` step before
`verify`. The real fix belongs in that bench file — it should import
`src/index.ts` like the rest of the repo, or be excluded from the program.

`bun run build` measures tree-shaking and writes `dist/size-report.json`;
`scripts/size-gate.ts` then checks the claim is actually true (unused primitives
really were dropped, with a positive control) and that every `exports` subpath has
a built file.

### Size budgets

Ceilings in KiB gzip, and the measured value against each, from `bun run build` on
an Apple M3, 2026-09-28. Every entry sits at 87% of its ceiling; the gate prints
the percentage on every run, so a regression is visible long before it fails.

| entry | ceiling | measured |
|---|---:|---:|
| `index` | 110.3 | 95.87 |
| `core/index` | 10.1 | 8.74 |
| `math/index` | 13.4 | 11.66 |
| `geometry/index` | 37.5 | 32.64 |
| `material/index` | 64.4 | 56.04 |
| `scene/index` | 18.8 | 16.34 |
| `render/index` | 80.4 | 69.95 |
| **tree-shaken app** | **81.9** | **71.23** |

The tree-shaken app is the headline: 71.23 KB gzip against a three.js PVR-cube
baseline of 133 KB, which is **1.87×**. It was 2.9× at 45.3 KB before the present
pass and the timestamp layer, which together cost 26 KB. That regression is the
price of fixing a black tone map and a stat that reported a GPU that was never
measured. `scripts/build.ts` and `scripts/size-gate.ts` each carry the ceilings
and a `checkBudgetAgrees` check fails if the two ever drift.

## Conventions

- TypeScript, `strict`, `noUnusedLocals`, `verbatimModuleSyntax` — so
  `import type` is required, not stylistic. Import paths carry `.ts`.
- ESM only; no CJS is emitted. `sideEffects: false` in `package.json` is a
  promise: no top-level work, no listener registration, no prototype patching
  anywhere in `src`. Keep it.
- String-literal unions, never `enum` (`'opaque' | 'transparent'`).
- One obvious way to call each thing. `Material.create` is the only way to build
  a material; the constructor is private on purpose.
- Errors go through `fail(code, message, { why, fix })` with a code that exists in
  `error-catalog.ts` **and** a blame classification that is right. Unknown options
  are rejected, not ignored. A code that cannot be reached is not a code.
- Comments explain why, not what. If a comment restates the next line, cut it.
- No classes in `src/math`; nothing there allocates outside `create`/`clone`.
- If you add a magic number, put the reason in a comment or a test.

## How to add a thing

- **A primitive** — new `src/geometry/primitives/<name>.ts` returning `MeshData`.
  Write through `VertexWriter` (it resolves the offsets for you) and
  `allocateIndices`. Re-export from `primitives/index.ts`, `geometry/index.ts`,
  and `src/index.ts`. `grid.ts` delegates; `torus.ts` is the one with a closed
  surface in both directions; `cone.ts` is defined in terms of `cylinder`. **Check
  the winding**: `capsule()` shipped with all 168 triangles inside-out, and no
  validation error anywhere catches it.
- **A material** — a function returning a `MaterialSpec`, following
  `basicMaterialSpec` / `pbrMaterialSpec`, with a `…MaterialSpec()` export beside
  the `…Material()` factory. Statements only in `vertex` and `fragment`; helper
  `fn`s go in `prelude`. A new spec option must be added to `SPEC_KEYS` in
  `scaffold.ts` or it will be rejected. Shared BRDF terms belong in
  `material/lighting.ts` so the next material does not re-derive them.
- **A vertex layout** — one entry in `VERTEX_FORMATS` in `geometry/layout.ts`,
  with the WGSL type, component count, typed array, and byte unit. If a
  `TANGENT_LAYOUT` default is involved, remember the `mat3x3f` column padding
  above.
- **An error code** — add a member to the `ErrorCodeEntry` union *and* an
  `ERROR_CATALOG` entry in `error-catalog.ts`; the catalog's mapped type makes the
  second a compile error if you forget. Then a `BLAME` entry, which is likewise
  mapped. Then `fail(code, …)`. If it is structurally unrepresentable, do not add
  it: put the guarantee in the type system, where it is stronger. `VARYING_MISMATCH`
  and `SHADER_NO_ENTRYPOINT` were deleted for exactly that reason.
- **A uniform slot type** — add the name to `SlotType` in `core/slot.ts` and its
  row to `UNIFORM_TYPES` in `core/uniform.ts`, which is keyed by `SlotType` so the
  compiler forces both. Take `align` and `size` from the WGSL uniform
  address-space rules; `alignUp` handles the rest.
- **A timing measurement** — one `GPUQuerySet`, a resolve buffer, and *at least
  two* staging buffers, because one means every frame's copy lands on the buffer
  the previous frame's `mapAsync` is still using. Encode, never await.
- **A claim in a document** — a number and the method that produced it, in the
  same sentence, or no number. A placeholder that says what is being measured is
  better than an estimate that gets trusted.
