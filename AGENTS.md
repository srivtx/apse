# AGENTS.md

A directory page, not a manual: it routes you to the right file and names the
invariants that are easy to break. `ARCHITECTURE.md` is the *why*.

## What apse is

apse is a WebGPU renderer: a scene graph, a material system that generates WGSL
from a declarative spec, a geometry layer, and a frame loop. WebGPU-only, no
WebGL2 fallback. It is **not** an engine — no editor, no asset pipeline, no
glTF loader, no animation system, no instancing, no batching. Start at
`Renderer.create`, the `geometry` primitives, and `Material.create`.

## The file tree

Each line is a module and the one invariant it owns.

```
src/index.ts              public surface. Root + 6 subpath entries, no side effects.
src/core/error-catalog.ts every failure mode, 42 codes, each with `why` + `fix`. Add here first.
src/core/error.ts         AseError + fail(). Five fields on every error, no bare Error.
src/core/result.ts        Result/ok/err. Wrong argument throws; missing capability returns.
src/core/resource.ts      ref-counted ownership. dispose() forces; unref() drops one ref.
src/core/uniform.ts       UNIFORM_TYPES + buildUniformBlock. WGSL layout rules live here.
src/core/slot.ts          the reserved blocks: FRAME_FIELDS, OBJECT_FIELDS, BIND_GROUP.
src/geometry/layout.ts    VertexLayout: one description → WGSL struct, GPU layout, CPU buffer.
src/geometry/mesh.ts      MeshData (pure CPU) vs GpuMesh (two GPUBuffers). Bounds at construction.
src/geometry/primitives/  box, sphere, plane, torus, cylinder, grid. Pure functions, no device.
src/material/scaffold.ts  spec → WGSL, byte-identical. Body validation + the two cache keys.
src/material/material.ts  Material, FrameUniforms, ObjectUniforms, DeviceCache.
src/material/basic.ts     unlit, two lines of WGSL per stage. The scaffold's smoke test.
src/material/pbr.ts       Cook-Torrance GGX, all bindings generated. The scaffold's proof.
src/material/texture-slot.ts  texture kind → WGSL type + view dimension.
src/scene/node.ts         Node/MeshNode. localVersion vs subtreeDirty. updateWorldMatrices.
src/scene/graph.ts        Scene, culling, pooled DrawItems, OBJECT_UNIFORM_STRIDE = 256.
src/scene/camera.ts       Camera. Clip z is [0,1], near at 0. Right-handed, looking down −Z.
src/render/renderer.ts    the frame order, in one module. render(), start(), capture().
src/render/device.ts      createDevice, COMPAT_LIMITS, compatRequiredLimits.
src/render/target.ts      RenderTargetImpl. MSAA needs two colour textures; canvas needs one.
src/render/context.ts     CanvasSizer. CSS px vs backing px vs DPR, never zero.
src/render/pipeline-state.ts  fixed-function state: topology, cull, depth, blend, targets.
src/render/types.ts       DrawItem / Drawable / RenderTarget — the material↔render boundary.
src/math/                 vec3, mat4, quat, sphere, frustum. TypedArrays, column-major, no Euler.
scripts/build.ts          esbuild + tsc, then the gzip budget gate. Fails over budget.
scripts/size-gate.ts      checks the size *claim* is true, and the exports map resolves.
bench/run.ts              headless Chrome, both engines, pixel readback, validation check.
bench/index.html          the in-page harness: apse and three.js, same process, same device.
test/                     354 tests, all headless.
examples/lit-cube.html    a working single-file app.
```

`src/index.ts` is a *subset* of the subpaths: `getNodeVisitCount`,
`validateBodyIdentifiers`, `VertexWriter`, `layoutCached`, and
`compareDrawItems` are exported from `apse/scene` / `apse/material` /
`apse/geometry` / `apse/render` but not from the root. Check the
`src/<dir>/index.ts` file before assuming a name is public.

## Before you change X

- **`DrawItem.model` is a live reference** to the node's `Float32Array(16)`,
  not a copy. Copy it and the transform freezes; fail to update it and the
  renderer packs identity. It is the only place a draw item meets the graph.
- **`FrameUniforms` and `ObjectUniforms` are shared per device** through
  `deviceCache(device)`, keyed by `DEFAULT_FRAME_LABEL` / `DEFAULT_OBJECT_LABEL`.
  A second instance means the renderer writes uniforms the materials' bind
  groups never read: no error, every object at identity.
- **Write the frame uniform through `FrameUniforms.set()`,** never `.block`.
  `set` is what marks it dirty; `flush` is a no-op otherwise. A zeroed
  `viewProj` makes every triangle degenerate while every draw succeeds.
- **`getMappedRange()` is detached by `unmap()`,** and `new Uint8Array(ab)` is
  a view. `.slice()` inside the mapped window or you read all zeroes.
- **`ObjectUniforms` stride is 256 bytes, not 124.** Every dynamic offset must
  be a multiple of `minUniformBufferOffsetAlignment`; halving the stride turns
  a compile-time constant into a validation error at the second object's draw.
- **`copyTextureToBuffer` needs `bytesPerRow` to be a multiple of 256.** Read
  the result at that stride, not `width * 4`.
- **A WebGPU canvas has no `preserveDrawingBuffer`.** Reading it after present
  gives black. Use `Renderer.capture()`.
- **Never pass `adapter.limits` as `requiredLimits`,** and never spread it:
  `GPUSupportedLimits` is prototype getters, so `{...adapter.limits}` is `{}`.
  `requiredLimits` also rejects unknown keys. Send the compatibility profile.
- **Compatibility mode zeroes `maxStorageBuffersInVertexStage`.** A vertex
  shader reading a storage buffer compiles on a laptop and fails on a phone.
- **`vec3<f32>` is 16-byte aligned but 12 bytes wide.** Take every offset from
  `buildUniformBlock` / `OBJECT_BLOCK`; never hand-compute one.
- **A material's pipeline is shared; never destroy it.** `Material.onDispose`
  frees only its own uniform buffer. `GPUBindGroup` has no `destroy()`.
- **Samplers with identical config share one binding under one name,** taken
  from the first slot in the group. `describeMaterial(spec)` is the authority.

## Commands

```
bun test              # 354 headless tests
bun run typecheck     # tsc --noEmit
bun run build         # bundle + size gate; exits non-zero over budget
bun run bench         # real headless Chrome; fails if nothing drew
bun run verify        # all four, in that order
```

`bun run build` also measures tree-shaking and writes `dist/size-report.json`;
`scripts/size-gate.ts` then checks the claim is actually true (unused primitives
really were dropped) and that every `exports` subpath has a built file. Budgets
(gzip KB): `index` 60, `tree-shaken app` 48, `render` 40, `material` 24, `scene`
13, `geometry` 12, `core` 10, `math` 8. Current: `index` 55.3, tree-shaken app
43.5.

## Conventions

- TypeScript, `strict`, `noUnusedLocals`, `verbatimModuleSyntax` — so
  `import type` is required, not stylistic. Import paths carry `.ts`.
- ESM only; no CJS is emitted. `sideEffects: false` in `package.json` is a
  promise: no top-level work, no listener registration, no prototype patching
  anywhere in `src`. Keep it.
- String-literal unions, never `enum` (`'opaque' | 'transparent'`).
- One obvious way to call each thing. `Material.create` is the only way to
  build a material; the constructor is private on purpose.
- Errors go through `fail(code, message, { why, fix })` with a code that exists
  in `error-catalog.ts`. Unknown options are rejected, not ignored.
- Comments explain why, not what. If a comment restates the next line, cut it.
- No classes in `src/math`; nothing there allocates outside `create`/`clone`.
- If you add a magic number, put the reason in a comment or a test.

## How to add a thing

- **A primitive** — new `src/geometry/primitives/<name>.ts` returning
  `MeshData`. Write through `VertexWriter` (it resolves the offsets for you) and
  `allocateIndices`. Re-export from `primitives/index.ts`, `geometry/index.ts`,
  and `src/index.ts`. `grid.ts` is 43 lines and delegates; `torus.ts` is the
  one with a closed surface in both directions.
- **A material** — a function returning a `MaterialSpec`, following
  `basicMaterialSpec` / `pbrMaterialSpec`. Statements only in `vertex` and
  `fragment`; helper `fn`s go in `prelude`. A new spec option must be added to
  `SPEC_KEYS` in `scaffold.ts` or it will be rejected.
- **An error code** — add a member to the `ErrorCodeEntry` union *and* an
  `ERROR_CATALOG` entry in `error-catalog.ts`; the catalog's mapped type makes
  the second a compile error if you forget. Then `fail(code, …)`.
- **A uniform slot type** — add the name to `SlotType` in `core/slot.ts` and
  its row to `UNIFORM_TYPES` in `core/uniform.ts`, which is keyed by `SlotType`
  so the compiler forces both. Take `align` and `size` from the WGSL uniform
  address-space rules; `alignUp` handles the rest.
- **A vertex format** — one entry in `VERTEX_FORMATS` in `geometry/layout.ts`,
  with the WGSL type, component count, typed array, and byte unit.
