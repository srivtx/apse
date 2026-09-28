# ARCHITECTURE.md

The *why*. `AGENTS.md` is the routing table; this is the reasoning, the
measurements, and the things that do not exist yet. Every number here comes from
[Performance model](#performance-model) or from a comment in the source.

---

## The four load-bearing abstractions

Each collapses a class of bug by making a disagreement between two
representations *unrepresentable*. The test is not elegance — it is whether two
representations can still tell different stories.

### 1. `VertexLayout` — one description, three consumers

A vertex attribute needs describing three times in a naive renderer: a WGSL struct
field, a `GPUVertexBufferLayout` entry, and a byte offset into the interleaved CPU
buffer. When those disagree the disagreement is silent — the pipeline validates
against the layout it was handed, the CPU fills at the offsets it assumed, and the
vertex stage reads floats that mean something else.

`src/geometry/layout.ts` produces all three from one record:

```ts
const L = layout({ position: 'float32x3', normal: 'float32x3', uv: 'float32x2' });
L.wgslStruct('VertexIn');   // struct VertexIn { @location(0) position : vec3<f32>, ... }
L.gpuLayout();              // { arrayStride: 32, attributes: [{ shaderLocation: 0, offset: 0, ... }] }
L.allocate(count);          // the interleaved Float32Array
```

`L.key` is the canonical string `position:float32x3|normal:float32x3|uv:float32x2`
— order-significant, because declaration order *is* the byte layout — and it is
what `pipelineStateKeyOf` hashes, so layout identity is a string compare.

`VERTEX_FORMATS` is the second half: each entry carries WGSL type, component
count, the typed array the CPU packs with, and byte size, so an unknown format is
rejected at construction rather than yielding a `byteSize` of `NaN`.
`float16x2`/`float16x4` are deliberately absent — WebGPU supports them, but
packing needs an f16 encoder no JS engine exposes, and a format with a slow path
is worse than no format.

`assertCompatible` (called from `Material.updateMesh`) makes the claim pay off at
runtime: layouts must match exactly, in names *and* formats. A partial match is a
WebGPU error at `setVertexBuffer` naming a shader location and a format and
nothing else; apse rejects it at bind time with the difference named.

### 2. `buildUniformBlock` — one table, WGSL and bytes

The same disagreement one layer down. A uniform field has a WGSL type, a byte
offset, a size, and a CPU packing routine. WGSL's uniform address space is
stricter than the storage address space, and the strictness is where hand-written
code goes wrong: **`vec3<f32>` is 16-byte aligned but only 12 bytes wide**, and
the next member starts at the next 16-byte boundary. `mat3x3<f32>` is three
16-byte-aligned columns — 48 bytes with a dead float in each.

`UNIFORM_TYPES` in `src/core/uniform.ts` is the single table: `wgsl`,
`components`, `align`, `size`, `kind`. `buildUniformBlock('MaterialData', { a:
'vec3f', b: 'f32' })` returns the struct text, per-field offsets, the struct size,
and — with `dynamic: true` — the 256-aligned stride. One call site computes the
offset; there is no second place that could compute it differently. `UniformBlock`
then writes *by name* through the same field table, validating component count and
rejecting non-finite values once, in one place, rather than at six scattered call
sites.

Those offsets are used verbatim by the generated WGSL, the bind group layout's
`minBindingSize`, `setSlot`'s bounds check, and the byte range `flushSlots`
uploads. Change the table and all of them change. The reserved blocks live in
`src/core/slot.ts`: `FRAME_BLOCK` (14 fields, 432 bytes), `OBJECT_BLOCK` (5 fields,
128 bytes, stride 256), `MATERIAL_BLOCK` (empty by default; the scaffold fills it
from the spec).

### 3. The material scaffold — statements in, WGSL out

This is the core innovation. A material here is a *specification*, not a shader
program. The user writes two statement lists; apse generates every WGSL struct,
every `@group`, every `@binding`, and the entry-point signatures.

**Why not string-surgery.** The alternative — handing users WGSL to patch — is
what makes shader customization unusable elsewhere: overriding a chunk means
knowing dozens of internal names that are not a stable API; the same uniform is
reachable through three paths, one an implementation detail; the cache key
becomes a hundred-component string join recomputed on every miss plus a mandatory
undocumented escape hatch; and a custom material hand-declares
`modelViewMatrix`, `projectionMatrix`, `normalMatrix`, `cameraPosition`, and
`#ifdef` blocks for fog, clipping, morph targets, and skinning — none of which
exist here. Every one of those is now generated from data.

**A worked example.** Three declarations and two statement lists:

```ts
const spec: MaterialSpec = {
  name: 'tint',
  varyings: { uv: 'vec2f', tint: 'vec3f' },
  slots: { color: { type: 'vec3f', default: [1, 0.4, 0.1] } },
  textures: { albedo: { kind: '2d' } },
  phase: 'opaque',
  targets: [{ format: 'rgba8unorm' }],

  vertex: `
out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.uv = in.uv;
out.tint = mat.color;
`,
  fragment: `
let t = textureSample(albedo, albedoSampler, in.uv);
return vec4f(t.rgb * in.tint, 1.0);
`,
};

const { code } = generateScaffold(spec);
```

The generated program, abbreviated only in `Frame` and `ObjectData`:

```wgsl
// Generated by apse — material "tint". Do not edit: every
// declaration, binding, and signature below is produced from the spec.
// vertex   body may use: in, frame, obj, out, mat, albedo, albedoSampler
// fragment body may use: in, frame, obj, mat, albedo, albedoSampler

struct Frame { view : mat4x4<f32>, /* …13 more fields… */ }
struct ObjectData { model : mat4x4<f32>, normalMatrix : mat3x3<f32>, /* … */ }
struct MaterialData { color : vec3<f32>, };

struct VertexIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
};

struct Varyings {
  @builtin(position) clip : vec4f,
  @location(0) uv : vec2f,
  @location(1) tint : vec3f,
}

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var<uniform> obj : ObjectData;
@group(2) @binding(0) var<uniform> mat : MaterialData;
@group(3) @binding(0) var albedo : texture_2d<f32>;
@group(3) @binding(1) var albedoSampler : sampler;

@vertex
fn vs(in : VertexIn) -> Varyings {
  var out : Varyings;
  out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
  out.uv = in.uv;
  out.tint = mat.color;
  return out;
}

@fragment
fn fs(in : Varyings) -> @location(0) vec4f {
  let t = textureSample(albedo, albedoSampler, in.uv);
  return vec4f(t.rgb * in.tint, 1.0);
}
```

**What the body can see, and nothing else.** In the vertex stage, `in` is the
`VertexIn` fields, `out` is the `Varyings` fields, `frame`/`obj`/`mat` are the
`Frame` / `ObjectData` / declared `slots`, and each declared texture is one
variable plus its sampler. In the fragment stage `in` is the `Varyings` fields,
`out` does not exist (it is an input there), and the rest are identical. Plus
WGSL's builtin library. The generated header lists the names per stage, so the
answer to "what may I write here" is in the program you are about to compile.

Bind group indices are fixed by `BIND_GROUP` — frame 0, object 1, material 2,
texture 3 — chosen by change frequency. Fixed, not allocated, so a material with
no textures can leave group 3 unbound and the renderer's positional `setBindGroup`
calls stay correct; an unused group is `undefined` in the pipeline layout, which
reserves the index without declaring anything in it.

**Validation as a feature.** `validateBodyIdentifiers` checks every `in.`, `out.`,
`frame.`, `obj.`, and `mat.` access against what the spec actually declared, before
compilation. A typo produces a typed `AseError` naming the field, naming the
container, and listing what exists:

```
SHADER_BODY_INVALID: The fragment body of material "tint" is invalid:
`in.uv` is not a field.
  why: in is the Varyings struct, i.e. the material's declared `varyings`, and
       that container declares no field called "uv". …
  fix: Declare "uv" on the material. Available on in: clip, tint.
```

Without it the same mistake reaches Tint or Naga and returns a WGSL error whose
line number points into *generated* code — a line the author never wrote and
cannot see. `SHADER_COMPILE_FAILED` says as much: the fault is almost always in
your body, and the reported line is inside it. `validateGeneratedWGSL` runs on
apse's own output before the driver sees it — balanced braces, every struct field
comma-terminated, `@group(n)` indices agreeing with `BIND_GROUP` — which turns a
template bug into a named apse failure rather than a driver message.

The scanner also rejects a body that declares a `fn`, a `struct`, or any
`@attribute`. That boundary is what makes the abstraction safe: if a body could
declare a binding, the bind group layout, the pipeline layout, and the generated
code would all have to be reconciled against text apse does not control — the exact
failure mode the module removes. Helpers go in `prelude`, inserted *after* the
generated declarations and *before* the entry points, so it can add code and cannot
redefine a binding. `pbr.ts` puts its three GGX functions there, which is why the
PBR material contains no `@group` and no `struct`.

**Determinism, and the two cache keys.** `generateScaffold` is a pure function of
its spec — byte-identical output in any process, no WebGPU global touched — so a
test can assert on generated WGSL with no device in sight. That property is what
makes the cache correct rather than hopeful. Two O(1) FNV-1a hashes:

- **`pipelineStateKey`** — structural state only: layout key, varying order and
  flatness, slot types, texture kinds and sample types, sampler keys, topology,
  cull, front face, depth, blend, target formats, sample count, depth format,
  strip index format. It **excludes the body**, because the bind group layouts a
  material needs depend on its declarations, not its statements. Verified: 500
  `pbrMaterialSpec` variants with distinct base colours produce **1** distinct state
  key, so they share roughly four `GPUBindGroupLayout` objects.
- **`pipelineKey`** — the state key plus a hash of the tokenised code (comments
  deleted, whitespace collapsed). It **includes the body**, because a
  `GPURenderPipeline` is a compiled, linked shader module plus fixed-function
  state, not a bundle of interchangeable settings. Sharing one across different
  bodies renders one material with the other's code — no diagnostic, no warning, a
  plausible-looking frame.

Stripping comments is what keeps the second question answerable: the material's
`name` goes into the generated header and bodies are liberally commented, so
without it every material would be its own pipeline and the cache decorative.
Comments cannot change what a shader does, so they cannot change its identity.
Verified: `pbrMaterialSpec({name:'a'})` and `{name:'b'}` share a `pipelineKey`.

**Sampler sharing.** Slots whose sampler configuration — sample type, mipmap
filtering, address mode, comparison — is byte-identical share **one** binding,
because a sampler is immutable state carrying no per-texture data. That halves a
typical multi-texture material's binding count and removes N−1 redundant
`GPUSampler` objects. A shared sampler has exactly one name, from the *first* slot
in the group; a body reaching for the second slot's `<name>Sampler` is rejected by
the validator rather than handed to WGSL as undefined.
`describeMaterial(spec).textures[i].sampler` is the authority.

### 4. One object buffer, addressed by dynamic offset

`ObjectData` is the same struct for every material, so it is not material-specific
and needs no buffer per material. It is one large `GPUBuffer` bound once with a
dynamic offset per draw. As laid out: `model` (mat4x4f, 64 bytes at offset 0),
`normalMatrix` (mat3x3f, 48 bytes at 64 — three 16-byte columns, 4 floats each),
`objectId` and `instanceId` (u32 at 112 and 116), `visibility` (f32 at 120). That
is 124 bytes of data and 128 of struct.

The stride between objects is **256 bytes**, not 124 and not 128.
`minUniformBufferOffsetAlignment` is 256 on every profile and every dynamic offset
must be a multiple of it. Object 1 at offset 128 is invalid, and it surfaces as a
validation error at the first draw of the *second* object, not at buffer creation.
Padding is a hard constraint, not tidiness: a std140-compatible layout with every
field padded out wastes the same memory and buys nothing, because the requirement
is imposed by the binding, not the layout.

The measured consequence, from `renderer.ts`: per-object uniform data is the
largest CPU cost in a naive WebGPU renderer, and the standard failure shape — one
uniform buffer plus one bind group per object — measures at roughly 8,000 objects
before the frame collapses on an M1. apse's per-object cost is a `setBindGroup`
with an integer offset and nothing else, and a frame's whole transform data goes
out in one `queue.writeBuffer` over a `Float32Array` the CPU already had to touch.

Two details make it correct, not just fast. The buffer is **allocated lazily on
first draw**, grown by doubling: a scene built and never rendered pays nothing, and
a scene oscillating around a threshold does not reallocate every frame — growth
invalidates the bind group, because a bind group captures its buffer, and
`ObjectUniforms` rebuilds it. And `pack()` writes a **CPU mirror**, with the
renderer calling `uploadFrom(count)` once after packing every object — one
`writeBuffer` for 2,000 objects, not 2,000. Field offsets come from `OBJECT_BLOCK`,
the same table the WGSL declares, resolved once at module load; a `find` in a
per-object loop is exactly the cost that makes a scene stutter.

The same reasoning gives the frame uniform a single instance: 500 materials
writing the same 432 bytes is 500 writes of 216 KB to say one thing 500 times. Both
blocks resolve through `deviceCache(device)`, keyed by `DEFAULT_FRAME_LABEL` and
`DEFAULT_OBJECT_LABEL`, so the renderer and the materials get the *same* objects.
Constructing a parallel pair is the one bug here with no diagnostic: the renderer
writes into a buffer the material's bind group never reads, no error is raised, and
every object draws at identity.

---

## The frame order

`src/render/renderer.ts` is the only module that knows it. Steps 2–5 are CPU time
inside a 16.67 ms budget already shared with the compositor, the input system, and
the collector — which is what makes a declared budget worth having. 0.3 ms CPU and
4 ms GPU is well-behaved; 3 ms CPU and 0.5 ms GPU is broken, and no amount of GPU
in the machine fixes it.

1. **size** — resize only if the `CanvasSizer` says the backing store changed.
   Resizing invalidates the swapchain texture, and WebGPU has no default
   framebuffer, so depth must be rebuilt too. Doing it per notification
   reallocates megabytes every time a scrollbar appears.
2. **cull** — `collectDrawItems` runs the pruned transform pass, then tests every
   mesh node's sphere against six planes. Two passes on purpose; see below.
3. **sort** — opaque front-to-back (early-Z rejection: a bandwidth optimisation),
   transparent back-to-front (blending is order-dependent: visual). Both grouped by
   material first, because `setPipeline` is expensive and one comparator
   comparison pays for itself after two items share a material. `objectId` is the
   final tiebreaker, making the sort stable against a deterministic total order — a
   comparator returning 0 for two distinct items produces a different picture on a
   different day.
4. **pack** — world matrix and inverse-transpose normal matrix per object into the
   CPU mirror. The normal matrix uses the adjugate expanded by hand: 27 multiplies
   against a general 4×4 inverse's ~120, and at a thousand objects that is a third
   of the frame. A degenerate transform falls back to identity, so the object looks
   un-transformed rather than invisible — a bug a user can see and report.
5. **upload** — one `writeBuffer` for the frame block, one for the object block.
   The frame block goes through `FrameUniforms.set()`, which marks it dirty, then
   `flush()`. Writing `.block` directly leaves the flag clear, `flush` no-ops, and
   the GPU reads a zeroed `viewProj`: every triangle degenerate, every draw
   successful, no error anywhere.
6. **encode** — one encoder. The pass is opened once for opaque and again for
   transparent, because depth state is baked into the pipeline and cannot change
   mid-pass; a pipeline change *within* a phase does not need a new pass. The
   colour format is checked against the target before `setPipeline` — a mismatch
   invalidates the whole command buffer with no exception, and Dawn's message names
   two format enums rather than the mistake.
7. **submit** — one `queue.submit`.
8. **stats** — `FrameStats` over a rolling window, plus budget checks. Averaged
   rather than per-frame because one slow frame is a GC pause, not a regression;
   the window is the difference between a budget that reports a problem and one
   that fires every time a shader finishes compiling.

**`render()` is synchronous on purpose.** Every caller that wants a frame *now* — a
test, a benchmark, a screenshot, one draw into a readback buffer — should not have
to reason about a `requestAnimationFrame` that may not fire for another 16 ms, or
at all if the tab is hidden. `start()` is the async layer over the same
synchronous `render()`.

**`start()` clamps dt to 100 ms** because a backgrounded tab resumes with a delta
of seconds and `position += velocity * dt` teleports. Clamping once, here, means no
user code has to. Note that the uniform's `frame.delta` is written from
`#lastDelta`, which the frame loop never advances — a shader reading `frame.delta`
sees 0 today. `frame.time` and `frame.elapsed` are correct.

`capture()` is the only supported way to get pixels out of a canvas, and it does so
by rendering to an offscreen `COPY_SRC` target and copying out of it. A WebGPU
canvas has no `preserveDrawingBuffer`: the swapchain texture expires at present, so
sampling the canvas afterwards yields an empty image. This is the most common
reason a WebGPU port of a working WebGL program "renders but the screenshot is
black". The readback target carries depth even though it is never read — a pass
with no depth attachment and a pipeline with `depthStencil` is a validation error,
the whole pass is discarded including the clear, and the symptom is a uniformly
black readback with no error anywhere.

---

## Dirty propagation in the scene graph

The design this replaces turns on one line. The conventional
`updateMatrixWorld(force)`:

```ts
updateMatrixWorld(force) {
  if (this.matrixAutoUpdate) this.updateMatrix();   // every frame
  this.matrixWorldNeedsUpdate = true;              // and always dirty
  if (this.matrixWorldNeedsUpdate || force) {
    force = true;                                  // <-- unconditional
    for (const c of this.children) c.updateMatrixWorld(force);
  }
}
```

`force = true` says "if I recomputed, everything below must recompute too", on the
assumption that a parent can only become dirty by moving. Wrong in the case that
matters: the scene root is always dirty, so the whole graph is traversed and every
matrix rewritten every frame, including the nine hundred static nodes nobody will
look at twice. A scene that has not changed costs exactly what a scene where
everything moves costs.

A world matrix is a pure function of the local transform and the parent's world
matrix, so it only needs rewriting when one of those changed. apse tracks **two
independent signals**, and the second is the whole trick:

- **`localVersion`** — did *this* node's own transform change? Bumped by this
  node's setters and nothing else. Decides whether to **write**.
- **`subtreeDirty`** — has anything at or below changed? Set on the node and every
  ancestor by every change below. Decides whether to **visit**.

```
visit this node?  subtreeDirty || parentChanged
write its world?  ownChanged || parentChanged
    ownChanged    = localVersion !== localVersionWritten
    parentChanged = parentWorldVersion !== parent.worldVersion
```

Collapsing the two into one flag — the obvious first cut, and what happens when
`localVersion` is bumped on every ancestor — forces a parent to be rewritten when a
child moves, and a rewritten parent invalidates *its* children. The distinction
between "this matrix is stale" and "somewhere below me is stale" is the entire
difference between O(depth) and O(n).

Measured in `test/scene.test.ts` by counting real writes and visits into real
`Float32Array`s, not by timing:

| scene | frame 1 | frame 2, static | frame 2, one thing moved |
|---|---:|---:|---:|
| 1000 nodes, all static | 1000 writes / 1000 visits | **0 writes, 1 visit** | — |
| 5-node chain, leaf moves 1 | — | — | **1 write, 5 visits** |
| 1000 nodes, leaf 5 deep moves 1 | 1000 / 1000 | — | **1 write**, 1000 visits |
| 103 nodes: a 2-deep branch + 100 static siblings, branch moves 1 | — | — | 2 writes, 103 visits |

The write count is the headline: one moving leaf costs **one** matrix write whether
the scene holds 5 nodes or 1,000, and the ancestors' `worldVersion` tokens are
unchanged — the test asserts that, which is the proof they were not rewritten. The
visit count is the honest companion: in the 1000-node case the traversal still
*examines* the 995 static siblings, each pruned on a single flag read. That is the
difference from the design above, where examining one would have cost a 4×4
multiply.

Three supporting decisions:

- **Float equality is never used.** Version-based, because comparing sixteen floats
  per node per frame to discover nothing moved costs as much as writing the matrix,
  and it cannot distinguish "moved and came back" from "never moved" — a transform
  animated out and back would be recorded as static and the two would then disagree
  about the next frame. The setters *are* the tracking mechanism: they mutate the
  exposed typed arrays **in place** — the reference never changes, because the
  uniform packer holds it — and bump the version. Writing into `local` directly is
  supported and needs a matching `markDirty()`; that is the one rule the module asks
  you to remember, and it is the same rule WebGPU imposes on `writeBuffer` inputs.
- **No per-object transform class.** A node's transform is a `Float32Array(16)`.
  For comparison, three.js `Object3D` carries 31 own properties
  (`getOwnPropertyNames`; 28 enumerable) and a `Mesh` shell 37 (34 enumerable). apse
  has no per-object transform object at all; a `MeshNode` adds only `mesh`,
  `material`, `castShadow`, `order`, `instanceCount`, `firstInstance`.
- **No Euler angles anywhere.** Rotation is a quaternion only, `[x, y, z, w]`, laid
  out as WGSL lays out a `vec4<f32>`. Euler triples are not closed under
  composition, do not interpolate along the shortest arc, and a bidirectional
  `rotation` property costs a trigonometry-heavy conversion per write plus a getter
  and setter closure per object. The one conversion that exists,
  `quat.fromEulerXYZ`, is applied at authoring time; nothing stores the angles.

**Culling is a separate pass and is not optimised away.** The transform pass is O(1)
on a static scene. The cull pass is O(nodes) on *any* scene, and no version
bookkeeping can change that: if the camera moved, the answer to "what is on screen"
changed, and answering it means asking about each object. Caching that needs a
static camera. What the cull pass does *not* do is recompute a matrix — it reads
world data the transform pass already brought up to date, so a static scene with a
moving camera costs 1000 sphere tests and zero matrix multiplies. The design above
would have cost 1000 of each.

Draw items are pooled, and that is not an optimisation detail: at 60 Hz with 5,000
visible objects, fresh `DrawItem`s are 300,000 short-lived objects per second, and
the collector's response is to stop the world for a few milliseconds at the worst
possible moment. The pool is indexed, not a stack — item *k* of this frame is the
same object as item *k* of last frame whenever count and order are stable, which
keeps identity comparison meaningful and makes it testable.

Bounding spheres are computed at `MeshData` construction, not in the cull test. The
cull loop is the hottest thing in a renderer — it runs for every object every frame
including the culled ones — and it reads a value that cannot change while the
vertex buffer is static; computing it there would pay 60 times a second, and again
on every re-upload. Two passes: AABB centre, then farthest vertex from it. The
centre need not be the centroid; any point inside the convex hull gives a containing
sphere, and the AABB centre costs six comparisons per vertex instead of a divide.

---

## Resource ownership

`Resource` reference-counts. This is not decoration; it removes the two mistakes that
leak GPU memory most often:

- **Sharing is safe.** Each user takes a reference with `ref()`.
- **Dropping a reference frees at zero.** `unref()` decrements, disposes at zero.
- **`dispose()` forces**, regardless of count. It is the escape hatch for "free this
  now", and what teardown uses.

GC is never relied on: browser GPU object lifetimes are not deterministic, and a
dropped wrapper can keep a multi-megabyte buffer alive indefinitely.

The contrast is specific. three.js requires manual `.dispose()` on every geometry,
material, texture, and target, and **`material.dispose()` does not free its
textures** — a shared texture stays resident after every material using it is gone.
In apse, dropping a mesh releases the references it holds on its geometry and
material, and GPU memory frees when the last holder lets go.

`ResourceScope` owns an ordered set and releases it in reverse acquisition order,
catching per-child failures so one broken child cannot strand its siblings. It is
how a mesh releases atomically: no partial state where the buffers are gone and the
pipeline is not.

**What is deliberately not freed.** `Material.onDispose` destroys only the
material's own uniform buffer. The render pipeline is *not* destroyed, because it is
shared with every other material of the same `pipelineKey`, and destroying it would
silently break those. The cost is that a disposed material's pipeline stays resident
until the device is destroyed — bounded by the number of *distinct* materials in the
application, which is the entire reason the cache exists. The texture bind group is
not destroyed and could not be: `GPUBindGroup` has no `destroy()`. Same reasoning
across the device cache: `disposeSharedUniforms` frees the two uniform buffers and
leaves layouts, pipelines, and samplers, none of which has a `destroy()`.

---

## The error contract

Every error carries five things: `code` (a stable slug — branch on this, never on
`message`), `message` (what you asked for and what was there, values interpolated),
`why` (the technical rule violated), `fix` (the single corrective action), and
`link` (the docs anchor for that code). All five are on every error; no code path
produces a bare `Error`. `toString()` is `CODE: message — fix: <fix>`; `toJSON()`
is the structured form. `isAseError` also matches on `name`, so it survives two
copies of the library in one graph.

**The catalog is a discriminated union and the type enforces completeness.** There
are **42** codes. `ErrorCodeEntry` declares one member per code with the signature
of its message function, and `ERROR_CATALOG` is typed
`{ [C in AseErrorCode]: ErrorGuidance }` — adding a code without its `why` and `fix`
is a compile error. `fail<C>(code, message, opts)` is likewise discriminated, so
passing params the code's template does not want is a type error rather than a
runtime `undefined`. The consequence worth stating: **the set of things that can go
wrong is a finite, enumerable, typed value**, so a caller or an agent can read
`ERROR_CODES` instead of the source to find the failure surface. `ErrorDetail` adds
a structured payload (`{ kind: 'numeric', field, value, min, max }` and friends) so
a handler can branch on the value rather than parse the message.

**Programmer error throws; capability failure returns.** *A wrong argument throws, a
missing capability returns.* `requestDevice` rejecting because the adapter lacks a
limit you asked for is a bug in your code. A loader failing to find a file, or a
probe discovering the device cannot do something, is a fact about the world. Making
both the same shape teaches callers to wrap everything in `try/catch`, which is
strictly worse than not handling either — a handler that catches everything cannot
tell a typo from a missing file. So they are different types and the compiler says
which before anything runs. `isOk`/`isErr` narrow, `unwrap`/`unwrapOr` are the
deliberate escape hatches, and `attempt` turns a throwing callback into a `Result`
for the places where an error must become a value — a loop over many assets where
one failure should not abandon the rest.

`validateBodyIdentifiers` and `validateGeneratedWGSL` are the agent-facing half of
this contract. Between them, the class of failure an agent is most likely to
produce — a hallucinated identifier in a shader body — becomes a typed error naming
the field and listing the alternatives, before any compiler is involved.

---

## Compatibility mode is the default target

WebGPU is not Baseline. Firefox on Linux is Nightly-only, Firefox on Android is
behind a flag, Firefox on Intel Macs is unsupported, Safari on iOS below A12 is
unsupported. The dominant mobile cohort is GLES 3.1-class hardware, so `device.ts`
asks for `featureLevel: 'compatibility'` first and reaches for `'core'` only when
the adapter advertises `core-features-and-limits` and `preferCore` is set.

`compatRequiredLimits()` sends the compatibility defaults **verbatim**, which is
safe by construction: a device is created with the defaults for its feature level,
and each `requiredLimits` entry raises a limit to `max(requested, default)`.
Requesting something *worse* than the default is legal and has no effect. So the
request yields compatibility defaults on a compatibility device, core defaults on a
core device, and device creation can never fail because a phone lacks something apse
only wanted.

Two traps, both tested. **`requiredLimits` rejects unknown keys outright**, so a
limit added to the spec next year is not safe to send today —
`maxBindGroupsPlusVertexBuffers` and `maxImmediateSize` are recent additions and are
deliberately absent. And **`{ ...adapter.limits }` copies nothing**:
`GPUSupportedLimits` is an interface of *prototype getters*, so the spread is `{}` —
not "all the limits", nothing. `copyLimits` copies field by field, one read per line,
and is tested against a fake whose own-property count is zero.

The limits that differ, and what each costs:

| limit | compat | core | consequence in apse |
|---|---:|---:|---|
| `maxTextureDimension2D` | 4096 | 8192 | a 5120×2880 canvas fails on a phone, not a laptop |
| `maxUniformBufferBindingSize` | 16 KiB | 64 KiB | `UNIFORM_BLOCK_OVERFLOW` well before core would |
| `maxColorAttachments` | 4 | 8 | the scaffold supports exactly one target |
| `maxInterStageShaderVariables` | 15 | 16 | **one fewer user varying**; `@builtin(position)` takes the other |
| `maxComputeInvocationsPerWorkgroup` | 128 | 256 | no compute path exists yet |
| `maxComputeWorkgroupSizeX/Y` | 128 | 256 | ditto |
| `minUniformBufferOffsetAlignment` | 256 | 256 | the object uniform stride |
| `maxStorageBuffersInVertexStage` | **0** | 8 | see below |

**No storage buffers in the vertex stage.** Compatibility mode defaults
`maxStorageBuffersInVertexStage` and `maxStorageTexturesInVertexStage` to zero,
because GLES 3.1 allows `MAX_VERTEX_SHADER_STORAGE_BLOCKS` to be zero and a
significant share of devices ship with that. A vertex shader reading a storage
buffer therefore **compiles on a laptop and fails on a phone** — a validation error
at `createBindGroupLayout` or `createPipelineLayout`, with the shader looking fine.
apse reads these limits from the *device* and never requests them; they are in
`COMPAT_VERTEX_STAGE_LIMITS` because they are the limits most likely to turn a
working shader into a broken one.

**Mip generation must be view-dimension agnostic.** Compatibility mode fixes the
binding view dimension at texture *creation* via `textureBindingViewDimension`, and
one draw may not bind two views of the same texture differing in `aspect`,
`baseMipLevel`, `mipLevelCount`, or `swizzle` — one parameter set per texture,
because GLES 3.1 has no texture views. A mip chain generated per view dimension
violates that. apse generates no mipmaps today, so this is a constraint on future
work rather than a current bug; it is recorded because the obvious implementation
breaks it.

Optional features are **detected, never required**: `timestamp-query` is on roughly
44% of devices and quantised to 100 µs, `subgroups` is Chromium-only, and requiring
either would make device creation fail on exactly the devices that lack them.

---

## Known limitations and what is not built yet

Everything here is absent, not merely rough. This is the section that stops someone
building on a capability that does not exist.

**No instancing, so every object is a draw call.** `DrawableGeometry` carries
`instanceCount` and `firstInstance`, `GpuMesh` accepts them, and `renderer.ts` passes
them to `draw`/`drawIndexed` — but nothing constructs an instanced draw and no
material reads a per-instance attribute. A 5,000-object scene is 5,000 draw calls.
`ObjectData` has an `instanceId` field and `firstInstance` is threaded through the
whole pipeline; what is missing is the vertex-side buffer and a `stepMode:
'instance'` layout.

**No batching or merging.** No atlas, no draw-call merging, no sorting beyond the
material grouping in step 3. Draw count equals visible object count.

**No post-processing.** No tone map, no bloom, no FXAA, no grading. `frame.exposure`
is in the block and the PBR material deliberately does *not* apply it, assuming a
present pass will — and that pass does not exist. Building one means rendering to a
`createColorTarget`, writing a material that samples it, and presenting it;
`createColorTarget` exists to make that possible.

**Shadows are not wired up in the default PBR path.** `pbrMaterial({ shadows: true
})` is real — it declares a `depth-2d` slot with a comparison sampler, adds a 3×3
PCF loop, and needs `mat.lightViewProj` set to the light's orthographic matrix. But
**apse has no shadow pass.** You must render the depth map yourself into a
`createDepthTarget`, then `material.setTexture('shadowDepth', view)` and
`material.setSlot('lightViewProj', m)`. Turning the option on without both gives a
material whose shadow term samples nothing. `MeshNode.castShadow` exists and
defaults to `true`, but no code reads it.

**No animation system.** No keyframes, skinning, morph targets, or blend trees. The
`renderer.start()` callback is the whole story, and it hands you `dt` already
clamped. Bounds are static for the same reason: a skinned or morphed mesh is the
case that would genuinely need dynamic bounds, and it will be an explicit
`updateBounds()` when morph targets exist.

**No loaders.** No glTF, no OBJ, no texture decoders, no environment maps, no I/O.
You bring `Float32Array`s or a `GPUTextureView` you made. `MeshData` takes either
interleaved data or one dense array per attribute — that is the whole import surface
for geometry.

**WebGPU only, no WebGL2 fallback.** `createDevice` fails with `WEBGPU_UNAVAILABLE` if
`navigator.gpu` is undefined, which includes every non-secure origin. The
compatibility *profile* is supported; a compatibility *fallback to WebGL* is not, and
never will be — the scaffold generates WGSL, and the alternative would be a second
shader language.

**GPU timing requires `timestamp-query`, feature-detected and often absent.**
`AseDevice.hasTimestampQuery` reports it. `FrameStats.gpu` is `0` on every frame
regardless: the query buffers are not implemented and `averageGpu` is the mean of
zeros. Treat the GPU column as unimplemented, not as a measurement of zero.
`timestamp-query` values quantise to 100 µs, so it is coarse even where it works.

**MSAA needs an offscreen target, because a canvas texture cannot be multisampled.**
`getCurrentTexture()` always returns a single-sample attachment, so
`createCanvasTarget` refuses `sampleCount: 4` rather than ignoring it. A 4× path
allocates two colour textures — the multisampled one you render into, and the
single-sample one the resolve writes into — and only the second is sampleable by a
later pass, because binding a multisampled texture as a sampled texture is a
validation error, not a blurry result. `createCanvasTarget` hard-codes `sampleCount:
1` and `render()`'s default target is the canvas, so **MSAA is unreachable from the
default path**; the plumbing is in `#encode` (it reads `target.sampleCount` and sets
`resolveTarget`), but you must pass an offscreen target and present it yourself.

**One colour attachment.** The scaffold rejects more than one `targets` entry,
because the generated fragment entry point returns a single `@location(0) vec4f`. A
second channel goes in a second pass that reads the first target as a texture.

**No compute.** The compute limits in `COMPAT_LIMITS` are there because the table is
a complete statement of apse's floor, not because anything uses them. There is no
compute pipeline, no storage buffer, no workgroup API.

**Format portability is not automatic.** `depth24plus` is the default because it is
renderable as a depth-stencil attachment everywhere; `depth32float` gives full range
but is not universally renderable. A material's colour format is baked into its
pipeline at creation, so it is permanently bound to that format — and the canvas
format is `bgra8unorm` on desktop, `rgba8unorm` on Android, so a hardcoded default is
wrong on one of them. Pass `targetFormat` explicitly for an offscreen target.

**Draw items need a real material.** `collectDrawItems` reads `node.material.phase`
and `node.material.renderPipeline`, so a test double must supply those fields.

---

## Performance model

### What is measured

`bench/run.ts` serves `bench/` on `http://localhost:8787`, drives real headless
Chrome, and runs **both engines in the same process on the same device**. The three.js
baseline is `WebGLRenderer`, because three.js's own documentation calls that the
recommended path for WebGL2 and labels `WebGPURenderer` WIP upstream; comparing
against a path its authors describe as unreleased would be a straw man.

The harness is not a microbenchmark. It guards against a specific failure: a benchmark
measuring a renderer that is quietly drawing nothing reports excellent numbers. So it
collects every `GPUUncapturedErrorEvent` and **fails the run** if any fired (a
validation error invalidates a pass without throwing), reads the framebuffer back and
**fails if the frame is entirely background-coloured**, and writes a screenshot.
`bun run bench` is the only test here that proves pixels came out the right colour.

### The numbers

Apple M-series GPU, headless Chrome 153, 1280×720, CPU milliseconds per frame, median
of 60 samples, same process and device. Objects are laid out in a grid, not at the
origin, so nothing is trivially culled.

| scene | apse | three.js | ratio |
|---|---:|---:|---:|
| 1000 cubes | 0.70 | 1.00 | 1.4× |
| 2000 cubes | 1.30 | 2.00 | 1.4× |
| 4000 cubes | 2.60 | 3.90 | 1.5× |
| 5000 cubes | 3.90 | 4.70 | 1.2× |
| 1000 spheres (720k tris) | 0.70 | 1.30 | 1.9× |

JS heap: apse 10.4–42.5 MB against three.js 42.0–61.3 MB for the same scenes.

**What these numbers do and do not show.** They are 1.1–1.9×, not 5×, and the reason
is in the draw counts: every scene above is one draw call per object, 1,000 to 5,000
of them. At that scale both libraries are bottlenecked by per-draw WebGPU/WebGL API
cost, not by scene-graph or cache overhead, so the savings from a static transform
pass or a shared uniform buffer are a small fraction of the frame. The 1,000-sphere
row is the most favourable because per-draw cost is spread over 720 triangles each.
Do not extrapolate this table to a scene apse is structurally better at, and do not
quote it as 5×.

The honest summary: **apse wins on bundle size and memory by a wide margin, wins
modestly on CPU frame time, and its advantage should grow where the per-draw cost is
amortised — which is instancing and batching, neither implemented.**

### Bundle size

This is the wide-margin result, and it is enforced rather than asserted.

| entry | gzip | budget |
|---|---:|---:|
| `index` (all of it, incl. every chunk) | 55.3 KB | 60 |
| tree-shaken app | 43.5 KB | 48 |
| `render` | 38.9 KB | 40 |
| `material` | 21.6 KB | 24 |
| `scene` | 10.4 KB | 13 |
| `geometry` | 9.7 KB | 12 |
| `core` | 7.7 KB | 10 |
| `math` | 6.6 KB | 8 |

Two measurement details make these honest. Sizes are for the **full reachable module
graph** per entry, not the entry file: measuring `index.js` alone reports 2.5 KB for a
complete renderer, because code splitting moved the substance into chunks and the
entry became a re-export list — a consumer downloads the entry and everything it
transitively imports. And compression is measured on the concatenation, because a
bundler serving all chunks in one response gets one shared dictionary and per-file
numbers overstate the cost.

The "tree-shaken app" row is the headline and the only measurement that tests the
claim the library actually makes. `bench/tree-shake.ts` is a realistic minimal app —
renderer, camera, one lit mesh, one animation loop — bundled, minified, and compared
against a hard ceiling. `sideEffects: false` is easy to write and easy to violate; a
bundler that silently keeps 300 KB of dead code reports no error, so the only defence
is a number in CI that fails when the claim stops being true.

`scripts/size-gate.ts` is the second half of that defence, and it exists because the
build can check its own arithmetic but not its own claims. It re-reads the report and
fails if the tree-shaking claim is false — that the bundle actually dropped the unused
primitives, checked by looking for their markers, with a positive control beside it
because a check that cannot fail is not a check. It also verifies every `exports`
subpath and every `typesVersions` target has a built file, which is otherwise
invisible until somebody runs `npm install`.

For comparison, three.js at 0.186.1 tree-shaken for an equivalent PBR scene —
`WebGLRenderer`, `MeshStandardMaterial`, `BoxGeometry`, `Scene`, `Mesh`,
`PerspectiveCamera`, `AmbientLight`, `DirectionalLight` — measures **131.5 KB
gzip** (524 KB raw) under the same bundler, recorded as the **133 KB** baseline
constant in `scripts/size-gate.ts`, against apse's 43.5 KB for that scene plus a
PBR material, a scene graph, and a frame loop.

### What would move the numbers

Roughly in order of leverage:

1. **Instancing and batching.** The largest available win, and the reason the CPU gap
   is only 1.4×. The plumbing is in place — `instanceCount`, `firstInstance`,
   `obj.instanceId`, the pass-through to `drawIndexed` — so this is a vertex-side
   buffer plus a `stepMode: 'instance'` layout, not a redesign.
2. **Culling beyond the frustum.** The current pass is a sphere test against six
   planes on every drawable node every frame. A BVH or a screen-space tile structure
   would turn the sphere scene's 1000 tests into a few hundred.
3. **A cheaper sort.** `Array.sort` over 5,000 items every frame with a five-key
   comparator. A pre-sorted array plus a dirty range, or a counting sort on
   `(material, order)`, would help; the depth key has to be recomputed regardless,
   since a real camera moves.
4. **Pre-compiling pipelines.** `Material.create` is already async — compilation is
   two to five seconds cold and must not block the main thread — but nothing warms
   the cache ahead of time. Building the materials a scene will need before it is
   first shown removes the hitch.
5. **Nothing in the frame loop allocates**, and that is worth keeping. Scratch buffers
   in `renderer.ts`, `node.ts`, and `graph.ts` are module-level and reused, walk
   stacks are pooled and re-entrant, draw items are pooled by index. Any new per-frame
   allocation is a regression the young generation hides on average and punishes at
   the worst moment.
