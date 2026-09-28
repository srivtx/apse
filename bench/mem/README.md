# Per-object heap cost: apse vs three.js

Answers one question: how many bytes does one scene-graph object cost in apse and in
three.js, and is the claim *"three.js's `Object3D` is a class instance with 31 own
properties and an `Object3D` shell measured around 1.8 KB, so three.js costs far more
memory per node"* true?

**The property count is true. The 1.8 KB is false.**

- `Object3D` really does have **31 own properties**. (A `Mesh` has 37.)
- An `Object3D` shell costs **136 bytes**, not 1,804. That claim is **13× too high**.
- A whole `Object3D` — the shell plus every object only it owns — costs
  **1,216 bytes**, still **1.5× smaller** than the claimed 1,804.
- The previous attempt's Bun figure of **450 B was not reproduced by any method here**,
  and the reason it is not reproducible is demonstrable: `process.memoryUsage().heapUsed`
  cannot measure a per-object cost in Bun. Details in *The 450* below.

The corrected comparison is **1,060 B/node (apse) vs 1,240 B/node (three.js)** — a
**1.17×** difference, not a multiple.

## Method

Chrome **153.0.8010.53**, headless, macOS arm64. Real Chrome, not a
puppeteer-managed download.

- **CDP `HeapProfiler`, through `page.createCDPSession()`.** `HeapProfiler.enable`,
  `HeapProfiler.collectGarbage` **twice** (300 ms apart) before every snapshot, then
  `HeapProfiler.takeHeapSnapshot({reportProgress:false, captureNumericValue:false,
  treatGlobalObjectsAsRoots:true})`. The snapshot JSON is parsed and `self_size` is
  summed per `(type, name)`. The 50,000-object snapshots are 85–204 MB of JSON.
- **`performance.memory` is not load-bearing.** It is whole-isolate and quantised, so a
  few-MB difference is inside its noise and a 1.5× correction would be invisible to it.
  It is recorded in the run output as a cross-check only.
- **One library per page, one page per data point.** Two libraries in one page
  contaminate every number, and reusing a page across object counts leaves the previous
  scene reachable from module scope. Every point is a fresh `browser.newPage()`.
- **Slope, not difference.** Each configuration is measured at **0, 10,000 and 50,000**
  objects. The headline is `(total(50k) − total(10k)) / 40,000`. Everything that does
  not scale with the object count — module code, the WebGPU device, the WebGL context,
  the snapshot's own fixed cost — cancels out of it. The 0→50k slope is printed beside
  the 10k→50k slope and **agrees to the byte** in all nine configurations measured, so
  the cost is linear in the object count over this range.
- **The per-class slopes sum to the total slope** (1,060 = 1,060; 1,240 = 1,240;
  1,388 ≈ 1,389). The decomposition is complete, not a top-N sample.
- **Two independent full runs produced byte-identical slopes** — 1,036 / 1,060 / 1,060 /
  1,160 / 1,216 / 1,240 / 1,389 in both. The absolute totals move by a few tens of KB
  between runs (GC timing, V8 internals); the slope does not, which is the whole reason
  for reporting a slope.

**This reports self size, not retained size.** Self size is the bytes an object occupies
at its own address, counted once. Retained size is everything an object keeps alive,
which for a node in an array is the next node in the array, so retained size is
meaningless as a per-object figure here. The slope is the fair comparison. The
per-class self sizes in §6 are supporting detail: each column sums to the total slope,
so each engine's cost can be accounted for in full, but no individual row is comparable
across engines on its own — a 16-byte `Matrix4` shell says nothing about what owns it.

`bun run bench/mem/run.ts`. Needs `dist/` built for the `dist` rows (`bun run build`).

## 1. Bytes per scene-graph object

Slope of total snapshot `self_size` between 10,000 and 50,000 objects. Raw totals in
bytes.

| configuration | 0 objects | 10,000 | 50,000 | **B/object** | 0→50k B/obj |
|---|---:|---:|---:|---:|---:|
| apse `Node` (bare, src) | 2,164,495 | 12,673,795 | 54,115,655 | **1,036** | 1,036 |
| apse `MeshNode` (stub geometry, src) | 2,165,375 | 12,914,587 | 55,322,299 | **1,060** | 1,060 |
| apse `MeshNode` (real `GpuMesh` + PBR material, src) | 2,505,927 | 13,259,771 | 55,664,191 | **1,060** | 1,060 |
| apse `MeshNode` (**dist**, minified bundle) | 2,161,758 | 12,914,726 | 55,317,386 | **1,060** | 1,060 |
| three `Object3D` (bare) | 5,731,189 | 17,974,337 | 66,611,889 | **1,216** | 1,216 |
| three `Mesh` (shared geometry + material) | 5,731,189 | 18,220,773 | 67,818,217 | **1,240** | 1,240 |

**Headline: apse 1,060 B/node, three.js 1,240 B/node, ratio 1.17×.**

Two controls that make the number trustworthy:

- **`src` and `dist` are identical** (1,060 both). The per-object cost is a property of
  the library, not of how it was built. `src` is measured because `dist` is minified
  and every class in a snapshot of it is called `g`, so the `src` mode — the real
  TypeScript, transpiled by the server as it is served — is what makes the per-class
  breakdown readable.
- **A stub geometry and a real uploaded `GpuMesh` are identical** (1,060 both). A mesh
  and a material are one pointer each and both are shared, so they cannot affect a
  per-node cost. They are measured to prove that, not to assume it.

Live heap objects per scene-graph object, from the same snapshots' `node_count`:

| | heap objects per node |
|---|---:|
| apse `MeshNode` | **21.0** |
| three `Mesh` | **57.0** |

three.js allocates **2.7× as many live heap objects** per node. That is the real
structural difference, and the byte totals are closer than it looks because apse's
objects are large individually (six typed arrays) and three.js's are numerous and small.

## 2. Bytes per drawable object

Measured, not inferred: a bare node and a drawable node of the same class, same page
harness, same snapshot method.

| | bare node | drawable node | **cost of being drawable** |
|---|---:|---:|---:|
| apse `Node` → `MeshNode` | 1,036 | 1,060 | **+24 B** |
| three `Object3D` → `Mesh` | 1,216 | 1,240 | **+24 B** |

Identical, and small: six extra own properties cost V8 24 bytes of object growth in
both. A shared geometry and a shared material are two pointers either way.

What **is** per-drawable is the render path, and here the two differ:

| | scene graph only | with render path | render path adds |
|---|---:|---:|---:|
| apse, `DrawItem` pool populated | 1,060 | **1,160** | +100 B |
| three.js, one `WebGLRenderer.render()` | 1,240 | **1,389** | +149 B |

apse's +100 B is a pooled `PooledDrawItem` (**60 B/object**, 12 fields) plus ~40 B of
sorting and offset state. three.js's +149 B is a live render list plus two plain
`Object`s of per-object WebGL state per node, which is 1.5× apse's render-path cost for
the same frame.

## 3. Property counts and per-instance closures

`getOwnPropertyNames(o).length` for one live instance. Per-instance closures are
**not** a property count: two instances of the same class are built and every function
reachable within three own-property hops is collected from each, then diffed **by
function identity**. Anything present in both is a prototype method, a module constant
or part of a shared object, and is not per-instance cost however many properties it
occupies.

| class | own props | of which are functions | **per-instance closures** |
|---|---:|---:|---|
| apse `Node` | 16 | 0 | **0** |
| apse `MeshNode` | 22 | 0 | **0** |
| apse `Scene` | 4 | 0 | **0** |
| three `Object3D` | 31 | 0 | **2** |
| three `Mesh` | 37 | 0 | **2** |
| three `Scene` | 41 | 0 | **2** |
| three `Vector3` / `Euler` / `Quaternion` / `Matrix4` / `Matrix3` / `Layers` | 3 / 5 / 5 / 1 / 1 / 1 | 0 | 0 |

**The closure count is the interesting part of the claim, and it is real.** three.js
0.186.1's `Object3D` constructor declares two functions *inside* the constructor body
(`src/core/Object3D.js:145` and `:151`) so that `Euler` and `Quaternion` can be
two-way synced, and hands them to `rotation._onChange(...)` /
`quaternion._onChange(...)`. Both are per-instance garbage that a property count does
not reveal:

| | count/obj | B/obj |
|---|---:|---:|
| `closure:onRotationChange` | 1.00 | 32 |
| `closure:onQuaternionChange` | 1.00 | 32 |
| `object:system / Context / scope` (the shared closure scope) | 1.00 | 24 |
| **total per-instance closure cost** | | **88 B** |

Note what is **not** per-instance: `onBeforeRender`, `onAfterRender`, `raycast` and
friends are **prototype methods** in 0.186.1, not instance properties. The measured
"of which are functions" column is 0 for every three.js class, which settles it. (The
claim that they are instance properties is not correct for this version; check again
before repeating it for a version where it might be.)

**88 B of three.js's 1,216 B — 7% — is two arrow-free closures and the scope they
share**, allocated so that an object whose rotation is never touched still pays for the
mechanism that would keep `rotation` and `quaternion` in sync. apse has zero per-instance
closures anywhere in its scene graph: `#composeBasis()` and `#decompose()` are prototype
methods and the synchronisation is a call, not a listener.

Also not visible in a property count: apse's `Node` has 16 own properties **plus three
private fields** (`#worldVersion`, `#parent`, `#children`), which V8 stores outside the
property backing store. The 136 B `Object3D` shell and the 92 B `Node` shell are
therefore measured totals, private slots included, not property-table arithmetic.

## 4. Total isolate heap, 10,000 objects, each library in its own page

| | 10,000 objects, whole isolate | of which the scene | 50,000, whole isolate | of which the scene | ratio (three/apse) |
|---|---:|---:|---:|---:|---:|
| apse, real mesh | 12.65 MB | 10.75 MB | 53.09 MB | 50.70 MB | — |
| **three.js, `Mesh`** | **17.38 MB** | 12.49 MB | 64.68 MB | 59.21 MB | **1.16×** |
| apse, `+pool` | 13.62 MB | 11.78 MB | 57.89 MB | 55.50 MB | — |
| three.js, `+WebGL` | 19.59 MB | 14.33 MB | 72.57 MB | 66.64 MB | 1.20× |

**Whole isolate at 10k objects: 12.65 MB (apse) vs 17.38 MB (three.js) = 1.37×.** That
ratio is *not* the per-object ratio, and the difference is the module baseline: 2.39 MB
for apse's page against 5.47 MB for three.js's, because `three.module.js` is one 648 KB
file of source. Subtract the 0-object control and the scene-attributable ratio is
**1.16×**.

Quoting 1.37× as a memory result would be quoting three.js's bundle size with extra
steps. The 1.16× is the number that describes the scene.

## 5. Verdict on the claim

**False on the memory figure, true on the property count.**

| claim | measured | verdict |
|---|---|---|
| `Object3D` has 31 own properties | 31 | **true** |
| an `Object3D` shell is ~1,804 B | **136 B** | **false, 13× overstated** |
| therefore three.js costs far more per node | 1,216 vs apse's 1,036 (node) / 1,240 vs 1,060 (drawable) | **overstated — 1.17×, not a multiple** |
| — | three.js allocates 57 live heap objects per node vs apse's 21 | **true, and the more interesting fact** |

**The correction is large and should be stated unambiguously: 1,804 B is wrong. The
honest figure for one `Object3D` in a scene is 1,216 B, and the shell on its own is
136 B.** A 13× error on the shell and a 1.5× error on the total. The corrected ratio is
1.17×, which is a real difference and worth a row in a table, but it is not the
difference the 1.8 KB figure implies, and it should not be presented as one.

The previous attempt's 450 B was not a measurement of an `Object3D` at all. See below.

## 6. What changes the slope

Per-object self size by class, from the same snapshots. These are **self** sizes: each
object counted once, at its own address. They are not retained sizes, and they are not
comparable across engines on their own — a `Matrix4` shell being 16 B says nothing
about what owns it. What the table is good for is arithmetic: each column sums to the
total slope, so each engine's cost can be accounted for in full.

**apse `MeshNode` — 1,060 B/node**

| class | count/obj | B/node | B/obj |
|---|---:|---:|---:|
| `object:Float32Array` | 6.00 | 60 | 360.0 |
| `object:ArrayBuffer` | 6.00 | 52 | 312.0 |
| `array:(object elements)` (the typed-array backing stores) | 6.00 | 39.3 | 236.1 |
| `object:MeshNode` — **the shell** | 1.00 | 116 | 116.0 |
| `string:*` — the node's `name` | 1.00 | 19.2 | 20.0 |
| `object:Array` — the `#children` array | 1.00 | 16 | 16.0 |
| **total** | | | **1,060.1** |

**three `Mesh` — 1,240 B/node**

| class | count/obj | B/node | B/obj |
|---|---:|---:|---:|
| `array:(object elements)` — backing stores for the matrix `elements` arrays | 3.00 | 138.6 | 416.1 |
| `number:*` — boxed doubles | 16.00 | 11.2 | 180.0 |
| `object:Mesh` — **the shell** | 1.00 | 160 | 160.0 |
| `object:Array` — `children`, `animations`, 4 × `elements` | 6.00 | 16 | 96.0 |
| `object:Vector3` — `position`, `scale`, `up` | 3.00 | 24 | 72.0 |
| `string:*` — mostly the 36-char `uuid` | 17.00 | 3.1 | 48.0 |
| `object:Matrix4` — `matrix`, `matrixWorld`, **`modelViewMatrix`** | 3.00 | 16 | 48.0 |
| `object:Euler` | 1.00 | 36 | 36.0 |
| `object:Quaternion` | 1.00 | 36 | 36.0 |
| `closure:onQuaternionChange` | 1.00 | 32 | 32.0 |
| `closure:onRotationChange` | 1.00 | 32 | 32.0 |
| `object:Object` — the empty `userData` | 1.00 | 28 | 28.0 |
| `object:system / Context / scope` — the closures' shared scope | 1.00 | 24 | 24.0 |
| `object:Matrix3` — **`normalMatrix`** | 1.00 | 16 | 16.0 |
| `object:Layers` | 1.00 | 16 | 16.0 |
| **total** | | | **1,240.1** |

One honest limit on the largest row: the snapshot reports the `(object elements)` bucket
as **3 backing-store nodes totalling 416.1 B per `Mesh`**, while the `object:Array` row
independently counts 6 `JSArray`s per `Mesh` — `children`, `animations` and the four
matrix `elements`. `children` and `animations` are empty and share V8's canonical empty
store, and the four `elements` stores do not all appear as separate nodes, so the
bucket cannot be attributed to individual matrices. Read it as **"416 B per `Mesh` in
array backing stores"**, which is the largest single line item in three.js's node at
34% of its total, and not as a per-matrix figure.

**The things that actually move the number:**

1. **apse's cost is six `Float32Array`s, and only those.** 908 of its 1,060 B — 86% — is
   `local`, `world`, `position`, `rotation`, `scale`, `worldPosition`. A calibration row
   in the harness measures a bare `Float32Array(16)` in isolation at **188 B** (60
   wrapper + 52 `ArrayBuffer` + 76 elements store, of which 64 is the payload), and six
   arrays of lengths 16/16/3/4/3/3 model to 900 B against the 908 B measured — 0.9% out.
   So apse's per-node cost is almost exactly *"six typed arrays, plus a 116-byte
   object"*. A `Float32Array` is not free in V8: each one is three separate heap
   objects. If apse ever wants a smaller node, this — not the property count — is the
   lever.
2. **three.js's cost is matrices, and two of them are the renderer's, not the node's.**
   `matrix` and `matrixWorld` belong to the scene graph. **`modelViewMatrix` and
   `normalMatrix` do not** — they are per-object renderer scratch, allocated in the
   `Object3D` constructor and rewritten every frame, and they are the reason
   `Object3D` has 31 properties rather than 29. Their shells are 32 B of the 48 B in
   the `Matrix4`/`Matrix3` rows, plus a share of the 416 B of `elements` stores. They
   are real cost and worth naming, and they are the clearest thing in either library's
   node that is not really part of the node.
3. **three.js keeps a redundant rotation representation.** `Euler` + `Quaternion` per
   node (72 B) *plus* the two closures and the scope that keep them in sync (88 B) =
   **160 B, 13% of the node**, to hold the same four numbers apse holds once in a
   `Float32Array(4)`.
4. **apse's `DrawItem` pool is real but small and it does not churn.** 60 B per draw
   slot, allocated once and reused: the pool is indexed, item *k* of this frame is the
   same object as item *k* of last frame, and it never shrinks. It is the only
   per-drawable allocation either library makes, and it is two-fifths of what three.js's
   render path costs for the same frame.
5. **three.js's `Object3D` has an empty `userData` `{}` (28 B) and a 36-char `uuid`
   (48 B) per node**, 6% between them, before an application stores anything in
   `userData`.
6. **the `EventDispatcher` listener map is not a cost here.** `_listeners` is created
   lazily on the first `addEventListener`, so a plain `Object3D` has none. The claim
   that an `Object3D` "retains an `EventDispatcher` listener map" is not true of an
   unlistened-to object. (The retained `children` `Array` is real, but it is 16 B in
   three.js and 16 B in apse.)
7. **both libraries store matrices differently, and that choice is where the 1.17× comes
   from.** three.js's `Matrix4.elements` is a 16-element plain JS array, which V8 backs
   with a double-precision store; apse's `local` is a `Float32Array`, whose 16 elements
   cost 4 bytes each. On raw bytes per matrix apse pays *more* — the calibration row
   puts a `Float32Array(16)` at 188 B against a measured 138.6 B average for the three
   element-store nodes three.js keeps per `Mesh`, which is what a double-precision store
   at 8 bytes per element predicts — and still comes out ahead, because it allocates six
   small arrays where three.js allocates three `Matrix4` objects *and* a `Vector3`,
   `Euler`, `Quaternion`, `Matrix3` and `Layers` around them. The two effects nearly
   cancel: take apse's six typed arrays out of its 1,060 B and its node is **152 B**
   against three.js's 1,240.

### Two reading notes on the snapshot

- V8 reports `self_size` 0 for a large number of `string` and `number` nodes
  (internalised strings, Smis). The `string:*` and `number:*` rows therefore have
  per-node averages far below the 16-byte minimum object size. The **row totals** are
  correct and are what the slope uses; the per-node averages in those two rows are not
  meaningful and should not be quoted.
- V8 names a closure's scope by address (`Context / scope @36663`), and the address
  changes every run. Left in, it makes the same class look like two different classes
  across the two snapshots and the per-class slopes stop summing to the total. The
  harness strips the address; without that the per-class sum is 6 B high.

## The 450

The previous attempt reported 450 B for an `Object3D` and could not reproduce the
1,804 B. **450 is not reproducible here by any method, and the reason is that Bun's
`process.memoryUsage().heapUsed` cannot measure this.** Demonstrated, not asserted:

- Run as a one-shot per class, Bun's `heapUsed` delta is sane: **1,223 B/object** for a
  50,000-`Object3D` allocation, within 1.4% of the 1,216 B Chrome reports. So the
  *totals* corroborate.
- Run with the classes measured in sequence in one process, the same script returns an
  identical **~81 B/object for an apse `Node`, an apse `MeshNode`, a three `Object3D`
  and a three `Mesh`** — four classes that differ by 3× in real cost. A number that
  cannot tell a `Node` from a `Mesh` is not measuring the objects.
- At **0 objects**, the same script reports 1.9 MB of heap growth, all of it module
  loading. There is no zero-object control in that shape of measurement, so a 1.9 MB
  constant is silently divided by the object count.

Both failure modes are the same failure: a whole-isolate counter, no forced GC
protocol, no zero control. The Chrome snapshot method above has all three, and its
per-class slopes sum to its total, which is the check that catches exactly this.

**Do not carry the 1,804 forward. Carry 1,216 B per `Object3D` in a scene, 1,240 B per
`Mesh`, 1,060 B per apse `MeshNode` — and the 1.17× ratio with it.**

## Files

```
bench/mem/index.html   the in-page harness. One library per page, one page per data
                       point. Builds N objects, holds them, and reports the property
                       and closure structure of each class.
bench/mem/run.ts       the driver. Static server (+ an esbuild route that transpiles
                       /src/**/*.ts on the way out so class names survive into the
                       snapshot), puppeteer, CDP HeapProfiler, snapshot parser, slopes.
bench/mem/README.md    this file.
```

The run writes a full JSON report — every class, every count, at every object count —
to `$TMPDIR/apse-mem-report.json` and prints the path. Nothing is written into the repo.
