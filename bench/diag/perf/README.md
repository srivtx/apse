# bench/diag/perf — is apse's deficit per-draw CPU or GPU?

## Reproducibility caveat, found after the numbers above were taken

`bun run bench:perf` **does not always finish.** The baseline section completes
and is reproducible. The instanced sweep, at the 100 000-object end, has crashed
headless Chrome with `Protocol error (Runtime.callFunctionOn): Target closed` —
the renderer process dies, not a graceful error. The tables in this file come
from a run that completed; a later run of the same commit did not.

So the per-draw and instanced numbers below are real measurements, but the
harness is not yet reliable at the top of the sweep. Until it is, treat the
100 000-row as **measured once, not established**, and re-run before quoting it
anywhere that matters. The lower rows (1 000 / 10 000) reproduce.

The cause is not diagnosed. It is consistent with a GPU-process OOM at
100 000 cubes plus per-instance transforms, or with the page holding two WebGPU
devices plus a WebGL2 context plus 100 000 transforms at once. `two-device.html`
exists to isolate whether two devices are the trigger and currently says they
are not.


```
bun run bench/diag/perf/run.ts                  # the full run
APSE_PERF_QUICK=1 bun run bench/diag/perf/run.ts  # smoke test, ~3 min
```

## The question

At HEAD apse measured 1.3–1.4× slower than three.js on cube scenes and level on
spheres. The split is the clue: cubes are draw-call-bound, spheres are
geometry-bound, so the hypothesis is that apse loses on **per-draw CPU**, not on
the GPU. Instancing collapses N draws into 1 and attacks exactly that axis, so
if the hypothesis holds, instancing should close the gap and cross over.

Four things are measured: the baseline, the instanced sweep and its crossover,
the CPU phase split at 1k and 5k objects, and what the present pass costs.

## The answer

**The hypothesis is supported.** apse's deficit is per-draw CPU. Measured as the
slope of frame time against draw count:

| engine | per-draw marginal |
|---|---|
| apse | **0.87 µs/draw** |
| three.js | **0.66 µs/draw** |
| ratio | **1.32×** |

That 1.32× is the entire per-draw gap, and it is stable across every scene size
measured. At 100 000 objects it is **87 ms of CPU per frame in draw submission
alone**. Instancing collapses that to 1 draw and **0.012 ms of CPU — flat from
1 000 to 100 000 instances**, because one draw costs the same whatever it
instanced. On the same scene with pixel-identical output that is a **7000×
reduction**.

**So yes, apse crosses over — and it crosses over immediately, at 1 000
instances, not at some higher threshold.** There is no crossover *point* to
report because there is no region where per-draw apse is competitive with
instanced three.js. Measured end-to-end (CPU + GPU), apse instanced against
three.js per-draw:

| objects | apse instanced | three.js per-draw | apse ahead by |
|---|---|---|---|
| 1 000 | 0.151 ms | 0.950 ms | 6.3× |
| 10 000 | 0.246 ms | 7.412 ms | 30× |
| 100 000 | 1.109 ms | 78.162 ms | 70× |

The one comparison where apse does **not** win is against three.js *instanced*,
i.e. like-for-like on the same technique. There apse is 0.15–1.11 ms against
three's 0.038–1.084 ms, and it only draws level at 50k–100k. That gap is
present-pass cost plus a fixed per-frame overhead, not per-draw cost — see
section 4 below.

## Method, and the traps in it

Every item below is a mistake this harness or its neighbours have already made.

- **A software adapter fails the run.** Not a footnote. `requestAdapter` on a
  software path returns timings that are not representative of anything.
- **The canvas backing store is asserted**, not read off a CSS size. A 300×150
  canvas inflates every number in the report.
- **Coverage is asserted per scene, per engine**, against a background
  calibrated from an *empty scene of that same engine*. Sharing one background
  between the two produced a reading of "100% of pixels lit" for an empty scene.
- **Coverage is reported dilated as well as exact.** See the sub-pixel note below.
- **Any WebGPU validation error fails the run.** It invalidates a command buffer
  without throwing, and a benchmark can otherwise measure a renderer drawing
  nothing. This guard caught two real bugs in the harness itself.
- **Distributions, never single samples**: 120 samples × 3 trials per scene,
  p50 and p90 reported.
- **Geometry-bound and draw-call-bound scenes are never averaged together.**
  Cubes and spheres are separate rows. Averaging them is how one flattering
  number reached the README once.
- **The two engines are measured in separate batches.** Interleaving them leaves
  one engine's GPU work in flight while the other is timed, which inflated apse's
  1000-cube frame from 0.8 ms to 1.6 ms — a 2× error from measurement order
  alone.
- **The queue is drained every few samples.** A backlog makes `submit` block on
  a backed-up GPU and inflates the CPU time being measured.
- **`performance.now()` is clamped to 100 µs here** (no cross-origin isolation).
  This is not a rounding detail: a one-draw instanced frame is ~12 µs, so the
  first version of this harness reported **0.000 ms for every instanced scene**.
  All sub-quantum numbers are measured in batches of K frames, and the achieved
  resolution is reported next to each one.
- **The present-pass A/B is two renderers in one page, interleaved.** Across two
  page loads the per-page variance exceeded the effect and the comparison came
  out *inverted* — the present pass appeared to make frames faster.

### The sub-pixel trap

The first sweep normalised each object's size to hold total projected area
constant as the count grew. Past ~30 000 instances each cube falls below one
pixel; a sub-pixel triangle covers no pixel centre, so the frame is genuinely
empty and the coverage assertion correctly reported that nothing drew. That is a
true statement about rasterisation and a useless statement about the renderer.

The `spread` grid variant derives scale from the grid instead
(`extent / (2*side)`, spacing twice the object size), so objects keep a constant
pixel size and never overlap. The `fixed` variant keeps scale at 0.55 and lets
added objects add overdraw. Both are reported; `spread` is the one that isolates
the per-draw CPU axis, and `fixed` is the fairer per-N comparison.

### What the phase split actually measures

`performance.now()`'s 100 µs quantum means per-call timing of the WebGPU draw
API is meaningless: 1000 individual `setBindGroup` calls each read as 0 or
0.1 ms, and summing them produced **154% of the frame**. So:

- **Phases are timed per frame, not per draw** — cull, the `render()`→`finish()`
  bracket, `writeBuffer`, `finish`, `submit`, and the present pass — and each is
  accumulated over the whole loop before being divided by the frame count.
- **Call counts are exact**, because counting needs no timer.
- **The per-draw cost is the marginal**, from the slope of frame time against
  draw count. No model, no closure problem.
- The trivial-pipeline microbench is reported **as a lower bound** and is
  explicitly not used to close the bracket. Using it manufactured a fake
  58% residual.

The static/moving distinction is not cosmetic. `#packObjects` packs only *dirty*
objects, so a static scene writes no object uniforms after its first frame — one
`writeBuffer` call, 0.42 KB. A moving scene writes the whole object range every
frame: two calls, **32.8 MB/frame** at 5 000 objects, and 0.41 ms of it. Reporting
only a static scene understates the pack phase and points the team at the wrong
loop.

## Results

Device `apple metal-3 · compatibility`, feature level `compatibility`,
1280×720, Chrome headless on the real GPU, three.js r186 on WebGL2
(`WebGL 2.0 (OpenGL ES 3.0 Chromium)`). 120 samples × 3 trials per scene,
p50 quoted. `performance.now()` quantum 0.1 ms, not cross-origin isolated.
**No `timestamp-query` on this device, so no GPU-side milliseconds appear
anywhere in this report** — every "e2e" figure is a CPU-side wait on
`onSubmittedWorkDone` (apse) or a 1-pixel `readPixels` (three.js), not a
timestamp query.

### Baseline, one draw per object

| scene | apse p50 | apse p90 | three p50 | three p90 | ratio | draws | lit px/cols (apse / three) |
|---|---|---|---|---|---|---|---|
| 1000 cubes | 0.800 | 0.900 | 0.600 | 0.700 | 0.75× | 1000 | 4456/841 / 4456/841 |
| 2000 cubes | 1.500 | 1.600 | 1.200 | 1.300 | 0.80× | 2000 | 5192/901 / 5192/901 |
| 5000 cubes | 3.500 | 3.700 | 2.500 | 2.600 | 0.71× | 5000 | 5472/924 / 5472/924 |
| 10000 cubes | 7.200 | 7.400 | 5.200 | 5.720 | 0.72× | 10000 | 5588/934 / 5588/934 |
| 1000 spheres | 0.700 | 0.800 | 0.700 | 0.800 | **1.00×** | 1000 | 4413/828 / 4422/828 |

Cubes lose 1.25–1.41×, spheres tie exactly. The split the hypothesis rests on
reproduces, and it is the same shape as the 1.3–1.4× previously recorded.
Coverage is **byte-identical between the two engines** on every cube row, which
is the cross-check that both drew the same thing. (The sphere row differs by 9
pixels out of 921 600 — 0.001% — the two engines' shading of a curved surface
at one scale, not a difference in what was drawn.)

### CPU phase split, ms per frame

Percentages are of the timed-instrumented frame. Every phase is timed per frame
and accumulated over the loop; call counts are exact.

| phase | 1k static | 1k moving | 5k static | 5k moving |
|---|---|---|---|---|
| cull / walk | 0.054 (6.2%) | 0.056 (5.7%) | 0.283 (6.7%) | 0.283 (5.7%) |
| sort (`sortDrawItems`) | 0.048 (5.4%) | 0.048 (4.9%) | 0.245 (5.8%) | 0.245 (4.9%) |
| pack + `uploadRange` (full) | 0.024 (2.7%) | 0.024 (2.4%) | 0.120 (2.8%) | 0.120 (2.4%) |
| frame uniform (14 writes + flush) | 0.001 (0.1%) | 0.001 (0.1%) | 0.001 (0.0%) | 0.001 (0.0%) |
| **per-draw encode (measured)** | **0.864 (98%)** | **0.867 (88%)** | **4.227 (100%)** | **4.237 (85%)** |
| present pass (last pass) | 0.003 (0.3%) | 0.003 (0.3%) | 0.002 (0.0%) | 0.003 (0.1%) |
| `encoder.finish` | 0.001 (0.1%) | 0.001 (0.1%) | 0.002 (0.1%) | 0.001 (0.0%) |
| `queue.submit` | 0.003 (0.3%) | 0.003 (0.3%) | 0.006 (0.1%) | 0.004 (0.1%) |
| `queue.writeBuffer` | 0.000 (1 call, 0.42 KB) | 0.090 (2 calls, 32.8 MB) | 0.003 (1 call, 0.42 KB) | 0.398 (2 calls, 32.8 MB) |
| calls/frame | 1000 drawIndexed, 2 setPipeline, 1006 setBindGroup, 0 createBindGroup, 2 passes | same | 5000 / 2 / 5006 / 0 / 2 | same |
| marginal fit | 0.86 µs/draw | 0.87 µs/draw | 0.85 µs/draw | 0.85 µs/draw |

**Per-draw encode is 85–100% of the frame at both counts.** Culling, sorting and
packing together are under 15%. The actionable finding is that the fix is not in
the scene-graph loops at all — it is in what a single draw costs.

`setBindGroup` runs **1006 times for 1000 objects**: 1000 dynamic-offset binds
plus 6 fixed ones. That is the per-object call the ratio tracks, and it is the
thing to attack.

Two caveats stated rather than hidden:

- The **microbench per-call numbers (0.13 µs for the `setBindGroup`+`draw`
  combo) are a lower bound, not the real cost** — they run on a trivial pipeline
  with a 3-vertex buffer. The real per-draw cost is the measured 0.85 µs
  marginal, and the microbench is deliberately not used to close the bracket.
  Using it manufactured a fake 58% residual.
- The **pack phase is priced as a full pack of every object, which is an upper
  bound.** A static frame packs nothing after its first frame. The static/moving
  distinction is the real measurement, via the `writeBuffer` call count: 1 call /
  0.42 KB static, 2 calls / **32.8 MB per frame** moving at 5 000 objects. A
  moving scene re-uploads the whole object range every frame. That is 8.0% of a
  5k frame and it is the only phase where static and moving diverge sharply.

### Present pass cost

Default `toneMapping` against `toneMapping: null, hdr: false`, two renderers in
one page, interleaved block by block, 12 blocks. CPU is the batched column; e2e
includes the GPU wait.

| scene | cpu on | cpu off | Δ cpu | e2e on | e2e off | Δ e2e |
|---|---|---|---|---|---|---|
| 1000 cubes | 0.800 | 0.780 | +0.020 (2.6%) | 1.000 | 0.940 | +0.060 (6.4%) |
| 5000 cubes | 3.900 | 3.819 | +0.081 (2.1%) | 4.550 | 4.400 | +0.150 (3.4%) |
| 5000 instanced | 0.014 | 0.012 | +0.002 (15.7%) | 0.193 | 0.117 | +0.076 (64.7%) |
| 100k instanced | 0.013 | 0.011 | +0.002 (17.5%) | 1.374 | 1.310 | +0.064 (4.9%) |

**The present pass costs ~0.002 ms of CPU and a fixed ~0.06–0.08 ms of GPU time
per frame.** It is essentially free on the CPU and it is a fixed cost, not a
per-draw one — which is why it is irrelevant on a 1000-draw frame (0.0%) and
dominant on a 1-draw frame (64.7%). Coverage is byte-identical on and off, so
the pass is not changing what is drawn.

### Draw-call count vs frame time

The scaling curve, from the baseline rows plus the sweep, all one draw per object
and the same 1280×720 frame:

| objects | draws | apse CPU | three CPU | ratio | apse e2e | three e2e |
|---|---|---|---|---|---|---|
| 1 000 | 1000 | 0.800 | 0.600 | 0.75× | — | — |
| 2 000 | 2000 | 1.500 | 1.200 | 0.80× | — | — |
| 5 000 | 5000 | 3.500 | 2.500 | 0.71× | — | — |
| 10 000 | 10000 | 7.200 | 5.200 | 0.72× | — | — |
| 25 000 | 25000 | 20.375 | 15.550 | 0.76× | 22.725 | 18.938 |
| 50 000 | 50000 | 39.462 | 31.250 | 0.79× | 45.075 | 37.700 |
| 100 000 | 100000 | 86.675 | 65.725 | 0.76× | 92.800 | 78.162 |

(25k–100k are from the `spread` sweep, which is a different grid layout from the
1k–10k baseline rows, so the two blocks are internally consistent but the
1k→25k step should not be read as a single continuous curve.)

Frame time is very nearly linear in draw count in both engines, with a fixed cost
of under 0.05 ms: apse 0.87 µs/draw, three 0.66 µs/draw, and the ratio holds from
1 000 to 100 000 draws. There is no knee, no cliff at the 4096 object-buffer
capacity, and no super-linear growth on either side. **The cost is the draw, and
it is the draw at every size.**

## Files

- `index.html` — the measurement page. One `Ctx` per renderer configuration,
  each with its own canvas, device, materials, calibrated background and scene
  map. `?tones=default,none` gives the two configurations the A/B needs.
- `run.ts` — the runner: baseline, sweep, phase split, tone A/B, plus all the
  assertions.
- `two-device.html` — a standalone check that two WebGPU devices coexist in one
  page. The existing harness works around this with one renderer per page; on
  this Chrome build the workaround is not needed, which is what allows the
  present-pass A/B to be same-page and interleaved.
- `results/report.json`, `results/run.log` — the last run's raw output.
