/**
 * The per-draw-CPU measurement.
 *
 * The question this exists to answer is narrow and falsifiable:
 *
 *   apse is 1.3-1.4x slower than three.js on cube scenes (draw-call-bound) and
 *   level on spheres (geometry-bound). So the loss is per-draw CPU, not GPU.
 *   Instancing collapses N draws into 1, so it should remove exactly that gap.
 *
 * Four things it measures:
 *   1. a clean apse-vs-three.js baseline, with coverage asserted per scene
 *   2. the instanced sweep, and where (if anywhere) it crosses over
 *   3. the CPU phase split at 1k and 5k objects, static and moving
 *   4. what the present pass costs, tone-mapped against not
 *
 * Method, because each of these has been got wrong here before:
 *   - a software adapter is a hard failure, not a footnote
 *   - the canvas backing store is asserted, not assumed
 *   - every timed scene is read back and its coverage asserted non-zero, against
 *     a background calibrated from an *empty* scene of the same engine
 *   - any WebGPU validation error fails the run
 *   - distributions, never single samples
 *   - geometry-bound and draw-call-bound scenes are never averaged together
 *   - the two engines are measured in separate batches, never interleaved:
 *     interleaving leaves one engine's GPU work in flight while the other is
 *     timed, which inflated apse's 1000-cube frame from 0.8ms to 1.6ms
 *   - the queue is drained every few samples, because a backlog makes submit
 *     block on the GPU and inflates the CPU time being measured
 *   - sub-clock-quantum frames are timed in batches, because performance.now()
 *     is clamped to 100us here and a one-draw instanced frame is cheaper than
 *     one quantum. Unbatched, every instanced scene reads 0.000ms.
 *   - the present-pass A/B is two renderers in one page, measured interleaved.
 *     Across two page loads the per-page variance was larger than the effect,
 *     and the A/B came out inverted.
 *
 * Usage: bun run bench/diag/perf/run.ts
 *        APSE_PERF_QUICK=1 bun run bench/diag/perf/run.ts    (smoke test)
 */

import { serve } from 'bun';
import puppeteer from 'puppeteer';
import type { Browser, Page } from 'puppeteer';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = new URL('../../..', import.meta.url).pathname;
const PORT = 8791;
const OUT = join(ROOT, 'bench/diag/perf/results');
const PAGE = '/bench/diag/perf/index.html';

const QUICK = process.env['APSE_PERF_QUICK'] === '1';
const WARMUP = QUICK ? 5 : 20;
const SAMPLES = QUICK ? 20 : 120;
const TRIALS = QUICK ? 1 : 3;

interface Dist {
  n: number; min: number; p50: number; p90: number; p99: number; max: number; mean: number; samples: number[];
}

interface Meas {
  name: string; engine: string; mode: string; objects: number; scale: number;
  animated: boolean; clockMs: number; batch: number; batchedResolutionMs: number;
  render: Dist; batched: Dist; fenced: Dist; animMsPerFrame: number;
  stats: { drawCalls: number; triangles: number; uniformBytes: number };
  hasGpuTiming: boolean; gpuMs?: number;
  coverage?: Record<string, number>;
}

const log: string[] = [];
function say(s: string): void {
  console.log(s);
  log.push(s);
}

async function openPage(browser: Browser, tones: string): Promise<{ page: Page; env: any }> {
  const page = await browser.newPage();
  page.on('console', (m) => {
    const t = m.text();
    if (t.startsWith('[perf]')) say('  ' + t);
  });
  page.on('pageerror', (e: unknown) => say(`  PAGE ERROR: ${e instanceof Error ? e.message : String(e)}`));
  await page.goto(`http://localhost:${PORT}${PAGE}?tones=${tones}`, { waitUntil: 'load' });
  // Boot explicitly rather than waiting on a flag: a flag turns a thrown error
  // into a timeout that reports "unknown".
  const env = await page.evaluate(() => (globalThis as any).__boot());
  const ready = await page.evaluate(() => (globalThis as any).__ready === true);
  if (!ready) {
    const err = await page.evaluate(() => String((globalThis as any).__err ?? 'unknown'));
    throw new Error(`perf page did not become ready (tones=${tones}): ${err}`);
  }
  return { page, env };
}

async function assertNoValidationErrors(page: Page, ctx: number | 'three', where: string): Promise<void> {
  const errs: string[] = await page.evaluate((c: any) => (globalThis as any).__errors(c), ctx);
  if (errs.length > 0) {
    say(`\n  VALIDATION ERROR during ${where}:`);
    for (const e of errs.slice(0, 3)) say('    ' + String(e).split('\n').slice(0, 8).join('\n    '));
    throw new Error(`${errs.length} WebGPU validation error(s) during ${where}; measurements are meaningless`);
  }
}

async function build(page: Page, ctx: number | 'three', engine: string, mode: string, spec: Record<string, unknown>) {
  return await page.evaluate(
    (c: any, e: string, m: string, s: any) => (globalThis as any).__build(c, e, m, s),
    ctx, engine, mode, spec,
  ) as { name: string; objects: number; scale: number; nodes?: number };
}

async function measure(page: Page, ctx: number | 'three', name: string, opts: Record<string, unknown>): Promise<Meas> {
  return await page.evaluate(
    (c: any, n: string, o: any) => (globalThis as any).__measure(c, n, o),
    ctx, name, opts,
  ) as Meas;
}

async function coverageOf(page: Page, ctx: number | 'three', name: string): Promise<Record<string, number>> {
  return await page.evaluate((c: any, n: string) => (globalThis as any).__coverage(c, n), ctx, name) as Record<string, number>;
}

/**
 * Measures one scene for one engine and asserts it drew something.
 *
 * The coverage assertion is per engine and per scene, against that engine's own
 * calibrated empty frame. A benchmark that measures a renderer drawing nothing
 * otherwise reports excellent numbers, which is the worst failure mode there is.
 */
async function runOne(
  page: Page, ctx: number | 'three', engine: string, mode: string, name: string, spec: Record<string, unknown>,
): Promise<Meas> {
  await build(page, ctx, engine, mode, spec);
  const m = await measure(page, ctx, name, { samples: SAMPLES, warmup: WARMUP, trials: TRIALS });
  const cov = await coverageOf(page, ctx, name);
  await assertNoValidationErrors(page, ctx, `${engine}/${mode} ${name}`);
  // Either an exact or a dilated hit counts as "drew something". Point sampling
  // alone reports an empty frame for legitimately sub-pixel geometry, which is
  // a fact about rasterisation and not about the renderer.
  if (cov.litPixels === 0 && (cov.litPixelsDilated ?? 0) === 0) {
    throw new Error(
      `${engine}/${mode} scene ${name} rendered nothing: 0 lit pixels and 0 dilated ` +
      `against its own empty frame (maxDelta ${cov.maxDelta})`,
    );
  }
  if (cov.litColumns < 2) {
    throw new Error(`${engine}/${mode} scene ${name} lit only ${cov.litColumns} column(s): a degenerate render, not a scene`);
  }
  m.coverage = cov;
  return m;
}

const f3 = (v: number) => v.toFixed(3);
const f2 = (v: number) => v.toFixed(2);
const f1 = (v: number) => v.toFixed(1);
const ratio = (three: number, apse: number) => (apse > 0 ? (three / apse) : 0);

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });

  const server = serve({
    port: PORT,
    idleTimeout: 0,
    routes: {
      '/*': async (req) => {
        const path = new URL(req.url).pathname;
        const file = Bun.file(join(ROOT, path === '/' ? PAGE.slice(1) : path.slice(1)));
        if (await file.exists()) {
          const type = path.endsWith('.ts') || path.endsWith('.js') || path.endsWith('.mjs')
            ? 'text/javascript; charset=utf-8'
            : path.endsWith('.html') ? 'text/html; charset=utf-8'
            : path.endsWith('.json') ? 'application/json; charset=utf-8'
            : 'application/octet-stream';
          return new Response(file, { headers: { 'content-type': type } });
        }
        return new Response('not found', { status: 404 });
      },
    },
  });

  const browser = await puppeteer.launch({
    headless: true,
    executablePath: process.env['CHROME_PATH'] ?? undefined,
    channel: process.env['CHROME_PATH'] === undefined ? 'chrome' : undefined,
    args: [
      // No software-rasteriser flags. They make requestAdapter() return null on
      // macOS, which looks like "WebGPU unsupported" and is actually "you asked
      // for a path that does not exist here". A software adapter is rejected
      // explicitly below instead.
      '--no-sandbox',
      '--disable-dev-shm-usage',
    ],
  });

  const report: Record<string, unknown> = {
    generatedAt: new Date().toISOString(),
    node: process.version,
    host: process.platform + ' ' + process.arch,
    samplesPerScene: SAMPLES, trials: TRIALS, warmup: WARMUP, quick: QUICK,
  };

  try {
    // Two contexts in one page: the shipped default, and toneMapping: null.
    const { page, env } = await openPage(browser, 'default,none');
    report.env = env;

    say('\n  ENVIRONMENT');
    say(`    adapter        ${env.adapter}`);
    say(`    featureLevel   ${env.featureLevel}`);
    for (const c of env.contexts) {
      say(`    ctx${c.index}           tone=${c.tone === 'none' ? 'toneMapping: null, hdr: false' : 'default (present pass ON)'}  sceneFormat ${c.sceneFormat}`);
    }
    say(`    backing store  ${env.backing.w}x${env.backing.h} per context`);
    say(`    clock quantum  ${env.clockMs.toFixed(4)}ms (performance.now resolution${env.crossOriginIsolated ? ', cross-origin isolated' : ', not isolated'})`);
    say(`    gpu timestamps ${env.hasTimestamps ? 'available' : 'NOT available — stats.gpu is null, so NO GPU-side ms appears in this report'}`);
    say(`    three.js       r${env.three.revision} on ${env.three.gl}`);
    say(`    date           ${env.date}`);

    if (env.software) {
      throw new Error(
        `requestAdapter returned a software adapter ("${env.adapter}"). ` +
        `Absolute timings from a software rasteriser are not representative and the run is invalid.`,
      );
    }
    if (env.backing.w !== 1280 || env.backing.h !== 720) {
      throw new Error(`canvas backing store is ${env.backing.w}x${env.backing.h}, expected 1280x720`);
    }
    await assertNoValidationErrors(page, 0, 'boot');
    await assertNoValidationErrors(page, 1, 'boot ctx1');

    // -----------------------------------------------------------------------
    // 1. Baseline. Cubes and spheres stay separate rows and are never averaged:
    // cubes are draw-call-bound, spheres geometry-bound, and averaging them is
    // how one flattering number reached the README once.
    // -----------------------------------------------------------------------
    say(`\n  1. BASELINE — one draw per object, 1280x720, p50 of ${SAMPLES} samples x${TRIALS} trials`);
    say('       ' + 'scene'.padEnd(15) + 'apseCPU'.padStart(9) + 'threeCPU'.padStart(9) + 'cpuRatio'.padStart(9) +
      '  apseE2E'.padStart(9) + 'threeE2E'.padStart(9) + 'e2eRatio'.padStart(9) +
      '  apseP90'.padStart(9) + 'threeP90'.padStart(9) + '  draws'.padStart(7) + '   litPx/cols');
    const baselineSpecs = [
      { name: 'b-1000-cube', objects: 1000, geometry: 'cube' },
      { name: 'b-2000-cube', objects: 2000, geometry: 'cube' },
      { name: 'b-5000-cube', objects: 5000, geometry: 'cube' },
      { name: 'b-10000-cube', objects: 10000, geometry: 'cube' },
      { name: 'b-1000-sphere', objects: 1000, geometry: 'sphere' },
    ];
    const baseline: any[] = [];
    for (const s of QUICK ? baselineSpecs.slice(0, 2) : baselineSpecs) {
      const a = await runOne(page, 0, 'apse', 'per-draw', s.name, s);
      const t = await runOne(page, 'three', 'three', 'per-draw', s.name, s);
      baseline.push({ spec: s, apse: a, three: t });
      const cpu = ratio(t.render.p50, a.render.p50);
      const e2e = ratio(t.fenced.p50, a.fenced.p50);
      const colour = cpu >= 1 ? '\x1b[32m' : '\x1b[31m';
      say('       ' + s.name.padEnd(15) +
        f3(a.render.p50).padStart(9) + f3(t.render.p50).padStart(9) +
        `${colour}${cpu.toFixed(2)}x\x1b[0m`.padStart(10) +
        f3(a.fenced.p50).padStart(10) + f3(t.fenced.p50).padStart(10) + e2e.toFixed(2).padStart(9) + 'x' +
        f3(a.render.p90).padStart(9) + f3(t.render.p90).padStart(9) +
        String(a.stats.drawCalls).padStart(8) +
        `   ${a.coverage!.litPixels}px/${a.coverage!.litColumns}c (three ${t.coverage!.litPixels}px/${t.coverage!.litColumns}c)`);
    }
    report.baseline = baseline;

    // -----------------------------------------------------------------------
    // 2. The instanced sweep, and the crossover.
    //
    // Two grid variants. `spread` holds spacing at twice the object size, so
    // objects keep a constant pixel size and never overlap: the fragment cost
    // does not grow with N, and the only thing that changes is the draw count.
    // Those rows answer the per-draw-CPU question. `fixed` is the baseline
    // layout, where added objects add overdraw; apse and three.js still see
    // identical geometry at each N, so it is the fairer per-N comparison but
    // the worse trend.
    //
    // Engines are swept in separate batches, for the contention reason above.
    // -----------------------------------------------------------------------
    say('\n  2. INSTANCED SWEEP — cpu/e2e ms per frame (cpu = batched CPU, e2e = incl. GPU)');
    const SWEEP = QUICK ? [1000, 10000] : [1000, 2000, 5000, 10000, 25000, 50000, 100000];
    const sweep: any[] = [];
    for (const objects of SWEEP) {
      for (const variant of QUICK ? ['spread'] : ['spread', 'fixed']) {
        const name = `s-${variant}-${objects}`;
        const spec = { name, objects, geometry: 'cube', variant };
        const rec: any = { objects, variant };

        const ap = await runOne(page, 0, 'apse', 'per-draw', name, spec);
        rec.apsePerDraw = ap;
        const ai = await runOne(page, 0, 'apse', 'instanced', name, spec);
        rec.apseInst = ai;
        const tp = await runOne(page, 'three', 'three', 'per-draw', name, spec);
        rec.threePerDraw = tp;
        const ti = await runOne(page, 'three', 'three', 'instanced', name, spec);
        rec.threeInst = ti;

        sweep.push(rec);
        const c = (m: Meas) => f3(m.batched.p50);
        const e = (m: Meas) => f3(m.fenced.p50);
        say(`    ${objects.toString().padStart(6)} (${variant}, scale ${ap.scale})  ` +
          `apse/draw ${c(ap)}/${e(ap)}  apse/inst ${c(ai)}/${e(ai)} (${ai.stats.drawCalls}d)  ` +
          `three/draw ${c(tp)}/${e(tp)}  three/inst ${c(ti)}/${e(ti)} (${ti.stats.drawCalls}d)`);
        const cov = (m: Meas) => `${m.coverage!.litPixels}px/${m.coverage!.litColumns}c`;
        const covD = (m: Meas) => `${m.coverage!.litPixelsDilated}px`;
        say(`           lit: apse/inst ${cov(ai)} (dilated ${covD(ai)})  apse/draw ${cov(ap)} (${covD(ap)})  three/inst ${cov(ti)} (${covD(ti)})  three/draw ${cov(tp)} (${covD(tp)})`);
      }
    }
    report.sweep = sweep;

    // -----------------------------------------------------------------------
    // 3. CPU phase split, static and moving, at 1k and 5k.
    //
    // Static matters because #packObjects packs only *dirty* objects, so a
    // static scene writes no object uniforms after its first frame. Reporting
    // only a static scene understates the pack phase and points the team at the
    // wrong loop. The moving variant marks every node dirty each frame, which is
    // what an animated scene does; its per-frame cost is measured apart from the
    // renderer's and reported separately.
    // -----------------------------------------------------------------------
    say('\n  3. CPU PHASE SPLIT (ms per frame)');
    const phases: any[] = [];
    for (const objects of (QUICK ? [1000] : [1000, 5000])) {
      for (const animated of [false, true]) {
        const name = `p-${objects}-${animated ? 'moving' : 'static'}`;
        await build(page, 0, 'apse', 'per-draw', { name, objects, geometry: 'cube' });
        const pr: any = await page.evaluate(
          (n: string, o: any) => (globalThis as any).__probe(0, n, o),
          name, { samples: QUICK ? 20 : 120, warmup: WARMUP, animated },
        );
        const clean = await measure(page, 0, name, { samples: SAMPLES, warmup: WARMUP, trials: 2, animated });
        const cov = await coverageOf(page, 0, name);
        await assertNoValidationErrors(page, 0, 'phases ' + name);
        if (cov.litPixels === 0) throw new Error(`phase scene ${name} rendered nothing`);
        phases.push({ objects, animated, clean, probe: pr, coverage: cov });

        const t = pr.timed, c = pr.controlRender, mb = pr.microbench, js = mb.jsPhases ?? {};
        const draws = t.counts.drawTotal;
        const modelPerDraw = (mb.perDrawComboUs / 1000) * draws;
        const frame = t.frameMs;
        // The three JS phases priced directly, plus the per-draw API cost
        // modelled from the microbench. What is left of the bracket is the
        // frame uniform write and render()'s own prologue.
        // The per-draw cost is the measured marginal, not the microbench: the
        // microbench runs a trivial pipeline and is a lower bound, so using it
        // to close the bracket would manufacture a large fake residual.
        const margMs = ((mb.marginal?.usPerDraw ?? 0) / 1000) * draws;
        const priced = (js.cullMs ?? 0) + (js.sortMs ?? 0) + (js.packFullMs ?? 0) +
          (js.frameUniformMs ?? 0) + (js.syncSizeMs ?? 0) + margMs;
        const residual = t.encodeBracketMs - priced - t.presentEncodeMs;
        say(`    ${objects} objects, ${animated ? 'MOVING (every object dirty each frame)' : 'STATIC (nothing dirty after frame 1)'}`);
        say(`      clean frame p50 ${f3(clean.render.p50)}ms   count-only control ${f3(c.p50)}ms   timed-instrumented ${f3(pr.timedRender.p50)}ms (timing cost ${f3(pr.timedRender.p50 - c.p50)}ms)`);
        say(`      encode bracket (render->finish) ${f3(t.encodeBracketMs).padStart(8)}ms  ${pct(t.encodeBracketMs, frame)}  = everything below, over ${f1(draws)} draws`);
        say(`        cull / walk                 ${f3(js.cullMs ?? 0).padStart(8)}ms  ${pct(js.cullMs ?? 0, frame)}   [priced directly over ${js.items} real draw items]`);
        say(`        sort (sortDrawItems)        ${f3(js.sortMs ?? 0).padStart(8)}ms  ${pct(js.sortMs ?? 0, frame)}   [priced directly]`);
        say(`        pack + uploadRange (full)   ${f3(js.packFullMs ?? 0).padStart(8)}ms  ${pct(js.packFullMs ?? 0, frame)}   [priced directly, ${js.packedPerCall ?? 0} objects; upper bound — a static frame packs none]`);
        say(`        frame uniform (14 writes + flush) ${f3(js.frameUniformMs ?? 0).padStart(8)}ms  ${pct(js.frameUniformMs ?? 0, frame)}   [priced directly]`);
        say(`        size sync (canvas read)     ${f3(js.syncSizeMs ?? 0).padStart(8)}ms  ${pct(js.syncSizeMs ?? 0, frame)}   [priced directly]`);
        say(`        per-draw encode (measured)  ${f3(margMs).padStart(8)}ms  ${pct(margMs, frame)}   [MEASURED: ${f2(mb.marginal?.usPerDraw ?? 0)}us/draw x ${f1(draws)} draws, from the slope of frame time vs draw count]`);
        say(`        per-draw encode (microbench lower bound) ${f3(modelPerDraw).padStart(8)}ms  ${pct(modelPerDraw, frame)}   [${f2(mb.perDrawComboUs)}us/draw on a trivial pipeline — a LOWER bound, not the real cost]`);
        say(`        present pass (last pass)    ${f3(t.presentEncodeMs).padStart(8)}ms  ${pct(t.presentEncodeMs, frame)}`);
        say(`        unattributed residual       ${f3(residual).padStart(8)}ms  ${pct(residual, frame)}   [bracket minus everything above; the marginal slope is a single linear fit, so curvature lands here]`);
        say(`      marginal fit: ${(mb.marginal?.points ?? []).map((pt: any) => `${pt.draws}d=${pt.ms}ms`).join('  ')}  ->  ${f2(mb.marginal?.usPerDraw ?? 0)}us/draw (${mb.marginal?.usPerDrawMethod}), fixed cost ${f3(mb.marginal?.fixedMsPerFrame ?? 0)}ms/frame; least-squares gives ${f2(mb.marginal?.usPerDrawLsq ?? 0)}us/draw`);
        say(`      encoder.finish              ${f3(t.finishMs).padStart(8)}ms  ${pct(t.finishMs, frame)}`);
        say(`      queue.submit                ${f3(t.submitMs).padStart(8)}ms  ${pct(t.submitMs, frame)}`);
        say(`      queue.writeBuffer           ${f3(t.writeBufferMs).padStart(8)}ms  ${pct(t.writeBufferMs, frame)}  (${f1(t.writeBufferCalls)} calls, ${f2(t.writeBufferKB)}KB/frame — 1 call = static, 2 = object range re-uploaded)`);
        say(`      calls/frame: drawIndexed ${f1(t.counts.drawIndexed)}  setPipeline ${f1(t.counts.setPipeline)}  setBindGroup ${f1(t.counts.setBindGroup)}  createBindGroup ${f1(t.counts.createBindGroup)}  renderPasses ${f1(t.passesPerFrame)}`);
        say(`      microbench per call: bindGroup ${f2(mb.bindGroupPerCallUs)}us  +pipeline ${f2(mb.pipelinePerCallUs)}us  +draw ${f2(mb.drawPerCallUs)}us  +drawIndexed ${f2(mb.indexedPerCallUs)}us  combo ${f2(mb.perDrawComboUs)}us  (${mb.reps} reps x 7 blocks, cumulative)`);
        if (animated) say(`      [caller's animation loop, not the renderer's: ${f3(clean.animMsPerFrame)}ms/frame]`);
      }
    }
    report.phases = phases;

    // -----------------------------------------------------------------------
    // 4. Present pass cost, tone-mapped against not, in the SAME page and
    //    interleaved. ctx0 is the default, ctx1 is { toneMapping: null, hdr:
    //    false } — the documented direct path, with no intermediate, no
    //    fullscreen pass and no pipeline. Interleaving is the point: across two
    //    page loads the per-page variance was larger than the effect and the
    //    comparison came out inverted.
    // -----------------------------------------------------------------------
    say('\n  4. PRESENT PASS COST — default toneMapping vs toneMapping: null (same page, interleaved)');
    const TONE_SCENES = [
      { name: 't-1000', objects: 1000, instanced: false },
      { name: 't-5000', objects: 5000, instanced: false },
      { name: 't-5000-inst', objects: 5000, instanced: true },
      { name: 't-100000-inst', objects: 100000, instanced: true },
    ].filter((s) => !(QUICK && s.objects > 10000));
    const tone: any[] = [];
    for (const s of TONE_SCENES) {
      const mode = s.instanced ? 'instanced' : 'per-draw';
      await build(page, 0, 'apse', mode, { name: s.name, objects: s.objects, geometry: 'cube' });
      await build(page, 1, 'apse', mode, { name: s.name, objects: s.objects, geometry: 'cube' });
      // Interleaved: alternate the two configurations sample-block by
      // sample-block so any drift in the machine hits both equally.
      // The *batched* CPU column, not render.p50: an instanced frame is well
      // under the 100us clock quantum, so render.p50 is 0.000 for both
      // configurations and the comparison would be 0 vs 0.
      const on: number[] = [], off: number[] = [], onE: number[] = [], offE: number[] = [];
      const blocks = QUICK ? 4 : 12;
      for (let b = 0; b < blocks; b++) {
        const a = await measure(page, 0, s.name, { samples: Math.max(4, Math.floor(SAMPLES / 4)), warmup: b === 0 ? WARMUP : 2, trials: 1 });
        const o = await measure(page, 1, s.name, { samples: Math.max(4, Math.floor(SAMPLES / 4)), warmup: b === 0 ? WARMUP : 2, trials: 1 });
        on.push(a.batched.p50); off.push(o.batched.p50);
        onE.push(a.fenced.p50); offE.push(o.fenced.p50);
      }
      const a0 = await measure(page, 0, s.name, { samples: 8, warmup: 4, trials: 1 });
      const o0 = await measure(page, 1, s.name, { samples: 8, warmup: 4, trials: 1 });
      const covOn = await coverageOf(page, 0, s.name);
      const covOff = await coverageOf(page, 1, s.name);
      await assertNoValidationErrors(page, 0, 'tone on ' + s.name);
      await assertNoValidationErrors(page, 1, 'tone off ' + s.name);
      if (covOn.litPixels === 0 || covOff.litPixels === 0) {
        throw new Error(`tone A/B scene ${s.name} rendered nothing (on ${covOn.litPixels}px, off ${covOff.litPixels}px)`);
      }
      const med = (a: number[]) => quantile(a.slice().sort((x, y) => x - y), 0.5);
      const rec = {
        name: s.name, objects: s.objects, instanced: s.instanced,
        onCpu: med(on), offCpu: med(off), onE2E: med(onE), offE2E: med(offE),
        onCpuRaw: med(on), offCpuRaw: med(off),
        onE2ERaw: med(onE), offE2ERaw: med(offE),
        batchOn: a0.batch, batchOff: o0.batch,
        quantumOn: a0.batchedResolutionMs, quantumOff: o0.batchedResolutionMs,
        cpuRatio: med(off) / med(on),
        onBlocks: on, offBlocks: off, coverageOn: covOn, coverageOff: covOff,
      };
      tone.push(rec);
      const pcpu = rec.offCpu > 0 ? ((rec.onCpu - rec.offCpu) / rec.offCpu * 100) : 0;
      const pe2e = rec.offE2E > 0 ? ((rec.onE2E - rec.offE2E) / rec.offE2E * 100) : 0;
      say(`    ${s.name.padEnd(15)} cpu: on ${f3(rec.onCpu)}ms  off ${f3(rec.offCpu)}ms  -> ${f3(rec.onCpu - rec.offCpu)}ms (${pcpu.toFixed(1)}%)   [batched, K=${rec.batchOn}, resolution ${rec.quantumOn.toFixed(4)}ms]`);
      say(`    ${' '.repeat(15)} e2e: on ${f3(rec.onE2E)}ms  off ${f3(rec.offE2E)}ms  -> ${f3(rec.onE2E - rec.offE2E)}ms (${pe2e.toFixed(1)}%)`);
      say(`    ${' '.repeat(15)} lit: on ${covOn.litPixels}px/${covOn.litColumns}c  off ${covOff.litPixels}px/${covOff.litColumns}c  (identical: same scene, tone map only changes the curve)`);
    }
    report.tone = tone;
  } finally {
    await browser.close();
    server.stop(true);
  }

  writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 2));
  writeFileSync(join(OUT, 'run.log'), log.join('\n') + '\n');
  say(`\n  report: ${join(OUT, 'report.json')}\n`);
}

function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? sorted[lo]! : sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (i - lo);
}

function pct(v: number, of: number): string {
  if (!(of > 0)) return '  (n/a)';
  return `(${(v / of * 100).toFixed(1).padStart(5)}%)`;
}

main().catch((e) => {
  console.error(`\n  perf run failed: ${e instanceof Error ? e.message : String(e)}\n`);
  process.exit(1);
});
