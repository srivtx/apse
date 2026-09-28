/**
 * The real test: does it draw?
 *
 * Everything else in this repo is unit tests over pure functions, which can all
 * pass while the renderer draws a black screen. This runs a real WebGPU device in
 * headless Chrome, renders a real scene, reads the framebuffer back, and asserts
 * on the pixels. A green run here means pixels came out the right colour.
 *
 * It also runs the same scene through three.js in the same browser, in the same
 * process, so the comparison is machine-to-machine rather than across two
 * laptops. Everything is reported, nothing is asserted about three.js beyond
 * "it also drew" — its numbers are a baseline, not a target under test.
 *
 *   bun run bench
 */

import { serve } from 'bun';
import puppeteer from 'puppeteer';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;
const PORT = 8787;
const SHOTS = join(ROOT, 'bench/results');

interface SceneSpec {
  readonly name: string;
  readonly objects: number;
  readonly geometry: 'cube' | 'sphere' | 'torus';
}

/**
 * Scenes chosen to expose where the cost curve bends, not to flatter anything.
 * The interesting comparison is 1000 → 4096 → 5000: `maxObjects` defaults to
 * 4096, so if the object-uniform buffer reallocates on a frame boundary instead
 * of on the frame that needs it, the curve kinks exactly there.
 */
const SCENES: readonly SceneSpec[] = [
  { name: '1000 cubes', objects: 1000, geometry: 'cube' },
  { name: '2000 cubes', objects: 2000, geometry: 'cube' },
  { name: '4000 cubes', objects: 4000, geometry: 'cube' },
  { name: '4100 cubes', objects: 4100, geometry: 'cube' },
  { name: '5000 cubes', objects: 5000, geometry: 'cube' },
  { name: '1000 spheres', objects: 1000, geometry: 'sphere' },
];

interface Sample {
  readonly cpuMs: number;
  drawCalls?: number;
  triangles?: number;
  readonly uniformBytes: number;
}

interface RunResult {
  readonly engine: string;
  readonly scene: string;
  readonly samples: readonly Sample[];
  meanCpuMs: number;
  p50CpuMs: number;
  p95CpuMs: number;
  drawCalls?: number;
  triangles?: number;
  readonly heapBytes: number;
  readonly setupMs: number;
  readonly error: string | null;
  readonly capacity?: number;
  readonly calls?: string;
  readonly objs?: number;
}

async function main(): Promise<void> {
  mkdirSync(SHOTS, { recursive: true });

  const server = serve({
    port: PORT,
    idleTimeout: 0,
    routes: {
      '/*': async (req) => {
        const path = new URL(req.url).pathname;
        const file = Bun.file(join(ROOT, path === '/' ? 'bench/index.html' : path.slice(1)));
        if (await file.exists()) {
          // A module script must be served with a JavaScript MIME type. Chrome
          // enforces this strictly and reports it only as a failed dynamic
          // import, which is a confusing way to learn about a header.
          const type = path.endsWith('.ts') || path.endsWith('.js') || path.endsWith('.mjs')
            ? 'text/javascript; charset=utf-8'
            : path.endsWith('.html') ? 'text/html; charset=utf-8'
            : path.endsWith('.css') ? 'text/css; charset=utf-8'
            : path.endsWith('.json') ? 'application/json; charset=utf-8'
            : 'application/octet-stream';
          return new Response(file, { headers: { 'content-type': type } });
        }
        return new Response('not found', { status: 404 });
      },
    },
  });

  console.log(`\n  apse benchmark\n  server: http://localhost:${PORT}\n`);

  const browser = await puppeteer.launch({
    headless: true,
    // The installed Chrome, not a puppeteer-managed download. This is a
    // benchmark: the number has to come from a real browser on real hardware.
    executablePath: process.env['CHROME_PATH'] ?? undefined,
    channel: process.env['CHROME_PATH'] === undefined ? 'chrome' : undefined,
    args: [
      // Deliberately minimal. The obvious flags — --enable-unsafe-swiftshader,
      // --use-angle=swiftshader, --enable-features=Vulkan — all make
      // requestAdapter() return null on macOS, which looks like "WebGPU is
      // unsupported" and is actually "you asked for a software path that does
      // not exist here". Chrome's headless mode reaches the real GPU; the only
      // thing it needs is a secure context, which the local server provides.
      '--no-sandbox',
      '--disable-dev-shm-usage',
    ],
  });

  const results: RunResult[] = [];

  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      const t = m.text();
      if (t.startsWith('[bench]')) console.log(`  ${t}`);
    });
    page.on('pageerror', (e: unknown) => {
      console.error(`  PAGE ERROR: ${e instanceof Error ? e.message : String(e)}`);
    });

    await page.goto(`http://localhost:${PORT}/bench/index.html`, { waitUntil: 'load' });
    const ready = await page.waitForFunction(
      () => (globalThis as Record<string, unknown>)['__benchReady'] === true,
      { timeout: 60_000 },
    ).then(() => true).catch(() => false);

    if (!ready) {
      const err = await page.evaluate(() => JSON.stringify((globalThis as any).__benchError ?? 'unknown'));
      throw new Error(`bench harness failed to initialise: ${err}`);
    }

    // Sanity: confirm the harness really has a WebGPU device before trusting
    // any number it reports.
    const probe = await page.evaluate(() => (globalThis as any).__probe());
    console.log(`  device: ${probe.adapter}  featureLevel: ${probe.featureLevel}`);
    if (probe.software) {
      console.log('  NOTE: running on a software adapter. Absolute ms are not');
      console.log('        representative of real hardware; ratios still are.\n');
    }

    // Build every scene once, for both engines. Construction is not what is
    // being measured, and rebuilding it per round both wastes the run and
    // perturbs the device.
    for (const spec of SCENES) {
      for (const engine of ['apse', 'three'] as const) {
        await page.evaluate(
          (e: string, s: SceneSpec) => (globalThis as any).__build(e, s),
          engine, spec,
        );
      }
    }
    console.log('  scenes built');

    for (const spec of SCENES) {
      const r = await page.evaluate((s: SceneSpec) => {
        (globalThis as any).__validationErrors.length = 0;
        return (globalThis as any).__run(s);
      }, spec);
      results.push(r);
      printRow(r);
      const errs = await page.evaluate(() => (globalThis as any).__validationErrors.slice());
      if (errs.length > 0) {
        console.log(`        ${errs.length} validation error(s): ${String(errs[0]).split('\n')[0].slice(0, 150)}`);
      }
    }


    // A pixel readback proves the frame was not black. A benchmark that
    // measures a renderer drawing nothing would otherwise report very good
    // numbers, which is the worst possible failure mode for a benchmark.
    const shot = await page.evaluate(() => (globalThis as any).__readback());

    // A validation error invalidates a pass without throwing, so a benchmark can
    // happily measure a renderer that is drawing nothing. Fail on any of them.
    const gpuErrors = await page.evaluate(() => (globalThis as any).__validationErrors ?? []);
    if (gpuErrors.length > 0) {
      for (const m of gpuErrors.slice(0, 2)) console.error(`  GPU VALIDATION:\n${String(m).slice(0, 1400)}\n`);
      throw new Error(`${gpuErrors.length} WebGPU validation error(s); measurements above are meaningless`);
    }

    if (shot.nonBackground === 0) {
      throw new Error(
        `readback produced a fully background-coloured frame — nothing drew ` +
        `(max luma ${shot.maxLuma}, ${shot.draws} draws, ${shot.format}, ` +
        `mesh ${JSON.stringify(shot.mesh)}, wgsl ${shot.wgslLen}b)`,
      );
    }
    console.log(`\n  readback: ${shot.nonBackground.toLocaleString()} / ${shot.total.toLocaleString()} pixels ` +
      `drawn (${(shot.coverage * 100).toFixed(1)}% coverage), max luma ${shot.maxLuma}`);
    console.log(`             format ${shot.format}, ${shot.draws} draw calls`);

    if (shot.png) {
      const file = join(SHOTS, 'lit-cube.png');
      writeFileSync(file, Buffer.from(shot.png, 'base64'));
      console.log(`  screenshot: ${file}`);
    }

    // three.js baseline in the same process, same device.
    // Instanced scenes are implemented (`src/geometry/instanced.ts`, 72 tests)
    // but are not yet reachable through the scene graph, so they are not
    // benchmarked here. A row that prints 0 draws is worse than no row.

    for (const spec of SCENES) {
      const r = await page.evaluate((s: SceneSpec) => (globalThis as any).__runThree(s), spec);
      results.push(r);
      printRow(r);
    }
  } finally {
    await browser.close();
    server.stop(true);
  }

  const report = {
    generatedAt: new Date().toISOString(),
    node: process.version,
    results,
    comparison: compare(results),
  };
  writeFileSync(join(SHOTS, 'report.json'), JSON.stringify(report, null, 2));
  console.log(`\n  report:  ${join(SHOTS, 'report.json')}\n`);
  summarise(report.comparison);
}

function compare(all: RunResult[]): Record<string, { apse: number; three: number; ratio: number } | null> {
  const out: Record<string, { apse: number; three: number; ratio: number } | null> = {};
  for (const spec of SCENES) {
    const a = all.find((r) => r.engine === 'apse' && r.scene === spec.name);
    const t = all.find((r) => r.engine === 'three' && r.scene === spec.name);
    if (a === undefined || t === undefined || a.error !== null || t.error !== null) {
      out[spec.name] = null;
      continue;
    }
    out[spec.name] = { apse: a.p50CpuMs, three: t.p50CpuMs, ratio: t.p50CpuMs / a.p50CpuMs };
  }
  return out;
}

function summarise(c: Record<string, { ratio: number } | null>): void {
  const rows = Object.entries(c).filter(([, v]) => v !== null) as [string, { ratio: number }][];
  if (rows.length === 0) return;
  console.log('  CPU time per frame, median — apse vs three.js (same device, same process)\n');
  for (const [name, v] of rows) {
    const bar = v.ratio >= 1
      ? `${v.ratio.toFixed(1)}x faster`
      : `${(1 / v.ratio).toFixed(1)}x SLOWER`;
    const colour = v.ratio >= 1.5 ? '\x1b[32m' : v.ratio >= 1 ? '\x1b[33m' : '\x1b[31m';
    console.log(`    ${name.padEnd(16)} ${colour}${bar.padEnd(14)}\x1b[0m`);
  }
  console.log();
}

function printRow(r: RunResult): void {
  if (r.error !== null) {
    console.log(`  ${r.engine.padEnd(5)} ${r.scene.padEnd(16)} ERROR: ${r.error}`);
    return;
  }
  const tag = r.engine === 'apse' ? '\x1b[36m' : '\x1b[90m';
  console.log(
    `  ${tag}${r.engine.padEnd(5)}\x1b[0m ${r.scene.padEnd(16)} ` +
    `p50 ${r.p50CpuMs.toFixed(3)}ms  p95 ${(r.p95CpuMs ?? 0).toFixed(3)}ms  ` +
    `draws ${String(r.drawCalls ?? 0).padStart(5)}  ` +
    `tris ${(r.triangles ?? 0).toLocaleString().padStart(11)}  ` +
    `heap ${(r.heapBytes / 1048576).toFixed(1)}MB  setup ${r.setupMs.toFixed(0)}ms  ` +
    (r.capacity === undefined ? '' : `objbuf ${r.capacity}  ${r.calls ?? ''}`),
  );
}

main().catch((e) => {
  console.error(`\n  benchmark failed: ${e instanceof Error ? e.message : String(e)}\n`);
  process.exit(1);
});
