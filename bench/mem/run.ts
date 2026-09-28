/**
 * Per-object heap cost of two scene graphs, from a real Chrome heap snapshot.
 *
 * The question is narrow: how many bytes does one scene-graph node cost in apse
 * and in three.js, and is the "three.js Object3D is 31 properties and about
 * 1.8 KB" claim true, overstated, or false?
 *
 * Method, because the method is the answer:
 *
 *   - CDP `HeapProfiler` through puppeteer. `takeHeapSnapshot` returns V8's
 *     own object census; every number here is a `self_size` sum out of it.
 *     `performance.memory` is not used for anything load-bearing. It is
 *     whole-isolate and quantised, so a few-MB difference is inside its noise
 *     and a 3x correction would be invisible to it.
 *   - One library per page. A page holds exactly one scene-graph
 *     implementation, so a snapshot can be attributed to it.
 *   - Two object counts, 10,000 and 50,000, and the *slope* between them. A
 *     slope is far more robust than a single difference: everything that does
 *     not scale with the object count — the module code, the WebGPU device, the
 *     WebGL context, the snapshot's own fixed cost — cancels out of it exactly.
 *   - `HeapProfiler.collectGarbage` twice before every snapshot.
 *   - A zero-object page per configuration, which gives the fixed cost of the
 *     isolate and a second, independent slope to check the first against.
 *
 * This reports **self size**: bytes an object occupies, counted once, at its
 * own address. It is not retained size. Retained size is what an Object3D
 * keeps alive, which is every number its own children hold, and it is
 * meaningless as a per-object figure here because one object's retained set
 * contains the next object in the array. The slope is the fair comparison and
 * it is the headline; the per-class self sizes are supporting detail.
 *
 *   bun run bench/mem/run.ts
 */

import { serve } from 'bun';
import puppeteer from 'puppeteer';
import { writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { transform } from 'esbuild';

const ROOT = new URL('../..', import.meta.url).pathname;
const PORT = 8799;

const COUNTS = [0, 10_000, 50_000] as const;

/**
 * Three points, and the two-point slope is the headline. The 10k->50k slope is
 * the number to quote; 0->50k is printed beside it as a check, and a large
 * disagreement between the two would mean the cost is not linear in the object
 * count, which is worth knowing.
 */
const LOW = 10_000;
const HIGH = 50_000;

interface Config {
  readonly engine: 'apse' | 'three';
  readonly variant: string;
  readonly build: 'src' | 'dist';
  readonly label: string;
}

const CONFIGS: readonly Config[] = [
  { engine: 'apse', variant: 'stub', build: 'src', label: 'apse  src  stub' },
  { engine: 'apse', variant: 'real', build: 'src', label: 'apse  src  real' },
  { engine: 'apse', variant: 'collect', build: 'src', label: 'apse  src  +pool' },
  { engine: 'apse', variant: 'calib', build: 'src', label: 'apse  calib f32(16)' },
  { engine: 'apse', variant: 'bare', build: 'src', label: 'apse  src  Node' },
  { engine: 'apse', variant: 'real', build: 'dist', label: 'apse  dist real' },
  { engine: 'three', variant: 'bare', build: 'dist', label: 'three bare obj' },
  { engine: 'three', variant: 'none', build: 'dist', label: 'three Mesh' },
  { engine: 'three', variant: 'webgl', build: 'dist', label: 'three +WebGL' },
];

interface ClassStat {
  selfBytes: number;
  count: number;
  minSize: number;
  maxSize: number;
}

interface SnapshotAgg {
  totalSelfBytes: number;
  nodeCount: number;
  edgeCount: number;
  classes: Record<string, ClassStat>;
}

interface Point {
  readonly n: number;
  readonly totalSelfBytes: number;
  readonly usedJSHeapSize: number;
  readonly agg: SnapshotAgg;
}

interface SlopeEntry {
  readonly key: string;
  readonly bytesPerObject: number;
  readonly countPerObject: number;
  readonly lowBytes: number;
  readonly highBytes: number;
  readonly lowCount: number;
  readonly highCount: number;
}

interface Slope {
  readonly low: number;
  readonly high: number;
  readonly totalPerObject: number;
  readonly totalFromZero: number;
  readonly classes: readonly SlopeEntry[];
}

// ---------------------------------------------------------------------------
// Heap snapshot → self-size census
// ---------------------------------------------------------------------------

/**
 * String-valued nodes are bucketed. 50,000 UUIDs would otherwise put 50,000
 * distinct keys in the map and make the class breakdown unreadable; the bytes
 * still sum correctly, which is all the slope needs.
 */
const BUCKETED = new Set(['string', 'concatenated string', 'sliced string', 'number', 'symbol', 'bigint']);

/**
 * V8 names a closure's scope by an address: `Context / scope @36663`. The
 * address changes every run, so leaving it in makes the same class look like
 * two different classes across the two snapshots, and the per-class slopes stop
 * summing to the total slope.
 */
const normalize = (name: string) => name.replace(/ @\d+$/, '');

/** Size of the last snapshot's JSON, so the cost of this method is visible. */
let lastSnapshotJsonBytes = 0;

function parseSnapshot(json: string): SnapshotAgg {
  const snap = JSON.parse(json) as {
    snapshot: {
      meta: { node_fields: string[]; node_types: string[][] };
      node_count: number;
      edge_count: number;
    };
    nodes: number[];
    strings: string[];
  };

  const meta = snap.snapshot.meta;
  const fields = meta.node_fields;
  const width = fields.length;
  const iType = fields.indexOf('type');
  const iName = fields.indexOf('name');
  const iSelf = fields.indexOf('self_size');
  if (iType < 0 || iName < 0 || iSelf < 0) {
    throw new Error(`unexpected node_fields: ${fields.join(',')}`);
  }
  const typeNames = meta.node_types[iType];
  const nodes = snap.nodes;
  const strings = snap.strings;
  const count = snap.snapshot.node_count;

  const classes: Record<string, ClassStat> = Object.create(null);
  let total = 0;

  for (let i = 0, o = 0; i < count; i++, o += width) {
    const type = typeNames[nodes[o + iType]] as string;
    const self = nodes[o + iSelf] as number;
    const name = BUCKETED.has(type) ? '*' : normalize(strings[nodes[o + iName]] as string);
    const key = `${type}:${name}`;
    let s = classes[key];
    if (s === undefined) s = classes[key] = { selfBytes: 0, count: 0, minSize: self, maxSize: self };
    s.selfBytes += self;
    s.count++;
    if (self < s.minSize) s.minSize = self;
    if (self > s.maxSize) s.maxSize = self;
    total += self;
  }

  return {
    totalSelfBytes: total,
    nodeCount: count,
    edgeCount: snap.snapshot.edge_count,
    classes,
  };
}

async function takeSnapshot(page: any): Promise<SnapshotAgg> {
  const client = await page.createCDPSession();
  const chunks: string[] = [];
  try {
    // Subscribed before the command: the chunks arrive while takeHeapSnapshot
    // is still in flight, and a listener attached afterwards loses the first
    // (largest) ones.
    client.on('HeapProfiler.addHeapSnapshotChunk', (e: { chunk: string }) => chunks.push(e.chunk));
    await client.send('HeapProfiler.enable');
    await client.send('HeapProfiler.collectGarbage');
    await new Promise((r) => setTimeout(r, 300));
    await client.send('HeapProfiler.collectGarbage');
    await new Promise((r) => setTimeout(r, 300));
    await client.send('HeapProfiler.takeHeapSnapshot', {
      reportProgress: false,
      captureNumericValue: false,
      treatGlobalObjectsAsRoots: true,
    });
    const json = chunks.join('');
    chunks.length = 0;
    const agg = parseSnapshot(json);
    lastSnapshotJsonBytes = json.length;
    return agg;
  } finally {
    await client.detach().catch(() => undefined);
  }
}

// ---------------------------------------------------------------------------
// Slopes
// ---------------------------------------------------------------------------

function slopeOf(a: Point, b: Point): Slope {
  const dn = b.n - a.n;
  const totalPerObject = (b.totalSelfBytes - a.totalSelfBytes) / dn;

  const keys = new Set([...Object.keys(a.agg.classes), ...Object.keys(b.agg.classes)]);
  const classes: SlopeEntry[] = [];
  for (const key of keys) {
    const lo = a.agg.classes[key];
    const hi = b.agg.classes[key];
    if ((hi?.count ?? 0) - (lo?.count ?? 0) < 1) continue;
    const perObject = ((hi?.selfBytes ?? 0) - (lo?.selfBytes ?? 0)) / dn;
    if (Math.abs(perObject) < 8) continue;
    classes.push({
      key,
      bytesPerObject: perObject,
      countPerObject: ((hi?.count ?? 0) - (lo?.count ?? 0)) / dn,
      lowBytes: lo?.selfBytes ?? 0,
      highBytes: hi?.selfBytes ?? 0,
      lowCount: lo?.count ?? 0,
      highCount: hi?.count ?? 0,
    });
  }
  classes.sort((x, y) => y.bytesPerObject - x.bytesPerObject);

  return {
    low: a.n,
    high: b.n,
    totalPerObject,
    totalFromZero: b.totalSelfBytes,
    classes,
  };
}

const fmt = (b: number) => `${(b / 1024).toFixed(1)} KB`;
const mb = (b: number) => `${(b / 1048576).toFixed(2)} MB`;

// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const server = serve({
    port: PORT,
    idleTimeout: 0,
    routes: {
      '/*': async (req) => {
        const path = new URL(req.url).pathname;
        // The apse `src` mode comes first, because `dist/index.js` is minified,
        // so every class in a snapshot of it is called `g` or `e` and the
        // per-class breakdown is unreadable. Transpiling the real sources on the
        // way out keeps the class names — `MeshNode`, `PooledDrawItem` — which
        // is the difference between a number and a number you can check.
        //
        // Import specifiers are left alone, so a `from './node.ts'` inside
        // src/index.ts comes back to this same route and is transpiled too.
        // Chrome enforces the MIME type, not the extension, so a `.ts` URL
        // served as JavaScript is a module and not a syntax error.
        if (path.startsWith('/src/') && path.endsWith('.ts')) {
          const src = await Bun.file(join(ROOT, path.slice(1))).text();
          const { code } = await transform(src, {
            loader: 'ts',
            format: 'esm',
            target: 'es2022',
            sourcefile: path,
          });
          return new Response(code, { headers: { 'content-type': 'text/javascript; charset=utf-8' } });
        }
        const file = Bun.file(join(ROOT, path === '/' ? 'bench/mem/index.html' : path.slice(1)));
        if (await file.exists()) {
          // A module script must be served with a JavaScript MIME type. Chrome
          // reports a wrong one only as a failed dynamic import, which is a
          // confusing way to learn about a header.
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

  console.log(`\n  per-object heap cost\n  server: http://localhost:${PORT}\n`);

  const browser = await puppeteer.launch({
    headless: true,
    executablePath: process.env['CHROME_PATH'] ?? undefined,
    channel: process.env['CHROME_PATH'] === undefined ? 'chrome' : undefined,
    protocolTimeout: 900_000,
    args: [
      '--no-sandbox',
      '--disable-dev-shm-usage',
      // performance.memory is still reported, but only as a cross-check, and
      // it is quantised unless the flag is set.
      '--enable-precise-memory-info',
    ],
  });

  const report: Record<string, unknown>[] = [];
  const allSlopes: { config: Config; slope: Slope; zero: number; used: number; points: Point[] }[] = [];

  try {
    // Structural facts first: cheap, and they are the part of the claim that
    // does not need a 50,000-object page to check.
    console.log('  structure');
    const structurePages: Record<string, unknown> = {};
    for (const engine of ['apse', 'three'] as const) {
      const page = await browser.newPage();
      page.on('pageerror', (e: unknown) => console.error(`  PAGE ERROR: ${e instanceof Error ? e.message : String(e)}`));
      await page.goto(`http://localhost:${PORT}/bench/mem/index.html?engine=${engine}&n=0`, { waitUntil: 'load' });
      await page.waitForFunction(() => (globalThis as any).__memReady === true, { timeout: 60_000 });
      const info = await page.evaluate(() => (globalThis as any).__memInfo());
      structurePages[engine] = info;
      await page.close();
    }
    for (const engine of ['apse', 'three'] as const) {
      const info = structurePages[engine] as { classes: { label: string; ownProps: number; perInstanceClosures: number; closurePaths: string[]; ownFunctionProps: number }[] };
      console.log(`\n    ${engine}`);
      console.log(`      ${'class'.padEnd(14)}${'own props'.padStart(10)}${'of which fns'.padStart(14)}${'per-inst closures'.padStart(20)}`);
      for (const c of info.classes) {
        console.log(`      ${c.label.padEnd(14)}${String(c.ownProps).padStart(10)}${String(c.ownFunctionProps).padStart(14)}${String(c.perInstanceClosures).padStart(20)}`);
        if (c.closurePaths.length > 0) {
          console.log(`      ${''.padEnd(14)}${c.closurePaths.map((p) => p.split('.').slice(1).join('.')).join(', ')}`);
        }
      }
    }
    console.log();
    report.push({ kind: 'structure', structure: structurePages });

    for (const config of CONFIGS) {
      const points: Point[] = [];
      for (const n of COUNTS) {
        // A fresh page per point. Reusing one and rebuilding would leave the
        // previous scene reachable from the module scope, and the snapshot
        // would sum both.
        const page = await browser.newPage();
        page.on('pageerror', (e: unknown) => console.error(`  PAGE ERROR: ${e instanceof Error ? e.message : String(e)}`));
        page.on('console', (m: { type: () => string; text: () => string }) => {
          if (m.type() === 'error') console.error(`  console: ${m.text()}`);
        });
        const t0 = Date.now();
        await page.goto(
          `http://localhost:${PORT}/bench/mem/index.html?engine=${config.engine}&n=${n}` +
          `&variant=${config.variant}&build=${config.build}`,
          { waitUntil: 'load' },
        );
        await page.waitForFunction(() => (globalThis as any).__memReady === true, { timeout: 60_000 });
        const built = await page.evaluate(() => (globalThis as any).__memBuild());
        const agg = await takeSnapshot(page);
        await page.close();
        points.push({
          n,
          totalSelfBytes: agg.totalSelfBytes,
          usedJSHeapSize: (built as { usedJSHeapSize: number }).usedJSHeapSize,
          agg,
        });
        console.log(
          `  ${config.label.padEnd(20)} n=${String(n).padStart(6)}  ` +
          `snapshot ${fmt(agg.totalSelfBytes).padStart(12)}  ` +
          `${String(agg.nodeCount).padStart(9)} nodes  ` +
          `json ${mb(lastSnapshotJsonBytes).padStart(9)}  ` +
          `${(built as { held: number }).held} held  ${((Date.now() - t0) / 1000).toFixed(1)}s`,
        );
      }
      const byN = new Map(points.map((p) => [p.n, p]));
      const slope = slopeOf(byN.get(LOW)!, byN.get(HIGH)!);
      const zero = byN.get(0)!;
      allSlopes.push({ config, slope, zero: zero.totalSelfBytes, used: byN.get(LOW)!.usedJSHeapSize, points });
      report.push({ kind: 'config', config, points, slope });
    }
  } finally {
    await browser.close();
    server.stop(true);
  }

  // -------------------------------------------------------------------------

  console.log(`\n  bytes per object — slope of total snapshot self_size, ${LOW} → ${HIGH}\n`);
  console.log(`    ${'configuration'.padEnd(20)}${'0 obj'.padStart(12)}${'10k'.padStart(12)}${'50k'.padStart(12)}${'slope B/obj'.padStart(14)}${'0→50k B/obj'.padStart(14)}${'perf.mem 10k'.padStart(15)}`);
  for (const s of allSlopes) {
    const p10k = s.slope.totalPerObject * LOW + s.zero;
    const p50k = s.slope.totalPerObject * HIGH + s.zero;
    const zeroSlope = (p50k - s.zero) / HIGH;
    console.log(
      `    ${s.config.label.padEnd(20)}${mb(s.zero).padStart(12)}${mb(p10k).padStart(12)}${mb(p50k).padStart(12)}` +
      `${s.slope.totalPerObject.toFixed(0).padStart(14)}${zeroSlope.toFixed(0).padStart(14)}${mb(s.used).padStart(15)}`,
    );
  }

  for (const s of allSlopes) {
    console.log(`\n  ${s.config.label} — per-object self size by class`);
    console.log(`    ${'class'.padEnd(34)}${'@10k bytes'.padStart(12)}${'@50k bytes'.padStart(12)}${'@10k n'.padStart(9)}${'@50k n'.padStart(9)}${'B/obj'.padStart(8)}${'cnt/obj'.padStart(8)}${'B/node'.padStart(8)}  sizes`);
    let sum = 0;
    for (const c of s.slope.classes) sum += c.bytesPerObject;
    for (const c of s.slope.classes.slice(0, 20)) {
      const a = s.points.find((p) => p.n === LOW)!.agg.classes[c.key];
      const b = s.points.find((p) => p.n === HIGH)!.agg.classes[c.key];
      const size = a && b
        ? (a.minSize === a.maxSize && b.minSize === b.maxSize ? `${a.minSize}` : `${a.minSize}..${b.maxSize}`)
        : '?';
      console.log(
        `    ${c.key.padEnd(34)}${fmt(c.lowBytes).padStart(12)}${fmt(c.highBytes).padStart(12)}` +
        `${String(c.lowCount).padStart(9)}${String(c.highCount).padStart(9)}` +
        `${c.bytesPerObject.toFixed(1).padStart(8)}${c.countPerObject.toFixed(2).padStart(8)}` +
        `${((b?.selfBytes ?? 0) / Math.max(1, b?.count ?? 1)).toFixed(1).padStart(8)}  ${size}`,
      );
    }
    console.log(`    ${'— sum of listed classes'.padEnd(34)}${''.padStart(12)}${''.padStart(12)}${''.padStart(9)}${''.padStart(9)}${sum.toFixed(0).padStart(8)}   (total slope ${s.slope.totalPerObject.toFixed(0)})`);
  }

  const file = join(tmpdir(), 'apse-mem-report.json');
  writeFileSync(file, JSON.stringify({ generatedAt: new Date().toISOString(), report }, null, 2));
  console.log(`\n  report: ${file}\n`);
}

main().catch((e) => {
  console.error(`\n  mem bench failed: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}\n`);
  process.exit(1);
});
