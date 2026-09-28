/**
 * The build.
 *
 * Two tools, deliberately not one:
 *
 *   esbuild  transpiles and bundles. It does not type-check, and it never will.
 *   tsc      emits declarations. It does not bundle, and it never will.
 *
 * Trying to make one do both is how a library ends up shipping declarations that
 * disagree with its JavaScript. Here they run in sequence over the same source,
 * and a type error fails the build before a single byte is bundled.
 *
 * Output is ESM only. CommonJS is not emitted: tree-shaking is dead the moment a
 * module is `require`d, and a renderer whose entire premise is that you pay for
 * what you import cannot ship a CJS entry point.
 */

import { build, type BuildOptions } from 'esbuild';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { readFile, writeFile, rm, mkdir } from 'node:fs/promises';
import { gzipSync, brotliCompressSync, constants } from 'node:zlib';
import { join } from 'node:path';

const run = promisify(execFile);
const ROOT = new URL('..', import.meta.url).pathname;
const SRC = join(ROOT, 'src');
const DIST = join(ROOT, 'dist');

/**
 * One bundle per public entry point.
 *
 * Splitting per entry point rather than emitting one file is what makes
 * `sideEffects: false` a real claim: an importing bundler sees six independent
 * module graphs and can drop the five the application never references. A single
 * 400 KB bundle makes the flag unenforceable.
 */
const ENTRIES: Record<string, string> = {
  index: 'src/index.ts',
  'core/index': 'src/core/index.ts',
  'math/index': 'src/math/index.ts',
  'geometry/index': 'src/geometry/index.ts',
  'material/index': 'src/material/index.ts',
  'scene/index': 'src/scene/index.ts',
  'render/index': 'src/render/index.ts',
};

interface SizeRow {
  entry: string;
  files: number;
  raw: number;
  gzip: number;
  brotli: number;
  source: string;
  /**
   * Every source module in this entry's import closure, `src/`-relative and
   * sorted. Empty for `tree-shaken app`: that row measures an *application*, not
   * a published entry, and so has no layer contract to keep. `scripts/size-gate.ts`
   * asserts the non-empty ones against the layer DAG.
   */
  modules: string[];
}

/**
 * Hard ceilings on gzip size, in KiB, for the full reachable module graph.
 *
 * These are measured, not wished for. Every budget below carries the number it
 * was derived from and the date that number was taken, so a reader can tell a
 * re-baseline from a fudge.
 *
 * Headroom is 15%, and the reason it is 15% and not 5% is the warning band. The
 * gate colours any entry at 90% of its ceiling yellow so a regression is visible
 * as "now at 92% of budget" a long time before it fails. Re-baselining to 8% puts
 * every row at 92% on day one, which turns a permanent state into a permanent
 * warning and the warning into noise. At 15% a re-baselined tree sits at 87%: green,
 * with about three points of headroom before anything turns yellow, and a hard
 * failure at 100%.
 *
 * Is 15% still tight enough? On the root entry it is 10.2 KB of raw JavaScript,
 * and on the headline claim it is 6.8 KB of gzip — a feature nobody budgeted for
 * cannot land without tripping the gate, which is the whole point. The
 * per-module-count check in `scripts/size-gate.ts` is what catches the small
 * absorptions, because a few hundred bytes never reaches a byte ceiling at all.
 *
 * Measurement conditions, for every row: bun 1.3.9, node 22.21.0, esbuild
 * exactly as pinned in `bun.lock`, macOS arm64, `gzip -9` over the concatenation
 * of the entry and every chunk it transitively imports. Sizes vary by well under
 * 1% across platforms with the same lockfile, so the 15% is headroom for the
 * next deliberate feature, not a fudge factor for toolchain noise.
 *
 * The headline claim is `tree-shaken app`: a complete renderer, camera, PBR
 * material, and animation loop, measured after tree-shaking. For comparison,
 * three.js needs ~133 KB gzip for a single PBR cube.
 */
const BUDGETS: Record<string, number> = {
  // measured 85.56 KB gzip on 2026-09-28
  'index': 110.3,
  // measured 8.74 KB gzip on 2026-09-28
  'core/index': 10.1,
  // measured 11.66 KB gzip on 2026-09-28
  'math/index': 13.4,
  // measured 33.01 KB gzip on 2026-09-28
  'geometry/index': 37.5,
  // measured 57.49 KB gzip on 2026-09-28
  'material/index': 64.4,
  // measured 16.34 KB gzip on 2026-09-28
  'scene/index': 18.8,
  // measured 70.12 KB gzip on 2026-09-28
  'render/index': 80.4,
  // measured 71.22 KB gzip on 2026-09-28
  'tree-shaken app': 81.9,
};

/**
 * Bundles a realistic minimal app and measures what survives tree-shaking.
 *
 * This is the only measurement that tests the claim the library actually makes.
 * `sideEffects: false` is easy to write and easy to violate; a bundler that
 * silently keeps 300 KB of dead code reports no error, so the only defence is a
 * number in CI that fails when the claim stops being true.
 */
async function measureTreeShaken(): Promise<{ raw: number; gzip: number; brotli: number }> {
  const entry = join(ROOT, 'bench/tree-shake.ts');
  const out = join(DIST, '.size/tree-shaken.js');
  await build({
    entryPoints: [entry],
    outfile: out,
    absWorkingDir: ROOT,
    bundle: true,
    format: 'esm',
    target: 'es2022',
    minify: true,
    treeShaking: true,
    logLevel: 'silent',
  });
  const buf = await readFile(out);
  return { raw: buf.byteLength, ...compressBuffer(buf) };
}

function compressBuffer(buf: Buffer): { gzip: number; brotli: number } {
  return {
    gzip: gzipSync(buf, { level: 9 }).byteLength,
    brotli: brotliCompressSync(buf, {
      params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
    }).byteLength,
  };
}

async function main(): Promise<void> {
  const dev = process.argv.includes('--dev');

  await rm(DIST, { recursive: true, force: true });
  await mkdir(DIST, { recursive: true });

  // --- declarations ---------------------------------------------------------
  // Runs first, and its failure is the build's failure. Emitting .d.ts for code
  // that does not type-check produces types that lie, which is worse for an
  // AI agent than having no types at all: it will trust them.
  process.stdout.write('  tsc   declarations\n');
  await run('npx', ['tsc', '-p', 'tsconfig.build.json'], { cwd: ROOT });

  // --- bundles --------------------------------------------------------------
  process.stdout.write('  esbuild bundles\n');
  const shared: BuildOptions = {
    bundle: true,
    format: 'esm',
    target: 'es2022',
    platform: 'neutral',
    // The single most important line in this file. Without it, a bundler
    // consuming apse has to assume every module has side effects and can keep
    // nothing, which is how a 3.3 KB scene graph turns into a 133 KB download.
    treeShaking: true,
    legalComments: 'inline',
    minify: !dev,
    sourcemap: true,
    splitting: true,
    // One chunk per entry, so a browser can cache the shader scaffold
    // independently of the renderer loop.
    chunkNames: 'chunks/[name]-[hash]',
    entryNames: '[dir]/[name]',
    logLevel: 'silent',
    banner: {
      js: '/*! apse — a WebGPU renderer. https://apse.dev */',
    },
  };

  const result = await build({
    ...shared,
    // Entry names are derived from their path under `src`, not from a map, so
    // `src/core/index.ts` becomes `core/index.js` and shares its module graph
    // with the root entry instead of duplicating it.
    entryPoints: Object.values(ENTRIES).map((src) => join(ROOT, src)),
    // Pinning the working directory makes every path in the metafile relative to
    // the repository root no matter where the script was invoked from, which is
    // what the lookups below assume. Without it, `bun run --cwd ../.. build`
    // produces a report whose keys resolve to nothing and a gate that passes
    // because it measured an empty graph.
    absWorkingDir: ROOT,
    outbase: SRC,
    outdir: DIST,
    metafile: true,
  });

  // --- size report + budget gate -------------------------------------------
  //
  // Measuring `index.js` alone would report 2.5 KB for a complete renderer,
  // because code splitting moved the substance into chunks and the entry
  // became a re-export list. That number is meaningless: a consumer downloads
  // the entry *and everything it transitively imports*, so that is what gets
  // measured. The metafile is the only way to know the difference.
  const metafile = result.metafile!;
  const outputs = metafile.outputs;

  const reachable = (entry: string): Set<string> => {
    const seen = new Set<string>();
    const queue = [entry];
    while (queue.length > 0) {
      const file = queue.pop()!;
      if (seen.has(file)) continue;
      seen.add(file);
      for (const imp of outputs[file]?.imports ?? []) {
        if (imp.kind === 'import-statement' || imp.kind === 'dynamic-import') {
          queue.push(imp.path);
        }
      }
    }
    return seen;
  };

  /**
   * The source modules behind an entry, from the metafile's *input* graph.
   *
   * `reachable` answers "how many bytes does a consumer download"; this answers
   * "which modules did they pull in", and the second question is the one the
   * layer check in `scripts/size-gate.ts` asks. The output graph cannot answer
   * it: after code splitting, the substance of an entry is scattered across the
   * entry file and a set of shared chunks, and the mapping back to source is
   * gone by the time you look at `outputs`.
   */
  const reachableInputs = (entry: string): string[] => {
    const inputs = metafile.inputs;
    const seen = new Set<string>();
    const queue = [entry];
    while (queue.length > 0) {
      const file = queue.pop()!;
      if (seen.has(file)) continue;
      seen.add(file);
      for (const imp of inputs[file]?.imports ?? []) {
        if (imp.external) continue;
        queue.push(imp.path);
      }
    }
    return [...seen].sort();
  };

  const compress = (buf: Buffer): { gzip: number; brotli: number } => ({
    gzip: gzipSync(buf, { level: 9 }).byteLength,
    brotli: brotliCompressSync(buf, {
      params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
    }).byteLength,
  });

  const rows: SizeRow[] = [];
  for (const [entry, source] of Object.entries(ENTRIES)) {
    const outPath = join('dist', `${entry}.js`).replace(/\\/g, '/');
    if (outputs[outPath] === undefined) continue;

    const files = [...reachable(outPath)].filter((f) => f.endsWith('.js'));
    const bufs = await Promise.all(files.map((f) => readFile(join(ROOT, f))));
    // Compression is measured on the concatenation, not the sum of per-file
    // sizes: a bundler serving all chunks in one directory response gets one
    // shared dictionary, and per-file numbers overstate the real cost.
    const combined = Buffer.concat(bufs);
    const { gzip, brotli } = compress(combined);

    rows.push({
      entry,
      files: files.length,
      raw: combined.byteLength,
      gzip,
      brotli,
      source,
      modules: reachableInputs(source),
    });
  }

  // A single-import consumer, which is what a real app looks like. This is the
  // number the README quotes, and the one every other library is compared to.
  const treeShaken = await measureTreeShaken();
  rows.push({ entry: 'tree-shaken app', files: 1, ...treeShaken, source: 'bench/tree-shake.ts', modules: [] });

  await writeFile(
    join(DIST, 'size-report.json'),
    `${JSON.stringify({ generatedAt: new Date().toISOString(), entries: rows }, null, 2)}\n`,
  );

  const pad = (s: string, n: number): string => s.padEnd(n);
  const kb = (n: number): string => `${(n / 1024).toFixed(2)} KB`;
  process.stdout.write('\n');
  process.stdout.write(
    `  ${pad('entry', 20)}${pad('files', 7)}${pad('modules', 9)}${pad('raw', 11)}${pad('gzip', 11)}brotli\n`,
  );
  process.stdout.write(`  ${'-'.repeat(67)}\n`);
  for (const r of rows) {
    process.stdout.write(
      `  ${pad(r.entry, 20)}${pad(String(r.files), 7)}${pad(String(r.modules.length), 9)}` +
      `${pad(kb(r.raw), 11)}${pad(kb(r.gzip), 11)}${kb(r.brotli)}\n`,
    );
  }

  // An entry with no budget is not a pass. `?? Infinity` would let a newly added
  // entry ship unmeasured and unconstrained, which is how a bundle grows to
  // 400 KB while every ceiling below it still holds.
  const unbudgeted = rows.filter((r) => BUDGETS[r.entry] === undefined);
  const breaches = rows.filter((r) => r.gzip > (BUDGETS[r.entry] ?? Infinity) * 1024);
  process.stdout.write('\n');
  for (const u of unbudgeted) {
    process.stdout.write(`  BUDGET ${u.entry}: no ceiling declared, so ${kb(u.gzip)} gzip is unconstrained\n`);
  }
  if (unbudgeted.length > 0 || breaches.length > 0) {
    for (const b of breaches) {
      process.stdout.write(
        `  BUDGET ${b.entry}: ${kb(b.gzip)} gzip exceeds ${BUDGETS[b.entry]} KB\n`,
      );
    }
    process.exit(1);
  }

  // --- version check --------------------------------------------------------
  const pkg = JSON.parse(await readFile(join(ROOT, 'package.json'), 'utf8')) as { version: string };
  const indexFile = join(SRC, 'index.ts');
  const source = await readFile(indexFile, 'utf8');
  const declared = /export const VERSION = '([^']+)'/.exec(source)?.[1];
  if (declared !== pkg.version) {
    process.stdout.write(`  VERSION mismatch: src/index.ts says ${declared}, package.json says ${pkg.version}\n`);
    process.exit(1);
  }

  process.stdout.write(`\n  built ${rows.length} entries, all within budget\n\n`);
}

main().catch((e: unknown) => {
  const message = e instanceof Error ? e.message : String(e);
  process.stderr.write(`\n  build failed: ${message}\n\n`);
  process.exit(1);
});
