/**
 * The size gate.
 *
 * `scripts/build.ts` builds, measures every entry's reachable graph, writes
 * `dist/size-report.json`, and exits non-zero over budget. This script does not
 * repeat any of that and it never runs esbuild — re-bundling here would only
 * reproduce the build's answer with more moving parts.
 *
 * The build can check its own arithmetic. It cannot check its own claims, because
 * the claim is about a *consumer's* download, not about the bytes on disk here.
 * Five things fall in that gap, and they are what this file is for:
 *
 *   1. **The headline number.** A complete renderer, camera, PBR material, scene
 *      graph, and animation loop, tree-shaken, under a hard ceiling. The budget
 *      is a named constant below, and the report prints the percentage of it
 *      used on every run — so a regression is visible as "now at 92% of budget"
 *      a long time before it fails at 100%.
 *   2. **Direction.** The same number against the recorded three.js baseline, so
 *      a regression in either direction shows up. Halving our size is as much a
 *      regression as doubling it.
 *   3. **That the size is real.** A bundle that leaked the five unused
 *      primitives is a bundle whose size does not describe what anybody
 *      downloads, and the size number would still look fine. So the tree-shaking
 *      *claim* is asserted, not inferred from a byte count — see
 *      {@link PRIMITIVE_MARKERS}. There is a positive control alongside it,
 *      because a check that cannot fail is not a check.
 *   4. **That the import graph has not been quietly rearranged.** Every entry is
 *      supposed to reach a known set of layers and a known number of modules. An
 *      entry that absorbs a dependency costs a few hundred bytes and trips no
 *      budget, so the shape of the graph is asserted directly — see
 *      {@link LAYER_GRAPH} and {@link checkModuleGraph}.
 *   5. **That the package is installable.** Every `exports` subpath and every
 *      `typesVersions` target must have a built file. A broken exports map is
 *      invisible until somebody runs `npm install`, at which point it is
 *      somebody else's afternoon.
 *
 * Run it under bun. Node-only, no new dependencies:
 *
 *     bun run scripts/size-gate.ts
 */

import { readFile, access, readdir } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';

// ---------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------

/**
 * The headline claim, in KiB gzip: a complete renderer + camera + PBR material
 * + scene graph + animation loop, tree-shaken, from a single import.
 *
 * Deliberately the same number as `BUDGETS['tree-shaken app']` in
 * `scripts/build.ts`. {@link checkBudgetAgrees} re-reads that file and fails if
 * the two ever drift, because two gates that disagree are worse than one gate.
 */
const TREE_SHAKEN_GZIP_BUDGET_KB = 52.4;

/**
 * Per-entry ceilings, in KiB gzip. Mirrors `BUDGETS` in `scripts/build.ts`.
 * The build fails over these; this script reports against them, so an entry
 * that has quietly outgrown its ceiling is visible before the next build fails.
 *
 * Same numbers, same provenance, same reason — see the table in `build.ts` for
 * the measured values, the date each was taken, and why the headroom is 15% and
 * not 5%. {@link checkBudgetAgrees} fails if these ever drift apart.
 */
const BUDGETS: Readonly<Record<string, number>> = {
  'index': 78.2,
  'core/index': 8.9,
  'math/index': 7.6,
  'geometry/index': 20.7,
  'material/index': 43.7,
  'scene/index': 12.1,
  'render/index': 57.8,
  'tree-shaken app': 52.4,
};

/**
 * The layer DAG, plus the exact number of source modules each entry may reach.
 *
 * This is the "no entry has silently absorbed a dependency" check, and it is the
 * one a byte count cannot do. A module that starts importing the renderer costs
 * maybe 300 bytes gzip, so no ceiling trips when `src/core/uniform.ts` grows an
 * import of `src/render/target.ts` — the entry still fits, the budget still
 * holds, and `import { fail } from 'apse/core'` has quietly become a dependency
 * on the whole renderer. The only thing that catches it is asserting the shape
 * of the import graph itself.
 *
 * The edges are the real ones, read off the metafile's input graph, and they
 * form a DAG. `core` depends on nothing, which is what makes `apse/core` usable
 * without the rest of the library. Everything else hangs off it:
 *
 *     core
 *      ├── math ──┬── geometry ──┐
 *      │          │              ├── material ──┐
 *      │          └── scene ─────┴──────────────┴── render
 *      │
 *      └── (index, the public barrel, which is allowed to reach all of it)
 *
 * `root` is `src/index.ts`, and no subpath may reach it — a subpath that imports
 * the barrel has imported everything, which defeats the point of subpaths. That
 * is why `src/core/index.ts` re-exports the six core modules individually rather
 * than the barrel, and why `src/index.ts` does the same for every layer.
 *
 * `modules` is the exact reachable-module count, not a ceiling. Bumping it is a
 * deliberate act: either a file was added to a layer, or a file was added
 * somewhere it does not belong and the layer check above caught the second case
 * first. It also cannot drift quietly in the way a byte budget can, because a
 * number has to be edited, not re-measured.
 *
 * Counts as of 2026-09-28, same build that produced the budgets above.
 */
const LAYER_GRAPH: Readonly<Record<string, { readonly layers: readonly string[]; readonly modules: number }>> = {
  'core/index': { layers: ['core'], modules: 7 },
  'math/index': { layers: ['core', 'math'], modules: 7 },
  'scene/index': { layers: ['core', 'math', 'scene'], modules: 8 },
  'geometry/index': { layers: ['core', 'math', 'geometry'], modules: 16 },
  'material/index': { layers: ['core', 'math', 'geometry', 'material'], modules: 18 },
  'render/index': { layers: ['core', 'math', 'geometry', 'material', 'scene', 'render'], modules: 25 },
  'index': {
    layers: ['core', 'math', 'geometry', 'material', 'scene', 'render', 'root'],
    modules: 39,
  },
};

/**
 * The row that measures an application rather than a published entry, and so is
 * the one entry without a layer contract.
 */
const APP_ENTRY = 'tree-shaken app';

/**
 * The comparison point for the size claim: three.js at ~133 KB gzip for a
 * single PBR cube, from three.js's own published build. Not measured here and
 * not re-measurable here — it is a recorded external figure, and it is recorded
 * so the ratio is printed on every run rather than quoted once in a README.
 */
const THREE_JS_GZIP_BASELINE_KB = 133;

/**
 * Printed when our own size stops being the headline. Below this ratio the size
 * advantage is still the story; above it, something in the reachable graph grew
 * and the number is no longer the thing to lead with.
 */
const MIN_INTERESTING_RATIO = 2;

/** `measureTreeShaken()` in `scripts/build.ts` writes here. */
const TREE_SHAKEN_BUNDLE = 'dist/.size/tree-shaken.js';

/** The app the size claim is about, and the file whose imports we read. */
const TREE_SHAKE_APP = 'bench/tree-shake.ts';

const ROOT = new URL('..', import.meta.url).pathname;
const REPORT = join(ROOT, 'dist/size-report.json');
const PKG = join(ROOT, 'package.json');
const BUILD = join(ROOT, 'scripts/build.ts');
const SRC = join(ROOT, 'src');

/**
 * Markers for the six primitives, and the only source of truth for which
 * primitive leaked.
 *
 * A size number cannot detect a tree-shaking failure: the bundle still builds,
 * the build still passes, and the report still prints a plausible figure. So the
 * claim is checked directly, against strings that only exist if a primitive's
 * body survived into the bundle.
 *
 * esbuild does not mangle property names and does not rename destructured
 * shorthand keys, so `capped` survives as `capped:d=!0` and `widthSegments` as
 * both `t.widthSegments` and `widthSegments:s=1`. Whitespace is stripped before
 * matching so a future `name: "sphere"` still matches `name:"sphere"`.
 *
 * `box` is the positive control: the app imports it, so its marker *must* be
 * present. If it is not, the marker set is stale or the bundle is empty, and
 * every "0 found" below it is meaningless — so that is a failure, not a pass.
 */
const PRIMITIVE_MARKERS: Readonly<Record<string, readonly string[]>> = {
  box: ['name:"box"'],
  sphere: ['widthSegments', 'heightSegments', 'name:"sphere"'],
  plane: ['depthSegments', 'widthSegments'],
  grid: ['depthSegments', 'name:"grid"'],
  torus: ['radialSegments', 'tubularSegments', 'name:"torus"'],
  cylinder: ['capped', 'radiusTop', 'radiusBottom', 'name:"cylinder"'],
};

// ---------------------------------------------------------------------------
// Report shape
// ---------------------------------------------------------------------------

interface SizeRow {
  entry: string;
  files: number;
  raw: number;
  gzip: number;
  brotli: number;
  source: string;
  /** Source modules in this entry's import closure, `src/`-relative. See LAYER_GRAPH. */
  modules: string[];
}

interface SizeReport {
  generatedAt: string;
  entries: SizeRow[];
}

interface Failure {
  readonly what: string;
  readonly action: string;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

const DIM = '\x1b[90m';
const BOLD = '\x1b[1m';
const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const OFF = '\x1b[0m';

const out = (line = ''): void => { process.stdout.write(`${line}\n`); };

const pad = (s: string, n: number): string => s.padEnd(n);
const rpad = (s: string, n: number): string => s.padStart(n);

/** KiB to two decimals, matching the build's own formatting. */
const kb = (bytes: number): string => `${(bytes / 1024).toFixed(2)} KB`;

/** For constants that are already in KiB, so the number is not divided again. */
const kib = (value: number): string => `${value.toFixed(2)} KB`;

const exists = async (path: string): Promise<boolean> => {
  try { await access(path); return true; } catch { return false; }
};

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

/**
 * Reads the report the build wrote, and refuses to proceed without it.
 *
 * A gate that reports a pass when it found nothing to measure is the worst
 * possible failure mode, so a missing report is a failure with the one command
 * that fixes it.
 */
async function loadReport(failures: Failure[]): Promise<SizeReport | null> {
  if (!(await exists(REPORT))) {
    failures.push({
      what: 'dist/size-report.json does not exist, so there is nothing to verify.',
      action: 'run `bun run build` first — it writes the report the gate reads.',
    });
    return null;
  }
  let report: SizeReport;
  try {
    report = JSON.parse(await readFile(REPORT, 'utf8')) as SizeReport;
  } catch (e: unknown) {
    failures.push({
      what: `dist/size-report.json is not valid JSON: ${e instanceof Error ? e.message : String(e)}`,
      action: 'delete dist/ and run `bun run build`.',
    });
    return null;
  }
  if (!Array.isArray(report.entries) || report.entries.length === 0) {
    failures.push({
      what: 'dist/size-report.json has no entries, so no size was measured.',
      action: 'delete dist/ and run `bun run build`.',
    });
    return null;
  }
  return report;
}

/** The tree-shaken row, identified by the app the build measured. */
function treeShakenRow(report: SizeReport): SizeRow | null {
  return report.entries.find((r) => r.entry === 'tree-shaken app' && r.source === TREE_SHAKE_APP) ??
    report.entries.find((r) => r.entry === 'tree-shaken app') ??
    null;
}

/**
 * Asserts the headline claim and prints how much room is left.
 *
 * The percentage is the point. A gate that only speaks at 100% gives one bit per
 * run; one that prints "91% of budget" every time gives an agent watching for a
 * regression somewhere to see it.
 */
async function checkHeadline(report: SizeReport, failures: Failure[]): Promise<void> {
  const row = treeShakenRow(report);
  if (row === null) {
    failures.push({
      what: 'The size report has no "tree-shaken app" row, so the headline claim cannot be measured.',
      action: 'check that scripts/build.ts still writes the bench/tree-shake.ts measurement.',
    });
    return;
  }

  const limit = TREE_SHAKEN_GZIP_BUDGET_KB * 1024;
  const used = (row.gzip / limit) * 100;
  const headroom = limit - row.gzip;
  const ratio = (THREE_JS_GZIP_BASELINE_KB * 1024) / row.gzip;

  out('  headline claim');
  out(`    tree-shaken realistic app  ${BOLD}${kb(row.gzip)} gzip${OFF}  ${DIM}(${row.raw.toLocaleString()} B raw, ${kb(row.brotli)} brotli)${OFF}`);
  out(`    ceiling                   ${BOLD}${kib(TREE_SHAKEN_GZIP_BUDGET_KB)}${OFF}   ${DIM}(TREE_SHAKEN_GZIP_BUDGET_KB in scripts/size-gate.ts)${OFF}`);
  out(`    used                      ${used >= 90 ? YELLOW : ''}${used.toFixed(1)}% of budget${OFF}  ${DIM}${headroom >= 0 ? `${kb(headroom)} left` : `${kb(-headroom)} over`}${OFF}`);
  out(`    three.js baseline         ${kib(THREE_JS_GZIP_BASELINE_KB)} gzip  —  ${ratio >= MIN_INTERESTING_RATIO ? `${BOLD}${ratio.toFixed(2)}x smaller` : `${YELLOW}${ratio.toFixed(2)}x smaller`}${OFF}`);
  out();

  if (row.gzip > limit) {
    failures.push({
      what: `The headline size claim is false: a tree-shaken realistic app is ${kb(row.gzip)} gzip, and the claim is that it is under ${kib(TREE_SHAKEN_GZIP_BUDGET_KB)}.`,
      action: 'trim the reachable graph, or raise TREE_SHAKEN_GZIP_BUDGET_KB and BUDGETS in scripts/build.ts deliberately — and update the number in README.md, which is the claim this gate exists to keep true.',
    });
  }
  if (ratio < MIN_INTERESTING_RATIO) {
    failures.push({
      what: `tree-shaken app is ${kb(row.gzip)} gzip, only ${ratio.toFixed(2)}x the ${kib(THREE_JS_GZIP_BASELINE_KB)} three.js baseline.`,
      action: 'something in the reachable graph grew. Check what the last few material, scene, or render modules added before re-raising the budget.',
    });
  }

  // A report that no longer describes the file on disk is worse than no report.
  const bundlePath = join(ROOT, TREE_SHAKEN_BUNDLE);
  if (await exists(bundlePath)) {
    const onDisk = (await readFile(bundlePath)).byteLength;
    if (onDisk !== row.raw) {
      failures.push({
        what: `${TREE_SHAKEN_BUNDLE} is ${onDisk.toLocaleString()} B but the report says ${row.raw.toLocaleString()} B. The report is stale.`,
        action: 'run `bun run build` and re-run this gate.',
      });
    }
  } else {
    failures.push({
      what: `${TREE_SHAKEN_BUNDLE} does not exist, so the tree-shaking check has no bundle to read.`,
      action: 'run `bun run build` — scripts/build.ts writes that bundle while measuring it.',
    });
  }
}

/** Prints every entry against its ceiling. Informational, except for breaches. */
function checkEntries(report: SizeReport, failures: Failure[]): void {
  out('  entries');
  out(`  ${DIM}${pad('entry', 20)}${pad('files', 6)}${rpad('gzip', 10)}${rpad('brotli', 10)}${rpad('budget', 8)} used${OFF}`);
  out(`  ${DIM}${'-'.repeat(65)}${OFF}`);

  for (const row of report.entries) {
    const budget = BUDGETS[row.entry];
    if (budget === undefined) {
      failures.push({
        what: `The size report has an entry "${row.entry}" with no budget in scripts/size-gate.ts.`,
        action: 'add it to BUDGETS, or remove the entry from scripts/build.ts ENTRIES.',
      });
      continue;
    }
    const limit = budget * 1024;
    const used = (row.gzip / limit) * 100;
    const over = row.gzip > limit;
    out(
      `  ${pad(row.entry, 20)}${pad(String(row.files), 6)}` +
      `${rpad(kb(row.gzip), 10)}${rpad(kb(row.brotli), 10)}` +
      `${rpad(`${budget} KB`, 8)} ${over ? RED : used >= 90 ? YELLOW : ''}${used.toFixed(0)}%${over ? `  OVER by ${kb(row.gzip - limit)}` : OFF}`,
    );
    if (over) {
      failures.push({
        what: `${row.entry} is ${kb(row.gzip)} gzip, over its ${budget} KB budget by ${kb(row.gzip - limit)}.`,
        action: 'trim the reachable graph, or raise the budget in scripts/build.ts with a comment explaining why.',
      });
    }
  }
  for (const entry of Object.keys(BUDGETS)) {
    if (!report.entries.some((r) => r.entry === entry)) {
      failures.push({
        what: `No size was reported for "${entry}", which has a ${BUDGETS[entry]} KB budget.`,
        action: 'check that scripts/build.ts still bundles this entry — an entry that stopped building is a silent hole in the gate.',
      });
    }
  }
  out();
}

/**
 * The named budgets in `scripts/build.ts`, read as text.
 *
 * The build is the thing that actually fails, so it is the thing that owns the
 * numbers. But it also cannot be imported — it calls `main()` at module scope —
 * and a literal copy here would drift silently the first time someone raised a
 * ceiling. So the file is parsed instead. Reading another script's source is
 * unglamorous; having the gate report 90% of a ceiling that the build enforces at
 * a different one is worse.
 */
async function checkBudgetAgrees(failures: Failure[]): Promise<void> {
  const source = await readFile(BUILD, 'utf8');
  const start = source.indexOf('const BUDGETS');
  if (start < 0) {
    failures.push({
      what: 'Could not find a BUDGETS table in scripts/build.ts.',
      action: 'if it was renamed, update checkBudgetAgrees() in scripts/size-gate.ts.',
    });
    return;
  }
  const body = source.slice(start, source.indexOf('};', start));
  const parsed = new Map<string, number>();
  // Anchored to the whole line so the measured-size and date comments sitting
  // beside each budget cannot be mistaken for budgets themselves. Decimals are
  // allowed because a ceiling tight enough to catch a real regression is closer
  // than 1 KB on the small entries, and rounding it up to the next whole KiB
  // would hand back a fifth of the headroom on `apse/math`.
  for (const m of body.matchAll(/^\s*'([^']+)'\s*:\s*(\d+(?:\.\d+)?)\s*,?\s*$/gm)) {
    parsed.set(m[1]!, Number(m[2]));
  }

  for (const [entry, kb] of Object.entries(BUDGETS)) {
    const theirs = parsed.get(entry);
    if (theirs === undefined) {
      failures.push({
        what: `scripts/build.ts has no budget for "${entry}"; scripts/size-gate.ts says ${kb} KB.`,
        action: 'add the entry to BUDGETS in scripts/build.ts, or remove it here.',
      });
    } else if (Math.abs(theirs - kb) > 1e-9) {
      failures.push({
        what: `Budget drift for "${entry}": scripts/build.ts says ${theirs} KB, scripts/size-gate.ts says ${kb} KB.`,
        action: 'make the two constants the same number, or delete the copy in size-gate.ts and read the build\'s table instead.',
      });
    }
  }
  for (const entry of parsed.keys()) {
    if (!(entry in BUDGETS)) {
      failures.push({
        what: `scripts/build.ts budgets "${entry}" but scripts/size-gate.ts has no entry for it.`,
        action: 'add it to BUDGETS in scripts/size-gate.ts so its ceiling is reported.',
      });
    }
  }
  // A third table now exists — LAYER_GRAPH — and it has to cover the same set,
  // minus the one entry that measures an app rather than a published entry.
  for (const entry of Object.keys(BUDGETS)) {
    if (entry === APP_ENTRY) continue;
    if (!(entry in LAYER_GRAPH)) {
      failures.push({
        what: `"${entry}" has a size budget but no entry in LAYER_GRAPH, so nothing checks which modules it pulls in.`,
        action: 'add it to LAYER_GRAPH with its layers and its reachable-module count.',
      });
    }
  }
  for (const entry of Object.keys(LAYER_GRAPH)) {
    if (!(entry in BUDGETS)) {
      failures.push({
        what: `LAYER_GRAPH has "${entry}" but BUDGETS does not, so it is unbudgeted.`,
        action: 'add a budget for it in both BUDGETS and the BUDGETS table in scripts/build.ts.',
      });
    }
  }
}

/** Named bindings in the `import { ... } from 'apse'` statement of the app. */
function readAppImports(source: string): string[] {
  const match = /import\s*\{([^}]*)\}\s*from\s*['"][^'"]+['"]/.exec(source);
  if (match === null) return [];
  return match[1]!
    .split(',')
    .map((s) => s.trim().split(/\s+as\s+/)[0]!.trim())
    .filter((s) => s.length > 0);
}

/**
 * Checks the tree-shaking claim, not just the size.
 *
 * A bundle that leaked the five unused primitives still builds, still passes the
 * budget, and still produces a number in `dist/size-report.json` that does not
 * describe what a consumer downloads. That failure is invisible to a byte count
 * by construction, so the unused primitives are looked for by name.
 *
 * `bench/tree-shake.ts` is parsed rather than hardcoded, so adding a primitive
 * to the app moves the marker set with it instead of producing a false alarm.
 */
async function checkTreeShaking(failures: Failure[]): Promise<void> {
  const appPath = join(ROOT, TREE_SHAKE_APP);
  if (!(await exists(appPath))) {
    failures.push({
      what: `${TREE_SHAKE_APP} does not exist, so the tree-shaking claim has no app to check.`,
      action: 'restore bench/tree-shake.ts, or point TREE_SHAKE_APP at the file the build measures.',
    });
    return;
  }
  const bundlePath = join(ROOT, TREE_SHAKEN_BUNDLE);
  if (!(await exists(bundlePath))) {
    // Already reported by checkHeadline. Skipped rather than reported twice.
    return;
  }

  const imports = readAppImports(await readFile(appPath, 'utf8'));
  if (imports.length === 0) {
    failures.push({
      what: `Could not read the imports of ${TREE_SHAKE_APP}.`,
      action: 'check readAppImports() in scripts/size-gate.ts against the file — the gate is currently checking nothing.',
    });
    return;
  }
  const bundle = (await readFile(bundlePath, 'utf8')).replace(/\s+/g, '');

  const used = Object.keys(PRIMITIVE_MARKERS).filter((p) => imports.includes(p));
  const unused = Object.keys(PRIMITIVE_MARKERS).filter((p) => !imports.includes(p));
  const count = (marker: string): number => {
    let n = 0;
    let at = bundle.indexOf(marker);
    while (at >= 0) { n++; at = bundle.indexOf(marker, at + marker.length); }
    return n;
  };

  out('  tree-shaking');
  out(`    ${DIM}app${OFF}        ${TREE_SHAKE_APP}`);
  out(`    ${DIM}imports${OFF}    ${imports.join(', ')}`);
  out(`    ${DIM}primitives${OFF} using ${used.join(', ') || 'none'}  ${DIM}must be absent:${OFF} ${unused.join(', ') || 'none'}`);
  out();

  // Positive control first. Without it, every zero below could be a marker set
  // that stopped matching anything at all.
  for (const primitive of used) {
    const markers = PRIMITIVE_MARKERS[primitive]!;
    const hits = markers.filter((m) => count(m) > 0);
    out(`    ${DIM}control${OFF}    ${pad(primitive, 10)} ${hits.length}/${markers.length} marker(s) present ${DIM}${hits.join(', ')}${OFF}`);
    if (hits.length === 0) {
      failures.push({
        what: `The tree-shake app imports \`${primitive}\`, but none of its markers (${markers.join(', ')}) appear in ${TREE_SHAKEN_BUNDLE}. The marker set has gone stale, so the leak checks below prove nothing.`,
        action: 're-derive PRIMITIVE_MARKERS from the current esbuild output by bundling a file that imports every primitive and looking at what survives minification.',
      });
    }
  }

  const leaked: { primitive: string; marker: string; n: number }[] = [];
  for (const primitive of unused) {
    const markers = PRIMITIVE_MARKERS[primitive]!;
    const found = markers.map((m) => ({ marker: m, n: count(m) })).filter((x) => x.n > 0);
    for (const f of found) leaked.push({ primitive, marker: f.marker, n: f.n });
    out(
      `    ${pad(primitive, 10)} ${pad(markers.join(', '), 58)}${found.length === 0 ? `${DIM}0 found${OFF}` : RED + `${found.length} found${OFF}`}`,
    );
  }
  out();

  if (leaked.length > 0) {
    failures.push({
      what: `Tree-shaking is broken: ${leaked.length} marker(s) for ${unused.length} unused primitive(s) survive in ${TREE_SHAKEN_BUNDLE} (${leaked.map((l) => `${l.primitive}: ${l.marker} x${l.n}`).join('; ')}). The size in the report above does not describe what a consumer downloads.`,
      action: 'find the module with top-level work or a side effect that keeps the whole primitives barrel alive — `sideEffects: false` in package.json cannot be enforced against a module that has side effects. Start with the files that export more than one thing.',
    });
  }
}

/**
 * Which layer a source module belongs to, as a graph edge name.
 *
 * `src/core/error.ts` is layer `core`. `src/index.ts` is the public barrel and
 * gets its own name, `root`, because it is the one module no subpath may reach:
 * a subpath that imports it has imported the whole library, which is the exact
 * mistake the subpaths exist to prevent. A path outside `src/` gets a name that
 * is in no layer's list, so it fails rather than passing unnoticed.
 */
function layerOf(module: string): string {
  const nested = /^src\/([^/]+)\//.exec(module);
  if (nested !== null) return nested[1]!;
  const top = /^src\/([^/]+)\.ts$/.exec(module);
  if (top === null) return `(outside src: ${module})`;
  return top[1] === 'index' ? 'root' : top[1];
}

/** Every `.ts` file under `src/`, `src/`-relative and forward-slashed. */
async function sourceModules(): Promise<string[]> {
  const found: string[] = [];
  const walk = async (dir: string): Promise<void> => {
    for (const dirent of await readdir(dir, { withFileTypes: true })) {
      const full = join(dir, dirent.name);
      if (dirent.isDirectory()) {
        await walk(full);
      } else if (dirent.isFile() && dirent.name.endsWith('.ts')) {
        found.push(relative(ROOT, full).split(sep).join('/'));
      }
    }
  };
  await walk(SRC);
  return found.sort();
}

/**
 * Asserts the shape of the import graph, not the size of the output.
 *
 * Three assertions, all of them things a byte count cannot see:
 *
 *   1. **No entry reaches a layer it is not allowed to reach.** This is the
 *      headline one. A new import inside `src/core/` that reaches the renderer
 *      adds a few hundred bytes and trips nothing; it turns `apse/core` — 7.72 KB
 *      of error types and a uniform-layout helper — into a dependency on the
 *      whole 67 KB library, and every consumer pays for it at runtime in a way
 *      only a graph check can name.
 *   2. **Each entry reaches exactly as many modules as {@link LAYER_GRAPH} says.**
 *      An exact count, not a ceiling. Adding a file to a layer is a deliberate
 *      act, and the number has to be edited rather than re-measured.
 *   3. **Every file in `src/` is reachable from some entry.** Read from the
 *      filesystem, not from the report, so the two are independent. It catches
 *      a module that was added and never exported — dead code that still gets
 *      type-checked, still gets reviewed, and still never ships. The check is
 *      over the *union* of the entries' closures rather than the root's alone,
 *      because the six subpath barrels and the type-only `render/types.ts` are
 *      deliberately absent from `src/index.ts`'s graph: the root re-exports the
 *      individual modules rather than the barrels, and a type-only module is
 *      erased before it reaches a bundler.
 */
async function checkModuleGraph(report: SizeReport, failures: Failure[]): Promise<void> {
  const rows = new Map(report.entries.map((r) => [r.entry, r]));

  out('  module graph');
  out(`  ${DIM}${pad('entry', 20)}${pad('modules', 9)}${pad('expected', 9)}${DIM}layers${OFF}`);
  out(`  ${DIM}${'-'.repeat(72)}${OFF}`);

  const reachable = new Set<string>();
  for (const row of report.entries) {
    if (Array.isArray(row.modules)) for (const m of row.modules) reachable.add(m);
  }

  for (const [entry, rule] of Object.entries(LAYER_GRAPH)) {
    const row = rows.get(entry);
    if (row === undefined) {
      failures.push({
        what: `No size report row for "${entry}", so its import graph cannot be checked.`,
        action: 'check that scripts/build.ts ENTRIES still bundles this entry.',
      });
      continue;
    }
    if (!Array.isArray(row.modules) || row.modules.length === 0) {
      failures.push({
        what: `"${entry}" reported no source modules, so the layer check for it proves nothing.`,
        action: 'check that scripts/build.ts still writes reachableInputs() into dist/size-report.json. This is what it calls "modules".',
      });
      continue;
    }

    const layers = new Set(row.modules.map(layerOf));
    const allowed = new Set<string>(rule.layers);
    const leaked = [...layers].filter((l) => !allowed.has(l)).sort();
    const countOk = row.modules.length === rule.modules;

    out(
      `  ${pad(entry, 20)}${pad(String(row.modules.length), 9)}${pad(String(rule.modules), 9)}` +
      `${DIM}${[...layers].sort().join(' ')}${OFF}`,
    );

    if (leaked.length > 0) {
      failures.push({
        what: `${entry} reaches ${leaked.map((l) => `layer ${l}`).join(' and ')}, which is not in its declared layer set (${rule.layers.join(' ')}). An entry has absorbed a dependency it should not have.`,
        action: 'find the import that crosses the layer boundary — it is usually the one convenience helper that seemed harmless. The fix is to pass the value across the boundary as a parameter, or to promote the import into a layer that is allowed to know about it. Update LAYER_GRAPH only if the edge is genuinely intended.',
      });
    }
    if (!countOk) {
      failures.push({
        what: `${entry} reaches ${row.modules.length} source module(s) but LAYER_GRAPH expects ${rule.modules}.`,
        action: leaked.length > 0
          ? 'fix the layer leak above first, then re-measure. Do not bump the count to match a graph you have not looked at.'
          : 'a file was added to or removed from this layer. Re-run `bun run build` and update the count in LAYER_GRAPH with the module that changed.',
      });
    }
  }

  // (3) Nothing in src/ may be unreachable from the published surface.
  const onDisk = await sourceModules();
  const onDiskSet = new Set(onDisk);
  const orphaned = onDisk.filter((m) => !reachable.has(m));
  const phantom = [...reachable].filter((m) => !onDiskSet.has(m));
  out(
    `  ${DIM}every src/ module reachable from some entry: ${reachable.size} of ${onDisk.length}` +
    `${orphaned.length === 0 && phantom.length === 0 ? '' : RED + `  (${orphaned.length} unreachable, ${phantom.length} phantom)` + OFF}`,
  );
  out();
  if (orphaned.length > 0) {
    failures.push({
      what: `${orphaned.length} file(s) in src/ are not reachable from any entry: ${orphaned.join(', ')}. They are type-checked and reviewed but can never be imported by a consumer.`,
      action: 'export them from the entry that owns them if they are public, or delete them if they are not. Unreachable code still costs review time and still shows up in coverage.',
    });
  }
  if (phantom.length > 0) {
    failures.push({
      what: `The report claims ${phantom.length} module(s) that do not exist on disk: ${phantom.join(', ')}.`,
      action: 'the report is stale, or scripts/build.ts is writing paths relative to the wrong directory. Run `bun run build` and re-run this gate.',
    });
  }
}

/**
 * Every subpath in `exports` and every target in `typesVersions` must have a
 * built file.
 *
 * This is the check that is invisible until it matters: nothing about a broken
 * exports map shows up in a build, a test, or a bundle. It shows up on a
 * consumer's machine, as `Cannot find module 'apse/geometry'`.
 */
async function checkExportsMap(failures: Failure[]): Promise<void> {
  const pkg = JSON.parse(await readFile(PKG, 'utf8')) as {
    exports?: Record<string, string | Record<string, string>>;
    typesVersions?: Record<string, Record<string, string[]>>;
  };

  out('  exports map');
  const targets: [string, string][] = [];
  for (const [subpath, entry] of Object.entries(pkg.exports ?? {})) {
    const conditions = typeof entry === 'string' ? { default: entry } : entry;
    for (const [condition, target] of Object.entries(conditions)) {
      targets.push([`${subpath} (${condition})`, target]);
    }
  }
  for (const [version, map] of Object.entries(pkg.typesVersions ?? {})) {
    for (const [subpath, list] of Object.entries(map)) {
      for (const target of list) targets.push([`typesVersions[${version}].${subpath}`, target]);
    }
  }

  let missing = 0;
  for (const [label, target] of targets) {
    if (target.startsWith('./package.json')) {
      out(`    ${pad(label, 30)} ${DIM}skip (not a build product)${OFF}`);
      continue;
    }
    const ok = await exists(join(ROOT, target));
    if (!ok) missing++;
    out(`    ${pad(label, 30)} ${pad(target, 34)}${ok ? DIM + 'ok' + OFF : RED + 'MISSING' + OFF}`);
  }
  out();

  const subpaths = Object.keys(pkg.exports ?? {}).length;
  if (missing > 0) {
    failures.push({
      what: `${missing} of ${targets.length} target(s) in the package exports map have no built file. Nothing in the build, the tests, or the bundle would have noticed.`,
      action: 'run `bun run build`. If the file is genuinely absent, scripts/build.ts ENTRIES and package.json exports have drifted apart — make one entry per subpath.',
    });
  }
  if (subpaths === 0) {
    failures.push({
      what: 'package.json declares no `exports`, so every subpath is unbudgeted and unchecked.',
      action: 'add the exports map, or accept that consumers can only import `apse/apse`.',
    });
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const failures: Failure[] = [];

  out();
  out(`  ${BOLD}size gate${OFF}  ${DIM}— apse${OFF}`);

  const report = await loadReport(failures);
  if (report === null) {
    reportFailures(failures);
    return;
  }
  out(`  ${DIM}report${OFF}  dist/size-report.json  ${DIM}generated ${report.generatedAt}${OFF}`);
  out();

  await checkBudgetAgrees(failures);
  checkEntries(report, failures);
  await checkHeadline(report, failures);
  await checkTreeShaking(failures);
  await checkModuleGraph(report, failures);
  await checkExportsMap(failures);

  if (failures.length === 0) {
    out(`  ${GREEN}PASS${OFF}  ${report.entries.length} entries within budget, tree-shaking intact, layer graph intact, exports map resolvable`);
    out();
    out(`  ${DIM}add to verify: bun run scripts/size-gate.ts${OFF}`);
    out();
    return;
  }
  reportFailures(failures);
}

function reportFailures(failures: Failure[]): void {
  out(`  ${RED}${BOLD}FAIL${OFF}  ${failures.length} problem${failures.length === 1 ? '' : 's'}`);
  for (const f of failures) {
    out();
    out(`  ${RED}·${OFF} ${f.what}`);
    out(`    ${DIM}do:${OFF} ${f.action}`);
  }
  out();
  out(`  ${DIM}add to verify: bun run scripts/size-gate.ts${OFF}`);
  out();
  process.exit(1);
}

main().catch((e: unknown) => {
  const message = e instanceof Error ? e.message : String(e);
  out();
  out(`  ${RED}${BOLD}FAIL${OFF}  size gate could not complete: ${message}`);
  out(`    ${DIM}do:${OFF} run \`bun run build\` and try again.`);
  out();
  process.exit(1);
});
