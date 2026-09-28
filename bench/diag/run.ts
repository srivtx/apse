/**
 * The diagnostic bench driver.
 *
 * Nine failure modes, triggered for real in both libraries, in the same browser
 * process on the same machine. For each one it records what a developer
 * actually sees: the exception and its fields, every console line either library
 * printed, every WebGPU validation error, and the pixels that came out.
 *
 * The point is not that one library has more codes than the other. The point is
 * whether a code, when it fires, tells you what to change — and what happens in
 * the cases where nothing fires at all.
 *
 *   bun run bench/diag/run.ts            # all nine
 *   bun run bench/diag/run.ts 4 6        # only those
 *
 * Prints the full record for every scenario. Writes nothing: the findings live
 * in bench/diag/README.md, and a JSON dump nobody reads is not evidence.
 */

import { serve } from 'bun';
import puppeteer from 'puppeteer';

const ROOT = new URL('..', import.meta.url).pathname.replace(/bench\/$/, '');
const PORT = 8788;

const only = process.argv.slice(2).map(Number).filter((n) => !Number.isNaN(n));

// ---------------------------------------------------------------------------

type ConsoleLine = { scenario: string | null; level: string; text: string };

/**
 * What the page managed to read off a thrown value.
 *
 * Every field is optional because the point of the harness is to compare a rich
 * typed error against a bare one: `code`, `why` and `fix` exist for apse and are
 * absent for three.js, and their absence is the measurement.
 */
interface ThrownShape {
  readonly thrown: boolean;
  readonly name?: string;
  readonly ctor?: string;
  readonly code?: string | number;
  readonly message?: string;
  readonly why?: string;
  readonly fix?: string;
  readonly link?: string;
  readonly detail?: string;
  readonly toString?: string;
  readonly toJSON?: string;
  readonly stackTop?: readonly string[];
  readonly value?: string;
}

interface Attempt {
  data: unknown;
  error: ThrownShape;
  console: ConsoleLine[];
  validation: string[];
}
interface Record_ {
  id: number;
  title: string;
  ms: number;
  apse: Attempt | null;
  three: Attempt | null;
}

const dim = (s: string) => `\x1b[90m${s}\x1b[0m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const cy = (s: string) => `\x1b[36m${s}\x1b[0m`;
const yl = (s: string) => `\x1b[33m${s}\x1b[0m`;
const rd = (s: string) => `\x1b[31m${s}\x1b[0m`;

function rule(ch = '─') { return dim(ch.repeat(78)); }

// ---------------------------------------------------------------------------

function printThrown(who: string, e: ThrownShape): void {
  if (e === undefined || e === null) return;
  if (e.thrown !== true) {
    console.log(`    ${who} threw nothing`);
    return;
  }
  console.log(`    ${who} threw:`);
  console.log(`      name      ${e.ctor ?? e.name ?? '?'}`);
  if (e.code !== undefined) console.log(`      code      ${yl(String(e.code))}`);
  console.log(`      message   ${e.message ?? ''}`);
  if (e.why !== undefined) console.log(`      why       ${wrap(e.why, 12)}`);
  if (e.fix !== undefined) console.log(`      fix       ${wrap(e.fix, 12)}`);
  if (e.link !== undefined) console.log(`      link      ${e.link}`);
  if (e.toString !== undefined) console.log(`      toString() ${wrap(e.toString, 16)}`);
  if (e.stackTop !== undefined && e.stackTop.length > 1) {
    console.log(`      stack     ${dim(e.stackTop[1] ?? '')}`);
  }
}

function wrap(text: string, indent: number, width = 96): string {
  const pad = ' '.repeat(indent);
  const words = String(text).split(/\s+/);
  const lines: string[] = [];
  let cur = '';
  for (const w of words) {
    if (cur.length + w.length + 1 > width - indent) { lines.push(cur); cur = w; }
    else cur = cur.length === 0 ? w : `${cur} ${w}`;
  }
  if (cur.length > 0) lines.push(cur);
  return lines.map((l, i) => (i === 0 ? l : pad + l)).join('\n');
}

function printConsole(who: string, lines: ConsoleLine[]): void {
  if (lines.length === 0) {
    console.log(`    ${who} console: ${dim('(nothing printed)')}`);
    return;
  }
  console.log(`    ${who} console: ${lines.length} line(s)`);
  for (const l of lines) {
    const tag = l.level === 'error' ? rd(l.level) : l.level === 'warn' ? yl(l.level) : dim(l.level);
    for (const [i, chunk] of wrap(l.text, 12).split('\n').entries()) {
      console.log(`${i === 0 ? `      ${tag} ` : '           '}${chunk}`);
    }
  }
}

function printValidation(who: string, msgs: string[]): void {
  if (msgs.length === 0) {
    console.log(`    ${who} WebGPU validation errors: ${dim('(none)')}`);
    return;
  }
  console.log(`    ${who} WebGPU validation errors: ${rd(String(msgs.length))}`);
  for (const m of msgs) console.log(`      ${wrap(m, 8)}`);
}

function printData(who: string, data: unknown): void {
  if (data === null || data === undefined) return;
  const text = JSON.stringify(data, (_k, v) => (v === undefined ? '<undefined>' : v), 1);
  if (text === '{}') return;
  console.log(`    ${who} measurements:`);
  for (const [i, chunk] of wrap(text, 8).split('\n').entries()) {
    console.log(`${i === 0 ? '      ' : '      '}${chunk}`);
  }
}

function printAttempt(who: string, a: Attempt | null): void {
  if (a === null) { console.log(`    ${who}: ${dim('not applicable')}`); return; }
  printThrown(who, a.error);
  printConsole(who, a.console);
  printValidation(who, a.validation);
  printData(who, a.data);
}

// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const server = serve({
    port: PORT,
    idleTimeout: 0,
    routes: {
      '/*': async (req) => {
        const path = new URL(req.url).pathname;
        const file = Bun.file(new URL(path === '/' ? 'bench/diag/index.html' : path.slice(1), `file://${ROOT}`));
        if (await file.exists()) {
          // Chrome enforces the JS MIME type for module scripts and reports a
          // violation only as a failed dynamic import, which is a confusing way
          // to learn about a header.
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

  console.log(`\n  ${bold('apse diagnostic bench')}`);
  console.log(`  server: http://localhost:${PORT}/bench/diag/index.html\n`);

  const browser = await puppeteer.launch({
    headless: true,
    // The installed Chrome, not a puppeteer-managed download: the question is
    // what a developer's browser says, on their hardware.
    executablePath: process.env['CHROME_PATH'] ?? undefined,
    channel: process.env['CHROME_PATH'] === undefined ? 'chrome' : undefined,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });

  const results: Record_[] = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e: unknown) => {
      console.error(`  ${rd('PAGE ERROR')}: ${e instanceof Error ? e.message : String(e)}`);
    });

    await page.goto(`http://localhost:${PORT}/bench/diag/index.html`, { waitUntil: 'load' });
    const ready = await page.waitForFunction(
      () => (globalThis as Record<string, unknown>)['__diagReady'] === true,
      { timeout: 60_000 },
    ).then(() => true).catch(() => false);

    if (!ready) {
      const err = await page.evaluate(() => JSON.stringify((globalThis as any).__diagError ?? 'unknown'));
      throw new Error(`harness failed to initialise: ${err}`);
    }

    const env = await page.evaluate(() => (globalThis as any).__diagEnv());
    console.log(rule());
    console.log(`  ${bold('environment')}`);
    console.log(`    apse        ${env.apse.version}  ${env.apse.adapter}  featureLevel ${env.apse.featureLevel}`);
    console.log(`    apse fmt    material default ${env.apse.preferredCanvasFormat}, device ${env.apse.canvasFormat}, ` +
      `backing ${env.apse.backing.w}x${env.apse.backing.h}`);
    console.log(`    apse stride ${env.apse.objectUniformStride} bytes, minUniformBufferOffsetAlignment ${env.apse.minUniformBufferOffsetAlignment}`);
    console.log(`    apse codes  ${env.apse.errorCodeCount} in the catalog`);
    console.log(`    three.js    r${env.three.version}  ${env.three.context.version}`);
    console.log(`    three gl    ${env.three.context.isWebGL2 ? 'WebGL2' : 'WebGL1'}  ` +
      `preserveDrawingBuffer ${env.three.preserveDrawingBuffer}  checkShaderErrors ${env.three.checkShaderErrorsDefault}`);
    console.log(rule());

    const ids = await page.evaluate(() => (globalThis as any).__diagScenarioIds()) as number[];
    for (const id of ids) {
      if (only.length > 0 && !only.includes(id)) continue;
      const rec = await page.evaluate((i: number) => (globalThis as any).__diag(i), id) as Record_;
      results.push(rec);
      console.log('');
      console.log(rule('═'));
      console.log(`  ${bold(`scenario ${rec.id}`)}  ${rec.title}  ${dim(`${rec.ms} ms`)}`);
      console.log(rule('═'));
      console.log(`  ${cy('apse')}`);
      printAttempt('apse', rec.apse);
      console.log('');
      console.log(`  ${bold('three.js')}`);
      printAttempt('three', rec.three);
    }
  } finally {
    await browser.close();
    server.stop(true);
  }

  console.log('');
  console.log(rule('═'));
  console.log(`  ${results.length} scenario(s) run.\n`);
}

main().catch((e) => {
  console.error(`\n  diagnostic bench failed: ${e instanceof Error ? e.message : String(e)}\n`);
  process.exit(1);
});
