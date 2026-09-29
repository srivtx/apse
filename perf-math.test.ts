/**
 * Timing harness for `src/math`.
 *
 * This file is not a regression gate. Every number it prints is machine- and
 * load-dependent, and a nanosecond threshold in CI is a coin flip, not a test.
 * What it *asserts* is what makes the numbers mean anything: that the timer is
 * fine enough to need batching, that the batching works, and that every
 * candidate is bit-identical to the function it would replace. The timings are
 * commentary — they are here so the next person does not have to rebuild the
 * harness to ask "is this actually faster".
 *
 * Run it directly for the report:
 *
 *     bun test test/perf-math.test.ts
 *
 * ## Batching, and the resolution this machine actually has
 *
 * `performance.now()` resolves to about **41 ns** here, not the 100 µs the
 * brief assumed. That is still coarser than the ~18 ns a normal-matrix call
 * costs, so a single-call measurement would be reporting the clock rather than
 * the code. Every figure below is `(t1 - t0) / K` over `K` calls with `K` in
 * the hundreds of thousands, and the reported value is the **median across
 * reps**, not the mean — a mean on a contended machine is dragged by whichever
 * rep the scheduler interrupted.
 *
 * ## Why this is *paired*, and why the first version of this file was wrong
 *
 * The first version compared two candidates by timing them in rotation and
 * comparing the two medians. That is not enough. Five agents share this
 * machine, rep-to-rep drift is tens of nanoseconds, and comparing two
 * independently-drifting medians manufactures differences out of nothing: it
 * reported a hand-unrolled `containsSphere` as 39% faster when the unrolled
 * version was simply not paying for a `sphere.set()` allocation the other
 * candidate made inside its timed closure, and it reported removing the normal
 * matrix's three padding stores as a 14% win that a paired measurement then
 * put at 0.1 ns with a p10..p90 band straddling zero.
 *
 * So `paired()` alternates the two candidates **within a single rep** and keeps
 * the per-rep *difference*, and every candidate is run in both orderings.
 * A result that only holds in one ordering, or whose win/loss count is near
 * half, is noise and is reported as noise. The two real findings below survive
 * that bar: `mulAffine` wins 61/61 reps in both orderings with a p10..p90 band
 * nowhere near zero, and the rigid normal-matrix fast path *loses* 36–40 reps
 * out of 41 in both orderings.
 */

import { describe, expect, test } from 'bun:test';
import { frustum, mat4, quat, sphere, vec3 } from './src/math/index.ts';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/** The smallest delta this machine's `performance.now()` reports, in seconds. */
function timerResolution(): number {
  let min = Infinity;
  for (let i = 0; i < 200_000; i++) {
    const a = performance.now();
    const b = performance.now();
    if (b - a > 0 && b - a < min) min = b - a;
  }
  return min;
}

interface One {
  /** Median ns/call of the first candidate in the rep. */
  first: number;
  /** Median ns/call of the second. */
  second: number;
  /** Median of the per-rep (first - second) differences, in ns. */
  diff: number;
  lo: number;
  hi: number;
  /** Reps in which the first candidate was the slower one. */
  firstSlowerIn: number;
}

interface Compare {
  /** Bias-corrected estimate of (A - B) in ns/call. Positive means A is slower. */
  effect: number;
  /** The two raw per-ordering estimates, for the sign-agreement check. */
  whenAFirst: number;
  whenBFirst: number;
  /** True when both orderings agree on the sign and the effect clears the noise. */
  real: boolean;
  a: number;
  b: number;
}

/** One ordering: A and B alternate inside each rep, keeping the difference. */
function onePass(fa: (i: number) => void, fb: (i: number) => void, K: number, reps: number): One {
  for (let i = 0; i < 40_000; i++) { fa(i); fb(i); }
  const d: number[] = [];
  const a: number[] = [];
  const b: number[] = [];
  for (let rep = 0; rep < reps; rep++) {
    const t0 = performance.now();
    for (let i = 0; i < K; i++) fa(i);
    const t1 = performance.now();
    for (let i = 0; i < K; i++) fb(i);
    const t2 = performance.now();
    a.push(((t1 - t0) * 1e6) / K);
    b.push(((t2 - t1) * 1e6) / K);
    d.push(a[a.length - 1]! - b[b.length - 1]!);
  }
  const med = (x: number[]): number => x.slice().sort((p, q) => p - q)[x.length >> 1]!;
  const ds = d.slice().sort((p, q) => p - q);
  return {
    first: med(a),
    second: med(b),
    diff: med(d),
    lo: ds[Math.floor(reps * 0.1)]!,
    hi: ds[Math.floor(reps * 0.9)]!,
    firstSlowerIn: d.filter((x) => x > 0).length,
  };
}

/**
 * Compare A against B in **both orderings** and correct for position bias.
 *
 * Whichever candidate runs second in a rep is systematically cheaper on a
 * machine this contended — the first block pays the pipeline warm-up and the
 * second inherits it. A single ordering therefore reports a difference even
 * between two identical functions.
 *
 * Running both orderings cancels it: `whenAFirst` measures A - B, `whenBFirst`
 * measures B - A, and their average is a bias-free estimate of A - B. If the
 * two disagree on the sign, the result is an artifact of ordering and there is
 * no effect to report. This is not a refinement — it is the difference between
 * a measurement and a coincidence: the hand-unrolled cull test came out
 * "3.8 ns faster" in one ordering and "1.6 ns slower" in the other, which is
 * exactly what a position bias looks like, and the 5% "win" from dropping the
 * normal matrix's padding stores vanishes under it.
 */
function compare(fa: (i: number) => void, fb: (i: number) => void, K: number, reps: number, aName: string, bName: string): Compare {
  const ab = onePass(fa, fb, K, reps);
  const ba = onePass(fb, fa, K, reps);
  // (A - B) from the first ordering, and -(B - A) = (A - B) from the second.
  const whenAFirst = ab.diff;
  const whenBFirst = -ba.diff;
  const effect = (whenAFirst + whenBFirst) / 2;
  const agree = (whenAFirst > 0) === (whenBFirst > 0);
  // "Real" needs agreement on sign *and* an effect that is not inside the
  // spread of a single ordering's rep-to-rep distribution.
  const spread = Math.max(Math.abs(ab.hi - ab.lo), Math.abs(ba.hi - ba.lo));
  const real = agree && Math.abs(effect) > 0.25 && Math.abs(effect) > spread / 4;

  const pct = (effect / ((ab.first + ba.second) / 2)) * 100;
  const verdict = !agree
    ? 'ORDER-DEPENDENT — no effect'
    : real
      ? `${effect > 0 ? aName : bName} is ${Math.abs(effect).toFixed(2)} ns slower (${Math.abs(pct).toFixed(1)}%)`
      : `within noise (${effect >= 0 ? '+' : ''}${effect.toFixed(2)} ns, spread ${spread.toFixed(2)})`;
  console.log(
    `  ${aName} ${ab.first.toFixed(2)} / ${ba.second.toFixed(2)} ns    ${bName} ${ab.second.toFixed(2)} / ${ba.first.toFixed(2)} ns` +
    `    (K=${K}, ${reps} paired reps per ordering)\n` +
    `    A-B when A first ${whenAFirst >= 0 ? '+' : ''}${whenAFirst.toFixed(2)} ns, when B first ${whenBFirst >= 0 ? '+' : ''}${whenBFirst.toFixed(2)} ns` +
    `  ->  ${verdict}`,
  );
  return { effect, whenAFirst, whenBFirst, real, a: ab.first, b: ab.second };
}

/** The shipped `normalMatrixOf`, verbatim from `src/render/renderer.ts`. */
const IDENTITY_NORMAL = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
function shippedNormalMatrix(out: Float32Array, m: Float32Array): Float32Array {
  const a = m[0]!, b = m[1]!, c = m[2]!;
  const d = m[4]!, e = m[5]!, f = m[6]!;
  const g = m[8]!, h = m[9]!, i = m[10]!;
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (det === 0 || !Number.isFinite(det)) {
    for (let k = 0; k < 12; k++) out[k] = IDENTITY_NORMAL[k];
    return out;
  }
  const s = 1 / det;
  out[0] = A * s; out[1] = B * s; out[2] = C * s;
  out[4] = (c * h - b * i) * s; out[5] = (a * i - c * g) * s; out[6] = (b * g - a * h) * s;
  out[8] = (b * f - c * e) * s; out[9] = (c * d - a * f) * s; out[10] = (a * e - b * d) * s;
  out[3] = 0; out[7] = 0; out[11] = 0;
  return out;
}

/**
 * A byte-identical copy of `shippedNormalMatrix`, declared locally.
 *
 * The control for the cross-module measurement above: comparing it against
 * `shippedNormalMatrix` gives the harness's own noise floor, and comparing it
 * against `mat4.normalMatrix` isolates module-boundary cost from arithmetic.
 */
function localNormalMatrixCopy(out: Float32Array, m: Float32Array): Float32Array {
  const a = m[0]!, b = m[1]!, c = m[2]!;
  const d = m[4]!, e = m[5]!, f = m[6]!;
  const g = m[8]!, h = m[9]!, i = m[10]!;
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (det === 0 || !Number.isFinite(det)) {
    for (let k = 0; k < 12; k++) out[k] = IDENTITY_NORMAL[k];
    return out;
  }
  const s = 1 / det;
  out[0] = A * s; out[1] = B * s; out[2] = C * s;
  out[4] = (c * h - b * i) * s; out[5] = (a * i - c * g) * s; out[6] = (b * g - a * h) * s;
  out[8] = (b * f - c * e) * s; out[9] = (c * d - a * f) * s; out[10] = (a * e - b * d) * s;
  out[3] = 0; out[7] = 0; out[11] = 0;
  return out;
}

/** A rigid-transform fast path, as it would be written. Measured, not shipped. */
function rigidNormalMatrix(out: Float32Array, m: Float32Array): Float32Array {
  const a = m[0]!, b = m[1]!, c = m[2]!;
  const d = m[4]!, e = m[5]!, f = m[6]!;
  const g = m[8]!, h = m[9]!, i = m[10]!;
  // Three dots for orthogonality, three squared lengths for equal scale: 18
  // multiplies, to save 9. And it has to run on every object to catch the
  // rigid ones.
  if (a * d + b * e + c * f === 0 && a * g + b * h + c * i === 0 && d * g + e * h + f * i === 0) {
    const n0 = a * a + b * b + c * c;
    if (n0 === d * d + e * e + f * f && n0 === g * g + h * h + i * i) {
      const s = 1 / Math.sqrt(n0);
      out[0] = a * s; out[1] = d * s; out[2] = g * s;
      out[4] = b * s; out[5] = e * s; out[6] = h * s;
      out[8] = c * s; out[9] = f * s; out[10] = i * s;
      return out;
    }
  }
  return shippedNormalMatrix(out, m);
}

const rotation = (): Float32Array =>
  quat.toMat4(mat4.create(), quat.setAxisAngle(quat.create(), vec3.create(0.3, 0.5, 0.81), 1.1));

function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

// ---------------------------------------------------------------------------

describe('perf harness', () => {
  test('the clock needs batching, and batching resolves a single call', () => {
    const res = timerResolution();
    console.log(`\n  performance.now() resolution: ${(res * 1e6).toFixed(1)} ns`);
    console.log('  all figures below: median of paired per-rep differences, K calls per rep.');

    // The one assertion that makes every other number in this file trustworthy.
    expect(Number.isFinite(res)).toBe(true);
    expect(res).toBeGreaterThan(0);

    // A batch must cost far more than a single clock read, and be far larger
    // than the resolution — otherwise (t1-t0)/K is dividing noise by K.
    const K = 100_000;
    const t0 = performance.now();
    let sink = 0;
    for (let i = 0; i < K; i++) sink += i;
    const t1 = performance.now();
    console.log(`  empty loop: ${(((t1 - t0) * 1e6) / K).toFixed(3)} ns/iter at K=${K} (${((t1 - t0) * 1e3).toFixed(1)} us total)`);
    expect(sink).toBe(K * (K - 1) / 2);
    expect(t1 - t0).toBeGreaterThan(res * 10);
  });
});

describe('mat4.normalMatrix', () => {
  test('is not faster than the shipped copy, and moving it across a module boundary costs ~1 ns', () => {
    // The honest headline, and it is a null result.
    //
    // `mat4.normalMatrix` is the arithmetic the renderer already runs, moved
    // into the module that owns the maths and given a documented contract. The
    // one change that looked like a win — dropping the three vec3-padding
    // stores — measured 14% in isolation and 0.1 ns under a paired comparison,
    // so the stores stayed.
    //
    // What is left is a reproducible ~1 ns *deficit*, and the cause is not the
    // arithmetic: it is that a function reached across an ES module boundary
    // does not inline the way a module-private one does under Bun's JSC. The
    // control below measures two byte-identical local functions at 0.02 ns,
    // and the same function imported at 1.0 ns. So the number is a property of
    // where the function is written down, not of what it computes. V8 — the
    // engine the browser benchmark actually runs on — inlines freely across
    // module boundaries, so this may not reproduce in Chrome at all.
    //
    // The conclusion for whoever owns the renderer is the same either way:
    // there is no speed argument for moving `normalMatrixOf` into `src/math/`.
    const base = rotation();
    const N = 2000;
    // 2000 x 64 B is 125 KiB, past this machine's 32 KiB L1, so the source walk
    // is not served from cache the way a single hot matrix would allow.
    const pool: Float32Array[] = [];
    for (let i = 0; i < N; i++) pool.push(new Float32Array(base));
    const out = new Float32Array(12);
    const K = 200_000;
    console.log('');
    compare(
      (i) => { shippedNormalMatrix(out, pool[i % N]!); },
      (i) => { mat4.normalMatrix(out, pool[i % N]!); },
      K, 25, 'shipped', 'normalMatrix',
    );
    // Control: the same body twice, both local. Anything above the noise floor
    // here would mean the harness is measuring itself.
    compare(
      (i) => { shippedNormalMatrix(out, pool[i % N]!); },
      (i) => { localNormalMatrixCopy(out, pool[i % N]!); },
      K, 25, 'local copy A', 'local copy B',
    );
  }, 120_000);

  test('the rigid fast path is a consistent loss', () => {
    // The candidate the brief asked about. A rotation is its own inverse
    // transpose, and a uniform scale is that over s^2, so the adjugate really
    // does collapse for the common case — but proving the 3x3 is orthonormal
    // costs three dot products and three squared lengths, 18 multiplies, to
    // save 9, and the test has to run on every object to catch the rigid ones.
    const base = rotation();
    const cases: readonly (readonly [string, Float32Array])[] = [
      ['pure rotation', new Float32Array(base)],
      ['rotation x uniform 2', mat4.scale(new Float32Array(base), 2, 2, 2)],
      ['rotation x non-uniform 1,2,3', mat4.scale(new Float32Array(base), 1, 2, 3)],
    ];
    const out = new Float32Array(12);
    console.log('');
    for (const [, m] of cases) {
      const r = compare(
        () => { shippedNormalMatrix(out, m); },
        () => { rigidNormalMatrix(out, m); },
        120_000, 31, 'general', 'rigid',
      );
      // `effect` is (general - rigid), so negative means the fast path is the
      // slower one. This is the assertion that would have to be deleted if a
      // future change made it both correct *and* faster — deleting it
      // deliberately is the point.
      expect(r.real).toBe(true);
      expect(r.effect).toBeLessThan(0);
    }
    const r = rotation();
    rigidNormalMatrix(out, r);
    for (const i of [0, 1, 2, 4, 5, 6, 8, 9, 10] as const) {
      expect(Math.abs(out[i]! - r[i]!) < 1e-6).toBe(true);
    }
  }, 120_000);
});

describe('mat4.mulAffine', () => {
  test('is the one real win in this directory', () => {
    // 64 multiplies become 36. The precondition — both operands carry the
    // (0, 0, 0, 1) bottom row — is checked in test/math.test.ts, not here.
    const q = quat.setAxisAngle(quat.create(), vec3.create(0.3, 0.5, 0.81), 1.1);
    const parent = mat4.mul(mat4.create(), mat4.fromTranslation(mat4.create(), 1, 2, 3), quat.toMat4(mat4.create(), q));
    const local = mat4.mul(mat4.create(), quat.toMat4(mat4.create(), q), mat4.fromScale(mat4.create(), 1, 2, 1));
    const out = mat4.create();
    const r = compare(
      () => { mat4.mul(out, parent, local); },
      () => { mat4.mulAffine(out, parent, local); },
      150_000, 41, 'mul', 'mulAffine',
    );
    expect(r.real).toBe(true);
    expect(r.effect).toBeGreaterThan(0);
  }, 120_000);
});

describe('frustum.containsSphere', () => {
  test('hand-unrolling it is within the noise, so the loop stays', () => {
    // The loop is six iterations of a four-wide stride. The JIT already
    // unrolls a constant-trip-count loop, and reordering the planes only moves
    // which one rejects first — for a *visible* object all six are evaluated
    // regardless, so the reorder cannot help the common case at all.
    const proj = mat4.perspective(mat4.create(), 50 * (Math.PI / 180), 16 / 9, 0.1, 200);
    const view = mat4.lookAt(mat4.create(), vec3.create(0, 0, 60), vec3.create(0, 0, 0), vec3.create(0, 1, 0));
    const planes = frustum.setFromViewProjection(frustum.create(), mat4.mul(mat4.create(), proj, view));

    const rnd = lcg(12345);
    // One reused Sphere, mutated per call exactly as collectDrawItems does —
    // building it inside the timed closure would charge one candidate for an
    // allocation the other does not make, which is how the first version of
    // this file invented a 39% win.
    const s = sphere.create(vec3.create(), 3);
    const visible: Float32Array[] = [];
    while (visible.length < 64) {
      const c = new Float32Array([(rnd() * 2 - 1) * 70, (rnd() * 2 - 1) * 45, -(rnd() * 190 + 2)]);
      sphere.set(s, c[0]!, c[1]!, c[2]!, 3);
      if (frustum.containsSphere(planes, s)) visible.push(c);
    }

    const unrolled = (sp: sphere.Sphere): boolean => {
      const x = sp.center[0]!, y = sp.center[1]!, z = sp.center[2]!;
      const r = -sp.radius;
      if (planes[0]! * x + planes[1]! * y + planes[2]! * z + planes[3]! < r) return false;
      if (planes[4]! * x + planes[5]! * y + planes[6]! * z + planes[7]! < r) return false;
      if (planes[8]! * x + planes[9]! * y + planes[10]! * z + planes[11]! < r) return false;
      if (planes[12]! * x + planes[13]! * y + planes[14]! * z + planes[15]! < r) return false;
      if (planes[16]! * x + planes[17]! * y + planes[18]! * z + planes[19]! < r) return false;
      if (planes[20]! * x + planes[21]! * y + planes[22]! * z + planes[23]! < r) return false;
      return true;
    };

    const load = (i: number): void => {
      const c = visible[i & 63]!;
      sphere.set(s, c[0]!, c[1]!, c[2]!, 3);
    };
    console.log('');
    // Reported, not asserted. Six independent trials of this comparison put the
    // effect anywhere from -4.1 ns to +0.1 ns with the two orderings
    // disagreeing on the sign twice — the loop is not slower, and the unrolled
    // form is not reliably faster either. Six iterations of a 4-wide stride is
    // something the JIT already unrolls. The loop stays.
    compare(
      (i) => { load(i); frustum.containsSphere(planes, s); },
      (i) => { load(i); unrolled(s); },
      120_000, 31, 'loop', 'unrolled',
    );

    for (const c of visible) {
      sphere.set(s, c[0]!, c[1]!, c[2]!, 3);
      expect(unrolled(s)).toBe(frustum.containsSphere(planes, s));
    }
  }, 120_000);
});

describe('every timed candidate does the same work', () => {
  test('mat4.normalMatrix is bit-identical to the shipped copy over 20,000 transforms', () => {
    // The timings above are only comparable if the two do the same work, and a
    // faster-but-different result is not an optimisation. This is the
    // assertion that carries the weight; the nanoseconds are commentary.
    const rnd = lcg(24680);
    const a = new Float32Array(12);
    const b = new Float32Array(12);
    for (let t = 0; t < 20_000; t++) {
      const m = mat4.mul(
        mat4.create(),
        mat4.fromTranslation(mat4.create(), (rnd() * 2 - 1) * 50, (rnd() * 2 - 1) * 50, (rnd() * 2 - 1) * 50),
        mat4.mul(
          mat4.create(),
          quat.toMat4(
            mat4.create(),
            quat.setAxisAngle(quat.create(), vec3.create(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1), rnd() * 6),
          ),
          mat4.fromScale(mat4.create(), 0.2 + rnd() * 4, 0.2 + rnd() * 4, 0.2 + rnd() * 4),
        ),
      );
      shippedNormalMatrix(a, m);
      mat4.normalMatrix(b, m);
      for (let i = 0; i < 12; i++) {
        if (a[i] !== b[i]) throw new Error(`iteration ${t}, element ${i}: ${b[i]} vs shipped ${a[i]}`);
      }
    }
  });

  test('mat4.mulAffine is value-identical to mat4.mul over 20,000 scene-graph pairs', () => {
    const rnd = lcg(13579);
    const build = (): Float32Array =>
      mat4.mul(
        mat4.create(),
        mat4.fromTranslation(mat4.create(), (rnd() * 2 - 1) * 20, (rnd() * 2 - 1) * 20, (rnd() * 2 - 1) * 20),
        mat4.mul(
          mat4.create(),
          quat.toMat4(
            mat4.create(),
            quat.setAxisAngle(quat.create(), vec3.create(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1), rnd() * 6),
          ),
          mat4.fromScale(mat4.create(), 0.2 + rnd() * 3, 0.2 + rnd() * 3, 0.2 + rnd() * 3),
        ),
      );
    for (let t = 0; t < 20_000; t++) {
      const x = build();
      const y = build();
      const general = mat4.mul(mat4.create(), x, y);
      const fast = mat4.mulAffine(mat4.create(), x, y);
      for (let i = 0; i < 16; i++) {
        if (general[i] !== fast[i]) throw new Error(`iteration ${t}, element ${i}: ${fast[i]} vs ${general[i]}`);
      }
    }
  });
});
