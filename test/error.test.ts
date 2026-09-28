/**
 * Error system tests.
 *
 * The bar here is *enforcement*, not presence. `ERROR_CATALOG` having 42 entries
 * with a `why` and a `fix` each proves nothing — the failure mode this file
 * exists to catch is a catalog that advertises a failure apse cannot produce, an
 * error that blames apse for the caller's typo, and a `fail()` that will happily
 * accept a code nobody can look up. Those are all runtime properties, so they
 * are all tested by triggering them.
 *
 * One test is compile-time only: the `@ts-expect-error` on a slug that is not in
 * the catalog fails `bun run typecheck` if the type-level constraint is ever
 * weakened, which a runtime test cannot see.
 */

import { describe, expect, test } from 'bun:test';
import { AseError, fail, isAseError, isErrorCode } from '../src/core/error.ts';
import {
  ERROR_BLAME,
  ERROR_CATALOG,
  ERROR_CODES,
  type AseErrorCode,
  type ErrorBlame,
} from '../src/core/error-catalog.ts';
import { Resource, ResourceScope, type Disposable, type DisposedCode } from '../src/core/resource.ts';
import { UniformBlock, buildUniformBlock } from '../src/core/uniform.ts';
import { FRAME_BLOCK, FRAME_FIELDS } from '../src/core/slot.ts';
import { attempt, err, isErr, isOk, ok, unwrap, unwrapOr } from '../src/core/result.ts';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function thrown(fn: () => unknown): AseError {
  let caught: unknown;
  let didThrow = false;
  try {
    fn();
  } catch (error) {
    caught = error;
    didThrow = true;
  }
  if (!didThrow || !isAseError(caught)) {
    throw new Error(`expected an AseError, got ${didThrow ? String(caught) : 'no throw'}`);
  }
  return caught;
}

/** A minimal concrete `Resource`, so the base class's own behaviour is testable. */
class TestResource extends Resource {
  readonly log: string[] = [];

  constructor(code?: DisposedCode, tag = 'released') {
    super(code);
    this.tag = tag;
  }

  readonly tag: string;

  /** The protected check, surfaced so a test can reach it the way a subclass does. */
  check(label: string, code?: DisposedCode): void {
    this.assertLive(label, code);
  }

  protected onDispose(): void {
    this.log.push(this.tag);
  }
}

// ---------------------------------------------------------------------------
// The code is enforced at the type level
// ---------------------------------------------------------------------------

describe('fail() — the code, at compile time', () => {
  // Never invoked. `bun test` sees a function; `tsc --noEmit` sees the directive
  // below, and fails if `fail` ever stops constraining `code` to the catalog.
  // Nothing a runtime test does can check that, which is why the constraint is
  // proved here rather than asserted.
  const wouldNotCompile = (): never => {
    // @ts-expect-error 'MESH_ALIGNMENT' is not in ERROR_CATALOG.
    return fail('MESH_ALIGNMENT', 'not a code', { why: 'w', fix: 'f' });
  };

  test('a slug outside the catalog is a compile error, not a runtime surprise', () => {
    expect(typeof wouldNotCompile).toBe('function');
  });

  test('a widened string is also a compile error, so a code cannot be built from data', () => {
    // The catalog is a union of literals, so `string` is not assignable to it.
    // This is the shape of "read the code out of a config file and pass it in",
    // and the @ts-expect-error is what proves the type-level half still holds
    // under `strict` and `verbatimModuleSyntax`.
    const fromData: string = 'NODE_CYCLE';
    const raise = (): never => {
      // @ts-expect-error `string` is wider than AseErrorCode.
      return fail(fromData, 'nope', { why: 'w', fix: 'f' });
    };
    // Widening the *type* does not invalidate the *value*: a string that happens
    // to hold a real code still raises it, which is what makes an untyped
    // boundary usable rather than merely safe.
    expect(thrown(raise).code).toBe('NODE_CYCLE');
  });
});

// ---------------------------------------------------------------------------
// ...and at runtime, for callers the type system cannot see
// ---------------------------------------------------------------------------

describe('fail() — the code, at runtime', () => {
  test('a code that is not in the catalog is refused, with the bad value named', () => {
    const e = thrown(() =>
      (fail as (c: unknown, m: string, o: { why: string; fix: string }) => never)(
        'MESH_ALIGNMENT',
        'nope',
        { why: 'w', fix: 'f' },
      ),
    );
    expect(e.code).toBe('INVALID_USAGE');
    expect(e.message).toContain('MESH_ALIGNMENT');
    expect(e.blame).toBe('caller');
    // Five fields, plus blame, on the error that guards the other errors.
    expect(e.why.length).toBeGreaterThan(0);
    expect(e.fix.length).toBeGreaterThan(0);
    expect(e.link).toContain('invalid-usage');
  });

  test('the guard is not a hole either: it names the catalog as the way out', () => {
    const e = thrown(() =>
      (fail as (c: unknown, m: string, o: { why: string; fix: string }) => never)(
        'NOT_A_CODE',
        'nope',
        { why: 'w', fix: 'f' },
      ),
    );
    expect(e.fix).toContain('ERROR_CODES');
  });

  test('a non-string code is refused as firmly as a wrong string', () => {
    for (const bogus of [undefined, null, 42, {}, [], Symbol('x')]) {
      const e = thrown(() =>
        (fail as (c: unknown, m: string, o: { why: string; fix: string }) => never)(
          bogus,
          'nope',
          { why: 'w', fix: 'f' },
        ),
      );
      expect(e.code).toBe('INVALID_USAGE');
    }
  });

  test('the constructor is the same chokepoint, because it is public and exported', () => {
    const e = thrown(() => new AseError('TOTALLY_MADE_UP' as AseErrorCode, 'nope', { why: 'w', fix: 'f' }));
    expect(e.code).toBe('INVALID_USAGE');
    expect(isAseError(e)).toBe(true);
  });

  test('isErrorCode accepts every catalog code and rejects inherited Object keys', () => {
    for (const code of ERROR_CODES) expect(isErrorCode(code)).toBe(true);
    // `ERROR_CATALOG` is a plain object, so an `in` check or a bare truthiness
    // test would accept 'constructor' and 'toString' — both of which a
    // config-driven code could easily contain.
    for (const bogus of ['constructor', 'toString', '__proto__', 'hasOwnProperty', '']) {
      expect(isErrorCode(bogus)).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// The catalog is a finite, enumerable, and honest failure surface
// ---------------------------------------------------------------------------

describe('the catalog', () => {
  test('every code has guidance, and every code is enumerable exactly once', () => {
    expect(ERROR_CODES.length).toBe(Object.keys(ERROR_CATALOG).length);
    for (const code of ERROR_CODES) {
      const guidance = ERROR_CATALOG[code];
      expect(typeof guidance.why).toBe('string');
      expect(typeof guidance.fix).toBe('string');
      // Guidance that says nothing is worse than none: it is a field the
      // contract promises is actionable.
      expect(guidance.why.length).toBeGreaterThan(20);
      expect(guidance.fix.length).toBeGreaterThan(10);
    }
  });

  test('the catalog and the blame table cannot drift apart', () => {
    for (const code of ERROR_CODES) {
      expect(ERROR_BLAME[code]).toBeDefined();
    }
    // And the reverse: a blame row for a code that no longer exists would be a
    // silent hole in the mapping, which is why both tables are mapped types.
    for (const code of Object.keys(ERROR_BLAME)) {
      expect(isErrorCode(code)).toBe(true);
    }
  });

  test('exactly one code is apse\'s fault, and it is INTERNAL_INVARIANT', () => {
    const library = ERROR_CODES.filter((code) => ERROR_BLAME[code] === 'library');
    expect(library).toEqual(['INTERNAL_INVARIANT']);
  });

  test('only INTERNAL_INVARIANT claims apse is always at fault', () => {
    // The defect this guards: INTERNAL_INVARIANT used to read "This always
    // indicates a bug in apse, not in your code" while being the most-raised
    // code in the library, so a developer's typo was relabelled a library fault
    // for anyone branching on `code`.
    for (const code of ERROR_CODES) {
      if (code === 'INTERNAL_INVARIANT') continue;
      expect(ERROR_BLAME[code]).not.toBe('library');
    }
    expect(ERROR_CATALOG.INTERNAL_INVARIANT.why).not.toContain('always indicates a bug in apse');
  });

  test('blame reaches the error, so a handler can branch on it', () => {
    const e = thrown(() => fail('OPTION_UNKNOWN', 'nope', { why: 'w', fix: 'f' }));
    expect(e.blame).toBe('caller');
    expect(e.toJSON().blame).toBe('caller');
    expect(e.link).toBe('https://apse.dev/errors/option-unknown');
    expect(e.toString()).toBe(`OPTION_UNKNOWN: nope — fix: f`);
  });

  test('the codes removed for being unreachable stay removed', () => {
    // Both were advertised and could not fire: one `varyings` declaration
    // generates the struct for both stages, so a cross-stage type mismatch has
    // no second place to come from, and the scaffold always emits `vs`/`fs`, so
    // a missing entry point is not a failure mode either. A code nobody can
    // reach is a lie in a public API.
    for (const removed of ['VARYING_MISMATCH', 'SHADER_NO_ENTRYPOINT']) {
      expect(isErrorCode(removed)).toBe(false);
      expect(ERROR_CATALOG).not.toHaveProperty(removed);
    }
  });

  test('the two lifecycle codes a disposed resource can now raise exist', () => {
    // These were the other two dead codes. `Resource.assertLive` takes the code
    // as a parameter, so a material reports MATERIAL_DISPOSED and a mesh
    // MESH_DISPOSED — the catalog entry is finally backed by a code path.
    expect(ERROR_BLAME.MATERIAL_DISPOSED).toBe('caller');
    expect(ERROR_BLAME.MESH_DISPOSED).toBe('caller');
    expect(ERROR_CATALOG.MESH_DISPOSED.fix).toContain('ref()');
  });
});

// ---------------------------------------------------------------------------
// Disposal is observable
// ---------------------------------------------------------------------------

describe('Resource — disposal', () => {
  test('ref() on a released resource is a caller error, not an apse bug', () => {
    const r = new TestResource();
    r.dispose();
    const e = thrown(() => r.ref());
    expect(e.code).toBe('RESOURCE_DISPOSED');
    expect(e.blame).toBe('caller');
    expect(e.message).toContain('disposed');
    expect(e.detail).toEqual({ kind: 'lifecycle', resource: 'TestResource', state: 'destroyed' });
  });

  test('disposal state is readable, not only raisable', () => {
    const r = new TestResource();
    expect(r.disposed).toBe(false);
    expect(r.disposedCode).toBe('INTERNAL_INVARIANT');
    r.dispose();
    expect(r.disposed).toBe(true);
    expect(r.refCount).toBe(0);
  });

  test('assertLive raises the code the caller names, so a material and a mesh differ', () => {
    // The mechanism defect 3 needed: one check, kind-specific code. Both codes
    // are in the catalog, so the message, why, fix and link are all filled in.
    const material = new TestResource();
    material.dispose();
    expect(thrown(() => material.check('Material "basic"', 'MATERIAL_DISPOSED')).code).toBe('MATERIAL_DISPOSED');
    expect(thrown(() => material.check('Material "basic"', 'MATERIAL_DISPOSED')).blame).toBe('caller');

    const mesh = new TestResource();
    mesh.dispose();
    const e = thrown(() => mesh.check('Mesh "box"', 'MESH_DISPOSED'));
    expect(e.code).toBe('MESH_DISPOSED');
    expect(e.message).toBe('Mesh "box" was used after it was disposed.');
    expect(e.link).toBe('https://apse.dev/errors/mesh-disposed');
  });

  test('a subclass that names itself in super() gets that code everywhere', () => {
    // The one-line migration for Material and GpuMesh: `super('MATERIAL_DISPOSED')`.
    class MyMaterial extends Resource {
      constructor() {
        super('MATERIAL_DISPOSED');
      }
      checkIt(): void {
        this.assertLive('Material "m"');
      }
      protected onDispose(): void {}
    }
    const m = new MyMaterial();
    expect(m.disposedCode).toBe('MATERIAL_DISPOSED');
    m.dispose();
    expect(thrown(() => m.checkIt()).code).toBe('MATERIAL_DISPOSED');
  });

  test('assertLive is a no-op while the resource is live', () => {
    const r = new TestResource();
    expect(() => r.check('Material "m"', 'MATERIAL_DISPOSED')).not.toThrow();
  });

  test('onDispose runs exactly once, and unref on a released resource is safe', () => {
    const r = new TestResource();
    r.ref();
    r.unref();
    expect(r.disposed).toBe(false);
    r.unref();
    expect(r.disposed).toBe(true);
    expect(r.log).toEqual(['released']);
    expect(() => r.unref()).not.toThrow();
    r.dispose();
    expect(r.log).toEqual(['released']);
  });
});

describe('ResourceScope', () => {
  test('releases owned resources in reverse acquisition order, once each', () => {
    const scope = new ResourceScope();
    const a = new TestResource(undefined, 'a');
    const b = new TestResource(undefined, 'b');
    scope.own(a);
    scope.own(b);
    expect(a.refCount).toBe(2);
    expect(scope.size).toBe(2);

    scope.dispose();
    // Reverse acquisition order, so a resource always outlives the thing that
    // was using it.
    expect(b.log).toEqual(['b']);
    expect(a.log).toEqual(['a']);
    expect(a.disposed).toBe(true);
    expect(scope.disposed).toBe(true);
    scope.dispose();
    expect(b.log).toEqual(['b']);
  });

  test('a child that throws on dispose does not strand its siblings', () => {
    const done: string[] = [];
    const bad: Disposable = { dispose: () => { throw new Error('boom'); } };
    const good: Disposable = { dispose: () => done.push('good') };
    const scope = new ResourceScope();
    scope.ownDisposable(bad);
    scope.ownDisposable(good);
    expect(() => scope.dispose()).not.toThrow();
    expect(done).toEqual(['good']);
  });

  test('owning into a disposed scope disposes the resource instead of stranding it', () => {
    const scope = new ResourceScope();
    scope.dispose();
    const r = new TestResource();
    expect(scope.own(r)).toBe(r);
    expect(r.disposed).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Uniform blocks: a bad field name is a caller's typo
// ---------------------------------------------------------------------------

describe('UniformBlock', () => {
  const spec = buildUniformBlock('Test', { scale: 'f32', colour: 'vec3f' });

  test('a field that does not exist is reported as the caller\'s, with the real names', () => {
    const block = new UniformBlock(spec);
    const e = thrown(() => block.set('scail', 2));
    expect(e.code).toBe('INVALID_USAGE');
    expect(e.blame).toBe('caller');
    expect(e.message).toContain('scail');
    expect(e.fix).toContain('scale');
    expect(e.fix).toContain('colour');
  });

  test('get() is checked the same way, because a typo there returns all zeroes', () => {
    const block = new UniformBlock(spec);
    expect(thrown(() => block.get('nope')).code).toBe('INVALID_USAGE');
  });

  test('a field that does exist is written, and read back', () => {
    const block = new UniformBlock(spec);
    block.set('scale', 2);
    block.set('colour', [1, 0.5, 0]);
    expect(block.get('scale')).toEqual([2]);
    expect(block.get('colour')).toEqual([1, 0.5, 0]);
  });

  test('the frame block every material draws with is a gapless, aligned field map', () => {
    // buildUniformBlock is the single source of truth for WGSL layout, so what
    // it produces is what the GPU reads. A field it cannot name is a caller's
    // typo; a field it can is at a real, aligned, non-overlapping offset.
    const spec = buildUniformBlock('FrameCopy', FRAME_FIELDS);
    expect(spec.fields.length).toBe(Object.keys(FRAME_FIELDS).length);
    let previousEnd = 0;
    for (const f of spec.fields) {
      expect(f.offset).toBeGreaterThanOrEqual(previousEnd);
      expect(f.offset % 4).toBe(0);
      expect(f.size).toBeGreaterThan(0);
      previousEnd = f.offset + f.size;
    }
    expect(spec.size).toBeGreaterThanOrEqual(previousEnd);
    // The reserved block that ships is built by the same function, so it agrees.
    expect(FRAME_BLOCK.size).toBe(spec.size);
    expect(FRAME_BLOCK.wgsl).toContain('struct Frame');
  });
});

// ---------------------------------------------------------------------------
// Result: a wrong argument throws, a missing capability returns
// ---------------------------------------------------------------------------

describe('Result', () => {
  test('a missing capability is a value, not an exception', () => {
    const r = err('NO_MSAA', 'This device cannot do 4x MSAA.', 'Fall back to sampleCount 1.');
    expect(isErr(r)).toBe(true);
    expect(isOk(r)).toBe(false);
    expect(r.ok ? 'ok' : r.code).toBe('NO_MSAA');
  });

  test('unwrap on a failure is a typed error, not a bare Error', () => {
    const e = thrown(() => unwrap(err('NO_MSAA', 'This device cannot do 4x MSAA.', 'Fall back to 1.')));
    expect(isAseError(e)).toBe(true);
    expect(e.code).toBe('INVALID_USAGE');
    // The failure the caller was handed survives the throw, in the message.
    expect(e.message).toContain('NO_MSAA');
    expect(e.message).toContain('Fall back to 1.');
  });

  test('unwrapOr substitutes without throwing, and unwrap passes the value through', () => {
    expect(unwrapOr(err('X', 'm', 'f'), 7)).toBe(7);
    expect(unwrap(ok(7))).toBe(7);
  });

  test('attempt turns a throw into a value without losing the reason', () => {
    const r = attempt(() => { throw new Error('no such file'); }, 'LOAD_FAILED', 'Check the path.');
    expect(isErr(r)).toBe(true);
    if (r.ok) throw new Error('expected a failure');
    expect(r.code).toBe('LOAD_FAILED');
    expect(r.message).toContain('no such file');
    expect(attempt(() => 3, 'LOAD_FAILED', 'x')).toEqual({ ok: true, value: 3 });
  });
});

describe('the blame taxonomy, end to end', () => {
  test('an error raised through fail() reports the blame of its code, not a constant', () => {
    const seen = new Map<ErrorBlame, AseErrorCode>();
    for (const code of ERROR_CODES) {
      // Every code must be constructible with a real why/fix, which is what
      // makes the table above a table of *reachability* rather than a wish.
      const e = new AseError(code, `raised ${code}`, ERROR_CATALOG[code]);
      expect(e.code).toBe(code);
      expect(e.blame).toBe(ERROR_BLAME[code]);
      seen.set(e.blame, code);
    }
    expect([...seen.keys()].sort()).toEqual(['caller', 'environment', 'library']);
  });
});
