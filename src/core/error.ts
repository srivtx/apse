/**
 * Typed error catalog.
 *
 * Every error apse throws carries five things:
 *   code    stable machine-readable slug — branch on this, never on `message`
 *   message what you asked for + what was actually there, values interpolated
 *   why     the technical rule that was violated
 *   fix     the single corrective action
 *   link    docs anchor for this exact code
 *
 * All five are present on every error. There is no code path that produces a
 * bare `Error`. If you find one, it is a bug in apse.
 *
 * Plus one derived field, {@link AseError.blame}, which says whose fault the
 * failure is. It exists because `INTERNAL_INVARIANT` is the most-raised code in
 * apse and its catalog text once claimed it "always indicates a bug in apse" —
 * which is false for most of the sites that raise it, and quietly relabels a
 * developer's own typo as a library fault for anyone branching on `code`.
 * `blame` is the field to branch on; `code` is the field to read.
 */

import { ERROR_BLAME, ERROR_CODES, ERROR_CATALOG } from './error-catalog.ts';
import type { AseErrorCode, ErrorBlame, ErrorDetail } from './error-catalog.ts';

export type { AseErrorCode, ErrorBlame, ErrorDetail } from './error-catalog.ts';

/** The options every catalog error is built from. */
export interface AseErrorOptions {
  why: string;
  fix: string;
  link?: string;
  detail?: ErrorDetail;
  cause?: unknown;
}

/**
 * Narrows an arbitrary value to a catalog code.
 *
 * The type of `code` is `AseErrorCode` everywhere it matters, but a type is not
 * a runtime check: a JavaScript caller, a value from JSON, or an `any` that
 * crossed an untyped boundary can still be a string that names no failure mode.
 * `fail` and the {@link AseError} constructor both refuse those at runtime, and
 * this is the predicate they use — also exported so a caller that *builds* a
 * code from data can check before raising it.
 */
export function isErrorCode(value: unknown): value is AseErrorCode {
  return typeof value === 'string' && Object.hasOwn(ERROR_CATALOG, value);
}

/** The error type every apse API throws. Narrow it with {@link isAseError}. */
export class AseError extends Error {
  readonly code: AseErrorCode;
  readonly why: string;
  readonly fix: string;
  readonly link: string;
  readonly detail: ErrorDetail | undefined;
  /** Whose fault this is: apse, the caller, or the environment. See `ERROR_BLAME`. */
  readonly blame: ErrorBlame;

  constructor(code: AseErrorCode, message: string, opts: AseErrorOptions) {
    // Checked in the constructor rather than in `fail()` because the constructor
    // is public and exported: `new AseError(someString, …)` would otherwise be
    // the one way to get an error whose `code` is not in the catalog, which
    // would void the typed-error guarantee for every handler downstream.
    if (!isErrorCode(code)) throw unrecognisedCode(code);
    super(message, opts.cause === undefined ? undefined : { cause: opts.cause });
    this.name = 'AseError';
    this.code = code;
    this.why = opts.why;
    this.fix = opts.fix;
    this.link = opts.link ?? `https://apse.dev/errors/${code.toLowerCase().replace(/_/g, '-')}`;
    this.detail = opts.detail;
    this.blame = ERROR_BLAME[code];

    // Keeps `instanceof` working when the library is consumed as a different
    // module instance (multiple copies in a dependency graph, dual bundles).
    Object.setPrototypeOf(this, AseError.prototype);
  }

  /** Single-line form: `CODE: message — fix: <fix>` */
  override toString(): string {
    return `${this.code}: ${this.message} — fix: ${this.fix}`;
  }

  /** Structured form for logging, JSON, or an error panel. */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      code: this.code,
      blame: this.blame,
      message: this.message,
      why: this.why,
      fix: this.fix,
      link: this.link,
      detail: this.detail,
    };
  }
}

/**
 * The error raised in place of one built from a code that does not exist.
 *
 * `INVALID_USAGE` because whatever passed the string is the thing that is
 * wrong: `fail` is apse's own constructor, so a slug outside the catalog is a
 * mistake at the call to `fail`, not a failure mode apse can report. Building it
 * from the catalog's own guidance keeps the `why` / `fix` text single-sourced.
 */
function unrecognisedCode(code: unknown): AseError {
  const guidance = ERROR_CATALOG.INVALID_USAGE;
  const shown = typeof code === 'string' ? `"${code}"` : `the value ${String(code)}`;
  return new AseError('INVALID_USAGE',
    `fail() was given ${shown}, which is not one of apse's error codes.`,
    {
      why: `${guidance.why} A code is the only thing a handler can branch on, and the set of codes is finite by construction — ${ERROR_CODES.length} of them — so a slug outside it cannot be handled.`,
      fix: 'Pass a code from ERROR_CODES (exported from apse and apse/core). If apse\'s own source produced the slug, it is a typo against ERROR_CATALOG.',
    });
}

export function isAseError(value: unknown): value is AseError {
  return value instanceof AseError ||
    (typeof value === 'object' && value !== null && (value as { name?: string }).name === 'AseError');
}

/**
 * Throws a catalog error. Signature is a discriminated union so TypeScript
 * checks that you passed the params the code's template actually needs —
 * a missing value is a compile error, not a runtime `undefined`.
 *
 * `code` is constrained to {@link AseErrorCode}, so a literal that is not in
 * the catalog fails to compile, and {@link AseError} refuses one at runtime.
 * The two together are what make "branch on `code`" safe: a value that reached
 * a handler is a code someone can look up.
 */
export function fail<C extends AseErrorCode>(code: C, message: string, opts: AseErrorOptions): never {
  throw new AseError(code, message, opts);
}
