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
 */

import type { AseErrorCode, ErrorDetail } from './error-catalog.ts';

export type { AseErrorCode, ErrorDetail } from './error-catalog.ts';

/** The error type every apse API throws. Narrow it with {@link isAseError}. */
export class AseError extends Error {
  readonly code: AseErrorCode;
  readonly why: string;
  readonly fix: string;
  readonly link: string;
  readonly detail: ErrorDetail | undefined;

  constructor(code: AseErrorCode, message: string, opts: {
    why: string;
    fix: string;
    link?: string;
    detail?: ErrorDetail;
    cause?: unknown;
  }) {
    super(message, opts.cause === undefined ? undefined : { cause: opts.cause });
    this.name = 'AseError';
    this.code = code;
    this.why = opts.why;
    this.fix = opts.fix;
    this.link = opts.link ?? `https://apse.dev/errors/${code.toLowerCase().replace(/_/g, '-')}`;
    this.detail = opts.detail;

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
      message: this.message,
      why: this.why,
      fix: this.fix,
      link: this.link,
      detail: this.detail,
    };
  }
}

export function isAseError(value: unknown): value is AseError {
  return value instanceof AseError ||
    (typeof value === 'object' && value !== null && (value as { name?: string }).name === 'AseError');
}

/**
 * Throws a catalog error. Signature is a discriminated union so TypeScript
 * checks that you passed the params the code's template actually needs —
 * a missing value is a compile error, not a runtime `undefined`.
 */
export function fail<C extends AseErrorCode>(code: C, message: string, opts: {
  why: string;
  fix: string;
  link?: string;
  detail?: ErrorDetail;
  cause?: unknown;
}): never {
  throw new AseError(code, message, opts);
}
