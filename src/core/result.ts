/**
 * A recoverable outcome, for failures that are expected rather than exceptional.
 *
 * apse's rule: **a wrong argument throws, a missing capability returns.**
 *
 * - `requestDevice` rejecting because the adapter lacks a limit you asked for is
 *   a bug in your code. Throw.
 * - A loader failing to find a file, or a capability probe discovering a device
 *   cannot do MSAA, is a fact about the world. Return.
 *
 * Making both the same shape teaches callers to wrap everything in `try/catch`,
 * which is strictly worse than not handling either. So they are different types,
 * and the compiler tells you which one you have before you run anything.
 */

import { fail } from './error.ts';

/** Recoverable failure. The set of reasons is finite and typed, not a string. */
export interface Err<Code extends string = string> {
  readonly ok: false;
  /** Branch on this. Never parse {@link Err.message}. */
  readonly code: Code;
  /** What was asked for, what was found, values interpolated. */
  readonly message: string;
  /** The one corrective action. */
  readonly fix: string;
  /** Anything the caller needs to decide, e.g. the reason string. */
  readonly context?: Readonly<Record<string, unknown>>;
}

export type Result<T, Code extends string = string> =
  | { readonly ok: true; readonly value: T }
  | Err<Code>;

/** Wraps a success. */
export function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

/** Wraps a failure. */
export function err<Code extends string>(code: Code, message: string, fix: string, context?: Record<string, unknown>): Err<Code> {
  return context === undefined ? { ok: false, code, message, fix } : { ok: false, code, message, fix, context };
}

/** Narrowing. Use in preference to testing `result.ok === true` in a branch. */
export function isOk<T, C extends string>(r: Result<T, C>): r is { ok: true; value: T } {
  return r.ok;
}

export function isErr<T, C extends string>(r: Result<T, C>): r is Err<C> {
  return !r.ok;
}

/** Unwraps, or throws with the failure's own message. The escape hatch. */
export function unwrap<T, C extends string>(r: Result<T, C>): T {
  if (r.ok) return r.value;
  // An `AseError`, not a bare `Error`: no bare `Error` escapes the public
  // surface, and a handler that catches everything must still be able to
  // branch on `code`. A Result's code belongs to whoever produced it rather than
  // to the catalog, so the code reported is the one for the *mistake* — asking
  // for the value of a failed Result — and the Err travels in the message, where
  // `unwrap` has always put it.
  fail('INVALID_USAGE', `${r.code}: ${r.message} — fix: ${r.fix}`, {
    why: 'unwrap() reads the value out of a Result that holds a failure. The failure is already fully described — its code, message and fix are the same three fields an AseError carries — so this throw exists only to make a missed check loud.',
    fix: 'Branch on the Result with isErr() and handle the failure, or use unwrapOr() to substitute a default.',
  });
}

/** Unwraps, or substitutes a default. */
export function unwrapOr<T, C extends string>(r: Result<T, C>, fallback: T): T {
  return r.ok ? r.value : fallback;
}

/**
 * Turns a callback that throws into a `Result`.
 *
 * For the places where an error has to become a value: a loop over many assets
 * where one failure should not abandon the rest, or an async boundary that must
 * not reject.
 */
export function attempt<T, C extends string = string>(
  fn: () => T,
  code: C,
  fix: string,
): Result<T, C> {
  try {
    return ok(fn());
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return err(code, message, fix, { cause: message });
  }
}
