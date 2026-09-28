/**
 * Reference-counted GPU resource ownership.
 *
 * This is the single biggest source of reported leaks in existing 3D
 * libraries: removing an object from a scene frees nothing, and calling
 * `dispose()` on a shared resource frees it for everyone. apse makes both
 * mistakes impossible.
 *
 *   - Sharing a resource is safe. Each user takes a reference.
 *   - Dropping a reference frees the GPU memory when the count reaches zero.
 *   - `dispose()` is still available for "free this now, I know I am done",
 *     and is what scene teardown uses.
 *
 * GC is never relied on. Browser GPU object lifetimes are not deterministic and
 * a dropped wrapper can keep a multi-megabyte buffer alive indefinitely.
 */

import { fail } from './error.ts';

export interface Disposable {
  /** Force-release this resource now, regardless of remaining references. */
  dispose(): void;
}

export abstract class Resource implements Disposable {
  #refs = 1;
  #disposed = false;

  /** Live references. Zero means the GPU resources are gone. */
  get refCount(): number {
    return this.#refs;
  }

  get disposed(): boolean {
    return this.#disposed;
  }

  /**
   * Take a reference. Call this when a second thing starts using the resource.
   * The returned value is `this`, so it can be assigned directly.
   */
  ref(): this {
    if (this.#disposed) {
      fail('INTERNAL_INVARIANT',
        `Tried to take a reference on a disposed ${this.constructor.name}.`, {
          why: 'Taking a reference on a released resource would hand out a handle to freed GPU memory.',
          fix: 'Check `.disposed` before re-using, or keep the resource alive with the reference you already hold.',
        });
    }
    this.#refs++;
    return this;
  }

  /**
   * Drop a reference. Releases GPU memory when the count reaches zero.
   * Safe to call on an already-disposed resource.
   */
  unref(): this {
    if (this.#disposed) return this;
    this.#refs--;
    if (this.#refs <= 0) this.dispose();
    return this;
  }

  /** Force-release immediately, whatever the reference count. */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#refs = 0;
    this.onDispose();
  }

  /** Release GPU resources. Called at most once. */
  protected abstract onDispose(): void;

  /** Throws a typed error if this resource has already been released. */
  protected assertLive(resource: string): void {
    if (this.#disposed) {
      fail('INTERNAL_INVARIANT',
        `${resource} was used after it was disposed.`, {
          why: 'The GPU buffers backing it are gone, so the resulting draw is undefined.',
          fix: 'Hold a reference with `.ref()` for as long as a second object still uses this resource.',
        });
    }
  }
}

/**
 * An ordered set of resources released together.
 *
 * Owning everything a mesh needs — geometry, material, per-mesh uniform
 * instance — in one scope means a mesh releases atomically. No partial state
 * where the buffers are gone but the pipeline is not.
 */
export class ResourceScope implements Disposable {
  readonly #owned = new Set<Disposable>();
  #disposed = false;

  get disposed(): boolean {
    return this.#disposed;
  }

  get size(): number {
    return this.#owned.size;
  }

  /** Take ownership of `r` and a reference to it. */
  own<T extends Resource>(r: T): T {
    if (this.#disposed) {
      r.dispose();
      return r;
    }
    if (!this.#owned.has(r)) {
      this.#owned.add(r);
      r.ref();
    }
    return r;
  }

  /** Take ownership of a plain disposable with no reference counting. */
  ownDisposable(d: Disposable): void {
    if (this.#disposed) {
      d.dispose();
      return;
    }
    this.#owned.add(d);
  }

  /** Release everything in reverse acquisition order. */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const items = [...this.#owned].reverse();
    this.#owned.clear();
    for (const r of items) {
      try {
        r.dispose();
      } catch {
        // A failing child must not strand its siblings.
      }
    }
  }
}
