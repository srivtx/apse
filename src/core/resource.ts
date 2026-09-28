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

/**
 * The catalog code a released resource reports when it is used again.
 *
 * `INTERNAL_INVARIANT` is the fallback for a subclass that has not named
 * itself: it still fails loudly, which is the important half, but it files the
 * caller's mistake as apse's. A subclass that can name itself should pass the
 * specific code to `super()` — a caller who keeps drawing a released material is
 * not an apse bug, and `MATERIAL_DISPOSED` says so where `INTERNAL_INVARIANT`
 * says the opposite.
 */
export type DisposedCode =
  | 'MATERIAL_DISPOSED'
  | 'MESH_DISPOSED'
  | 'RESOURCE_DISPOSED'
  | 'INTERNAL_INVARIANT';

export abstract class Resource implements Disposable {
  #refs = 1;
  #disposed = false;
  readonly #disposedCode: DisposedCode;

  constructor(disposedCode: DisposedCode = 'INTERNAL_INVARIANT') {
    this.#disposedCode = disposedCode;
  }

  /** Live references. Zero means the GPU resources are gone. */
  get refCount(): number {
    return this.#refs;
  }

  /**
   * True once the GPU resources are gone. Readable, not just raisable: a
   * consumer that can ask "is this still alive" does not have to provoke an
   * error to find out, which is the whole difference between an observable
   * resource and an unobservable one.
   */
  get disposed(): boolean {
    return this.#disposed;
  }

  /**
   * The code {@link assertLive} raises for this resource. Exposed so a layer
   * that holds one can name the failure before it happens — in a log line, in a
   * frame-stats counter, or in a debug overlay — without catching a throw.
   */
  get disposedCode(): DisposedCode {
    return this.#disposedCode;
  }

  /**
   * Take a reference. Call this when a second thing starts using the resource.
   * The returned value is `this`, so it can be assigned directly.
   */
  ref(): this {
    if (this.#disposed) {
      // Two decisions, one place. The kind comes from the class because the
      // message is about this resource — though it is minified in a bundle, so
      // it is a hint and the code is the identifier. And the code is the generic
      // one rather than this class's own `disposedCode`, because taking a new
      // reference on something already released is one lifecycle mistake
      // whatever kind it is; the kind-specific codes describe *drawing* a
      // released resource.
      const kind = this.constructor.name;
      fail('RESOURCE_DISPOSED',
        `Tried to take a reference on a disposed ${kind}.`, {
        why: 'The reference count reached zero and the GPU memory behind it was released, so a new reference would hand out a handle to freed memory. Disposal happens when the last reference drops, not when you call dispose().',
        fix: 'Take the reference before the last one drops, or check `.disposed` before re-using a resource whose lifetime you are not tracking.',
        detail: { kind: 'lifecycle', resource: kind, state: 'destroyed' },
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

  /**
   * Throws if this resource has already been released.
   *
   * `code` is a parameter rather than a constant because the caller of the check
   * is the only thing that knows what kind of resource it is holding: a material
   * reports `MATERIAL_DISPOSED`, a mesh `MESH_DISPOSED`. `INTERNAL_INVARIANT`
   * remains the fallback for a subclass that has not named itself — it still
   * fails loudly, which is the important half, but it files the caller's mistake
   * as apse's. The base class will not guess a kind on its behalf, because a
   * wrong guess is worse than the code a subclass has always raised: pass one
   * here, or name the class in its `super()` call.
   */
  protected assertLive(resource: string, code: DisposedCode = this.#disposedCode): void {
    if (!this.#disposed) return;
    fail(code, `${resource} was used after it was disposed.`, {
      why: 'The GPU buffers backing it are gone, so the resulting draw reads freed memory — which in WebGPU is not a crash but a silently discarded command, and an empty frame.',
      fix: 'Hold a reference with `.ref()` for as long as a second object still uses this resource, and check `.disposed` before re-using one.',
      detail: { kind: 'lifecycle', resource, state: 'destroyed' },
    });
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
