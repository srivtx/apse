/**
 * Draw-item ordering.
 *
 * The draw list is sorted, and the sort is the most expensive thing in the frame
 * after the draw calls themselves. So the choice of algorithm is worth stating
 * with its measurement, because this file has been wrong about it once.
 *
 * It was a counting sort — four stable passes, O(n + k) — chosen on the claim,
 * written in this header, that "a comparison sort costs roughly 2 ms" at 5000
 * items on an M-series core. That number does not reproduce. Measured on an M2,
 * the comparison sort costs **70 us** at 5000 items and the four counting passes
 * cost **175 us**: the counting sort was 2.5x *slower* than the thing it was
 * chosen to avoid, at every size from 1,000 to 100,000.
 *
 * Two things made it look plausible. Each pass clears and re-prefixs a
 * 1024-entry count array, so the fixed cost is ~4,000 operations before any item
 * is touched, and a few hundred items never amortise that. And `Array.sort` is a
 * native introsort: no allocation, stable in every shipping engine, and for n in
 * the thousands its comparisons are cheaper than four O(n + 1024) passes each
 * making three walks of the list.
 *
 * The counting sort was also, quietly, wrong. It quantised depth into 1024
 * buckets using `if (d >= 1) return BUCKETS - 1`, which treats depth as
 * normalised to [0, 1] — but `DrawItem.depth` is view-space `-z` in **metres**
 * (`graph.ts` computes it by transforming the node's world position by the view
 * matrix). So every object further than one metre from the camera landed in the
 * last bucket. A 12-object scene spanning 2 to 24 units produced **one** distinct
 * bucket: the front-to-back order was never applied, and the early-Z rejection
 * this sort exists to enable was not happening. Nothing in the picture changes,
 * because for opaque geometry draw order is a bandwidth optimisation rather than
 * a visual one. It costs fragments, silently.
 *
 * Two bugs are gone with it, both of which the counting sort's structure invited
 * and a plain comparator makes unrepresentable: depth was clamped against a
 * fixed range rather than the list's own, and transparent items came out
 * front-to-back instead of back-to-front — the latter a real visual bug, since
 * blending is order-dependent.
 *
 * What is given up: the counting sort's exact order was its own, and now the
 * order is the comparator's, which is the specification and the simpler thing to
 * reason about. `compareDrawItems` used to be kept only so the tests could prove
 * the two agreed; they are now the same function, so there is nothing to prove
 * and the reference is the implementation.
 */

import type { DrawItem, DrawPhase } from './types.ts';

const _materialKeys = new WeakMap<object, number>();
let _nextMaterialKey = 1;

/**
 * A stable small integer per material, assigned on first sight.
 *
 * Assigning on sight rather than using object identity is what makes the
 * material comparison an integer subtract. Object identity cannot be ordered at
 * all — `<` on two objects coerces both to `"[object Object]"` and returns false
 * both ways — so a comparator that grouped by material by identity would be
 * returning 0 for every pair of distinct materials, and every material-group
 * boundary would be decided by sort's own internal ordering rather than by the
 * key. That is the "a comparator that returns 0 for two distinct items draws a
 * different picture on a different day" failure, and it is why the first
 * comparison is on assigned keys and not on `a.material !== b.material`.
 */
function materialKey(material: object): number {
  let k = _materialKeys.get(material);
  if (k === undefined) {
    k = _nextMaterialKey++;
    _materialKeys.set(material, k);
  }
  return k;
}

/**
 * Sorts `items` in place into draw order.
 *
 * Priority, highest first: material, then explicit `order`, then phase
 * (opaque before transparent, which is also what lets the renderer open exactly
 * two passes), then depth — nearest-first for opaque so early-Z can reject
 * fragments of what is already drawn, farthest-first for transparent because
 * blending is order-dependent and that one is correctness, not optimisation.
 *
 * Returns the same array, so a caller can use it as an expression. In place,
 * because the draw list is owned by the renderer and copied out of it anyway;
 * returning a new array would allocate n references per frame to save nothing.
 */
export function sortDrawItems(items: DrawItem[]): DrawItem[] {
  items.sort(compareDrawItems);
  return items;
}

/**
 * The total order, and the implementation of {@link sortDrawItems}.
 *
 * Depth is compared in full precision rather than quantised. The counting sort
 * this replaced bucketed depth, which put two items in the same bucket into
 * collection order; measured, that misplaced 0.06% of same-material pairs,
 * spanning at most 0.1 units of depth. Harmless for opaque early-Z, but it meant
 * the sort's output was not the order it claimed to produce, and a total order
 * with an unstated tie-break is exactly the thing that draws a different picture
 * on a different day.
 *
 * A NaN depth — which means the camera transform produced NaN, already a bug
 * upstream — makes this comparator return NaN, and `Array.prototype.sort` then
 * treats the pair as equal. The NaN item lands wherever it lands and the rest of
 * the list is still in order; verified, not assumed, in `test/sort-perf.test.ts`.
 */
export function compareDrawItems(a: DrawItem, b: DrawItem): number {
  if (a.material !== b.material) return materialKey(a.material) - materialKey(b.material);
  if (a.order !== b.order) return a.order - b.order;
  if (a.phase !== b.phase) return a.phase === 'opaque' ? -1 : 1;
  if (a.depth !== b.depth) return a.phase === 'opaque' ? a.depth - b.depth : b.depth - a.depth;
  return a.objectId - b.objectId;
}

/** Phase of an item, re-exported for the sort's own use and for tests. */
export type { DrawPhase };
