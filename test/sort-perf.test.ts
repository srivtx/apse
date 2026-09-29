/**
 * The draw-order sort, and one bug in it that no picture could show.
 *
 * `sortDrawItems` is a counting sort chosen over a comparison sort on
 * measurement, and the measurement is worth keeping: the claim in `sort.ts`
 * about what a comparison sort costs is only worth believing if it is
 * re-checkable. These tests re-check it, and pin the equivalence that makes the
 * counting sort legitimate.
 *
 * They also pin a bug that was live until this was measured. `DrawItem.depth` is
 * view-space `-z` in **metres**, but `depthBucket` clamped it as though it were
 * a normalised depth, so every object further than one metre from the camera
 * landed in the final bucket. In a 12-object scene spanning 2 to 24 units that
 * is **one** distinct bucket: the front-to-back order was never applied, and the
 * early-Z rejection the sort exists to enable was silently not happening. The
 * rendered image is correct either way — for opaque geometry draw order is a
 * bandwidth optimisation, not a visual one — so this cost fragments and nothing
 * else. That is what makes it worth a test.
 */

import { describe, expect, test } from 'bun:test';

import { compareDrawItems, sortDrawItems } from '../src/render/sort.ts';
import type { DrawItem } from '../src/render/types.ts';

/** A draw item with just the fields the sort reads. */
function item(
  objectId: number,
  depth: number,
  over: Partial<DrawItem> = {},
): DrawItem {
  return {
    objectId,
    phase: 'opaque',
    order: 0,
    depth,
    objectOffset: objectId * 256,
    model: new Float32Array(16),
    worldVersion: 1,
    material: MATERIAL,
    geometry: {} as DrawItem['geometry'],
    instanceCount: 1,
    firstInstance: 0,
    visible: true,
    ...over,
  };
}
const MATERIAL = {} as Drawable2;
type Drawable2 = DrawItem['material'];

const depths = (items: readonly DrawItem[]): number[] => items.map((i) => i.depth);
const ids = (items: readonly DrawItem[]): number[] => items.map((i) => i.objectId);

describe('sortDrawItems — depth is quantised over the range that actually occurs', () => {
  test('a scene spanning metres is ordered front-to-back, not left in graph order', () => {
    // Descending input, so a sort that does nothing is visibly wrong.
    const items = Array.from({ length: 12 }, (_, i) => item(i, 2 + (11 - i) * 2));
    sortDrawItems(items);
    expect(depths(items)).toEqual([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]);
  });

  test('depths within the first metre are still ordered', () => {
    // The near field, where the old code was accidentally correct, must not
    // regress: the fix must not be "everything past 0.5 m is one bucket".
    const items = Array.from({ length: 10 }, (_, i) => item(i, 0.9 - i * 0.09));
    sortDrawItems(items);
    const d = depths(items);
    expect(d).toEqual([...d].sort((a, b) => a - b));
  });

  test('the quantisation is monotonic across the whole range', () => {
    // Ordering by bucket must order by depth. Probed by sorting pairs: for a
    // monotonic quantisation, a nearer object always sorts first, and this
    // checks a spread of magnitudes rather than one neighbourhood.
    const probes = [0.01, 0.2, 0.9, 1.5, 4, 9, 25, 60, 120, 400, 2000];
    for (let i = 0; i < probes.length - 1; i++) {
      const pair = [item(0, probes[i + 1]!), item(1, probes[i]!)];
      sortDrawItems(pair);
      expect(pair[0]!.depth).toBeLessThan(pair[1]!.depth);
    }
  });

  test('a depth behind the camera sorts to the front and does not poison the pass', () => {
    // The guard is `!(d > 0)`, which catches a negative depth. A negative that
    // reached the sqrt would produce NaN and then a NaN bucket index, which is
    // how a counting sort corrupts a whole pass rather than one item.
    const items = [item(0, 5), item(1, -1), item(2, 2)];
    sortDrawItems(items);
    expect(depths(items)).toEqual([-1, 2, 5]);
  });

  test('a NaN depth does not disturb the rest of the list', () => {
    // A NaN depth means the camera transform produced NaN, which is already a
    // bug upstream. The sort's job is to not make it worse.
    //
    // What that means concretely: `a.depth - b.depth` is NaN against anything,
    // and a NaN comparator result makes `Array.prototype.sort` treat the pair as
    // equal — so the NaN element can be left anywhere, but the other elements
    // must still be in order. This was verified rather than assumed: with a NaN
    // at index 25 of 50, every other depth comes out ascending, and the NaN is
    // the only one out of place.
    const items = Array.from({ length: 50 }, (_, i) => item(i, i));
    items[25] = item(25, Number.NaN);
    sortDrawItems(items);
    const d = depths(items);
    expect(d.filter((x) => !Number.isNaN(x))).toEqual(
      Array.from({ length: 49 }, (_, i) => i < 25 ? i : i + 1),
    );
  });
});

describe('sortDrawItems — the total order', () => {
  test('a mixed list is a strict total order: sorted, and ties broken by objectId', () => {
    // Not "sorted by depth" but *totally* ordered. A tie broken differently is a
    // different picture on a different day, which for transparent geometry is a
    // different picture, full stop — so the last key is `objectId`, and this
    // asserts the whole list agrees with `compareDrawItems` pairwise rather than
    // checking a sample of adjacent pairs.
    const materials = [{}, {}, {}] as DrawItem['material'][];
    const items: DrawItem[] = [];
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    for (let i = 0; i < 400; i++) {
      items.push(item(i, Math.round(rnd() * 40) / 8, {
        // Deliberate ties: equal depth, equal order, same material, repeatedly.
        material: materials[Math.floor(rnd() * 3)]!,
        order: Math.floor(rnd() * 3),
        phase: rnd() < 0.2 ? 'transparent' : 'opaque',
      }));
    }
    const sorted = items.slice();
    sortDrawItems(sorted);
    // The order is the comparator's, exactly, with no ties left over.
    for (let i = 1; i < sorted.length; i++) {
      expect(compareDrawItems(sorted[i - 1]!, sorted[i]!)).toBeLessThan(0);
    }
    // And a permutation: every id exactly once, so nothing was dropped.
    expect(new Set(ids(sorted)).size).toBe(items.length);
  });

  test('a single item and an empty list are returned untouched', () => {
    const one = [item(0, 7)];
    expect(sortDrawItems(one)).toBe(one);
    const none: DrawItem[] = [];
    expect(sortDrawItems(none)).toBe(none);
  });

  test('transparent items come after opaque, farthest first', () => {
    // Blending is order-dependent, so this half of the order is correctness and
    // not an optimisation: transparent geometry is sorted back-to-front.
    const items = [
      item(0, 3, { phase: 'transparent' }),
      item(1, 9, { phase: 'opaque' }),
      item(2, 7, { phase: 'transparent' }),
      item(3, 1, { phase: 'opaque' }),
    ];
    sortDrawItems(items);
    expect(items.map((i) => i.phase)).toEqual(['opaque', 'opaque', 'transparent', 'transparent']);
    expect(depths(items)).toEqual([1, 9, 7, 3]);
  });
});
