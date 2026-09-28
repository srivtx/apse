# Diagnosis: why apse is 1.32x slower per draw

## Measured, not inferred

Call census at 1000 boxes (920 drawn, 80 culled), from the project's own
FakeFrameDevice recording every WebGPU call:

    drawIndexed      920
    setBindGroup:1   921     <- object, one per draw (dynamic offset)
    setBindGroup:0     2     <- frame, once per pass
    setBindGroup:2     2     <- material, once per pipeline change
    setBindGroup:3     1     <- texture
    setPipeline        2
    setVertexBuffer:0  2
    setIndexBuffer     1

So per draw apse issues exactly:
  1 x setBindGroup(1, dynamicOffset)   <- the per-draw one
  1 x setVertexBuffer / setIndexBuffer (suppressed when unchanged)
  1 x drawIndexed

## The gap

apse per-draw marginal cost: 0.87 us. three.js: 0.66 us. Ratio 1.32x, stable
from 1k to 100k draws.

**Not the cause:**
- the scene-graph walk, cull and sort together are 6.2% / 5.4% of the frame
- uniform pack + upload is 2.7%
- the GPU is not it: spheres tie exactly, and instanced apse at 100k spends
  0.012 ms CPU against 87 ms for the per-draw path

**The cause:** per-draw encode is 85-100% of the frame, and it is dominated by
`setBindGroup`. apse uses **4 bind groups** (@0 frame, @1 object, @2 material,
@3 texture) and binds at least two of them per draw: @1 always, and @0/@2 when
the pipeline changes.

## The fix, and why it is structural

three.js packs per-object data into ONE uniform buffer addressed by dynamic
offset, in a single bind group. apse splits it across two: frame state in @0,
object state in @1. Because @0 must be bound too, the per-draw cost is two
setBindGroup calls where one suffices.

Merge @0 and @1 into one buffer and one bind group:
- one setBindGroup per draw instead of two
- the frame data is written once per frame at offset 0 of that buffer
- object i is at offset (i+1) * OBJECT_STRIDE

FRAME_BLOCK is ~400 bytes, OBJECT_BLOCK is 256 (stride). Both are far under
maxUniformBufferBindingSize (64 KiB) and the merged buffer is
(1 + maxObjects) * 256 bytes, which at 100k objects is 25.6 MB -- within
maxBufferSize but the ceiling on batch size must be documented and checked.

This is a real architectural change with a blast radius across
material.ts (layout + bind group construction), scaffold.ts (WGSL group
indices), renderer.ts (bind order, dynamic offsets), core/slot.ts (group
indices), and every generated shader. It is NOT a small patch.

## Second, smaller finding

`GpuMesh.vertexBuffer` / `.indexBuffer` are accessors that call assertLive.
renderer.ts read each one twice per draw (once in the comparison, once in the
call), so the liveness check ran twice where once suffices. Fixed in
src/render/renderer.ts by hoisting into locals; ~4 redundant accessor calls per
draw removed.

## Also noted

`AseError: A matrix handed to invert() is not invertible (determinant 0)` is
reported as INTERNAL_INVARIANT, which is `blame: 'library'`. A camera whose
lookAt target equals its own position is a caller error and should be
INVALID_USAGE. Found by tripping it while writing a probe.


## Correction, after implementing the fix

**The premise above was wrong, and implementing it proved it.**

The census shows `setBindGroup` for the *frame* group at **2** calls for 920 draws,
not 921. The `lastFrameBG` guard already suppressed it to once per pass. So
before this change the per-draw cost was **one** `setBindGroup` (the object group),
not two, and merging the frame into the object group removed a call that was
already not being made.

The per-draw cost, measured with 500 *distinct* meshes so nothing is suppressible:

    setBindGroup:0    501   scene (frame + object, one dynamic offset)
    setVertexBuffer:0 501
    setIndexBuffer    500
    drawIndexed       500
                      ---
    4.00 calls per draw

Four per draw, and **none of them is redundant**:

- one `setBindGroup` with the object's dynamic offset
- one `setVertexBuffer` — a different vertex buffer each time, because the
  meshes are different meshes
- one `setIndexBuffer` — likewise
- one `drawIndexed`

three.js's WebGLRenderer gets to roughly 4 too. This is close to the floor for
N distinct meshes on either API: you cannot draw a different buffer set without
binding it. The 0.87 us vs 0.66 us gap is **not** two bind groups. It is
probably the sort comparator, the draw-list bookkeeping, or the per-draw
`assertDrawable` and cache-line behaviour around the two buffer getters — none
of which has been isolated yet.

**What the merge is still worth, on its own terms:** one bind group instead of
two, one buffer instead of two, and no separate frame bind to keep correct. That
is real but it is a small constant, not the 1.32x. Presenting it as the fix
would be a second false claim in a file whose whole purpose is to stop them.

**Also found and fixed while doing this:** `ObjectUniforms.dispose()` and
`FrameUniforms.dispose()` destroyed the shared scene buffer, so disposing a
`PresentPass` freed the renderer's uniform storage and every other material on
the device silently lost its camera. The renderer allocates that buffer now, so
the renderer frees it.
