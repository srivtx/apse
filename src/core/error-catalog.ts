/**
 * The error catalog — the single list of every failure apse can report.
 *
 * Adding a failure mode means adding a member here. That is the whole point:
 * the set of things that can go wrong is a finite, enumerable, typed value, so
 * a caller (or an agent) can discover it without reading the source.
 *
 * Two rules make the list honest rather than aspirational:
 *
 *   1. **Every code is reachable, and every member carries its `why` and `fix`.**
 *      A failure apse cannot actually produce is not a code. Where a mistake is
 *      structurally unrepresentable — one `varyings` declaration generates the
 *      struct for both stages, so they cannot disagree — the code is *absent*,
 *      and the guarantee comes from the type system instead.
 *   2. **Every code is classified by blame.** `ErrorBlame` says whose fault the
 *      failure is, and `ERROR_BLAME` is a mapped table, so the classification
 *      cannot drift from the code list. `INTERNAL_INVARIANT` is the only code
 *      classified `'library'`: a handler that treats it as the general-purpose
 *      "something went wrong" bucket ends up filing its users' typos under
 *      "apse is broken", which is the one thing a typed catalog must not do.
 */

/** Detail payload for codes that carry structured context. */
export type ErrorDetail =
  | { kind: 'attribute'; attribute: string; format: string; expected: string; got: number }
  | { kind: 'uniform'; block: string; field: string; offset: number; size: number }
  | { kind: 'varying'; varying: string; declared: string; used: string }
  | { kind: 'gpu-validation'; scope: string; raw: string }
  | { kind: 'capability'; feature: string; required: string; available: string }
  | { kind: 'numeric'; field: string; value: number; min?: number; max?: number }
  | { kind: 'lifecycle'; resource: string; state: 'created' | 'destroyed' | 'in-flight' };

export type ErrorCodeEntry =
  | { code: 'WEBGPU_UNAVAILABLE'; message: (o: { reason: string }) => string }
  | { code: 'ADAPTER_UNAVAILABLE'; message: (o: { reason: string }) => string }
  | { code: 'DEVICE_LOST'; message: (o: { reason: string }) => string }
  | { code: 'DEVICE_REQUEST_FAILED'; message: (o: { reason: string; requested: string }) => string }
  | { code: 'GPU_VALIDATION_FAILED'; message: (o: { operation: string; raw: string }) => string }
  | { code: 'CANVAS_CONTEXT_INVALID'; message: (o: { actual: string }) => string }
  | { code: 'CANVAS_CONTEXT_ALREADY_TAKEN'; message: (o: { prior: string }) => string }
  | { code: 'CANVAS_SIZE_INVALID'; message: (o: { width: number; height: number; max: number }) => string }
  | { code: 'ATTRIBUTE_FORMAT_UNKNOWN'; message: (o: { attribute: string; format: string }) => string }
  | { code: 'ATTRIBUTE_LAYOUT_OVERFLOW'; message: (o: { attribute: string; offset: number; stride: number; max: number }) => string }
  | { code: 'ATTRIBUTE_MISSING'; message: (o: { attribute: string; expected: string[]; got: string[] }) => string }
  | { code: 'LAYOUT_MISMATCH'; message: (o: { attribute: string; expected: string; got: string }) => string }
  | { code: 'MESH_INDEX_MISALIGNED'; message: (o: { vertexCount: number; components: number }) => string }
  | { code: 'MESH_EMPTY'; message: (o: { which: string }) => string }
  | { code: 'MESH_NO_POSITION'; message: (o: { attributes: string[] }) => string }
  | { code: 'MESH_DATA_TOO_LARGE'; message: (o: { bytes: number; max: number }) => string }
  | { code: 'SLOT_TYPE_UNKNOWN'; message: (o: { field: string; type: string; known: string[] }) => string }
  | { code: 'SLOT_VALUE_WRONG_LENGTH'; message: (o: { field: string; type: string; expected: number; got: number }) => string }
  | { code: 'SLOT_VALUE_NOT_FINITE'; message: (o: { field: string; value: number }) => string }
  | { code: 'SLOT_DEFAULT_INVALID'; message: (o: { field: string; reason: string }) => string }
  | { code: 'UNIFORM_BLOCK_OVERFLOW'; message: (o: { block: string; needed: number; max: number }) => string }
  | { code: 'VARYING_TOO_MANY'; message: (o: { count: number; max: number; locations: string }) => string }
  | { code: 'VARYING_LOCATION_OVERFLOW'; message: (o: { varying: string; location: number; max: number }) => string }
  | { code: 'VARYING_TYPE_UNSUPPORTED'; message: (o: { varying: string; type: string; supported: string[] }) => string }
  | { code: 'SHADER_COMPILE_FAILED'; message: (o: { stage: string; line: number; column: number; message: string }) => string }
  | { code: 'SHADER_BODY_INVALID'; message: (o: { stage: string; reason: string }) => string }
  | { code: 'TEXTURE_SLOT_MISSING'; message: (o: { name: string; declared: string[] }) => string }
  | { code: 'TEXTURE_SLOT_TYPE_INVALID'; message: (o: { name: string; type: string; supported: string[] }) => string }
  | { code: 'RENDER_TARGET_SIZE_INVALID'; message: (o: { width: number; height: number }) => string }
  | { code: 'RENDER_TARGET_FORMAT_MISMATCH'; message: (o: { target: string; expected: string; got: string }) => string }
  | { code: 'MATERIAL_DISPOSED'; message: (o: { name: string }) => string }
  | { code: 'MESH_DISPOSED'; message: (o: { name: string }) => string }
  | { code: 'RESOURCE_DISPOSED'; message: (o: { kind: string }) => string }
  | { code: 'NODE_REPARENTED'; message: (o: { name: string; from: string; to: string }) => string }
  | { code: 'NODE_CYCLE'; message: (o: { name: string }) => string }
  | { code: 'NODE_NOT_ATTACHED'; message: (o: { name: string }) => string }
  | { code: 'CAMERA_NOT_SET'; message: () => string }
  | { code: 'RENDERER_ALREADY_DISPOSED'; message: () => string }
  | { code: 'BUDGET_EXCEEDED'; message: (o: { budget: string; actual: number; limit: number; unit: string }) => string }
  | { code: 'OPTION_UNKNOWN'; message: (o: { option: string; known: string[] }) => string }
  | { code: 'INVALID_USAGE'; message: (o: { call: string; rule: string }) => string }
  | { code: 'INTERNAL_INVARIANT'; message: (o: { invariant: string; actual: string }) => string };

export type AseErrorCode = ErrorCodeEntry['code'];

/**
 * Whose fault a failure is.
 *
 * - `library` — apse is wrong. Its own tables, generated code, or caches
 *   disagree with each other. Report it; do not work around it.
 * - `caller` — the call is wrong. A documented precondition was violated, and
 *   the correction is in the caller's source.
 * - `environment` — a fact about the machine, the driver, or the browser. No
 *   code change fixes it; a different device, a capability probe, or a fallback
 *   does.
 *
 * The distinction is a separate value rather than an implication of the code so
 * that a handler can branch on it without a 42-entry switch. `INTERNAL_INVARIANT`
 * is `'library'`; every other code is `'caller'` or `'environment'`.
 */
export type ErrorBlame = 'library' | 'caller' | 'environment';

/** Static, non-templated half of an error: the `why` / `fix` pair. */
export interface ErrorGuidance {
  why: string;
  fix: string;
}

/**
 * Every code, and its static, non-templated half.
 *
 * Annotated rather than written as `Object.freeze({…})`: freshness — and with
 * it the excess-property check that makes a code in `ErrorCodeEntry` with no
 * entry here, or an entry here for a code that no longer exists, a compile
 * error — is lost the moment an object literal is passed through a function
 * call. The literal is annotated first and frozen second, so both halves of that
 * guarantee survive.
 */
const CATALOG: { [C in AseErrorCode]: ErrorGuidance } = {
  WEBGPU_UNAVAILABLE: {
    why: '`navigator.gpu` is undefined, so the browser exposes no WebGPU implementation. This is either a browser without support, a context where WebGPU is disabled by policy, or a non-secure origin.',
    fix: 'Serve over https:// or http://localhost, and confirm the browser is Chrome/Edge 113+, Safari 26+, or Firefox 141+. Check `chrome://gpu` if you are on Linux.',
  },
  ADAPTER_UNAVAILABLE: {
    why: 'WebGPU is present but `requestAdapter()` resolved to null. Usually a blocklisted driver, a headless run without a GPU, or a page in a background tab.',
    fix: 'Pass `powerPreference: "high-performance"`, or pass `forceFallbackAdapter: true` to run on a software adapter. In headless Chrome add `--enable-unsafe-swiftshader`.',
  },
  DEVICE_LOST: {
    why: 'The GPUDevice was lost. The browser reclaimed it after a driver reset, a tab was backgrounded too long, or `device.destroy()` was called.',
    fix: 'Re-create the Renderer. Guard the recreate behind a device-lost listener and repopulate every GPU resource — buffers and textures do not survive.',
  },
  DEVICE_REQUEST_FAILED: {
    why: 'The adapter does not satisfy the limits or features you asked for in `requestDevice()`.',
    fix: 'Request only what you strictly need. apse already requests compatibility-level limits; read the thrown message to see which specific limit was rejected.',
  },
  GPU_VALIDATION_FAILED: {
    why: 'A WebGPU validation error was raised during this operation. The raw Dawn/ANGLE text names the exact descriptor field that was wrong.',
    fix: 'Read the raw text — it is the authoritative message — and enable the WebGPU DevTools panel. apse wraps each risky call in an error scope during development builds.',
  },
  CANVAS_CONTEXT_INVALID: {
    why: 'A canvas can only ever have one context type. This canvas already has a context of a different kind, so `getContext("webgpu")` returned null.',
    fix: 'Use a fresh <canvas> element, or pass the same element consistently. Do not reuse a canvas that another library has already drawn to.',
  },
  CANVAS_CONTEXT_ALREADY_TAKEN: {
    why: 'A previous Renderer already called `getContext("webgpu")` on this canvas. WebGPU context configuration is not re-entrant across renderer instances.',
    fix: 'Dispose the previous Renderer first, or give each Renderer its own canvas.',
  },
  CANVAS_SIZE_INVALID: {
    why: 'A WebGPU surface cannot exceed the device `maxTextureDimension2D` limit.',
    fix: 'Clamp the backing-store size, or reduce the device pixel ratio cap.',
  },
  ATTRIBUTE_FORMAT_UNKNOWN: {
    why: 'The attribute format string is not one of the vertex formats apse knows how to pack and describe to the GPU.',
    fix: 'Use one of the formats in `VERTEX_FORMATS`. Adding a new one means adding it to that table in src/geometry/layout.ts.',
  },
  ATTRIBUTE_LAYOUT_OVERFLOW: {
    why: 'Interleaving the declared attributes produced a vertex stride larger than the device `maxVertexBufferArrayStride` limit.',
    fix: 'Drop or pack an attribute. For a normal+color mesh, 8-bit normalized attributes (unorm8x2/snorm8x4) usually fit where float32x4 does not.',
  },
  ATTRIBUTE_MISSING: {
    why: 'The material declares attributes the mesh does not provide, so the vertex buffer cannot satisfy the pipeline layout.',
    fix: 'Add the attribute to the geometry, or remove it from the material\'s `attributes`. Layouts must match exactly — there is no partial binding.',
  },
  LAYOUT_MISMATCH: {
    why: 'A mesh is being drawn with a material built for a different vertex layout. The attribute is present on both sides but has a different format.',
    fix: 'Use one shared layout constant for both. `STANDARD_LAYOUT` covers position/normal/uv for most meshes.',
  },
  MESH_INDEX_MISALIGNED: {
    why: 'A triangle topology needs a vertex count that is a multiple of 3 per component, and this mesh is neither indexed nor padded to one.',
    fix: 'Call `mesh.setIndex(...)` to index the geometry, or pad the position count up to a multiple of 3.',
  },
  MESH_EMPTY: {
    why: 'The mesh has no drawable data, so there is nothing to encode.',
    fix: 'Populate the buffer before adding the mesh to a scene, or skip adding it while it is still being built.',
  },
  MESH_NO_POSITION: {
    why: 'Every material needs a position attribute to place vertices, and this mesh has none.',
    fix: 'Every primitive in `apse/geometry` includes `position`. Custom meshes must too.',
  },
  MESH_DATA_TOO_LARGE: {
    why: 'The vertex or index buffer exceeds the device `maxBufferSize` limit.',
    fix: 'Split the mesh into smaller chunks, or run on a device with a larger limit.',
  },
  SLOT_TYPE_UNKNOWN: {
    why: 'The uniform slot type is not in the type table, so apse cannot compute its WGSL size, alignment, or packing.',
    fix: 'Use one of the listed types. Adding one means adding an entry to `UNIFORM_TYPES` in src/core/uniform.ts.',
  },
  SLOT_VALUE_WRONG_LENGTH: {
    why: 'The array you assigned to a uniform slot does not have the component count that slot\'s type requires.',
    fix: 'A vec3f slot needs exactly 3 numbers. The message states how many you passed.',
  },
  SLOT_VALUE_NOT_FINITE: {
    why: 'NaN or Infinity in a uniform silently poisons every fragment it touches — usually from a division by zero or an uninitialised field.',
    fix: 'Guard the computation that produced it. A non-finite uniform is a bug upstream, not in apse.',
  },
  SLOT_DEFAULT_INVALID: {
    why: 'The `default` you gave a uniform slot does not match the slot\'s type.',
    fix: 'Match the component count and numeric range of the declared type.',
  },
  UNIFORM_BLOCK_OVERFLOW: {
    why: 'The uniform block needs more bytes than `maxUniformBufferBindingSize` allows (64 KiB core, 16 KiB compatibility).',
    fix: 'Move large or rarely-read fields into a storage buffer, or split them behind a separate bind group.',
  },
  VARYING_TOO_MANY: {
    why: 'The fragment stage only has 16 inter-stage shader variables on the core profile (15 in compatibility), and each varying consumes one.',
    fix: 'Pack related varyings into a single vec4, or move the data to a storage buffer sampled by fragment index.',
  },
  VARYING_LOCATION_OVERFLOW: {
    why: 'Inter-stage locations are assigned sequentially from 0 and this one landed past the device limit.',
    fix: 'Remove varyings, or pass high-frequency data through a flat storage buffer instead.',
  },
  VARYING_TYPE_UNSUPPORTED: {
    why: 'This varying type cannot cross the rasteriser boundary.',
    fix: 'Use f32, i32, u32, vec2f, vec3f, vec4f, or mat4x4f, each marked `@interpolate(flat)` in the generated struct where appropriate.',
  },
  SHADER_COMPILE_FAILED: {
    why: 'The generated WGSL did not compile. Because apse generates the bindings, uniform structs, varying struct, and entry-point signatures, the error is almost always in your shader *body*.',
    fix: 'Read the reported line and column. They point into your body, not into generated code — pass `scaffold: true` to the material to dump the full generated WGSL.',
  },
  SHADER_BODY_INVALID: {
    why: 'The shader body is not a statement list, or it tries to declare a binding itself.',
    fix: 'Write statements only — no `fn`, no `@group`, no `@binding`. apse owns the signature and every declaration; your body only reads and writes `in`, `out`, `frame`, `obj`, `mat`, and textures.',
  },
  TEXTURE_SLOT_MISSING: {
    why: 'The shader body referenced a texture that the material never declared in `textures`.',
    fix: 'Declare it: `textures: { albedo: { kind: "2d" } }`. apse cannot generate a binding for a name it does not know about.',
  },
  TEXTURE_SLOT_TYPE_INVALID: {
    why: 'A texture slot declared a kind this material cannot bind.',
    fix: 'Use one of the supported kinds. The message lists them.',
  },
  RENDER_TARGET_SIZE_INVALID: {
    why: 'A render target was created with a zero or negative extent.',
    fix: 'Give the target a positive size, and guard against a zero-size canvas on first layout pass.',
  },
  RENDER_TARGET_FORMAT_MISMATCH: {
    why: 'The material\'s declared output format does not match the render target it is being drawn into.',
    fix: 'Set the material `targets: [{ format }]` to the target format, or pick a different target.',
  },
  MATERIAL_DISPOSED: {
    why: 'This material\'s GPU resources were released. Disposal happens when the last reference is dropped, not when you call `.dispose()`.',
    fix: 'If other meshes still use this material, hold a reference: `material.ref()`. Check `material.refCount`.',
  },
  MESH_DISPOSED: {
    why: 'This mesh\'s buffers were released and it can no longer be drawn. Disposal happens when the last reference drops, not when you call `.dispose()`.',
    fix: 'Hold a reference with `mesh.ref()` for as long as a second object still draws this mesh, and check `mesh.disposed` before re-using one.',
  },
  RESOURCE_DISPOSED: {
    why: 'This ref-counted resource has already been released, so taking another reference on it would hand out a handle to freed GPU memory. Disposal happens when the last reference drops, not when you call `.dispose()`.',
    fix: 'Take the reference before the resource is released, and check `.disposed` — which is readable on every `Resource` — before re-using one you are unsure about.',
  },
  NODE_REPARENTED: {
    why: 'A node was moved to a new parent while it was mid-traversal.',
    fix: 'Defer reparenting to the start of a frame, or mutate the hierarchy outside of `node.update()`.',
  },
  NODE_CYCLE: {
    why: 'The requested parent change would make a node its own ancestor.',
    fix: 'The scene graph is a tree, not a graph. Every node has exactly one parent and no ancestor may be a descendant.',
  },
  NODE_NOT_ATTACHED: {
    why: 'This operation requires the node to be in a scene, and it is not.',
    fix: 'Add it to the scene with `scene.add(node)` first.',
  },
  CAMERA_NOT_SET: {
    why: 'A render was requested without an active camera.',
    fix: 'Pass one: `renderer.render(scene, camera)`, or set `renderer.camera = camera` once.',
  },
  RENDERER_ALREADY_DISPOSED: {
    why: 'This Renderer was disposed. Its device, pipelines, and targets are gone.',
    fix: 'Create a new Renderer. apse does not support re-acquiring a device on a disposed renderer.',
  },
  BUDGET_EXCEEDED: {
    why: 'A declared performance budget was exceeded. apse tracks these in dev builds so regressions surface immediately instead of as a slow demo.',
    fix: 'Raise the budget deliberately with a comment, or fix the regression. Do not silence it.',
  },
  OPTION_UNKNOWN: {
    why: 'An option key was passed that apse does not recognise. This is almost always a typo or an option from a different version.',
    fix: 'Check the spelling. apse rejects unknown options rather than ignoring them, so typos never fail silently.',
  },
  INVALID_USAGE: {
    why: 'The call violated a documented precondition of an apse API, and no code in the catalog names that particular rule more precisely. The fault is in the call, not in apse — this code is classified `blame: "caller"`.',
    fix: 'Correct the argument at the call named in the message. The `why` states the rule that was broken, and the API that raised it documents that rule.',
  },
  INTERNAL_INVARIANT: {
    why: 'An expectation internal to apse did not hold: a generated table, a resolved field list, and the code that reads it disagree. This is the *only* code classified `blame: "library"`, so a handler that files every failure here as an apse bug also files its users\' typos there.',
    fix: 'Please report it with the operation you were performing and the full generated WGSL if a material was involved. If the message describes an argument *you* passed rather than an apse table, this is a misclassification in apse and the code you want is `INVALID_USAGE` or something more specific.',
  },
};

/** Frozen so the runtime membership check in `isErrorCode` cannot be defeated. */
export const ERROR_CATALOG: { [C in AseErrorCode]: ErrorGuidance } = Object.freeze(CATALOG);

/**
 * Blame for every code, keyed the same way as the catalog.
 *
 * Mapped rather than kept as a parallel hand-maintained list: a code added to
 * `ErrorCodeEntry` with no row here is a compile error, so the classification
 * cannot fall behind the failure surface. See `INTERNAL_INVARIANT` for why the
 * `'library'` row is exactly one.
 */
const BLAME: { [C in AseErrorCode]: ErrorBlame } = {
  WEBGPU_UNAVAILABLE: 'environment',
  ADAPTER_UNAVAILABLE: 'environment',
  DEVICE_LOST: 'environment',
  DEVICE_REQUEST_FAILED: 'environment',

  GPU_VALIDATION_FAILED: 'caller',
  CANVAS_CONTEXT_INVALID: 'caller',
  CANVAS_CONTEXT_ALREADY_TAKEN: 'caller',
  CANVAS_SIZE_INVALID: 'caller',
  ATTRIBUTE_FORMAT_UNKNOWN: 'caller',
  ATTRIBUTE_LAYOUT_OVERFLOW: 'caller',
  ATTRIBUTE_MISSING: 'caller',
  LAYOUT_MISMATCH: 'caller',
  MESH_INDEX_MISALIGNED: 'caller',
  MESH_EMPTY: 'caller',
  MESH_NO_POSITION: 'caller',
  MESH_DATA_TOO_LARGE: 'caller',
  SLOT_TYPE_UNKNOWN: 'caller',
  SLOT_VALUE_WRONG_LENGTH: 'caller',
  SLOT_VALUE_NOT_FINITE: 'caller',
  SLOT_DEFAULT_INVALID: 'caller',
  UNIFORM_BLOCK_OVERFLOW: 'caller',
  VARYING_TOO_MANY: 'caller',
  VARYING_LOCATION_OVERFLOW: 'caller',
  VARYING_TYPE_UNSUPPORTED: 'caller',
  SHADER_COMPILE_FAILED: 'caller',
  SHADER_BODY_INVALID: 'caller',
  TEXTURE_SLOT_MISSING: 'caller',
  TEXTURE_SLOT_TYPE_INVALID: 'caller',
  RENDER_TARGET_SIZE_INVALID: 'caller',
  RENDER_TARGET_FORMAT_MISMATCH: 'caller',
  MATERIAL_DISPOSED: 'caller',
  MESH_DISPOSED: 'caller',
  RESOURCE_DISPOSED: 'caller',
  NODE_REPARENTED: 'caller',
  NODE_CYCLE: 'caller',
  NODE_NOT_ATTACHED: 'caller',
  CAMERA_NOT_SET: 'caller',
  RENDERER_ALREADY_DISPOSED: 'caller',
  BUDGET_EXCEEDED: 'caller',
  OPTION_UNKNOWN: 'caller',
  INVALID_USAGE: 'caller',

  INTERNAL_INVARIANT: 'library',
};

export const ERROR_BLAME: { [C in AseErrorCode]: ErrorBlame } = Object.freeze(BLAME);

/** Every code, for runtime enumeration. Agents can call this to discover the failure surface. */
export const ERROR_CODES: readonly AseErrorCode[] = Object.freeze(Object.keys(ERROR_CATALOG) as AseErrorCode[]);
