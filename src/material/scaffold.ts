/**
 * The material scaffold — apse's core abstraction.
 *
 * A material here is a *specification*, not a shader program. The user writes
 * two statement lists; apse generates every WGSL struct, every `@group`, every
 * `@binding`, and the entry-point signatures around them. The user never writes
 * `fn`, never writes `@group`, and never writes `struct`.
 *
 * This exists because the alternative — handing users WGSL to string-surgery —
 * is what makes shader customization in other libraries unusable:
 *
 *   - Overriding a chunk means knowing dozens of internal chunk names that are
 *     not a stable API and change between releases.
 *   - The same uniform value is reachable through three different paths, one of
 *     which is an implementation detail.
 *   - The program cache key is a hundred-component string join, recomputed on
 *     every cache miss, plus a mandatory undocumented escape hatch.
 *   - A custom material has to hand-declare `modelViewMatrix`, `projectionMatrix`,
 *     `modelMatrix`, `normalMatrix`, `cameraPosition`, and `#ifdef` blocks for
 *     fog, clipping, morph targets, and skinning — none of which exist here.
 *
 * Every one of those is impossible below, because every one of them is something
 * apse now generates from data.
 *
 * # What the body can see, and nothing else
 *
 * | name          | vertex stage                    | fragment stage         |
 * |---------------|---------------------------------|------------------------|
 * | `in`          | `VertexIn` fields (attributes)  | `Varyings` fields      |
 * | `out`         | `Varyings` fields               | — (not available)      |
 * | `frame`       | `FRAME_BLOCK` fields            | same                   |
 * | `obj`         | `OBJECT_BLOCK` fields           | same                   |
 * | `mat`         | the declared `slots`            | same                   |
 * | `<texture>`   | one variable per texture slot   | same                   |
 * | `<texture>Sampler` | matching sampler variable   | same                   |
 *
 * Plus WGSL's builtin library: `textureSample`, `textureSampleLevel`,
 * `textureSampleBias`, `dot`, `mix`, `normalize`, and the rest.
 *
 * # Validation is a feature, not a chore
 *
 * The body validator here is the single most valuable thing in this module for
 * anyone — human or agent — writing a material. A typo like `frame.viewProjj`
 * or a stray `fn helper()` fails as a typed {@link AseError} naming the field
 * and listing what exists, instead of surfacing as a WGSL compile error whose
 * line number points into generated code. See {@link validateBodyIdentifiers}.
 *
 * # Determinism
 *
 * {@link generateScaffold} is a pure function of its spec: same spec in,
 * byte-identical string out, for any process. That is what makes the pipeline
 * cache in `material.ts` correct rather than hopeful.
 *
 * This module is import-safe in a plain Node or Bun process. Nothing here
 * touches `navigator.gpu` or any WebGPU global at import or call time, so a
 * test can assert on generated WGSL with no device in sight.
 */

import { fail } from '../core/error.ts';
import {
  BIND_GROUP,
  FRAME_BLOCK,
  MATERIAL_BLOCK,
  OBJECT_BLOCK,
  RESERVED_SLOT_NAMES,
  collectSlotDefaults,
  resolveSlotTypes,
} from '../core/slot.ts';
import type { SlotDefs, SlotType } from '../core/slot.ts';
import { buildUniformBlock, UNIFORM_TYPES } from '../core/uniform.ts';
import type { UniformBlockSpec } from '../core/uniform.ts';
import { STANDARD_LAYOUT } from '../geometry/layout.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import {
  TEXTURE_KINDS,
  samplerTypeName,
  textureViewDimension,
  wgslTextureDecl,
} from './texture-slot.ts';
import type {
  ResolvedTextureSlot,
  TextureKind,
  TextureSampleType,
  TextureSlotSpec,
} from './texture-slot.ts';
import type { DrawPhase } from '../render/types.ts';
import type {
  BlendSpec,
  CullMode,
  DepthSpec,
  FrontFace,
  PrimitiveTopology,
  TargetSpec,
} from '../render/pipeline-state.ts';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * `GPUColorWrite.ALL`, inlined.
 *
 * `src/render/pipeline-state.ts` exports the same value, but that module
 * evaluates `GPUColorWrite.ALL` at module scope, so importing it for a runtime
 * value would make this file unimportable outside a browser — and therefore
 * untestable without a GPU. The bit pattern is fixed by the WebGPU spec.
 */
const COLOR_WRITE_ALL: GPUColorWriteFlags = 0xf as GPUColorWriteFlags;

/**
 * Inter-stage limits. The fragment stage's input struct may use at most
 * `maxInterStageShaderVariables` variables — and `@builtin(position)` is one of
 * them, which is why the user-facing budget is one lower than the raw number.
 *
 * Core profile: 16 variables, 60 components. Compatibility profile: 15
 * variables. `@builtin(position)` is a vec4, so it also spends 4 of the 60
 * components.
 */
export const CORE_INTER_STAGE = Object.freeze({ variables: 16, components: 60 });
export const COMPAT_INTER_STAGE = Object.freeze({ variables: 15, components: 60 });

/** Varying types that can cross the rasteriser boundary. */
export const VARYING_TYPES = Object.freeze({
  f32: { wgsl: 'f32', components: 1, flat: false },
  i32: { wgsl: 'i32', components: 1, flat: true },
  u32: { wgsl: 'u32', components: 1, flat: true },
  vec2f: { wgsl: 'vec2f', components: 2, flat: false },
  vec3f: { wgsl: 'vec3f', components: 3, flat: false },
  vec4f: { wgsl: 'vec4f', components: 4, flat: false },
  mat4x4f: { wgsl: 'mat4x4f', components: 16, flat: true },
} as const satisfies Readonly<Record<string, { wgsl: string; components: number; flat: boolean }>>);

export type VaryingType = keyof typeof VARYING_TYPES;

export const VARYING_TYPE_NAMES = Object.keys(VARYING_TYPES) as VaryingType[];

/** `webgpu: ${GpuVendor}`-independent, present on every `GPUAdapter`. */
export interface InterStageLimits {
  readonly maxInterStageShaderVariables?: number;
  readonly maxInterStageComponents?: number;
}

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

/** `{ worldPos: 'vec3f', uv: 'vec2f' }` — declaration order fixes `@location`. */
export type VaryingDefs = Readonly<Record<string, VaryingType>>;

/**
 * A material.
 *
 * `vertex` and `fragment` are **statement lists**, not functions. They are
 * injected into the body of a generated `fn vs` / `fn fs`; declaring a
 * function, a struct, a binding, or an entry point inside one is an error.
 * Helper functions belong in `prelude`.
 */
export interface MaterialSpec {
  /** Label for tooling and error messages. Defaults to `'material'`. */
  readonly name?: string;
  /** Vertex layout. Defaults to {@link STANDARD_LAYOUT}. */
  readonly layout?: VertexLayout;
  /**
   * Inter-stage data. Locations are assigned in declaration order from 0;
   * `@builtin(position) clip` is always present and consumes no location.
   *
   * `i32`, `u32`, and `mat4x4f` are emitted with `@interpolate(flat)`,
   * because integer and matrix varyings are illegal without it.
   */
  readonly varyings?: VaryingDefs;
  /** Typed fields of the generated `MaterialData` uniform block. */
  readonly slots?: SlotDefs;
  /** Texture declarations. The body may reference each by its variable name. */
  readonly textures?: Readonly<Record<string, TextureSlotSpec>>;
  /**
   * WGSL inserted after the generated declarations and before the entry
   * points. This is the escape hatch: helper `fn`s, `const`s, and `struct`s
   * that belong to *this* material go here. It is the only place a user may
   * write a declaration, because only apse may write `@group`, `@binding`,
   * and the entry-point attributes.
   */
  readonly prelude?: string;
  /** Draw ordering bucket. Defaults to `'opaque'`. */
  readonly phase?: DrawPhase;
  /** Defaults to `'triangle-list'`. */
  readonly topology?: PrimitiveTopology;
  /** Defaults to `'back'`. */
  readonly cull?: CullMode;
  /** Defaults to `'ccw'`. */
  readonly frontFace?: FrontFace;
  /** Defaults to write+`less` for opaque, no-write+`less` for transparent. */
  readonly depth?: Partial<DepthSpec>;
  /** `null` (the default) replaces the target; a spec alpha-blends. */
  readonly blend?: BlendSpec | null;
  /**
   * Colour attachments. Defaults to the canvas preferred format. Exactly one
   * target is supported, because the generated fragment entry point returns a
   * single `vec4f`.
   */
  readonly targets?: readonly TargetSpec[];
  /**
   * Depth attachment format. Baked into the pipeline, so it is part of the
   * pipeline cache key. Defaults to `'depth24plus'`, which every core-profile
   * device supports; set it to `'depth32float'` or `'depth16unorm'` to match
   * the render target you are drawing into.
   */
  readonly depthFormat?: GPUTextureFormat;
  /** Required by WebGPU for a strip topology. Ignored otherwise. */
  readonly stripIndexFormat?: GPUIndexFormat;
  /** MSAA sample count. Defaults to 1. */
  readonly sampleCount?: number;
  /** Statements for the generated vertex entry point. Must assign `out.clip`. */
  readonly vertex: string;
  /** Statements for the generated fragment entry point. Must `return a vec4f`. */
  readonly fragment: string;
  /** Log the full generated WGSL at creation. Diagnostics only; not cached. */
  readonly scaffold?: boolean;
}

/** Every key {@link resolveSpec} accepts. Unknown keys are rejected. */
const SPEC_KEYS: readonly string[] = [
  'name', 'layout', 'varyings', 'slots', 'textures', 'prelude', 'phase',
  'topology', 'cull', 'frontFace', 'depth', 'blend', 'targets', 'sampleCount',
  'depthFormat', 'stripIndexFormat', 'vertex', 'fragment', 'scaffold',
];

/**
 * The canvas format a material should target when the caller does not say.
 *
 * Read at call time rather than captured at module load, and guarded for
 * non-browser contexts so importing a material factory in a Node test does not
 * throw. `rgba8unorm` is the documented fallback for anywhere there is no canvas
 * — an offscreen target, where the caller must pass the format anyway.
 *
 * Exported because five shipped material factories all default to it and a
 * sixth private copy of these five lines is how they drift apart. It lives here
 * rather than in one of them because this is the module every material already
 * imports, and because {@link resolveTargets} computes the identical value.
 */
export function preferredTargetFormat(): GPUTextureFormat {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined'
    ? navigator.gpu.getPreferredCanvasFormat()
    : 'rgba8unorm';
}

// ---------------------------------------------------------------------------
// Resolved form
// ---------------------------------------------------------------------------

export interface ResolvedVarying {
  readonly name: string;
  readonly type: VaryingType;
  /** WGSL declaration text, e.g. `vec3f`. */
  readonly wgsl: string;
  /** `@location(n)`. `-1` for the always-present builtin position. */
  readonly location: number;
  /** True when the declaration carries `@interpolate(flat)`. */
  readonly flat: boolean;
  readonly components: number;
}

/**
 * One shared sampler.
 *
 * Samplers with identical configuration share a single binding — see
 * {@link resolveTextureSlots} for the rule.
 */
export interface ResolvedSampler {
  /** Stable key: `sampleType|mipmapFilter|addressMode|compare`. */
  readonly key: string;
  /** The single WGSL variable name, derived from the group's first slot. */
  readonly varName: string;
  readonly bindingIndex: number;
  /** WGSL type: `sampler` or `sampler_comparison`. */
  readonly wgslType: string;
  /** `GPUBindGroupLayoutEntry.sampler.type`. */
  readonly layoutType: 'filtering' | 'non-filtering' | 'comparison';
  readonly mipmapFilter: boolean;
  readonly addressMode: 'repeat' | 'clamp-to-edge' | 'mirror-repeat';
  readonly compare: boolean;
  /** Slot names sharing this binding, in declaration order. */
  readonly slotNames: readonly string[];
}

export interface ResolvedTarget {
  readonly format: GPUTextureFormat;
  readonly blend: BlendSpec | null;
  readonly writeMask: GPUColorWriteFlags;
}

/** A spec with every default filled in and every field validated. */
export interface ResolvedMaterialSpec {
  readonly name: string;
  readonly layout: VertexLayout;
  readonly varyings: readonly ResolvedVarying[];
  /** The user-declared varyings, without the builtin position. */
  readonly userVaryings: readonly ResolvedVarying[];
  readonly slotTypes: Record<string, SlotType>;
  readonly slotDefaults: ReadonlyMap<string, number | ArrayLike<number>>;
  readonly textures: readonly ResolvedTextureSlot[];
  readonly samplers: readonly ResolvedSampler[];
  /** slotName -> the shared sampler that slot must use. */
  readonly samplerForSlot: ReadonlyMap<string, ResolvedSampler>;
  readonly prelude: string;
  readonly phase: DrawPhase;
  readonly topology: PrimitiveTopology;
  readonly cull: CullMode;
  readonly frontFace: FrontFace;
  readonly depth: DepthSpec;
  readonly blend: BlendSpec | null;
  readonly targets: readonly ResolvedTarget[];
  readonly sampleCount: number;
  readonly depthFormat: GPUTextureFormat;
  readonly stripIndexFormat: GPUIndexFormat | undefined;
  readonly vertexBody: string;
  readonly fragmentBody: string;
  readonly scaffoldLog: boolean;
  /** The generated `MaterialData`, or null when the material declares no slots. */
  readonly materialBlock: UniformBlockSpec | null;
}

/** The output of {@link generateScaffold}. */
export interface GeneratedShader {
  /** The full WGSL program. */
  readonly code: string;
  readonly materialBlock: UniformBlockSpec;
  readonly frameBlock: UniformBlockSpec;
  readonly objectBlock: UniformBlockSpec;
  readonly slotTypes: Record<string, SlotType>;
  readonly textureSlots: ResolvedTextureSlot[];
  /** Everything the spec resolved to. Cheap to read; already validated. */
  readonly resolved: ResolvedMaterialSpec;
  /** Key for pipeline layout / bind group layout caches. Body-independent. */
  readonly pipelineStateKey: number;
  /** Key for the `GPURenderPipeline` cache. Includes the body. */
  readonly pipelineKey: number;
  /** `code.length`. */
  readonly byteLength: number;
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

const resolvedCache = new WeakMap<MaterialSpec, ResolvedMaterialSpec>();

/**
 * Fills in defaults and validates. Memoised per spec object, so calling it
 * twice is free and two materials from one spec share the resolution.
 */
export function resolveSpec(spec: MaterialSpec): ResolvedMaterialSpec {
  if (spec === null || typeof spec !== 'object') {
    fail('OPTION_UNKNOWN', 'A material spec must be an object.', {
      why: `generateScaffold() takes a single MaterialSpec object. Known keys: ${SPEC_KEYS.join(', ')}.`,
      fix: 'Call material({ name, vertex, fragment, ... }).',
    });
  }
  const hit = resolvedCache.get(spec);
  if (hit !== undefined) return hit;

  for (const key of Object.keys(spec)) {
    if (!SPEC_KEYS.includes(key)) {
      fail('OPTION_UNKNOWN', `Unknown material option "${key}".`, {
        why: 'apse rejects unrecognised spec keys so that a typo never fails silently as "that option was ignored", which is how a material ends up untextured with no error anywhere.',
        fix: `Use one of: ${SPEC_KEYS.join(', ')}. To add a new option, add it to SPEC_KEYS and to the resolver in src/material/scaffold.ts.`,
      });
    }
  }

  const resolved = resolveUncached(spec);
  resolvedCache.set(spec, resolved);
  return resolved;
}

function resolveUncached(spec: MaterialSpec): ResolvedMaterialSpec {
  const name = spec.name ?? 'material';
  const layout = spec.layout ?? STANDARD_LAYOUT;
  const phase: DrawPhase = spec.phase ?? 'opaque';
  const topology: PrimitiveTopology = spec.topology ?? 'triangle-list';
  const cull: CullMode = spec.cull ?? 'back';
  const frontFace: FrontFace = spec.frontFace ?? 'ccw';
  const sampleCount = spec.sampleCount ?? 1;
  const blend = spec.blend ?? null;

  const userVaryings = resolveVaryings(spec.varyings, name);
  const slotTypes = resolveSlots(spec.slots, name);
  const { textures, samplers, samplerForSlot } = resolveTextureSlots(spec.textures, name);

  const materialBlock = Object.keys(slotTypes).length === 0
    ? null
    : buildUniformBlock(MATERIAL_BLOCK.structName, slotTypes, { maxBindingSize: 65536 });

  // Mirrors `resolveDepth` in src/render/pipeline-state.ts. It is duplicated
  // rather than imported because that module evaluates `GPUColorWrite.ALL` at
  // module scope, which would make this file unimportable — and therefore
  // untestable — outside a browser.
  const depthFallback = phase === 'transparent'
    ? { write: false, compare: 'less' as const }
    : { write: true, compare: 'less' as const };
  const depth: DepthSpec = spec.depth === undefined
    ? depthFallback
    : { write: spec.depth.write ?? depthFallback.write, compare: spec.depth.compare ?? depthFallback.compare };

  const targets = resolveTargets(spec.targets, name);

  const vertexBody = spec.vertex ?? '';
  const fragmentBody = spec.fragment ?? '';

  validateBody(vertexBody, 'vertex', name);
  validateBody(fragmentBody, 'fragment', name);

  const resolved: ResolvedMaterialSpec = {
    name,
    layout,
    varyings: [POSITION_VARYING, ...userVaryings],
    userVaryings,
    slotTypes,
    slotDefaults: collectSlotDefaults(spec.slots),
    textures,
    samplers,
    samplerForSlot,
    prelude: spec.prelude ?? '',
    phase,
    topology,
    cull,
    frontFace,
    depth,
    blend,
    targets,
    sampleCount,
    depthFormat: spec.depthFormat ?? 'depth24plus',
    stripIndexFormat: spec.stripIndexFormat,
    vertexBody,
    fragmentBody,
    scaffoldLog: spec.scaffold === true,
    materialBlock,
  };

  // Identifier validation needs the resolved field lists, so it runs after
  // resolution rather than inside `resolveBody`.
  validateBodyIdentifiers(vertexBody, resolved, 'vertex', name);
  validateBodyIdentifiers(fragmentBody, resolved, 'fragment', name);

  return resolved;
}

const POSITION_VARYING: ResolvedVarying = Object.freeze({
  name: 'clip',
  type: 'vec4f' as VaryingType,
  wgsl: 'vec4f',
  location: -1,
  flat: false,
  components: 4,
});

function resolveVaryings(varyings: VaryingDefs | undefined, material: string): ResolvedVarying[] {
  if (varyings === undefined) return [];
  const out: ResolvedVarying[] = [];
  let location = 0;
  for (const [vname, vtype] of Object.entries(varyings)) {
    assertIdentifier(vname, `varying`, material);
    if (vname === POSITION_VARYING.name) {
      fail('VARYING_TYPE_UNSUPPORTED', `Varying "${vname}" collides with the reserved builtin.`, {
        why: 'apse always emits `@builtin(position) clip` as the first field of the Varyings struct, and the builtin is not addressable as a user varying.',
        fix: 'Rename your varying. `clip` is the generated name of the clip-space position.',
      });
    }
    const info = (VARYING_TYPES as Record<string, { wgsl: string; components: number; flat: boolean } | undefined>)[vtype];
    if (info === undefined) {
      fail('VARYING_TYPE_UNSUPPORTED', `Varying "${vname}" has unsupported type "${vtype}".`, {
        why: 'Only a fixed set of types can cross the rasteriser boundary. An arbitrary struct cannot, and a float type list would not be interpolatable.',
        fix: `Use one of: ${VARYING_TYPE_NAMES.join(', ')}. Note that i32, u32, and mat4x4f are emitted with @interpolate(flat) automatically.`,
      });
    }
    out.push(Object.freeze({
      name: vname,
      type: vtype,
      wgsl: info.wgsl,
      location: location++,
      flat: info.flat,
      components: info.components,
    }));
  }
  return out;
}

function resolveSlots(slots: SlotDefs | undefined, material: string): Record<string, SlotType> {
  if (slots === undefined) return {};
  for (const [sname, sspec] of Object.entries(slots)) {
    assertIdentifier(sname, 'slot', material);
    if (RESERVED_SLOT_NAMES.has(sname)) {
      fail('OPTION_UNKNOWN', `Slot "${sname}" shadows a reserved apse field name.`, {
        why: 'apse generates the frame block (frame.*) and object block (obj.*) and already declares these names in them. A material slot with the same name would silently collide with generated state.',
        fix: `Rename the slot. Reserved: ${[...RESERVED_SLOT_NAMES].join(', ')}.`,
      });
    }
    const type = typeof sspec === 'string' ? sspec : sspec.type;
    const info = UNIFORM_TYPES[type];
    if (info === undefined) {
      fail('SLOT_TYPE_UNKNOWN', `Slot "${sname}" has unknown type "${String(type)}".`, {
        why: 'apse computes the byte offset, the WGSL declaration, and the packing from this type. An unknown type has no size, so it cannot be placed.',
        fix: `Use one of: ${Object.keys(UNIFORM_TYPES).join(', ')}. To add a type, add an entry to UNIFORM_TYPES in src/core/uniform.ts.`,
      });
    }
    if (typeof sspec !== 'string' && sspec.default !== undefined) {
      validateDefault(sname, sspec.default, info.components, material);
    }
  }
  return resolveSlotTypes(slots);
}

function validateDefault(field: string, value: number | ArrayLike<number>, components: number, material: string): void {
  const length = typeof value === 'number' ? 1 : value.length;
  if (length !== components) {
    fail('SLOT_DEFAULT_INVALID',
      `Slot "${field}" on material "${material}" is a ${components}-component type but its default has ${length}.`, {
      why: 'Defaults are written into the uniform block at the same byte offset as any other value, so a wrong component count would spill into the neighbouring field.',
      fix: components === 1
        ? `Give "${field}" a single number, or omit the default entirely.`
        : `Give "${field}" exactly ${components} numbers.`,
    });
  }
  const at = (i: number): number => (typeof value === 'number' ? value : value[i]);
  for (let i = 0; i < length; i++) {
    const v = at(i);
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      fail('SLOT_DEFAULT_INVALID',
        `Slot "${field}" on material "${material}" has a default of ${String(v)} at component ${i}.`, {
        why: 'NaN and Infinity propagate through every fragment that reads the uniform and blank the whole draw, so a bad default is much worse than a thrown error.',
        fix: 'Guard the computation, or omit the default so the field starts zeroed.',
      });
    }
  }
}

/**
 * Resolves texture declarations and assigns binding indices.
 *
 * **Texture bindings come first, in declaration order, one per slot.** Then
 * come sampler bindings, one per *distinct sampler configuration*.
 *
 * Two texture slots whose sampler configuration — sample type, mipmap
 * filtering, address mode, comparison — is byte-identical share ONE binding,
 * because the sampler is immutable state that carries no per-texture data.
 * Sharing is not a micro-optimisation: it halves the binding count of a
 * typical multi-texture material and removes N-1 redundant `GPUSampler`
 * objects.
 *
 * The consequence is that a shared sampler has exactly one variable name,
 * derived from the *first* slot in its group. A body that reaches for the
 * second slot's `<name>Sampler` is rejected by
 * {@link validateBodyIdentifiers} with a message naming the right identifier,
 * rather than being handed to WGSL as an undefined variable.
 */
function resolveTextureSlots(
  textures: Readonly<Record<string, TextureSlotSpec>> | undefined,
  material: string,
): {
  textures: ResolvedTextureSlot[];
  samplers: ResolvedSampler[];
  samplerForSlot: Map<string, ResolvedSampler>;
} {
  const out: ResolvedTextureSlot[] = [];
  const draft: SamplerDraft[] = [];
  const samplerForSlot = new Map<string, ResolvedSampler>();
  if (textures === undefined) return { textures: out, samplers: [], samplerForSlot };

  for (const [slotName, tspec] of Object.entries(textures)) {
    const kind: TextureKind = tspec.kind ?? '2d';
    if (!TEXTURE_KINDS.includes(kind)) {
      fail('TEXTURE_SLOT_TYPE_INVALID', `Texture slot "${slotName}" has unsupported kind "${kind}".`, {
        why: 'apse derives the WGSL texture type, the `textureViewDimension` of the bind group layout entry, and the dimension of the view you must supply from this value.',
        fix: `Use one of: ${TEXTURE_KINDS.join(', ')}.`,
      });
    }
    const sampleType: TextureSampleType = tspec.sampleType ?? 'float';
    const compare = tspec.compare === true;
    if (compare && sampleType !== 'depth') {
      fail('TEXTURE_SLOT_TYPE_INVALID',
        `Texture slot "${slotName}" sets compare: true on a "${sampleType}" texture.`, {
        why: 'A comparison sampler produces a 0/1 visibility result, which is only defined for depth textures. WGSL rejects `textureSampleCompare` on a non-depth texture at compile time.',
        fix: 'Use `sampleType: "depth"` with a `depth-2d` kind, or drop `compare`.',
      });
    }
    const addressMode = tspec.addressMode ?? (kind === '2d' ? 'repeat' : 'clamp-to-edge');
    const mipmapFilter = tspec.mipmapFilter !== false;
    // The *variable* must be a usable identifier. The slot name need not be:
    // a `name` override exists precisely so a slot can be called `albedo-map`
    // and still be reachable from a body as `albedo`.
    const varName = tspec.name ?? slotName;
    assertIdentifier(varName, 'texture variable', material);

    const key = `${sampleType}|${mipmapFilter}|${addressMode}|${compare}`;
    let entry = draft.find((s) => s.key === key);
    if (entry === undefined) {
      entry = {
        key,
        varName: `${varName}Sampler`,
        bindingIndex: 0,
        wgslType: samplerTypeName(sampleType, compare),
        layoutType: compare ? 'comparison' as const
          : sampleType === 'float' ? 'filtering' as const
          : 'non-filtering' as const,
        mipmapFilter,
        addressMode,
        compare,
        slotNames: [slotName],
      };
      draft.push(entry);
    } else {
      entry.slotNames.push(slotName);
    }

    out.push(Object.freeze({
      slotName,
      varName,
      kind,
      sampleType,
      mipmapFilter,
      addressMode,
      compare,
      label: tspec.label ?? slotName,
      bindingIndex: out.length,
    }));
  }

  // Sampler bindings follow every texture binding. Assigned before freezing so
  // the exposed objects are immutable.
  for (let i = 0; i < draft.length; i++) draft[i].bindingIndex = out.length + i;
  const samplers: ResolvedSampler[] = draft.map((s) => Object.freeze({ ...s, slotNames: Object.freeze([...s.slotNames]) }));
  for (const t of out) {
    const s = samplers.find((x) => x.key === `${t.sampleType}|${t.mipmapFilter}|${t.addressMode}|${t.compare ? 'true' : 'false'}`);
    if (s !== undefined) samplerForSlot.set(t.slotName, s);
  }

  // A texture variable must not shadow a shared sampler variable, or one would
  // be a duplicate `@group(3) @binding(n) var` declaration.
  const samplerNames = new Set(samplers.map((s) => s.varName));
  for (const t of out) {
    if (samplerNames.has(t.varName)) {
      fail('OPTION_UNKNOWN',
        `Texture slot "${t.slotName}" uses the variable name "${t.varName}", which collides with a generated sampler binding.`, {
        why: 'apse derives a sampler variable named "<first-slot>Sampler" per shared sampler. A texture variable with that exact name would produce two declarations at the same @group(3) binding.',
        fix: 'Give the texture slot a different `name`, or rename the conflicting slot. Generated sampler names: ' + [...samplerNames].join(', ') + '.',
      });
    }
  }

  return { textures: out, samplers, samplerForSlot };
}

/** Mutable accumulator; frozen into a {@link ResolvedSampler} once binding indices are known. */
interface SamplerDraft {
  key: string;
  varName: string;
  bindingIndex: number;
  wgslType: string;
  layoutType: 'filtering' | 'non-filtering' | 'comparison';
  mipmapFilter: boolean;
  addressMode: 'repeat' | 'clamp-to-edge' | 'mirror-repeat';
  compare: boolean;
  slotNames: string[];
}

/**
 * The sampler variable name a body must use for a texture slot.
 *
 * Not always `<slotName>Sampler`: two slots whose sampler configuration is
 * identical share one binding, and a shared binding has exactly one name, taken
 * from the **first** slot in the group. A material factory that generates a body
 * has to know which name that is, and the only honest way to find out is to ask
 * the same resolver the generator uses — re-deriving the rule here would be
 * exactly the drift the sharing exists to prevent, and a wrong answer is a WGSL
 * error naming an undeclared identifier on a line nobody wrote.
 *
 * A pure function of `textures`, so calling it while the spec is still being
 * assembled is safe: `resolveSpec` runs the identical computation later and
 * reaches the identical answer.
 */
export function samplerNameFor(
  textures: Readonly<Record<string, TextureSlotSpec>> | undefined,
  slotName: string,
): string {
  if (textures === undefined) return `${slotName}Sampler`;
  const { samplerForSlot } = resolveTextureSlots(textures, 'sampler-probe');
  return samplerForSlot.get(slotName)?.varName ?? `${slotName}Sampler`;
}

function resolveTargets(targets: readonly TargetSpec[] | undefined, material: string): ResolvedTarget[] {
  // The canvas preferred format, not a hardcoded rgba8unorm: it is bgra8unorm
  // on desktop, and a pipeline built for the wrong one fails at setPipeline with
  // a message that names a format instead of the mistake.
  const list: readonly TargetSpec[] = targets ?? [{ format: preferredTargetFormat() }];
  if (list.length === 0) {
    fail('RENDER_TARGET_FORMAT_MISMATCH', `Material "${material}" declares no colour targets.`, {
      why: 'A render pipeline needs at least one colour attachment, and the generated fragment entry point returns a single `@location(0) vec4f`.',
      fix: 'Pass `targets: [{ format }]` matching the render target you draw into.',
    });
  }
  if (list.length > 1) {
    fail('RENDER_TARGET_FORMAT_MISMATCH',
      `Material "${material}" declares ${list.length} targets; the generated fragment entry point returns one colour.`, {
      why: 'The scaffold emits `@fragment fn fs(...) -> @location(0) vec4f`. Multiple attachments would need a generated output struct and a body that fills every member.',
      fix: 'Use one target, or move the extra channels into a second pass reading the first target as a texture.',
    });
  }
  return list.map((t) => ({
    format: t.format,
    blend: t.blend ?? null,
    writeMask: t.writeMask ?? COLOR_WRITE_ALL,
  }));
}

// ---------------------------------------------------------------------------
// Body validation
// ---------------------------------------------------------------------------

/**
 * Replaces comment content with spaces, preserving newlines and length.
 *
 * Every body scan runs on this, so a `fn` or a `@group` mentioned in a comment
 * is not a false positive, and a reported line number still points at the
 * right line.
 */
export function stripComments(src: string): string {
  let out = '';
  let i = 0;
  const n = src.length;
  while (i < n) {
    const two = src[i] === '/' ? src[i + 1] : '';
    if (two === '/') {
      while (i < n && src[i] !== '\n') { out += ' '; i++; }
      continue;
    }
    if (two === '*') {
      out += '  ';
      i += 2;
      while (i < n && !(src[i] === '*' && src[i + 1] === '/')) {
        out += src[i] === '\n' ? '\n' : ' ';
        i++;
      }
      if (i < n) { out += '  '; i += 2; }
      continue;
    }
    out += src[i];
    i++;
  }
  return out;
}

/**
 * Removes comments entirely, rather than blanking them.
 *
 * Used for hashing, and this is what makes the pipeline cache key mean what it
 * should. A comment has no effect on the compiled program, so two materials
 * that differ only in their comments — or in their `name`, which appears in
 * the generated header — compile to the same shader and must share a pipeline.
 * Blanking comments instead of deleting them would still change the hash,
 * because a longer comment means more spaces.
 *
 * Newlines inside a comment are kept, so a removal is still line-preserving for
 * anything that cares.
 */
export function removeComments(src: string): string {
  let out = '';
  let i = 0;
  const n = src.length;
  while (i < n) {
    const two = src[i] === '/' ? src[i + 1] : '';
    if (two === '/') {
      while (i < n && src[i] !== '\n') i++;
      continue;
    }
    if (two === '*') {
      i += 2;
      while (i < n && !(src[i] === '*' && src[i + 1] === '/')) {
        if (src[i] === '\n') out += '\n';
        i++;
      }
      i += 2;
      continue;
    }
    out += src[i];
    i++;
  }
  return out;
}

const FN_DECL = /\bfn\s+[A-Za-z_]/;
const STRUCT_DECL = /\bstruct\s+[A-Za-z_]/;
const ATTRIBUTE = /@([A-Za-z_][A-Za-z0-9_]*)/g;
const RETURN_STMT = /\breturn\b/;

function bodyFail(stage: string, reason: string, why: string, fix: string, material: string): never {
  return fail('SHADER_BODY_INVALID',
    `The ${stage} body of material "${material}" is invalid: ${reason}`, { why, fix });
}

/**
 * Rejects a body that tries to take ownership of a declaration.
 *
 * Everything apse owns is off limits in a body, because if a body could
 * declare a binding then the bind group layout, the pipeline layout, and the
 * generated code would all have to be reconciled against text apse does not
 * control — which is exactly the failure mode this module removes.
 */
function validateBody(body: string, stage: 'vertex' | 'fragment', material: string): void {
  if (body.trim().length === 0) {
    bodyFail(stage, `it is empty.`,
      'An empty body produces a vertex stage that never assigns out.clip, or a fragment stage that returns nothing, so the pipeline would fail to compile.',
      stage === 'vertex'
        ? 'Write at least `out.clip = <clip-space position>;`.'
        : 'Write at least `return vec4f(1.0, 0.0, 0.0, 1.0);`.',
      material);
  }

  const src = stripComments(body);

  if (FN_DECL.test(src)) {
    bodyFail(stage, 'it declares a function.',
      'apse generates the entry points and owns every declaration, so a body is spliced into a function body. A nested `fn` is not valid there and would collide with the generated scope.',
      'Move the helper into the `prelude` option, which apse inserts after the generated declarations and before the entry points. Helper functions belong there.',
      material);
  }

  if (STRUCT_DECL.test(src)) {
    bodyFail(stage, 'it declares a struct.',
      'apse generates every struct — Frame, ObjectData, MaterialData, VertexIn, and Varyings — from the spec, so a hand-written one could not be given a binding or matched to a vertex layout.',
      'Describe what you need as `slots` (uniform) or `varyings` (inter-stage) and apse will generate the struct. Use `prelude` only for types that are internal to your shader and never need binding.',
      material);
  }

  ATTRIBUTE.lastIndex = 0;
  const attr = ATTRIBUTE.exec(src);
  if (attr !== null) {
    bodyFail(stage, `it uses the \`@${attr[1]}\` attribute.`,
      'apse owns every attribute: @group, @binding, @builtin, @location, @interpolate, @vertex, and @fragment are all produced from the spec. Writing one in a body would desynchronise the WGSL from the bind group layout apse builds.',
      'Declare what you need through the spec — `slots` becomes a @group(2) uniform field, `textures` become @group(3) bindings, `varyings` become @location fields. For entry-point and binding control in a helper, use `prelude` and rely on the parameters it is given.',
      material);
  }

  if (stage === 'vertex' && !src.includes('out.clip')) {
    bodyFail(stage, 'it never assigns `out.clip`.',
      'The vertex entry point returns a Varyings, and `out.clip` is its `@builtin(position)`. A vertex stage that does not set it has no clip-space position, and the rasteriser will discard every triangle without a usable diagnostic.',
      'Add a line like `out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);` — `frame` and `obj` are already bound, so no uniform plumbing is needed.',
      material);
  }

  if (stage === 'fragment' && !RETURN_STMT.test(src)) {
    bodyFail(stage, 'it never returns a colour.',
      'The generated fragment entry point ends with `return` of whatever your body produces, and a WGSL function with a return type must return on every path.',
      'End the body with `return vec4f(r, g, b, a);`.',
      material);
  }

  if (stage === 'fragment' && /\bout\s*\./.test(src)) {
    bodyFail(stage, 'it writes to `out`.',
      '`out` is the vertex stage\'s Varyings value. The fragment stage reads the same data through `in`, and the inter-stage struct is an input there, so there is nothing to write.',
      'Use `in.<varying>` in the fragment stage. `out` exists only in the vertex stage.',
      material);
  }
}

/**
 * Checks every `frame.`, `obj.`, `mat.`, `in.`, and `out.` field access in a
 * body against what the spec actually declared.
 *
 * This is the most valuable guard in the module. Without it, a typo or a
 * hallucinated field name reaches Tint or Naga and comes back as a WGSL
 * compile error whose line and column point into *generated* code — a message
 * that names a line the user cannot see and never wrote. With it, the failure
 * is a typed {@link AseError} that names the field, says the prefix it was
 * read from, and lists the fields that do exist.
 *
 * The same scan catches the reverse: a body reaching for a sampler variable
 * that was deduplicated away, or an `in.` field in the wrong stage.
 */
export function validateBodyIdentifiers(
  body: string,
  resolved: ResolvedMaterialSpec,
  stage: 'vertex' | 'fragment',
  material: string,
): void {
  const src = stripComments(body);
  const access = /\b(in|out|frame|obj|mat)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)/g;

  const inFields = stage === 'vertex'
    ? resolved.layout.attributes.map((a) => a.name)
    : resolved.varyings.map((v) => v.name);
  const outFields = [...resolved.userVaryings.map((v) => v.name), POSITION_VARYING.name];
  const frameFields = FRAME_FIELD_NAMES;
  const objFields = OBJECT_FIELD_NAMES;
  const matFields = Object.keys(resolved.slotTypes);

  let m: RegExpExecArray | null;
  while ((m = access.exec(src)) !== null) {
    const prefix = m[1];
    const field = m[2];
    let known: readonly string[];
    let container: string;
    switch (prefix) {
      case 'in':
        known = inFields;
        container = stage === 'vertex'
          ? `the vertex input struct, which is generated from the material's layout (${resolved.layout.key})`
          : 'the Varyings struct, i.e. the material\'s declared `varyings`';
        break;
      case 'out':
        known = outFields;
        container = 'the Varyings struct, i.e. the material\'s declared `varyings`';
        break;
      case 'frame':
        known = frameFields;
        container = 'the generated Frame uniform block (@group(0))';
        break;
      case 'obj':
        known = objFields;
        container = 'the generated ObjectData uniform block (@group(1))';
        break;
      default:
        known = matFields;
        container = 'the generated MaterialData uniform block (@group(2))';
        break;
    }

    if (!known.includes(field)) {
      const near = closest(field, known);
      bodyFail(stage, `\`${prefix}.${field}\` is not a field.`,
        `${prefix} is ${container}, and that container declares no field called "${field}". Every ${prefix}.field access in a body is checked against the spec before the shader is compiled, so an undeclared one is a mistake in the material spec rather than a WGSL error.`,
        known.length === 0
          ? `${prefix} has no fields at all. ${prefix === 'mat' ? 'Declare them with `slots: { ' + field + ': ... }`.' : 'Check the material spec.'}`
          : `${near === undefined ? `Declare "${field}" on the material.` : `Did you mean \`${prefix}.${near}\`?`} Available on ${prefix}: ${known.join(', ')}.`,
        material);
    }
  }

  // Sampler variable names. A shared sampler exists under one name only.
  const samplerUse = new RegExp(`\\b(${resolved.textures.map((t) => t.varName).join('|')})\\s*\\.\\s*([A-Za-z_][A-Za-z0-9_]*)`, 'g');
  if (resolved.textures.length > 0) {
    let s: RegExpExecArray | null;
    while ((s = samplerUse.exec(src)) !== null) {
      const slot = resolved.textures.find((t) => t.varName === s![1]);
      if (slot === undefined) continue;
      if (s[2] === 'sample' || s[2] === 'sampleLevel' || s[2] === 'sampleBias' || s[2] === 'sampleCompare' || s[2] === 'sampleCompareLevel' || s[2] === 'load' || s[2] === 'dimensions') continue;
      const sampler = resolved.samplerForSlot.get(slot.slotName);
      if (sampler === undefined) continue;
      bodyFail(stage, `\`${slot.varName}.${s[2]}\` is not a texture operation.`,
        '`texture` variables in a body are handled by apse\'s generated bindings; the only valid members are the WGSL texture builtins (sample, sampleLevel, sampleBias, sampleCompare, load, dimensions).',
        `Use \`textureSample${slot.compare ? 'Compare' : ''}(${slot.varName}, ${sampler.varName}, ...)\`.`,
        material);
    }
  }

  // A body that names a per-slot sampler for a slot in a shared group would
  // otherwise be handed to WGSL as an undeclared identifier.
  for (const [slotName, sampler] of resolved.samplerForSlot) {
    if (sampler.slotNames[0] === slotName) continue;
    const wrong = `${slotName}Sampler`;
    if (new RegExp(`\\b${wrong}\\b`).test(src)) {
      bodyFail(stage, `\`${wrong}\` is not defined.`,
        `Texture slots "${sampler.slotNames.join('" and "')}" share one sampler binding because their sampler configuration is identical, and a shared binding has exactly one variable name.`,
        `Use \`${sampler.varName}\` for every slot in that group: ${sampler.slotNames.join(', ')}. Call describeMaterial() to get the correct sampler name for any slot.`,
        material);
    }
  }
}

/** Cheap "did you mean", Levenshtein distance 1 or 2. */
function closest(word: string, candidates: readonly string[]): string | undefined {
  let best: string | undefined;
  let bestD = 3;
  for (const c of candidates) {
    const d = editDistance(word, c);
    if (d < bestD) { bestD = d; best = c; }
  }
  return best;
}

function editDistance(a: string, b: string): number {
  let prev = new Array<number>(b.length + 1);
  let cur = new Array<number>(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[b.length];
}

const FRAME_FIELD_NAMES: readonly string[] = [
  'view', 'proj', 'viewProj', 'invView', 'invProj', 'invViewProj',
  'camPos', 'time', 'delta', 'elapsed', 'resolution', 'viewport', 'exposure', 'alpha',
];

const OBJECT_FIELD_NAMES: readonly string[] = [
  'model', 'normalMatrix', 'objectId', 'instanceId', 'visibility',
];

/**
 * Names apse refuses as a slot, varying, or texture variable.
 *
 * The first block is WGSL's *keywords*; the second is its much longer list of
 * **reserved words**, which Tint rejects exactly as hard. The second block is the
 * one that matters, because its members read like ordinary variable names —
 * `operator`, `sample`, `shared`, `filter`, `typedef`, `union` — and a material
 * that declared a slot called any of them would fail at shader-compile time with
 * a message pointing at generated code, rather than here where the fix can be
 * named.
 *
 * It is not the whole reserved list: it is the members that are plausible field
 * names in a graphics API. The cost of a false positive is a renamed slot, and
 * the cost of a false negative is a shader that does not compile.
 */
const WGSL_KEYWORDS: ReadonlySet<string> = new Set([
  // --- keywords ---
  'in', 'out', 'let', 'var', 'const', 'fn', 'struct', 'return', 'if', 'else', 'for',
  'while', 'loop', 'switch', 'case', 'default', 'break', 'continue', 'discard',
  'type', 'alias', 'override', 'true', 'false', 'array', 'atomic', 'ptr', 'sampler',
  'texture_2d', 'texture_cube', 'vec2f', 'vec3f', 'vec4f', 'mat4x4f', 'f32', 'i32', 'u32',
  // --- reserved words a graphics API might plausibly name a field ---
  'operator', 'common', 'filter', 'get', 'set', 'shared', 'static', 'typedef', 'typeid',
  'union', 'template', 'class', 'interface', 'namespace', 'using', 'module', 'new',
  'delete', 'public', 'private', 'protected', 'virtual', 'explicit', 'export',
  'external', 'interface', 'match', 'mut', 'ref', 'require', 'resource', 'self',
  'sizeof', 'super', 'this', 'typeof', 'unsized', 'use', 'where', 'with', 'yield',
  'async', 'await', 'become', 'cast', 'catch', 'coherent', 'compile', 'concept',
  'consteval', 'constexpr', 'crate', 'do', 'dynamic_cast', 'enum', 'fallthrough',
  'final', 'finally', 'friend', 'from', 'impl', 'implements', 'import', 'inline',
  'instanceof', 'layout', 'macro', 'meta', 'mod', 'move', 'mutable', 'noexcept',
  'null', 'of', 'package', 'partition', 'pass', 'patch', 'precise', 'precision',
  'priv', 'pub', 'readonly', 'register', 'reinterpret_cast', 'require', 'resource',
  'restrict', 'snorm', 'std', 'subroutine', 'target', 'throw', 'trait', 'try',
  'typeid', 'typename', 'unorm', 'unsafe', 'varying', 'volatile', 'wgsl',
  // --- the builtin type and function names a slot would shadow ---
  'vec2', 'vec3', 'vec4', 'mat2x2', 'mat3x3', 'mat4x4', 'bitcast', 'f16', 'texture_1d',
  'texture_2d_array', 'texture_3d', 'texture_cube_array', 'texture_storage_2d',
  'texture_depth_2d', 'texture_depth_cube', 'texture_depth_2d_array',
]);

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;

function assertIdentifier(name: string, what: string, material: string): void {
  if (!IDENTIFIER.test(name)) {
    fail('OPTION_UNKNOWN', `${what} name "${name}" on material "${material}" is not a usable WGSL identifier.`, {
      why: 'Slot, varying, and texture names are emitted verbatim as WGSL identifiers, so they must match the identifier grammar.',
      fix: 'Use a letter or underscore followed by letters, digits, and underscores. Give a texture slot a `name` override if its slot name cannot be changed.',
    });
  }
  if (WGSL_KEYWORDS.has(name)) {
    // Named separately from the grammar because the fix is different and the
    // reason is not obvious: `operator`, `filter`, `shared` and `sample` are
    // perfectly good JavaScript property names and perfectly illegal WGSL ones.
    fail('OPTION_UNKNOWN', `${what} name "${name}" on material "${material}" is a WGSL reserved word.`, {
      why: 'WGSL reserves a long list of words — including ordinary-looking ones like `operator`, `filter`, `shared`, `get`, `set` and `typedef` — and Tint rejects them as identifiers. The failure would otherwise surface as a shader-compile error whose line number points into generated code.',
      fix: `Rename it. \`${name}\` is reserved by the WGSL specification regardless of what it means in JavaScript.`,
    });
  }
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

/**
 * Re-indents a body for splicing into a `fn`.
 *
 * The user's block is dedented to its own common margin and then re-indented,
 * so a template literal written flush-left in a source file, or one written at
 * column 6 inside an object literal, both come out right. Without the dedent
 * step, a body written with any source indentation would appear in the
 * generated WGSL at that indentation plus two — which is not wrong, just
 * visibly wrong in a file people read to learn the library.
 */
function indent(body: string, spaces = 2): string {
  const lines = body.replace(/\t/g, '  ').split('\n').map((l) => l.replace(/\s+$/, ''));
  const nonEmpty = lines.filter((l) => l.trim().length > 0);
  if (nonEmpty.length === 0) return '';
  const margin = Math.min(...nonEmpty.map((l) => l.length - l.trimStart().length));
  const pad = ' '.repeat(spaces);
  return lines
    .map((l) => (l.trim().length === 0 ? '' : pad + l.slice(margin)))
    .join('\n')
    .replace(/^\n+|\n+$/g, '');
}

function varyingsStruct(resolved: ResolvedMaterialSpec): string {
  const lines = resolved.varyings.map((v) => {
    if (v.location < 0) return `  @builtin(position) ${v.name} : ${v.wgsl},`;
    return `  @location(${v.location})${v.flat ? ' @interpolate(flat)' : ''} ${v.name} : ${v.wgsl},`;
  });
  return `struct Varyings {\n${lines.join('\n')}\n};`;
}

function bindings(resolved: ResolvedMaterialSpec): string[] {
  const lines: string[] = [];
  lines.push(`@group(${BIND_GROUP.frame}) @binding(0) var<uniform> frame : Frame;`);
  lines.push(`@group(${BIND_GROUP.object}) @binding(0) var<uniform> obj : ObjectData;`);
  if (resolved.materialBlock !== null) {
    lines.push(`@group(${BIND_GROUP.material}) @binding(0) var<uniform> mat : ${resolved.materialBlock.structName};`);
  }
  for (const t of resolved.textures) lines.push(wgslTextureDecl(t.varName, t));
  for (const s of resolved.samplers) {
    lines.push(`@group(${BIND_GROUP.texture}) @binding(${s.bindingIndex}) var ${s.varName} : ${s.wgslType};`);
  }
  return lines;
}

/** Names a body may reference, for the generated header and describeMaterial. */
function bodyNames(resolved: ResolvedMaterialSpec, stage: 'vertex' | 'fragment'): string[] {
  const names = ['in', 'frame', 'obj'];
  if (stage === 'vertex') names.push('out');
  if (resolved.materialBlock !== null) names.push('mat');
  // A shared sampler is one name, so a group of N slots on one sampler
  // contributes N texture names and *one* sampler name. Collecting into a Set
  // rather than pushing per slot keeps the header an accurate list — a sampler
  // printed three times reads as three samplers, and the header is the answer to
  // "what may I write in this body".
  const samplers = new Set<string>();
  for (const t of resolved.textures) {
    names.push(t.varName);
    const s = resolved.samplerForSlot.get(t.slotName);
    if (s !== undefined) samplers.add(s.varName);
  }
  for (const name of samplers) names.push(name);
  return names;
}

/**
 * Generates the complete WGSL program for a material spec.
 *
 * Pure, deterministic, and device-free: the same spec always produces the same
 * bytes, which is what lets `material.ts` cache pipelines by a hash instead of
 * re-deriving a hundred-field key on every cache miss.
 *
 * `limits` is optional. Passing `device.limits` tightens the inter-stage checks
 * to the actual device; it never changes the emitted text, so determinism is
 * unaffected.
 */
export function generateScaffold(spec: MaterialSpec, limits?: InterStageLimits): GeneratedShader {
  const resolved = resolveSpec(spec);

  checkVaryingBudget(resolved, limits);

  const header = [
    `// Generated by apse — material "${resolved.name}". Do not edit: every`,
    '// declaration, binding, and signature below is produced from the spec.',
    `// vertex   body may use: ${bodyNames(resolved, 'vertex').join(', ')}`,
    `// fragment body may use: ${bodyNames(resolved, 'fragment').join(', ')}`,
    '// ---------------------------------------------------------------------',
  ].join('\n');

  const parts: string[] = [header, ''];

  parts.push('// ---- uniform blocks ----');
  parts.push(FRAME_BLOCK.wgsl);
  parts.push(OBJECT_BLOCK.wgsl);
  if (resolved.materialBlock !== null) parts.push(resolved.materialBlock.wgsl);
  parts.push('');

  parts.push(`// ---- vertex input (${resolved.layout.key}) ----`);
  parts.push(resolved.layout.wgslStruct('VertexIn'));
  parts.push('');

  parts.push('// ---- varyings ----');
  parts.push(varyingsStruct(resolved));
  parts.push('');

  parts.push('// ---- bindings ----');
  parts.push(bindings(resolved).join('\n'));
  parts.push('');

  if (resolved.prelude.trim().length > 0) {
    parts.push('// ---- prelude (user-supplied declarations) ----');
    parts.push(indent(resolved.prelude, 0).trim());
    parts.push('');
  }

  parts.push('// ---- generated by apse: vertex stage ----');
  parts.push('@vertex');
  parts.push('fn vs(in : VertexIn) -> Varyings {');
  parts.push('  var out : Varyings;');
  parts.push(indent(resolved.vertexBody));
  parts.push('  return out;');
  parts.push('}');
  parts.push('');

  parts.push('// ---- generated by apse: fragment stage ----');
  parts.push('@fragment');
  parts.push('fn fs(in : Varyings) -> @location(0) vec4f {');
  parts.push(indent(resolved.fragmentBody));
  parts.push('}');
  parts.push('');

  const code = parts.join('\n');
  const stateKey = pipelineStateKeyOf(resolved);

  return {
    code,
    materialBlock: resolved.materialBlock ?? MATERIAL_BLOCK,
    frameBlock: FRAME_BLOCK,
    objectBlock: OBJECT_BLOCK,
    slotTypes: resolved.slotTypes,
    textureSlots: resolved.textures as ResolvedTextureSlot[],
    resolved,
    pipelineStateKey: stateKey,
    pipelineKey: pipelineKeyOf(resolved, code),
    byteLength: code.length,
  };
}

/**
 * The inter-stage budget.
 *
 * `@builtin(position)` is one of the fragment stage's input variables, so the
 * user-facing budget is `maxInterStageShaderVariables - 1`: 15 on the core
 * profile, 14 in compatibility. It is also a `vec4`, so it spends 4 of the 60
 * `maxInterStageComponents`, leaving 56 for the user's own varyings.
 */
export function checkVaryingBudget(resolved: ResolvedMaterialSpec, limits?: InterStageLimits): void {
  const maxVars = limits?.maxInterStageShaderVariables ?? CORE_INTER_STAGE.variables;
  const maxComps = limits?.maxInterStageComponents ?? CORE_INTER_STAGE.components;
  const count = resolved.userVaryings.length;
  if (count + 1 > maxVars) {
    fail('VARYING_TOO_MANY',
      `Material "${resolved.name}" declares ${count} varyings, but the fragment stage allows ${maxVars} input variables in total and \`@builtin(position)\` takes one of them.`, {
      why: 'Each inter-stage variable is a @location in the Varyings struct, and the fragment stage may declare at most maxInterStageShaderVariables of them (16 on the core profile, 15 in compatibility). @builtin(position) is counted even though it consumes no location.',
      fix: `Declare at most ${maxVars - 1} varyings. Pack related values into one vec4, or pass high-frequency data through a flat storage buffer indexed by the varying that is already there.`,
    });
  }
  const components = resolved.userVaryings.reduce((sum, v) => sum + v.components, 0) + 4;
  if (components > maxComps) {
    fail('VARYING_TOO_MANY',
      `Material "${resolved.name}" uses ${components} inter-stage components, over the ${maxComps} allowed.`, {
      why: 'Beyond the variable count, WebGPU also caps the total components crossing the rasteriser at maxInterStageComponents (60 on the core profile). @builtin(position) spends 4.',
      fix: 'Drop a varying, or pack values into wider vectors so they share one component budget more efficiently than several narrow ones.',
    });
  }
}

// ---------------------------------------------------------------------------
// Cache keys
// ---------------------------------------------------------------------------

/**
 * FNV-1a, 32-bit.
 *
 * Chosen because it is three lines, has no table, and — the part that matters —
 * is order-sensitive and well-distributed, so two different structural shapes
 * do not collide by accident. Callers do the lookup in a `Map<number, …>`, so
 * comparison is an integer compare, never a string compare.
 */
export function fnv1a(...parts: readonly string[]): number {
  let h = 0x811c9dc5;
  for (const part of parts) {
    for (let i = 0; i < part.length; i++) {
      h ^= part.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    h ^= 0x1f; // field separator, so ["ab","c"] and ["a","bc"] differ
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function blendKey(blend: BlendSpec | null): string {
  if (blend === null) return 'null';
  const c = blend.color;
  const a = blend.alpha;
  return `${c.srcFactor},${c.dstFactor},${c.operation}/${a.srcFactor},${a.dstFactor},${a.operation}`;
}

/**
 * The key for the **pipeline layout and bind group layout** caches.
 *
 * A hash over the *resolved* structural fields only: vertex layout, varying
 * declaration order, slot type map, texture kinds, fixed-function state,
 * target formats, sample count. It deliberately excludes the shader body,
 * because the binding layouts a material needs depend on its declarations and
 * not on its statements — so two materials that differ only in how they shade
 * share one `GPUBindGroupLayout` per group, and one `GPUPipelineLayout`.
 *
 * Computed once per spec (memoised on the resolved form) and then only used as
 * a `Map` key, so the lookup itself is an integer compare.
 */
export function pipelineStateKeyOf(resolved: ResolvedMaterialSpec): number {
  return fnv1a(
    'apse-state-v1',
    resolved.layout.key,
    resolved.varyings.map((v) => `${v.location}:${v.name}:${v.wgsl}:${v.flat ? 1 : 0}`).join('|'),
    Object.entries(resolved.slotTypes).map(([k, v]) => `${k}:${v}`).join('|'),
    resolved.textures
      .map((t) => `${t.bindingIndex}:${t.varName}:${t.kind}:${t.sampleType}:${t.compare ? 1 : 0}`)
      .join('|'),
    resolved.samplers
      .map((s) => `${s.bindingIndex}:${s.key}:${s.wgslType}`)
      .join('|'),
    resolved.topology,
    resolved.cull,
    resolved.frontFace,
    `${resolved.depth.write ? 1 : 0}:${resolved.depth.compare}`,
    blendKey(resolved.blend),
    resolved.targets.map((t) => `${t.format}:${blendKey(t.blend)}:${t.writeMask}`).join('|'),
    String(resolved.sampleCount),
    resolved.depthFormat,
    resolved.stripIndexFormat ?? '-',
  );
}

/**
 * The key for the **`GPURenderPipeline`** cache.
 *
 * This is the pipeline state key *plus* a hash of the generated program, with
 * comments removed first.
 *
 * The two are separate keys on purpose, and the reason is the reason this
 * module exists. A `GPURenderPipeline` is not a bundle of interchangeable
 * state: it is a compiled, linked `GPUShaderModule` plus the fixed-function
 * state it was built with. Two materials whose generated WGSL differs in the
 * body have *different* shaders, and handing them the same pipeline object
 * would silently render one of them with the other's code — a bug with no
 * WebGPU diagnostic, no warning, and a plausible-looking frame.
 *
 * So: the state key answers "can these two materials share bind group
 * layouts?", which is a yes whenever the declarations match. The pipeline key
 * answers "is this the same compiled program?", which is a yes only when the
 * generated bytes match. Sharing is preserved wherever sharing is provably
 * correct, and given up only where correctness requires it.
 *
 * Stripping comments and collapsing whitespace is what keeps the second
 * question answerable. The material's `name` is emitted into the generated
 * header, and a body is liberally commented — so without this, every material
 * would be its own pipeline and the cache would be decorative. Comments cannot
 * change what a shader does, so they cannot change its identity; and WGSL is
 * free-form, so neither can the run of spaces a deleted comment leaves behind.
 */
export function pipelineKeyOf(resolved: ResolvedMaterialSpec, code: string): number {
  return fnv1a('apse-pipeline-v1', String(pipelineStateKeyOf(resolved)), tokenKeyOf(code));
}

/**
 * The semantic form of a WGSL program for hashing: comments deleted, every run
 * of whitespace collapsed to one space.
 *
 * Two programs with the same token sequence compile to the same shader. Two
 * programs that differ in a token do not — and a changed token always changes
 * the collapsed string, because `a;` and `b;` are not equal after collapsing.
 */
export function tokenKeyOf(code: string): string {
  return removeComments(code).replace(/\s+/g, ' ');
}

// ---------------------------------------------------------------------------
// describeMaterial
// ---------------------------------------------------------------------------

export interface DescribedAttribute {
  readonly name: string;
  readonly format: string;
  readonly location: number;
  readonly offset: number;
  readonly wgsl: string;
}

export interface DescribedVarying {
  readonly name: string;
  readonly type: string;
  readonly location: number;
  readonly flat: boolean;
  readonly builtin: boolean;
}

export interface DescribedSlot {
  readonly name: string;
  readonly type: SlotType;
  readonly wgsl: string;
  readonly components: number;
  readonly offset: number;
  readonly size: number;
  /** The variable name a body uses: `mat.<name>`. */
  readonly access: string;
}

export interface DescribedTexture {
  readonly slotName: string;
  /** The variable name a body uses. */
  readonly varName: string;
  readonly kind: TextureKind;
  readonly viewDimension: GPUTextureViewDimension;
  readonly sampleType: TextureSampleType;
  readonly binding: number;
  /** The one sampler variable this slot must use. */
  readonly sampler: string;
  readonly samplerBinding: number;
  /** Other slots sharing this sampler binding, if any. */
  readonly samplerSharedWith: readonly string[];
  readonly declaration: string;
}

export interface DescribedBindGroup {
  readonly index: number;
  readonly name: string;
  readonly present: boolean;
  readonly entries: readonly { binding: number; resource: string; visibility: string }[];
}

export interface MaterialDescription {
  readonly name: string;
  readonly layout: { readonly key: string; readonly stride: number; readonly attributes: readonly DescribedAttribute[] };
  readonly varyings: readonly DescribedVarying[];
  readonly slots: readonly DescribedSlot[];
  readonly textures: readonly DescribedTexture[];
  readonly samplers: readonly { name: string; binding: number; type: string; sharedBy: readonly string[] }[];
  readonly bindGroups: readonly DescribedBindGroup[];
  readonly blocks: {
    readonly frame: { readonly name: string; readonly size: number; readonly fields: number };
    readonly object: { readonly name: string; readonly size: number; readonly stride: number; readonly fields: number };
    readonly material: { readonly name: string; readonly size: number; readonly fields: number } | null;
  };
  /** Exactly what a body may reference, per stage. */
  readonly bodyNames: { readonly vertex: readonly string[]; readonly fragment: readonly string[] };
  readonly state: {
    readonly phase: DrawPhase;
    readonly topology: PrimitiveTopology;
    readonly cull: CullMode;
    readonly frontFace: FrontFace;
    readonly depth: DepthSpec;
    readonly blend: BlendSpec | null;
    readonly targets: readonly ResolvedTarget[];
    readonly sampleCount: number;
    readonly depthFormat: GPUTextureFormat;
  };
  readonly wgslBytes: number;
  readonly pipelineStateKey: number;
  readonly pipelineKey: number;
}

const describeCache = new WeakMap<MaterialSpec, MaterialDescription>();

/**
 * A machine-readable summary of what a spec generates.
 *
 * This is the anti-hallucination surface. It answers, without reading any
 * source: what slots exist and where they sit in memory, what varyings exist
 * and at which location, which sampler variable each texture must use, what
 * each bind group contains, and how long the generated program is.
 *
 * Memoised per spec object, so calling it every frame in a dev build is one
 * `WeakMap` lookup.
 */
export function describeMaterial(spec: MaterialSpec, limits?: InterStageLimits): MaterialDescription {
  const hit = describeCache.get(spec);
  if (hit !== undefined) return hit;
  const generated = generateScaffold(spec, limits);
  const r = generated.resolved;
  const vis = 'VERTEX | FRAGMENT';

  const described: MaterialDescription = {
    name: r.name,
    layout: {
      key: r.layout.key,
      stride: r.layout.stride,
      attributes: r.layout.attributes.map((a) => ({
        name: a.name,
        format: a.format,
        location: a.location,
        offset: a.offset,
        wgsl: a.info.wgsl,
      })),
    },
    varyings: r.varyings.map((v) => ({
      name: v.name,
      type: v.wgsl,
      location: v.location,
      flat: v.flat,
      builtin: v.location < 0,
    })),
    slots: generated.materialBlock.fields.map((f) => ({
      name: f.name,
      type: f.type,
      wgsl: UNIFORM_TYPES[f.type].wgsl,
      components: f.components,
      offset: f.offset,
      size: f.size,
      access: `mat.${f.name}`,
    })),
    textures: r.textures.map((t) => {
      const s = r.samplerForSlot.get(t.slotName);
      return {
        slotName: t.slotName,
        varName: t.varName,
        kind: t.kind,
        viewDimension: textureViewDimension(t.kind),
        sampleType: t.sampleType,
        binding: t.bindingIndex,
        sampler: s === undefined ? '' : s.varName,
        samplerBinding: s === undefined ? -1 : s.bindingIndex,
        samplerSharedWith: s === undefined ? [] : s.slotNames.filter((n) => n !== t.slotName),
        declaration: wgslTextureDecl(t.varName, t),
      };
    }),
    samplers: r.samplers.map((s) => ({
      name: s.varName,
      binding: s.bindingIndex,
      type: s.wgslType,
      sharedBy: s.slotNames,
    })),
    bindGroups: [
      {
        index: BIND_GROUP.frame,
        name: 'frame',
        present: true,
        entries: [{ binding: 0, resource: `uniform ${FRAME_BLOCK.structName} (${FRAME_BLOCK.size} B)`, visibility: vis }],
      },
      {
        index: BIND_GROUP.object,
        name: 'object',
        present: true,
        entries: [{
          binding: 0,
          resource: `uniform ${OBJECT_BLOCK.structName} (dynamic offset, stride ${OBJECT_BLOCK.stride} B)`,
          visibility: vis,
        }],
      },
      {
        index: BIND_GROUP.material,
        name: 'material',
        present: r.materialBlock !== null,
        entries: r.materialBlock === null ? [] : [{
          binding: 0,
          resource: `uniform ${r.materialBlock.structName} (${r.materialBlock.size} B)`,
          visibility: vis,
        }],
      },
      {
        index: BIND_GROUP.texture,
        name: 'texture',
        present: r.textures.length > 0,
        entries: [
          ...r.textures.map((t) => ({
            binding: t.bindingIndex,
            resource: `${t.kind} / ${t.sampleType} (${textureViewDimension(t.kind)})`,
            visibility: vis,
          })),
          ...r.samplers.map((s) => ({
            binding: s.bindingIndex,
            resource: `sampler ${s.wgslType} (${s.addressMode}${s.mipmapFilter ? ', mipmapped' : ''})`,
            visibility: vis,
          })),
        ],
      },
    ],
    blocks: {
      frame: { name: FRAME_BLOCK.structName, size: FRAME_BLOCK.size, fields: FRAME_BLOCK.fields.length },
      object: {
        name: OBJECT_BLOCK.structName,
        size: OBJECT_BLOCK.size,
        stride: OBJECT_BLOCK.stride,
        fields: OBJECT_BLOCK.fields.length,
      },
      material: r.materialBlock === null ? null : {
        name: r.materialBlock.structName,
        size: r.materialBlock.size,
        fields: r.materialBlock.fields.length,
      },
    },
    bodyNames: { vertex: bodyNames(r, 'vertex'), fragment: bodyNames(r, 'fragment') },
    state: {
      phase: r.phase,
      topology: r.topology,
      cull: r.cull,
      frontFace: r.frontFace,
      depth: r.depth,
      blend: r.blend,
      targets: r.targets,
      sampleCount: r.sampleCount,
      depthFormat: r.depthFormat,
    },
    wgslBytes: generated.byteLength,
    pipelineStateKey: generated.pipelineStateKey,
    pipelineKey: pipelineKeyOf(r, generated.code),
  };

  describeCache.set(spec, described);
  return described;
}

/**
 * A structural check on generated WGSL: balanced braces, every struct field
 * terminated with a comma, and `@group(n)` indices that agree with
 * {@link BIND_GROUP}.
 *
 * These are the invariants a generator can violate while looking perfectly
 * reasonable. Cheap to check, and it turns a syntax error in generated code
 * into a named apse failure.
 */
export function validateGeneratedWGSL(code: string): { readonly ok: true } {
  let depth = 0;
  for (let i = 0; i < code.length; i++) {
    if (code[i] === '{') depth++;
    else if (code[i] === '}') {
      depth--;
      if (depth < 0) {
        fail('INTERNAL_INVARIANT', 'Generated WGSL has an unmatched closing brace.', {
          why: `at character ${i}`,
          fix: 'Report this with the material spec that produced it.',
        });
      }
    }
  }
  if (depth !== 0) {
    fail('INTERNAL_INVARIANT', `Generated WGSL has ${depth} unclosed brace(s).`, {
      why: 'The scaffold builds every struct and entry point from a fixed template, so a brace count other than zero means a spec field broke the template.',
      fix: 'Report this with the material spec that produced it.',
    });
  }

  for (const m of code.matchAll(/struct\s+(\w+)\s*\{([^}]*)\}/g)) {
    for (const line of m[2].split('\n')) {
      const t = line.trim();
      if (t.length === 0) continue;
      if (!t.endsWith(',')) {
        fail('INTERNAL_INVARIANT', `Field "${t}" in generated struct ${m[1]} is missing its trailing comma.`, {
          why: 'WGSL requires a comma after every struct member; the generated structs come from a fixed template that always emits one.',
          fix: 'Report this with the material spec that produced it.',
        });
      }
    }
  }

  // Groups 0-2 each expose exactly one variable, fixed by BIND_GROUP. Group 3
  // is the texture group and holds a variable per texture and per shared
  // sampler, so it is checked for membership rather than for a single name.
  const fixedName = new Map<number, string>([
    [BIND_GROUP.frame, 'frame'],
    [BIND_GROUP.object, 'obj'],
    [BIND_GROUP.material, 'mat'],
  ]);
  const groups = new Set<number>([...fixedName.keys(), BIND_GROUP.texture]);

  for (const m of code.matchAll(/@group\((\d+)\)\s*@binding\((\d+)\)\s*var\s*(?:<(\w+)>\s*)?(\w+)/g)) {
    const group = Number(m[1]);
    const name = m[4];
    if (!groups.has(group)) {
      fail('INTERNAL_INVARIANT', `Generated WGSL binds @group(${group}), which is not in BIND_GROUP.`, {
        why: `BIND_GROUP is { frame: ${BIND_GROUP.frame}, object: ${BIND_GROUP.object}, material: ${BIND_GROUP.material}, texture: ${BIND_GROUP.texture} } and the two must not drift, or the renderer's positional layouts would bind the wrong thing.`,
        fix: 'Report this with the material spec that produced it.',
      });
    }
    const want = fixedName.get(group);
    if (want !== undefined && name !== want) {
      fail('INTERNAL_INVARIANT', `Generated WGSL binds "${name}" at @group(${group}); that group must be "${want}".`, {
        why: 'The bind group layouts the renderer builds are positional, so the variable a group exposes has to be the one the layout is described against.',
        fix: 'Report this with the material spec that produced it.',
      });
    }
  }

  return { ok: true };
}
