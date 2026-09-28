/**
 * The instanced material — the same unlit shader as {@link ./basic.ts}, with
 * the per-instance transform added to it.
 *
 * ## How the instance data reaches the vertex stage
 *
 * Through the spec's `layout`, which is a {@link VertexLayout} carrying the
 * per-instance attributes as well as the mesh's. That is the whole mechanism:
 * apse generates the `VertexIn` struct from the layout, the struct is
 * `@location(3..6)` for a `mat4x4f` and `@location(7)` for an optional colour,
 * and the pipeline gets two `GPUVertexBufferLayout` slots. No storage buffer, no
 * bind group, no vertex-stage storage read — which is the only reason this
 * works on the compatibility profile at all.
 *
 * ## How the body reads it, and why there is a prelude
 *
 * The generated vertex entry point is `fn vs(in : VertexIn)`, and the body is a
 * statement list spliced into it. It can therefore name `in.position`,
 * `in.normal`, `in.uv` — but only the fields the body validator knows about,
 * which are the *vertex* attributes.
 *
 * Rather than change that, the two accessors below are declared in the
 * `prelude` and take the whole `VertexIn` value. The body then reads
 * `instanceModel(in)` and never names a field it is not allowed to name, so the
 * scaffold's identifier validation — the thing that turns a typo into a typed
 * apse error instead of a WGSL compile error pointing into generated code — stays
 * intact and unchanged.
 *
 * The cost is that a typo in the prelude's own field names is caught by Tint
 * rather than by apse. It is two field names in a file that is the only place
 * they appear, and the alternative is a change to `scaffold.ts` that every
 * material pays for. See the note on `inFields` in `validateBodyIdentifiers`:
 * adding the instance attribute names to that list, and nothing else, would let
 * a body write `mat4x4f(in.instanceTransform0, …)` directly.
 *
 * ## The composition order
 *
 * `frame.viewProj · obj.model · instance · position`. The node's world matrix is
 * the parent and each instance transform is a child of it, which means a
 * `MeshNode` with an instance buffer places all of its instances under its own
 * transform — the same relationship a `Node` parent would have — and a node at
 * identity is a no-op, which is what a non-instanced scene relies on.
 */

import { fail } from '../core/error.ts';
import {
  INSTANCE_ATTRIBUTES,
  INSTANCE_ATTRIBUTES_COLORED,
  instancedLayout,
} from '../geometry/instanced.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import { Material } from './material.ts';
import type { MaterialOptions } from './material.ts';
import type { MaterialSpec } from './scaffold.ts';

export interface InstancedMaterialOptions extends MaterialOptions {
  /** Defaults to `'instanced'`. */
  name?: string;
  /**
   * The combined layout: mesh attributes plus instance attributes.
   *
   * Defaults to {@link instancedLayout}, which is `STANDARD_LAYOUT`'s three
   * vertex attributes plus the four transform columns — 7 of the 16 locations
   * the device allows. It must be one of the two layouts apse generates a body
   * for; anything else is rejected rather than compiled into a shader that
   * cannot draw.
   */
  layout?: VertexLayout;
  /**
   * Declare the per-instance colour attribute and pass it to the fragment
   * stage. Ignored when `layout` is given explicitly, since the layout is then
   * the authority on which attributes exist.
   */
  instanceColor?: boolean;
  /** Base tint the per-instance colour multiplies. Defaults to white. */
  color?: readonly [number, number, number];
  /** Per-material alpha. Defaults to 1. */
  opacity?: number;
  /** Alpha-blend and skip depth writes. Defaults to false. */
  transparent?: boolean;
  /** Disable back-face culling. Defaults to false. */
  doubleSided?: boolean;
  /**
   * Colour target format. Defaults to `navigator.gpu.getPreferredCanvasFormat()`,
   * exactly as `basicMaterial` does — see the note there.
   */
  targetFormat?: GPUTextureFormat;
}

/**
 * The attribute names the generated body reads, and the two accessors it reads
 * them through.
 *
 * Both are generated rather than written out, so the body and the struct apse
 * emits cannot disagree about a name: the struct comes from the same layout, and
 * a layout that does not declare these names is rejected before any of it runs.
 */
const TRANSFORM_ACCESSOR = `
fn instanceModel(v : VertexIn) -> mat4x4f {
  return mat4x4f(v.instanceTransform0, v.instanceTransform1, v.instanceTransform2, v.instanceTransform3);
}`;

const COLOR_ACCESSOR = `
fn instanceTint(v : VertexIn) -> vec4f {
  return v.instanceColor;
}`;

/**
 * The spec `instancedMaterial` builds. Exported so it can be generated, read,
 * and tested with no device — a material is its spec.
 */
export function instancedMaterialSpec(opts: InstancedMaterialOptions = {}): MaterialSpec {
  const {
    name = 'instanced',
    instanceColor = false,
    color = [1, 1, 1],
    opacity = 1,
    transparent = false,
    doubleSided = false,
    targetFormat = preferredFormat(),
  } = opts;
  const layout = opts.layout ?? instancedLayout({ color: instanceColor });

  // The body is generated from the canonical instance attribute names, so the
  // layout has to be the one this material knows how to shade. A caller who
  // built their own combined layout is told exactly what is missing rather than
  // handed a shader that references fields the struct does not have — which
  // would be a WGSL compile error, not an apse one.
  const attributes = instanceColor ? INSTANCE_ATTRIBUTES_COLORED : INSTANCE_ATTRIBUTES;
  const wanted = Object.entries(attributes).map(([n, f]) => `${n}:${f}`);
  const declared = layout.instanceAttributes.map((a) => `${a.name}:${a.format}`);
  if (declared.join() !== wanted.join()) {
    fail('ATTRIBUTE_MISSING',
      `Instanced material "${name}" needs the per-instance attributes ${wanted.join(', ')}, but the layout it was given declares ${declared.join(', ') || 'none'}.`, {
      why: 'The generated vertex body reads the instance transform and the optional tint by name, through the prelude accessors. apse generates that body and the `VertexIn` struct from the layout together, so a layout with different per-instance attributes would produce a program that reads fields the struct does not have — or reads four bytes per column where the shader expects sixteen.',
      fix: `Build the layout with \`instancedLayout({ color: ${instanceColor} })\`, or declare exactly these attributes, in this order: ${Object.entries(attributes).map(([n, f]) => `${n}: "${f}"`).join(', ')}.`,
    });
  }

  if (layout.attribute('position') === undefined) {
    fail('ATTRIBUTE_MISSING',
      `Instanced material "${name}" was given a layout with no "position" attribute.`, {
      why: 'The generated vertex body multiplies `in.position` by the instance transform. There is nothing else it could place, and a mesh without positions has no vertices to step the instance buffer over.',
      fix: 'Build the layout from `STANDARD_ATTRIBUTES` or `POSITION_LAYOUT`, or declare at least `position: "float32x3"`.',
    });
  }

  return {
    name,
    layout,
    // No uv varying: this material is unlit and samples nothing, so an
    // inter-stage varying would be a location and a bandwidth spent on a value
    // no one reads. That also means the material works with any vertex layout
    // that has a position, rather than demanding a uv the shader would discard.
    varyings: instanceColor ? { color: 'vec4f' } : {},
    slots: {
      tint: { type: 'vec3f', default: color },
      opacity: { type: 'f32', default: opacity },
    },
    prelude: TRANSFORM_ACCESSOR + (instanceColor ? COLOR_ACCESSOR : ''),
    phase: transparent ? 'transparent' : 'opaque',
    topology: 'triangle-list',
    cull: doubleSided ? 'none' : 'back',
    depth: transparent ? { write: false, compare: 'less' } : { write: true, compare: 'less' },
    blend: transparent
      ? {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      }
      : null,
    targets: [{ format: targetFormat }],
    sampleCount: 1,

    // Statements only, as everywhere else. `instanceModel(in)` reads the four
    // per-instance columns through the prelude accessor; the rest is the same
    // unlit shader `basicMaterial` generates.
    vertex: `
out.clip = frame.viewProj * obj.model * instanceModel(in) * vec4f(in.position, 1.0);
${instanceColor ? 'out.color = instanceTint(in);' : ''}
`,

    fragment: instanceColor
      ? `
// The instance tint multiplies the material tint, so a material-wide colour
// still works and the per-instance value only ever darkens or recolours.
return vec4f(in.color.rgb * mat.tint, in.color.a * mat.opacity * frame.alpha);
`
      : `
// Unlit, exactly like basicMaterial. The instance transform has already been
// applied; the colour is the material's own.
return vec4f(mat.tint, mat.opacity * frame.alpha);
`,
  };
}

/**
 * Creates the instanced material.
 *
 * Returns a promise because pipeline compilation must not block the main thread
 * — see the comment at the top of `material.ts`.
 */
export function instancedMaterial(
  device: GPUDevice,
  opts: InstancedMaterialOptions = {},
): Promise<Material> {
  const { frame, object, maxObjects } = opts;
  return Material.create(device, instancedMaterialSpec(opts), { frame, object, maxObjects });
}

/**
 * The canvas format a material should target when the caller does not say.
 * Same reasoning and same fallback as `basicMaterial`.
 */
function preferredFormat(): GPUTextureFormat {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined'
    ? navigator.gpu.getPreferredCanvasFormat()
    : 'rgba8unorm';
}
