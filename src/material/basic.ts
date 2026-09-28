/**
 * The unlit material — the "hello cube" path.
 *
 * It exists to be the shortest possible demonstration that the scaffold works:
 * a colour, an opacity, and two lines of WGSL per stage. It is written through
 * exactly the same declarative path as {@link ../pbr.ts} — no hand-written
 * WGSL, no bypass, no special case in the generator. If `basic` and `pbr` both
 * work, the abstraction scales past a toy.
 */

import { STANDARD_LAYOUT } from '../geometry/layout.ts';
import type { VertexLayout } from '../geometry/layout.ts';
import { Material } from './material.ts';
import type { MaterialOptions } from './material.ts';
import type { MaterialSpec } from './scaffold.ts';

export interface BasicMaterialOptions extends MaterialOptions {
  /** Defaults to `'basic'`. */
  name?: string;
  /** Unlit surface colour, linear. Defaults to white. */
  color?: readonly [number, number, number];
  /** Per-material alpha. Defaults to 1. */
  opacity?: number;
  /**
   * Declare an `albedo` texture slot and tint the colour by it. The generated
   * program differs by one line, which is the point: changing what a material
   * does is a change to its spec, not to a shader string.
   */
  textured?: boolean;
  /** Alpha-blend and skip depth writes. Defaults to false. */
  transparent?: boolean;
  /** Disable back-face culling. Defaults to false. */
  doubleSided?: boolean;
  /** Defaults to {@link STANDARD_LAYOUT}. */
  layout?: VertexLayout;
  /**
   * Colour target format. Defaults to `navigator.gpu.getPreferredCanvasFormat()`.
   *
   * WebGPU bakes the attachment format into a pipeline, so a material is
   * permanently bound to the format it was compiled for. The preferred canvas
   * format is `bgra8unorm` on desktop and `rgba8unorm` on Android, so a
   * hardcoded default is wrong on one of them, and the failure is a Dawn
   * validation error at `setPipeline` that names a format rather than the
   * mistake.
   *
   * Pass this explicitly when drawing into an offscreen target — an HDR
   * `rgba16float` post-processing chain, for instance — since the canvas format
   * is the wrong answer there.
   */
  targetFormat?: GPUTextureFormat;
}

/**
 * The spec `basicMaterial` builds. Exported so it can be inspected, tested, and
 * diffed without a device — the whole point of the scaffold is that the
 * material *is* its spec.
 */
export function basicMaterialSpec(opts: BasicMaterialOptions = {}): MaterialSpec {
  const {
    name = 'basic',
    color = [1, 1, 1],
    opacity = 1,
    textured = false,
    transparent = false,
    doubleSided = false,
    layout = STANDARD_LAYOUT,
    targetFormat = preferredFormat(),
  } = opts;

  return {
    name,
    layout,
    varyings: { uv: 'vec2f' },
    slots: {
      tint: { type: 'vec3f', default: color },
      opacity: { type: 'f32', default: opacity },
    },
    ...(textured ? { textures: { albedo: { kind: '2d' } as const } } : {}),
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

    // Statements only. `in.position` is the @location(0) attribute, `out` is
    // the generated Varyings, and `frame`/`obj`/`mat` are already bound.
    vertex: `
out.clip = frame.viewProj * obj.model * vec4f(in.position, 1.0);
out.uv = in.uv;
`,

    fragment: textured
      ? `
// Unlit, tinted by the texture. The sampler variable is generated from the
// texture declaration, so there is nothing to bind by hand.
let texel = textureSample(albedo, albedoSampler, in.uv);
return vec4f(texel.rgb * mat.tint, texel.a * mat.opacity * frame.alpha);
`
      : `
// Unlit. frame.alpha is the global alpha, so a UI fade does not need a
// per-material edit.
return vec4f(mat.tint, mat.opacity * frame.alpha);
`,
  };
}

/**
 * Creates the unlit material.
 *
 * Returns a promise because pipeline compilation must not block the main
 * thread — see the comment at the top of `material.ts`.
 */
export function basicMaterial(
  device: GPUDevice,
  opts: BasicMaterialOptions = {},
): Promise<Material> {
  const { frame, object, maxObjects } = opts;
  return Material.create(device, basicMaterialSpec(opts), { frame, object, maxObjects });
}

/**
 * The canvas format a material should target when the caller does not say.
 *
 * Read at call time rather than captured at module load, and guarded for
 * non-browser contexts so importing a material factory in a Node test does not
 * throw. `rgba8unorm` is the documented fallback for anywhere there is no
 * canvas — an offscreen target, where the caller must pass the format anyway.
 */
function preferredFormat(): GPUTextureFormat {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined'
    ? navigator.gpu.getPreferredCanvasFormat()
    : 'rgba8unorm';
}
