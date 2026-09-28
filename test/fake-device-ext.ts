/**
 * A fake GPU device that also speaks pipelines, buffers, and shader modules.
 *
 * `test/fake-device.ts` records textures and error scopes, which is everything
 * `RenderTargetImpl` needs and nothing `Material` needs. `Material.create` goes
 * further: it creates a bind group layout, a pipeline layout, a shader module,
 * two buffers, a sampler, and an **async** render pipeline, and then — on the
 * path this file exists for — asks the shader module for its compilation
 * diagnostics. None of that is in the shared fake, and putting it there would
 * mean editing a file another agent owns, so this extends `FakeGPUDevice` rather
 * than replacing it.
 *
 * # The three shapes of "the compiler said something"
 *
 * The whole point of the diagnostics path is that it has to cope with an
 * implementation that answers and with one that does not, so all three are
 * expressible:
 *
 *   - `compilationInfo: 'present'` — the normal case. The fake returns whatever
 *     messages the test put in {@link FakePipelineDeviceOptions.compilationMessages}.
 *   - `'absent'` — the method is **not defined at all** on the returned module.
 *     Several shipping implementations have no `getCompilationInfo`, and code
 *     that assumes it exists throws a `TypeError` from inside its own error
 *     handler, replacing a useful message with `getCompilationInfo is not a
 *     function`.
 *   - `'throws'` — the method exists and rejects, which is what a lost device or
 *     an implementation that validates lazily produces.
 *
 * # What is *not* faked
 *
 * The fake never parses or validates WGSL. It reports the diagnostics the test
 * dictates, which is the only way to assert that apse routes a diagnostic to the
 * right stage: a fake that actually compiled the shader would be a second WGSL
 * front end, and a test against it would only prove that the two agree with each
 * other. The stage attribution is apse's own `stageAtLine` walking apse's own
 * section banners, so the test pins that function and the line numbers it is
 * given — which is the part that can actually regress.
 */

import { FakeGPUDevice, fakeLimits } from './fake-device.ts';
import type { FakeDeviceOptions } from './fake-device.ts';

/** One message from `getCompilationInfo()`. */
export interface FakeCompilationMessage {
  readonly type: 'error' | 'warning' | 'info';
  readonly message: string;
  /** 1-based, as the real API reports it. */
  readonly lineNum: number;
  /** 1-based column, as the real API reports it. */
  readonly linePos: number;
}

/** How the fake's shader module answers a compilation-info request. */
export type CompilationInfoMode = 'present' | 'absent' | 'throws';

export interface FakePipelineDeviceOptions extends FakeDeviceOptions {
  /** Messages `getCompilationInfo()` returns, in order. */
  readonly compilationMessages?: readonly FakeCompilationMessage[];
  readonly compilationInfo?: CompilationInfoMode;
  /** Make `createRenderPipelineAsync` reject with this. Defaults to succeeding. */
  readonly pipelineError?: Error | null;
}

/** One `writeBuffer` call, with its bytes copied out. */
export interface RecordedWrite {
  readonly label: string;
  readonly bufferOffset: number;
  readonly size: number;
  /** A copy, so a later mutation of the caller's mirror cannot change history. */
  readonly bytes: Uint8Array;
}

export class FakePipelineDevice extends FakeGPUDevice {
  /** Every shader module handed out, in creation order. */
  readonly modules: { label: string; code: string }[] = [];
  /** `createRenderPipelineAsync` descriptors, in order. */
  readonly pipelineDescs: GPURenderPipelineDescriptor[] = [];
  readonly bindGroupLayouts: GPUBindGroupLayoutDescriptor[] = [];
  readonly pipelineLayouts: GPUPipelineLayoutDescriptor[] = [];
  readonly bindGroups: GPUBindGroupDescriptor[] = [];
  readonly buffers: { label: string; size: number; destroyed: boolean }[] = [];
  readonly samplers: GPUSamplerDescriptor[] = [];
  readonly writes: RecordedWrite[] = [];

  readonly #messages: readonly FakeCompilationMessage[];
  readonly #mode: CompilationInfoMode;
  readonly #pipelineError: Error | null;
  #compilationInfoCalls = 0;

  constructor(opts: FakePipelineDeviceOptions = {}) {
    super({ limits: opts.limits ?? fakeLimits(), features: opts.features, errorScopes: opts.errorScopes });
    this.#messages = opts.compilationMessages ?? [];
    this.#mode = opts.compilationInfo ?? 'present';
    this.#pipelineError = opts.pipelineError ?? null;
  }

  /** How many times apse asked a module for its diagnostics. */
  get compilationInfoCalls(): number {
    return this.#compilationInfoCalls;
  }

  get liveBuffers(): number {
    return this.buffers.filter((b) => !b.destroyed).length;
  }

  createShaderModule(desc: GPUShaderModuleDescriptor): GPUShaderModule {
    this.modules.push({ label: desc.label ?? '', code: desc.code });
    const mode = this.#mode;
    const messages = this.#messages;
    const device = this;
    const module: Record<string, unknown> = { label: desc.label, code: desc.code };
    if (mode !== 'absent') {
      module.getCompilationInfo = (): Promise<GPUCompilationInfo> => {
        device.#compilationInfoCalls++;
        if (mode === 'throws') {
          return Promise.reject(new Error('getCompilationInfo is not supported by this device'));
        }
        return Promise.resolve({
          messages: messages.map((m) => ({ ...m })),
        } as unknown as GPUCompilationInfo);
      };
    }
    return module as unknown as GPUShaderModule;
  }

  createRenderPipelineAsync(desc: GPURenderPipelineDescriptor): Promise<GPURenderPipeline> {
    this.pipelineDescs.push(desc);
    if (this.#pipelineError !== null) return Promise.reject(this.#pipelineError);
    return Promise.resolve({ label: desc.label } as unknown as GPURenderPipeline);
  }

  createBindGroupLayout(desc: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
    this.bindGroupLayouts.push(desc);
    return { label: desc.label } as unknown as GPUBindGroupLayout;
  }

  createPipelineLayout(desc: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
    this.pipelineLayouts.push(desc);
    return { label: desc.label } as unknown as GPUPipelineLayout;
  }

  createBindGroup(desc: GPUBindGroupDescriptor): GPUBindGroup {
    this.bindGroups.push(desc);
    return { label: desc.label } as unknown as GPUBindGroup;
  }

  createSampler(desc: GPUSamplerDescriptor): GPUSampler {
    this.samplers.push(desc);
    return { label: desc.label } as unknown as GPUSampler;
  }

  createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
    const record = { label: desc.label ?? '', size: desc.size, destroyed: false };
    this.buffers.push(record);
    return {
      label: desc.label,
      size: desc.size,
      usage: desc.usage,
      destroy: () => {
        if (record.destroyed) {
          throw new Error(`FakePipelineDevice: buffer "${record.label}" destroyed twice`);
        }
        record.destroyed = true;
      },
    } as unknown as GPUBuffer;
  }

  get queue(): GPUQueue {
    const device = this;
    return {
      writeBuffer(
        buffer: GPUBuffer,
        bufferOffset: number,
        data: unknown,
        dataOffset?: number,
        size?: number,
      ): void {
        // `data` is an ArrayBuffer in every call apse makes, and dataOffset and
        // size are then byte counts. The copy matters: apse hands over a live CPU
        // mirror and writes to it again on the next frame, so a retained view
        // would make every recorded frame read the last one's values.
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data.slice(dataOffset ?? 0, (dataOffset ?? 0) + (size ?? data.byteLength)))
          : new Uint8Array(0);
        device.writes.push({
          label: (buffer as unknown as { label: string }).label ?? '',
          bufferOffset,
          size: bytes.byteLength,
          bytes,
        });
      },
      submit(): void { /* nothing is submitted in these tests */ },
    } as unknown as GPUQueue;
  }
}

/** Type-narrowing helper, matching the one in `test/fake-device.ts`. */
export function asPipelineDevice(device: FakePipelineDevice): GPUDevice {
  return device as unknown as GPUDevice;
}

/** The 1-based line number of `fn vs(`, the vertex entry point. */
export function vertexEntryLineOf(code: string): number {
  return lineOf(code, 'fn vs(');
}

/** The 1-based line number of `fn fs(`, the fragment entry point. */
export function fragmentEntryLineOf(code: string): number {
  return lineOf(code, 'fn fs(');
}

/** The 1-based line number of the prelude section banner. */
export function preludeBannerLineOf(code: string): number {
  return lineOf(code, '// ---- prelude');
}

/** The 1-based line number of the vertex section banner. */
export function vertexBannerLineOf(code: string): number {
  return lineOf(code, '// ---- generated by apse: vertex stage');
}

/** The 1-based line number of the fragment section banner. */
export function fragmentBannerLineOf(code: string): number {
  return lineOf(code, '// ---- generated by apse: fragment stage');
}

/**
 * The 1-based line number of the first *statement* of the user-supplied prelude.
 *
 * Two lines past the banner, because the generator puts one blank line between
 * the section label and the text. A diagnostic aimed at the banner itself would
 * be aimed at apse's own text rather than at the author's.
 */
export function preludeFirstLineOf(code: string): number {
  return preludeBannerLineOf(code) + 2;
}

function lineOf(code: string, prefix: string): number {
  const at = code.split('\n').findIndex((l) => l.startsWith(prefix));
  if (at < 0) throw new Error(`fake-device-ext: no line starting with ${JSON.stringify(prefix)}`);
  return at + 1;
}
