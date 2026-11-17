/* global AudioWorkletProcessor, registerProcessor, sampleRate */
/**
 * PCM capture processor — runs on the browser's real-time audio thread.
 *
 * It only converts audio into the wire format; all analysis (VAD,
 * segmentation, transcription) happens in Python on the server.
 *
 *   input (any rate/channels) → mono → [resample to 16 kHz if needed]
 *     → int16 PCM in 20 ms chunks → transferred (zero-copy) to the main thread
 */

/** RBJ low-pass biquad (Transposed Direct Form II) used as anti-aliasing filter. */
function createLowPass(rate, cutoffHz, q) {
  const w0 = (2 * Math.PI * cutoffHz) / rate;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha;
  const b0 = (1 - cos) / 2 / a0;
  const b1 = (1 - cos) / a0;
  const a1 = (-2 * cos) / a0;
  const a2 = (1 - alpha) / a0;
  let z1 = 0;
  let z2 = 0;
  return (x) => {
    const y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b0 * x - a2 * y;
    return y;
  };
}

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetSampleRate || 16000;
    this.chunkSize = opts.chunkSamples || 320; // 20 ms at 16 kHz

    // Input samples per output sample (3 for 48 kHz → 16 kHz).
    this.step = sampleRate / this.targetRate;
    this.resampling = Math.abs(this.step - 1) > 1e-9;
    // Two cascaded biquads with Butterworth Qs = 4th-order anti-aliasing filter.
    const cutoff = this.targetRate * 0.45;
    this.antiAlias = this.resampling
      ? [createLowPass(sampleRate, cutoff, 0.5412), createLowPass(sampleRate, cutoff, 1.3066)]
      : [];

    this.position = 0; // read position of the next output sample within the current block
    this.previous = 0; // last sample of the previous block (index -1 for interpolation)
    this.mono = new Float32Array(128);
    this.chunk = new Int16Array(this.chunkSize);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;
    const block = this.downmix(input);
    if (this.resampling) this.resample(block);
    else for (let i = 0; i < block.length; i++) this.push(block[i]);
    return true;
  }

  downmix(channels) {
    const length = channels[0].length;
    if (this.mono.length !== length) this.mono = new Float32Array(length);
    if (channels.length === 1) {
      this.mono.set(channels[0]);
      return this.mono;
    }
    for (let i = 0; i < length; i++) {
      let sum = 0;
      for (let c = 0; c < channels.length; c++) sum += channels[c][i];
      this.mono[i] = sum / channels.length;
    }
    return this.mono;
  }

  resample(block) {
    const n = block.length;
    for (let i = 0; i < n; i++) {
      let s = block[i];
      for (let f = 0; f < this.antiAlias.length; f++) s = this.antiAlias[f](s);
      block[i] = s;
    }
    let t = this.position;
    while (t < n - 1) {
      const i = Math.floor(t);
      const frac = t - i;
      const a = i < 0 ? this.previous : block[i];
      this.push(a + (block[i + 1] - a) * frac);
      t += this.step;
    }
    this.position = t - n;
    this.previous = block[n - 1];
  }

  push(sample) {
    const s = Math.max(-1, Math.min(1, sample));
    this.chunk[this.filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.filled === this.chunkSize) {
      this.port.postMessage(this.chunk, [this.chunk.buffer]);
      this.chunk = new Int16Array(this.chunkSize);
      this.filled = 0;
    }
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
