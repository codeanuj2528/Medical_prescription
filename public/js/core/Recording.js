/** Keeps the whole session (speech + silence) in memory for "Save Recording". */
export class Recording {
  #chunks = [];
  #samples = 0;

  constructor(sampleRate = 16_000, maxMs = 45 * 60 * 1000) {
    this.sampleRate = sampleRate;
    this.maxSamples = (maxMs / 1000) * sampleRate;
  }

  get durationMs() {
    return (this.#samples / this.sampleRate) * 1000;
  }

  get isEmpty() {
    return this.#samples === 0;
  }

  /** @param {Int16Array} pcm */
  append(pcm) {
    if (this.#samples >= this.maxSamples) return;
    this.#chunks.push(pcm);
    this.#samples += pcm.length;
  }

  clear() {
    this.#chunks = [];
    this.#samples = 0;
  }

  /**
   * PCM from `fromMs` to now as ≤ 1 s views (WebSocket-message sized), for
   * replaying audio a dropped connection never finished transcribing.
   * @returns {Int16Array[]}
   */
  sliceFrom(fromMs) {
    const start = Math.max(0, Math.round((fromMs / 1000) * this.sampleRate));
    const pieces = [];
    let offset = 0;
    for (const chunk of this.#chunks) {
      const end = offset + chunk.length;
      if (end > start) pieces.push(chunk.subarray(Math.max(0, start - offset)));
      offset = end;
    }
    const merged = new Int16Array(pieces.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const piece of pieces) {
      merged.set(piece, at);
      at += piece.length;
    }
    const views = [];
    for (let i = 0; i < merged.length; i += this.sampleRate) views.push(merged.subarray(i, i + this.sampleRate));
    return views;
  }

  /** 44-byte RIFF header + PCM16 data (chunks are already little-endian int16). */
  toWavBlob() {
    const dataBytes = this.#samples * 2;
    const header = new DataView(new ArrayBuffer(44));
    const ascii = (offset, text) => [...text].forEach((c, i) => header.setUint8(offset + i, c.charCodeAt(0)));
    ascii(0, "RIFF");
    header.setUint32(4, 36 + dataBytes, true);
    ascii(8, "WAVE");
    ascii(12, "fmt ");
    header.setUint32(16, 16, true);
    header.setUint16(20, 1, true); // PCM
    header.setUint16(22, 1, true); // mono
    header.setUint32(24, this.sampleRate, true);
    header.setUint32(28, this.sampleRate * 2, true);
    header.setUint16(32, 2, true);
    header.setUint16(34, 16, true);
    ascii(36, "data");
    header.setUint32(40, dataBytes, true);
    return new Blob([header.buffer, ...this.#chunks], { type: "audio/wav" });
  }
}
