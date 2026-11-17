/**
 * Live VAD view: energy bars for the last 4 s, coloured by the server's VAD
 * decision, plus the adaptive threshold line. Draws on requestAnimationFrame
 * and only when new telemetry arrived.
 */
const HISTORY = 200; // frames of 20 ms = 4 s
const DB_MIN = -75;
const DB_MAX = -15;
const COLORS = { utterance: "#6366f1", onset: "#a5adfc", noise: "#cbd5e1", threshold: "#f59e0b" };

export class Waveform {
  #canvas;
  #ctx;
  #frames = []; // [energyDb, thresholdDb, isSpeech, inUtterance]
  #dirty = true;

  constructor(canvas) {
    this.#canvas = canvas;
    this.#ctx = canvas.getContext("2d");
    new ResizeObserver(() => this.#resize()).observe(canvas);
    this.#resize();
    const loop = () => {
      if (this.#dirty) this.#paint();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  push(frames) {
    this.#frames.push(...frames);
    if (this.#frames.length > HISTORY) this.#frames.splice(0, this.#frames.length - HISTORY);
    this.#dirty = true;
  }

  clear() {
    this.#frames = [];
    this.#dirty = true;
  }

  #resize() {
    const dpr = window.devicePixelRatio || 1;
    this.#canvas.width = Math.round(this.#canvas.clientWidth * dpr);
    this.#canvas.height = Math.round(this.#canvas.clientHeight * dpr);
    this.#ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.#dirty = true;
  }

  #paint() {
    this.#dirty = false;
    const ctx = this.#ctx;
    const width = this.#canvas.clientWidth;
    const height = this.#canvas.clientHeight;
    const mid = height / 2;
    const barWidth = width / HISTORY;
    const offset = HISTORY - this.#frames.length; // newest frame at the right edge

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.noise;
    ctx.fillRect(0, mid - 0.5, width, 1);

    this.#frames.forEach(([energyDb, , isSpeech, inUtterance], i) => {
      const amplitude = Math.max(1, level(energyDb) * (mid - 6));
      ctx.fillStyle = inUtterance ? COLORS.utterance : isSpeech ? COLORS.onset : COLORS.noise;
      ctx.fillRect((offset + i) * barWidth, mid - amplitude, Math.max(1, barWidth - 1), amplitude * 2);
    });

    const latest = this.#frames.at(-1);
    if (!latest) return;
    const y = level(latest[1]) * (mid - 6);
    ctx.strokeStyle = COLORS.threshold;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, mid - y);
    ctx.lineTo(width, mid - y);
    ctx.moveTo(0, mid + y);
    ctx.lineTo(width, mid + y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function level(db) {
  return Math.min(1, Math.max(0, (db - DB_MIN) / (DB_MAX - DB_MIN)));
}
