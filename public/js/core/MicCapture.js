/**
 * Microphone → 16 kHz mono int16 PCM chunks (100 ms each).
 *
 * The AudioContext is opened at 16 kHz so the browser's own high-quality
 * resampler does the work; browsers that refuse (older Firefox) fall back to
 * the device rate and the worklet resamples instead.
 */
const SAMPLE_RATE = 16_000;
// 100 ms per WebSocket message: Vercel's WebSocket proxy drops connections far more
// often at 50 messages/s than at 10; the cost is ≤ 100 ms later pause detection.
const CHUNK_SAMPLES = 1600;
const WORKLET_URL = "/worklets/pcm-capture.worklet.js";

// Browser "call" processing (noise suppression, echo cancellation) is tuned for
// video calls and swallows soft syllables and word endings, which hurts
// recognition. Our VAD already gates noise and Whisper is robust to it, so we
// keep the voice raw and only let the browser even out the level.
const CONSTRAINTS = {
  channelCount: 1,
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: true,
};

export class MicCapture {
  #stream = null;
  #context = null;
  #source = null;
  #node = null;

  /** @param {(pcm: Int16Array) => void} onChunk  @param {() => void} onEnded */
  async start(onChunk, onEnded) {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      throw new Error("This browser cannot capture audio (needs getUserMedia and AudioWorklet).");
    }
    try {
      this.#stream = await navigator.mediaDevices.getUserMedia({ audio: CONSTRAINTS, video: false });
    } catch (error) {
      throw new Error(describeMicError(error));
    }
    try {
      await this.#connect(onChunk, SAMPLE_RATE);
    } catch {
      await this.#teardown();
      await this.#connect(onChunk);
    }
    this.#stream.getAudioTracks().forEach((track) => track.addEventListener("ended", onEnded));
  }

  async stop() {
    await this.#teardown();
    this.#stream?.getTracks().forEach((track) => track.stop());
    this.#stream = null;
  }

  async #connect(onChunk, sampleRate) {
    const context = new AudioContext({ latencyHint: "interactive", ...(sampleRate ? { sampleRate } : {}) });
    this.#context = context;
    await context.audioWorklet.addModule(WORKLET_URL);
    this.#source = context.createMediaStreamSource(this.#stream);
    this.#node = new AudioWorkletNode(context, "pcm-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      channelCountMode: "explicit",
      processorOptions: { targetSampleRate: SAMPLE_RATE, chunkSamples: CHUNK_SAMPLES },
    });
    this.#node.port.onmessage = (event) => onChunk(event.data);
    // The worklet outputs silence; connecting it keeps every browser pulling audio through it.
    this.#source.connect(this.#node).connect(context.destination);
    if (context.state === "suspended") await context.resume();
  }

  async #teardown() {
    if (this.#node) {
      this.#node.port.onmessage = null;
      this.#node.disconnect();
    }
    this.#source?.disconnect();
    if (this.#context && this.#context.state !== "closed") await this.#context.close().catch(() => {});
    this.#node = this.#source = this.#context = null;
  }
}

function describeMicError(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "SecurityError":
      // Chrome says "Permission denied by system" when the OS (not the site) blocks the browser.
      return /system/i.test(error.message ?? "")
        ? "Your operating system is blocking the browser from the microphone. On macOS: System Settings → Privacy & Security → Microphone → turn on your browser, then restart it."
        : "Microphone access is blocked for this page. Allow it from the site-settings icon in the address bar, then reload. In-app/embedded browsers may not allow microphones at all — open this page in Chrome, Edge or Safari.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone was found.";
    case "NotReadableError":
    case "AbortError":
      return "The microphone is busy or unavailable (used by another app?).";
    default:
      return error?.message || "Could not start the microphone.";
  }
}
