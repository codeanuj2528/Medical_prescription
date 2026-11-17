/**
 * WebSocket link to the Python pipeline at /ws/transcribe.
 *
 * - Audio captured before the socket is open (or while reconnecting) is
 *   queued, up to ~10 s, and flushed in order once connected.
 * - If the connection drops while recording (network blip, the serverless
 *   function reaching its max duration, or the platform's proxy dropping it)
 *   it reconnects at once (then with backoff), sends a fresh "start", and
 *   replays the audio the old connection never finished transcribing — so a
 *   drop costs a moment of delay, not lost words.
 * - Server messages are re-emitted as DOM events named after their "type".
 */
const MAX_QUEUED_BYTES = 10 * 32_000; // 10 s of 16 kHz PCM16
const STOP_TIMEOUT_MS = 20_000;

export class StreamClient extends EventTarget {
  #url;
  #ws = null;
  #active = false;
  #startPayload = () => ({});
  #replay = null;
  #connections = 0;
  #queue = [];
  #queuedBytes = 0;
  #retryDelay = 500;
  #reconnectTimer = null;
  #stopWaiter = null;

  constructor(url = defaultUrl()) {
    super();
    this.#url = url;
  }

  /**
   * @param {() => object} startPayload  called on every (re)connect
   * @param {() => {fromMs: number, chunks: Int16Array[]}} [replay]  on reconnects: audio to send again
   */
  open(startPayload, replay = null) {
    this.#active = true;
    this.#startPayload = startPayload;
    this.#replay = replay;
    this.#retryDelay = 0; // first reconnect is immediate
    this.#connect();
  }

  /** @param {Int16Array} pcm */
  sendAudio(pcm) {
    if (this.#ws?.readyState === WebSocket.OPEN) {
      this.#ws.send(pcm.buffer);
      return;
    }
    this.#queue.push(pcm.buffer);
    this.#queuedBytes += pcm.byteLength;
    while (this.#queuedBytes > MAX_QUEUED_BYTES) this.#queuedBytes -= this.#queue.shift().byteLength;
  }

  /** Re-send a "start": the server drops its session state and continues from this offset. */
  restart() {
    this.#sendJson(this.#startPayload());
  }

  retry(chunkId) {
    this.#sendJson({ type: "retry", id: chunkId });
  }

  /** Graceful stop: the server flushes the last utterance and finishes pending transcriptions. */
  async close() {
    this.#active = false;
    clearTimeout(this.#reconnectTimer);
    const ws = this.#ws;
    if (ws?.readyState === WebSocket.OPEN) {
      const stopped = new Promise((resolve) => {
        this.#stopWaiter = resolve;
        setTimeout(resolve, STOP_TIMEOUT_MS);
      });
      ws.send(JSON.stringify({ type: "stop" }));
      await stopped;
    }
    ws?.close();
    this.#ws = null;
    this.#queue = [];
    this.#queuedBytes = 0;
  }

  #connect() {
    const ws = new WebSocket(this.#url);
    ws.binaryType = "arraybuffer";
    this.#ws = ws;

    ws.onopen = () => {
      this.#retryDelay = 0;
      const reconnect = this.#connections++ > 0 && this.#replay !== null;
      // On a reconnect the replay already contains the audio queued while offline.
      const replay = reconnect ? this.#replay() : null;
      ws.send(JSON.stringify(this.#startPayload()));
      const flushedBytes = this.#queuedBytes;
      if (replay) for (const view of replay.chunks) ws.send(view);
      else for (const buffer of this.#queue) ws.send(buffer);
      this.#queue = [];
      this.#queuedBytes = 0;
      this.#emit("connected", { reconnect, flushedBytes, fromMs: replay?.fromMs ?? null });
    };

    ws.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (message.type === "stopped") this.#stopWaiter?.();
      this.#emit(message.type, message);
    };

    ws.onclose = () => {
      if (this.#ws !== ws) return;
      this.#stopWaiter?.();
      this.#emit("disconnected", {});
      if (!this.#active) return;
      this.#emit("reconnecting", { delayMs: this.#retryDelay });
      this.#reconnectTimer = setTimeout(() => this.#connect(), this.#retryDelay);
      this.#retryDelay = Math.min(Math.max(this.#retryDelay * 2, 500), 8_000);
    };
  }

  #sendJson(payload) {
    if (this.#ws?.readyState === WebSocket.OPEN) this.#ws.send(JSON.stringify(payload));
  }

  #emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}

function defaultUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws/transcribe`;
}
