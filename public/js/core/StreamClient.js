/**
 * WebSocket link to the Python pipeline at /ws/transcribe.
 *
 * - Audio captured before the socket is open (or while reconnecting) is
 *   queued, up to ~10 s, and flushed in order once connected.
 * - If the connection drops while recording (network blip, or the serverless
 *   function reaching its max duration) it reconnects with backoff and sends
 *   a fresh "start" with the current chunk offset and transcript tail.
 * - Server messages are re-emitted as DOM events named after their "type".
 */
const MAX_QUEUED_BYTES = 10 * 32_000; // 10 s of 16 kHz PCM16
const STOP_TIMEOUT_MS = 20_000;

export class StreamClient extends EventTarget {
  #url;
  #ws = null;
  #active = false;
  #startPayload = () => ({});
  #queue = [];
  #queuedBytes = 0;
  #retryDelay = 500;
  #reconnectTimer = null;
  #stopWaiter = null;

  constructor(url = defaultUrl()) {
    super();
    this.#url = url;
  }

  /** @param {() => object} startPayload  called on every (re)connect */
  open(startPayload) {
    this.#active = true;
    this.#startPayload = startPayload;
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
      this.#retryDelay = 500;
      ws.send(JSON.stringify(this.#startPayload()));
      const flushedBytes = this.#queuedBytes;
      for (const buffer of this.#queue) ws.send(buffer);
      this.#queue = [];
      this.#queuedBytes = 0;
      this.#emit("connected", { flushedBytes });
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
      this.#retryDelay = Math.min(this.#retryDelay * 2, 8_000);
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
