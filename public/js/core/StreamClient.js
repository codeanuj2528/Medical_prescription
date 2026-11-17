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
 * - A connection can also die *silently* (still "open", but nothing gets
 *   through). A heartbeat catches that: a "ping" every 2 s, and if the server
 *   has said nothing at all for 5 s the connection is treated as dropped.
 * - Server messages are re-emitted as DOM events named after their "type".
 */
const MAX_QUEUED_BYTES = 10 * 32_000; // 10 s of 16 kHz PCM16
const STOP_TIMEOUT_MS = 20_000;
const HEARTBEAT_MS = 2_000;
const SILENT_SERVER_MS = 5_000; // the server normally talks ~10×/s (VAD telemetry), and answers every ping

export class StreamClient extends EventTarget {
  #url;
  #ws = null;
  #active = false;
  #stopping = false;
  #sentAudio = false; // nothing to flush on stop if the mic never delivered audio
  #startPayload = () => ({});
  #replay = null;
  #connections = 0;
  #queue = [];
  #queuedBytes = 0;
  #retryDelay = 500;
  #reconnectTimer = null;
  #heartbeat = null;
  #lastHeardAt = 0;
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
    this.#heartbeat = setInterval(() => this.#checkHeartbeat(), HEARTBEAT_MS);
    this.#connect();
  }

  /** @param {Int16Array} pcm */
  sendAudio(pcm) {
    this.#sentAudio = true;
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

  /**
   * Graceful stop: the server flushes the last utterance and finishes pending
   * transcriptions. If the connection drops while it does, it reconnects,
   * replays the unfinished audio and asks again — the last sentence is not lost.
   */
  async close() {
    clearTimeout(this.#reconnectTimer);
    const reachable = this.#connections > 0 || this.#ws?.readyState === WebSocket.CONNECTING;
    if (this.#active && this.#sentAudio && reachable) {
      this.#stopping = true;
      const stopped = new Promise((resolve) => {
        this.#stopWaiter = resolve;
        setTimeout(resolve, STOP_TIMEOUT_MS);
      });
      if (this.#ws?.readyState === WebSocket.OPEN) this.#sendJson({ type: "stop" });
      else if (this.#ws?.readyState !== WebSocket.CONNECTING) this.#connect(); // onopen sends the stop
      await stopped;
    }
    this.#active = false;
    this.#stopping = false;
    this.#stopWaiter = null;
    clearTimeout(this.#reconnectTimer);
    clearInterval(this.#heartbeat);
    const ws = this.#ws;
    this.#ws = null;
    ws?.close();
    this.#queue = [];
    this.#queuedBytes = 0;
  }

  #connect() {
    const ws = new WebSocket(this.#url);
    ws.binaryType = "arraybuffer";
    this.#ws = ws;

    ws.onopen = () => {
      this.#retryDelay = 0;
      this.#lastHeardAt = performance.now();
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
      if (this.#stopping) ws.send(JSON.stringify({ type: "stop" })); // dropped while stopping: ask again
    };

    ws.onmessage = (event) => {
      this.#lastHeardAt = performance.now();
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
      if (this.#ws === ws) this.#dropped();
    };
  }

  /** Every 2 s: ping, and give up on a connection the server has gone silent on. */
  #checkHeartbeat() {
    const ws = this.#ws;
    if (ws?.readyState !== WebSocket.OPEN) return;
    if (performance.now() - this.#lastHeardAt > SILENT_SERVER_MS) {
      ws.onclose = ws.onmessage = null; // a dead socket can take long to report "closed": don't wait
      ws.close();
      this.#dropped();
      return;
    }
    ws.send(JSON.stringify({ type: "ping" }));
  }

  #dropped() {
    this.#emit("disconnected", {});
    if (!this.#active) return;
    this.#emit("reconnecting", { delayMs: this.#retryDelay });
    this.#reconnectTimer = setTimeout(() => this.#connect(), this.#retryDelay);
    this.#retryDelay = Math.min(Math.max(this.#retryDelay * 2, 500), 8_000);
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
