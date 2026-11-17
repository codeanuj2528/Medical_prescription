import { summarize } from "./core/api.js";
import { MicCapture } from "./core/MicCapture.js";
import { formatPrescription, hasContent } from "./core/prescription.js";
import { Recording } from "./core/Recording.js";
import { StreamClient } from "./core/StreamClient.js";
import { SummaryView } from "./ui/SummaryView.js";
import { Toaster } from "./ui/Toaster.js";
import { TranscriptView } from "./ui/TranscriptView.js";
import { Waveform } from "./ui/Waveform.js";
import { copyText, countWords, downloadBlob, formatClock, icon, timestampSlug } from "./util.js";

const MIN_TRANSCRIPT_CHARS = 20;
const CONTEXT_CHARS = 240;
const BYTES_PER_MS = 32; // 16 kHz × 2 bytes
const MAX_REPLAY_MS = 30_000; // never resend more than this after a reconnect
const LANGUAGE_KEY = "mediscribe.language";

/**
 * Browser-side controller. It owns UI state and wiring only:
 *   mic → WebSocket (Python does VAD + segmentation + Whisper) → views.
 */
export class ScribeApp {
  #mic = new MicCapture();
  #recording = new Recording();
  #stream = null;
  #status = "idle"; // idle | starting | listening | stopping
  #chunks = new Map(); // id → { id, status, text, startMs, endMs, latencyMs, error, retryable, connection }
  #lastChunkId = 0; // ids keep increasing across reconnects and clears
  #ignoreUpTo = 0; // chunk ids at or below this belong to a cleared transcript
  #connection = 0; // server sessions so far; failed chunks are retryable only within their own
  #timelineBaseMs = 0; // recording position where the current server session's audio starts
  #clock = null;
  #summary = { status: "idle", summary: null, error: null, meta: null };
  #summaryRequest = null;

  constructor(doc) {
    this.el = {
      start: doc.querySelector("#btn-start"),
      stop: doc.querySelector("#btn-stop"),
      clear: doc.querySelector("#btn-clear"),
      save: doc.querySelector("#btn-save"),
      process: doc.querySelector("#btn-process"),
      copy: doc.querySelector("#btn-copy"),
      summaryClear: doc.querySelector("#btn-summary-clear"),
      chip: doc.querySelector("#vad-chip"),
      sessionError: doc.querySelector("#session-error"),
      language: doc.querySelector("#language"),
      words: doc.querySelector("#stat-words"),
      chunks: doc.querySelector("#stat-chunks"),
      latency: doc.querySelector("#stat-latency"),
      sent: doc.querySelector("#stat-sent"),
      time: doc.querySelector("#stat-time"),
    };
    this.transcript = new TranscriptView(doc.querySelector("#transcript"), (id) => this.retry(id));
    this.waveform = new Waveform(doc.querySelector("#waveform"));
    this.summaryView = new SummaryView(doc, (text) => this.#editClinicalSummary(text));
    this.toast = new Toaster(doc.querySelector("#toasts"));
    this.#bindControls();
    this.#watchMicPermission();
    this.#restoreLanguage();
    this.#render();
  }

  /** A new language applies from the next utterance: the server session restarts with it. */
  #onLanguageChange() {
    try {
      localStorage.setItem(LANGUAGE_KEY, this.el.language.value);
    } catch {
      // storage unavailable (private mode) — the choice just isn't remembered
    }
    if (this.#stream) {
      this.#stream.restart();
      this.toast.show(`Language: ${this.el.language.selectedOptions[0].text}`, "info");
    }
  }

  #restoreLanguage() {
    try {
      const saved = localStorage.getItem(LANGUAGE_KEY);
      if (saved && [...this.el.language.options].some((o) => o.value === saved)) this.el.language.value = saved;
    } catch {
      // storage unavailable — keep the default
    }
  }

  /** Clears the "blocked" error as soon as the user allows the microphone in site settings. */
  #watchMicPermission() {
    navigator.permissions
      ?.query({ name: "microphone" })
      .then((status) => {
        status.onchange = () => {
          if (status.state !== "denied") this.#showSessionError(null);
        };
      })
      .catch(() => {}); // Permissions API not supported for "microphone" in this browser
  }

  // ── Recording lifecycle ────────────────────────────────────────────────

  async start() {
    if (this.#status !== "idle") return;
    this.#showSessionError(null);
    this.#setStatus("starting");

    const stream = new StreamClient();
    this.#wireStream(stream);
    this.#stream = stream;
    stream.open(
      () => ({
        type: "start",
        chunkOffset: this.#lastChunkId,
        context: this.#transcriptText().slice(-CONTEXT_CHARS),
        language: this.el.language.value,
      }),
      () => this.#prepareReplay(),
    );

    try {
      await this.#mic.start(
        (pcm) => this.#onAudio(pcm),
        () => this.stop("The microphone was disconnected."),
      );
    } catch (error) {
      this.#stream = null;
      await stream.close();
      this.#setStatus("idle");
      this.#showSessionError(error.message);
      return;
    }
    this.#setStatus("listening");
    this.#clock = setInterval(() => this.#renderStats(), 500);
  }

  /** Stops the mic; the server flushes the last utterance and finishes pending transcriptions. */
  async stop(errorMessage = null) {
    if (this.#status !== "listening") return;
    this.#setStatus("stopping");
    clearInterval(this.#clock);
    await this.#mic.stop();
    await this.#stream?.close();
    this.#stream = null;
    this.#failPendingChunks("Connection closed before this chunk was transcribed.");
    this.#setStatus("idle");
    if (errorMessage) this.#showSessionError(errorMessage);
  }

  clear() {
    this.#ignoreUpTo = this.#lastChunkId;
    this.#chunks.clear();
    this.#recording.clear();
    this.transcript.clear();
    this.waveform.clear();
    this.#showSessionError(null);
    if (this.#stream) this.#stream.restart(); // fresh server session; ids continue after #lastChunkId
    this.#timelineBaseMs = 0;
    this.#render();
  }

  retry(id) {
    const chunk = this.#chunks.get(id);
    if (!chunk?.retryable || !this.#stream) return;
    this.#stream.retry(id);
  }

  saveRecording() {
    if (this.#recording.isEmpty) return;
    downloadBlob(this.#recording.toWavBlob(), `consultation-${timestampSlug()}.wav`);
  }

  // ── AI summary ─────────────────────────────────────────────────────────

  async processTranscript() {
    const transcript = this.#transcriptText();
    if (transcript.length < MIN_TRANSCRIPT_CHARS) {
      this.toast.show("Nothing to analyse yet — start the mic and describe the consultation.", "info");
      return;
    }
    const pending = [...this.#chunks.values()].filter((c) => c.status === "pending" || c.status === "partial").length;
    if (pending) this.toast.show(`${pending} chunk(s) still transcribing — using the text so far.`, "info");

    this.#summaryRequest?.abort();
    const request = new AbortController();
    this.#summaryRequest = request;
    this.#setSummary({ ...this.#summary, status: "loading", error: null });

    try {
      const result = await summarize(transcript, request.signal);
      this.#setSummary({
        status: "ready",
        summary: result.summary,
        error: null,
        meta: { model: result.model, latencyMs: result.latency_ms, partial: result.partial },
      });
      this.toast.show("Summary ready", "success");
    } catch (error) {
      if (error.name !== "AbortError") this.#setSummary({ ...this.#summary, status: "error", error: error.message });
    }
  }

  clearSummary() {
    this.#summaryRequest?.abort();
    this.#setSummary({ status: "idle", summary: null, error: null, meta: null });
  }

  async copyPrescription() {
    const { summary } = this.#summary;
    if (!summary || !hasContent(summary)) {
      this.toast.show("The summary is empty — nothing to copy.", "info");
      return;
    }
    const copied = await copyText(formatPrescription(summary));
    this.toast.show(
      copied ? "Prescription copied to clipboard" : "Clipboard access was blocked by the browser.",
      copied ? "success" : "error",
    );
  }

  // ── Stream events ──────────────────────────────────────────────────────

  #wireStream(stream) {
    stream.addEventListener("connected", (e) => {
      this.#connection++;
      this.#timelineBaseMs = e.detail.reconnect
        ? e.detail.fromMs
        : Math.max(0, this.#recording.durationMs - e.detail.flushedBytes / BYTES_PER_MS);
      this.#renderChip(false);
    });
    stream.addEventListener("vad", (e) => {
      this.waveform.push(e.detail.frames);
      this.#renderChip(Boolean(e.detail.frames.at(-1)?.[3]));
    });
    stream.addEventListener("chunk", (e) => this.#onChunk(e.detail));
    stream.addEventListener("error", (e) => this.toast.show(e.detail.message, "error"));
    // A drop while listening is repaired by the reconnect + replay (see #prepareReplay);
    // pending chunks are only failed if the session stops before that happens.
    stream.addEventListener("reconnecting", () => {
      if (this.#status === "listening") this.#renderChip(false, true);
    });
  }

  #onAudio(pcm) {
    this.#recording.append(pcm);
    this.#stream?.sendAudio(pcm);
  }

  #onChunk(message) {
    if (message.id <= this.#ignoreUpTo) return;
    this.#lastChunkId = Math.max(this.#lastChunkId, message.id);

    const chunk = this.#chunks.get(message.id) ?? { id: message.id, connection: this.#connection, text: "", latencyMs: null };
    chunk.status = message.status;
    if (message.startMs != null) chunk.startMs = this.#timelineBaseMs + message.startMs;
    if (message.endMs != null) chunk.endMs = this.#timelineBaseMs + message.endMs;
    if (message.status === "partial") {
      chunk.text = message.text; // live text, replaced by the final
      chunk.stable = Boolean(message.stable); // committed at a pause: shown as normal text
    }
    if (message.status === "done") {
      chunk.text = message.text;
      chunk.latencyMs = message.latencyMs;
      chunk.language = message.language;
    }
    chunk.error = message.status === "failed" ? message.error : null;
    chunk.retryable = message.status === "failed" && chunk.connection === this.#connection;

    this.#chunks.set(chunk.id, chunk);
    this.transcript.upsert(chunk);
    this.#renderStats();
    this.#renderControls();
  }

  /**
   * Called when a dropped connection reconnects: drop the chunks it never
   * finished and return the audio to send again, from the start of the first
   * unfinished utterance (or the end of the last finished one).
   */
  #prepareReplay() {
    const chunks = [...this.#chunks.values()];
    const unfinished = chunks.filter((c) => c.status === "pending" || c.status === "partial");
    const finishedEnds = chunks.filter((c) => c.status === "done" && c.endMs != null).map((c) => c.endMs);
    let fromMs = unfinished.length
      ? Math.min(...unfinished.map((c) => c.startMs ?? this.#timelineBaseMs))
      : Math.max(this.#timelineBaseMs, ...finishedEnds);
    fromMs = Math.max(fromMs, this.#recording.durationMs - MAX_REPLAY_MS, 0);

    // Everything the replay covers is re-transcribed under new ids — including a
    // later utterance that finished before an earlier one did (no duplicates).
    const replayed = chunks.filter((c) => unfinished.includes(c) || (c.startMs ?? -1) >= fromMs);
    for (const chunk of replayed) {
      this.#chunks.delete(chunk.id);
      this.transcript.remove(chunk.id);
    }
    this.#renderStats();
    return { fromMs, chunks: this.#recording.sliceFrom(fromMs) };
  }

  #failPendingChunks(reason) {
    for (const chunk of this.#chunks.values()) {
      if (chunk.status !== "pending" && chunk.status !== "partial") continue;
      Object.assign(chunk, { status: "failed", error: reason, retryable: false });
      this.transcript.upsert(chunk);
    }
  }

  // ── State helpers ──────────────────────────────────────────────────────

  #transcriptText() {
    return [...this.#chunks.values()]
      .filter((c) => c.status === "done" && c.text)
      .sort((a, b) => a.id - b.id)
      .map((c) => c.text)
      .join(" ");
  }

  #setStatus(status) {
    this.#status = status;
    this.transcript.setListening(status === "listening");
    this.#render();
  }

  #setSummary(state) {
    this.#summary = state;
    this.summaryView.render(state);
    this.#renderProcessButton();
  }

  #editClinicalSummary(text) {
    if (this.#summary.summary) this.#summary.summary.clinical_summary = text;
  }

  #showSessionError(message) {
    this.el.sessionError.hidden = !message;
    this.el.sessionError.textContent = message ?? "";
  }

  // ── Rendering ──────────────────────────────────────────────────────────

  #render() {
    this.#renderControls();
    this.#renderStats();
    this.#renderChip(false);
    this.#renderProcessButton();
    this.summaryView.render(this.#summary);
  }

  #renderControls() {
    const { start, stop, clear, save } = this.el;
    const status = this.#status;
    start.disabled = status !== "idle";
    start.classList.toggle("is-live", status === "listening");
    start.innerHTML =
      status === "listening"
        ? `${icon("mic")}<span>Listening</span><span class="live-dot"></span>`
        : status === "starting"
          ? `${icon("loader-circle", "spinner")}<span>Starting…</span>`
          : `${icon("mic")}<span>Start Mic</span>`;
    stop.disabled = status !== "listening";
    clear.disabled = status === "stopping" || (this.#chunks.size === 0 && this.#recording.isEmpty);
    save.disabled = this.#recording.isEmpty || status === "starting";
  }

  #renderStats() {
    const chunks = [...this.#chunks.values()];
    const latencies = chunks.map((c) => c.latencyMs).filter((v) => v != null);
    const speechMs = chunks.reduce((sum, c) => sum + Math.max(0, (c.endMs ?? 0) - (c.startMs ?? 0)), 0);
    const capturedMs = this.#recording.durationMs;

    this.el.words.textContent = String(countWords(this.#transcriptText()));
    this.el.chunks.textContent = String(chunks.length);
    this.el.latency.textContent = latencies.length
      ? `${Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length)} ms`
      : "—";
    this.el.sent.textContent = capturedMs > 0 ? `${Math.min(100, Math.round((speechMs / capturedMs) * 100))}%` : "—";
    this.el.time.textContent = formatClock(capturedMs);
    this.#renderControls();
  }

  #renderChip(inUtterance, reconnecting = false) {
    const live = this.#status === "listening";
    const state = !live ? "off" : reconnecting ? "reconnecting" : inUtterance ? "speech" : "listening";
    if (this.el.chip.dataset.state === state) return;
    this.el.chip.dataset.state = state;
    this.el.chip.querySelector(".label").textContent = {
      off: "Mic off",
      listening: "Listening — silence is not sent",
      speech: "Speech detected — recording chunk",
      reconnecting: "Reconnecting…",
    }[state];
  }

  #renderProcessButton() {
    const loading = this.#summary.status === "loading";
    this.el.process.disabled = loading;
    this.el.process.innerHTML = loading
      ? `${icon("loader-circle", "spinner")}<span>Analysing transcript…</span>`
      : `${icon("zap")}<span>Process Transcript with AI</span>`;
  }

  #bindControls() {
    this.el.start.addEventListener("click", () => this.start());
    this.el.stop.addEventListener("click", () => this.stop());
    this.el.clear.addEventListener("click", () => this.clear());
    this.el.save.addEventListener("click", () => this.saveRecording());
    this.el.process.addEventListener("click", () => this.processTranscript());
    this.el.copy.addEventListener("click", () => this.copyPrescription());
    this.el.summaryClear.addEventListener("click", () => this.clearSummary());
    this.el.language.addEventListener("change", () => this.#onLanguageChange());
    window.addEventListener("pagehide", () => this.#mic.stop());
  }
}
