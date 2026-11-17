import { escapeHtml, formatClock, icon } from "../util.js";

/**
 * Keyed list of transcript chunks. Chunk ids increase in speech order, so a
 * new id is appended and a known id is updated in place — results that come
 * back out of order (parallel transcription) still land in the right slot.
 */
export class TranscriptView {
  #root;
  #list;
  #placeholder;
  #placeholderText;
  #rows = new Map();

  constructor(root, onRetry) {
    this.#root = root;
    this.#list = root.querySelector("#chunk-list");
    this.#placeholder = root.querySelector(".placeholder");
    this.#placeholderText = root.querySelector("#placeholder-text");
    this.#list.addEventListener("click", (event) => {
      const button = event.target.closest("[data-retry]");
      if (button) onRetry(Number(button.dataset.retry));
    });
  }

  upsert(chunk) {
    let row = this.#rows.get(chunk.id);
    if (!row) {
      row = document.createElement("li");
      this.#rows.set(chunk.id, row);
      this.#list.append(row);
    }
    // Utterances the recogniser found empty (noise the VAD let through) are hidden.
    row.hidden = chunk.status === "done" && !chunk.text;
    row.className = `chunk ${chunk.status}`;
    row.innerHTML = `
      <time>${formatClock(chunk.startMs)}</time>
      <div class="body">${this.#body(chunk)}</div>
      ${chunk.latencyMs != null ? `<span class="latency" title="Last spoken word → text on screen">${chunk.latencyMs} ms</span>` : ""}`;
    this.#refreshPlaceholder();
    this.#scrollToEnd();
  }

  remove(id) {
    this.#rows.get(id)?.remove();
    this.#rows.delete(id);
    this.#refreshPlaceholder();
  }

  clear() {
    this.#rows.clear();
    this.#list.replaceChildren();
    this.#refreshPlaceholder();
  }

  setListening(listening) {
    this.#placeholderText.textContent = listening
      ? "Listening… speak naturally — every pause sends a chunk for transcription."
      : "Ready — click Start Mic and speak naturally. VAD will auto-chunk your speech.";
  }

  #body(chunk) {
    const typing = `<span class="typing"><i></i><i></i><i></i></span>`;
    if (chunk.status === "pending") return `Listening ${typing}`;
    if (chunk.status === "partial") {
      return chunk.stable
        ? escapeHtml(chunk.text) // final-quality text committed at the pause
        : `<span class="live-text">${escapeHtml(chunk.text)}</span> ${typing}`;
    }
    if (chunk.status === "failed") {
      const retry = chunk.retryable
        ? `<button class="link-btn" type="button" data-retry="${chunk.id}">${icon("rotate-cw")}Retry</button>`
        : "";
      return `${escapeHtml(chunk.error || "Transcription failed.")}${retry}`;
    }
    const badge = chunk.language ? `<span class="lang-badge">${escapeHtml(chunk.language.toUpperCase())}</span>` : "";
    return `${badge}${escapeHtml(chunk.text)}`;
  }

  #refreshPlaceholder() {
    const anyVisible = [...this.#rows.values()].some((row) => !row.hidden);
    this.#placeholder.hidden = anyVisible;
  }

  #scrollToEnd() {
    const el = this.#root;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 140) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }
}
