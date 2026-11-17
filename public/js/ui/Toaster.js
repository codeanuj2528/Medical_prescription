import { escapeHtml, icon } from "../util.js";

const ICONS = { success: "circle-check", error: "circle-alert", info: "info" };
const DISMISS_MS = 3_500;

export class Toaster {
  #root;

  constructor(root) {
    this.#root = root;
  }

  show(message, tone = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${tone}`;
    toast.setAttribute("role", "status");
    toast.innerHTML = `${icon(ICONS[tone])}<span>${escapeHtml(message)}</span>`;
    this.#root.append(toast);
    while (this.#root.children.length > 3) this.#root.firstElementChild.remove();
    setTimeout(() => toast.remove(), DISMISS_MS);
  }
}
