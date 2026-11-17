import { escapeHtml, icon } from "../util.js";

/** Every summary section after "Clinical Summary", described once. */
const SECTIONS = [
  { field: "chief_complaints", title: "Chief Complaints", icon: "messages-square", kind: "bullets", empty: "No complaints extracted yet" },
  { field: "symptoms", title: "Symptoms", icon: "thermometer", kind: "chips", empty: "No symptoms extracted yet" },
  { field: "clinical_impression", title: "Clinical Impression", icon: "stethoscope", kind: "text", empty: "No diagnosis inferred" },
  { field: "medications", title: "Medications", icon: "pill", kind: "meds", empty: "No medications extracted yet" },
  { field: "examinations", title: "Examinations & Tests", icon: "microscope", kind: "bullets", empty: "No examinations extracted yet" },
  { field: "allergies", title: "Allergies", icon: "triangle-alert", kind: "warn-chips", empty: "No allergies extracted yet" },
  { field: "past_medical_history", title: "Past Medical History", icon: "history", kind: "bullets", empty: "No past medical history extracted" },
  { field: "follow_up", title: "History & Follow-up", icon: "calendar-days", kind: "bullets", empty: "No follow-up extracted yet" },
];

const RENDERERS = {
  bullets: (items) => `<ul class="bullets">${items.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`,
  chips: (items) => `<ul class="chips">${items.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`,
  "warn-chips": (items) => `<ul class="chips warn">${items.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`,
  text: (value) => `<p class="impression">${escapeHtml(value)}</p>`,
  meds: (meds) =>
    `<ol class="meds">${meds
      .map(
        (med) => `<li class="med"><span class="med-icon">${icon("pill")}</span><div>
          <p class="med-name">${escapeHtml(med.name || "Unnamed medicine")}</p>
          <p class="med-detail">${escapeHtml([med.dosage, med.frequency, med.duration].filter(Boolean).join(" · "))}</p>
          ${med.instructions ? `<p class="med-note">${escapeHtml(med.instructions)}</p>` : ""}
        </div></li>`,
      )
      .join("")}</ol>`,
};

const isEmpty = (value) => (Array.isArray(value) ? value.length === 0 : !String(value || "").trim());
const SKELETON = `<div class="skeleton" aria-hidden="true"><i></i><i></i></div>`;

export class SummaryView {
  #el;
  #bodies = new Map();

  constructor(doc, onEditSummary) {
    this.#el = {
      name: doc.querySelector("#patient-name"),
      age: doc.querySelector("#patient-age"),
      gender: doc.querySelector("#patient-gender"),
      meta: doc.querySelector("#summary-meta"),
      error: doc.querySelector("#summary-error"),
      partial: doc.querySelector("#summary-partial"),
      text: doc.querySelector("#clinical-summary"),
      sections: doc.querySelector("#summary-sections"),
      copy: doc.querySelector("#btn-copy"),
      clear: doc.querySelector("#btn-summary-clear"),
    };
    this.#el.text.addEventListener("input", () => onEditSummary(this.#el.text.value));
    for (const section of SECTIONS) this.#mountSection(section);
  }

  /** @param {{status: string, summary: object|null, error: string|null, meta: object|null}} state */
  render({ status, summary, error, meta }) {
    const loading = status === "loading";
    const patient = summary?.patient ?? {};
    this.#el.name.textContent = patient.name || "—";
    this.#el.age.textContent = patient.age || "—";
    this.#el.gender.textContent = patient.gender || "—";

    this.#el.meta.hidden = !meta;
    if (meta) this.#el.meta.textContent = `Generated in ${meta.latencyMs} ms · ${meta.model} · 2 parallel extractors`;
    this.#el.error.hidden = !error;
    this.#el.error.textContent = error ?? "";
    this.#el.partial.hidden = !meta?.partial;

    if (this.#el.text.value !== (summary?.clinical_summary ?? "")) this.#el.text.value = summary?.clinical_summary ?? "";
    this.#el.text.readOnly = !summary;

    for (const section of SECTIONS) {
      const value = summary?.[section.field];
      this.#bodies.get(section.field).innerHTML = loading
        ? SKELETON
        : isEmpty(value)
          ? `<p class="empty">${section.empty}</p>`
          : RENDERERS[section.kind](value);
    }

    this.#el.copy.disabled = !summary || loading;
    this.#el.clear.disabled = status === "idle";
  }

  #mountSection({ field, title, icon: iconName }) {
    const section = document.createElement("section");
    section.className = "sum-section";
    section.innerHTML = `<div class="sec-head">${icon(iconName)}<h3>${title}</h3></div><div class="sec-body"></div>`;
    this.#el.sections.append(section);
    this.#bodies.set(field, section.querySelector(".sec-body"));
  }
}

