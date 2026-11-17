/** Plain-text prescription built from the summary, for pasting into an EMR. */

export function describeMedication(med) {
  const head = [med.name, med.dosage].filter(Boolean).join(" ");
  const schedule = [med.frequency, med.duration && `× ${med.duration}`].filter(Boolean).join(" ");
  return [head, schedule && `— ${schedule}`, med.instructions && `(${med.instructions})`].filter(Boolean).join(" ");
}

export function hasContent(summary) {
  return Object.values(summary).some((value) =>
    Array.isArray(value) ? value.length > 0 : typeof value === "string" ? value.trim() !== "" : false,
  );
}

export function formatPrescription(summary, date = new Date()) {
  const { patient } = summary;
  const lines = [
    `PRESCRIPTION — ${date.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" })}`,
    `Patient: ${patient.name || "—"} | Age: ${patient.age || "—"} | Gender: ${patient.gender || "—"}`,
  ];
  const list = (title, items) => items.length && lines.push("", `${title}:`, ...items.map((item) => `- ${item}`));
  const field = (title, value) => value.trim() && lines.push("", `${title}: ${value.trim()}`);

  list("Chief complaints", summary.chief_complaints);
  list("Symptoms", summary.symptoms);
  field("Clinical impression", summary.clinical_impression);
  if (summary.medications.length) {
    lines.push("", "Rx");
    summary.medications.forEach((med, i) => lines.push(`${i + 1}. ${describeMedication(med)}`));
  }
  list("Examinations & tests", summary.examinations);
  list("Allergies", summary.allergies);
  list("Past medical history", summary.past_medical_history);
  list("Follow-up & advice", summary.follow_up);
  field("Clinical summary", summary.clinical_summary);
  return lines.join("\n");
}
