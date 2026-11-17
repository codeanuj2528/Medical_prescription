/** REST calls to the FastAPI backend. */

export async function summarize(transcript, signal) {
  let response;
  try {
    response = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript }),
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("Network error — check your connection.");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.error?.message ?? `Request failed (HTTP ${response.status})`);
  return body; // { summary, model, latency_ms, partial }
}
