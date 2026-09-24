// Client for the operator API (docs/api_reference.md). Relative URLs: same origin in production (/app served by the API), Vite proxy in dev.
const BASE = "/api/v1";

async function call(path, opts = {}) {
  const res = await fetch(BASE + path, { headers: { "content-type": "application/json" }, ...opts });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const d = data?.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `HTTP ${res.status}`);
  }
  return data;
}
const post = (path, body) => call(path, { method: "POST", body: JSON.stringify(body) });
const q = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== "")).toString();

/** POST /chat/stream and read the Server-Sent Events: handlers {choice, step, answer, error}. Resolves when the stream ends. */
async function chatStream(body, handlers) {
  const res = await fetch(BASE + "/chat/stream", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const ev = /^event: (.*)$/m.exec(frame)?.[1];
      const data = /^data: (.*)$/m.exec(frame)?.[1];
      if (ev && data && handlers[ev]) handlers[ev](JSON.parse(data));
    }
  }
}

export const api = {
  // analytics pages
  status: () => call("/status"),
  stations: () => call("/stations"),
  edges: () => call("/network/edges"),
  flowsDaily: () => call("/flows/daily"),
  hourlyProfile: () => call("/flows/hourly-profile"),
  flowsHeatmap: (lines) => call(`/flows/heatmap?${q({ lines })}`),
  flowsStation: (name) => call(`/flows/station/${encodeURIComponent(name)}`),
  closures: () => call("/closures"),
  energy: () => call("/energy"),
  centrality: (n = 15) => call(`/centrality?n=${n}`),
  cascade: (body) => post("/cascade", body),
  // operator desk (replay)
  topology: () => call("/ops/topology"),
  timeline: () => call("/ops/timeline"),
  snapshot: (at) => call(`/ops/snapshot?${q({ at })}`),
  series: (date) => call(`/ops/series?${q({ date })}`),
  // chat
  chat: (message, operator_id, session_id, opts = {}) => post("/chat", { message, operator_id, session_id, ...opts }),
  chatStream,
  conversations: (operator_id) => call(`/conversations?${q({ operator_id, limit: 40 })}`),
  conversation: (session_id) => call(`/conversations/${encodeURIComponent(session_id)}`),
  // feedback loop
  score: (turnId, operator_id, score, comment) => post(`/turns/${turnId}/score`, { operator_id, score, comment }),
  action: (turnId, operator_id, body) => post(`/turns/${turnId}/action`, { operator_id, ...body }),
  pending: (operator_id) => call(`/feedback/pending?${q({ operator_id, limit: 20 })}`),
  feedbackStats: (operator_id) => call(`/feedback/stats?${q({ operator_id })}`),
  patchFeedback: (id, body) => call(`/feedback/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  precedents: (question, category) => call(`/precedents?${q({ question, category })}`),
  health: () => call("/health"),
};
