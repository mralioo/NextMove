// Thin client for the operator API (docs/operator_feedback_api.md). Relative URLs: same origin in production (/app served by the API), Vite proxy in dev.
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

export const api = {
  topology: () => call("/ops/topology"),
  timeline: () => call("/ops/timeline"),
  snapshot: (at) => call(`/ops/snapshot?${q({ at })}`),
  series: (date) => call(`/ops/series?${q({ date })}`),
  chat: (message, operator_id, session_id, opts = {}) => post("/chat", { message, operator_id, session_id, ...opts }),
  conversations: (operator_id) => call(`/conversations?${q({ operator_id, limit: 40 })}`),
  conversation: (session_id) => call(`/conversations/${encodeURIComponent(session_id)}`),
  score: (turnId, operator_id, score, comment) => post(`/turns/${turnId}/score`, { operator_id, score, comment }),
  action: (turnId, operator_id, body) => post(`/turns/${turnId}/action`, { operator_id, ...body }),
  pending: (operator_id) => call(`/feedback/pending?${q({ operator_id, limit: 20 })}`),
  patchFeedback: (id, body) => call(`/feedback/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  precedents: (question, category) => call(`/precedents?${q({ question, category })}`),
  health: () => call("/health"),
};
