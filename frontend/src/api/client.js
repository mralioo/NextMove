// Use relative path so Vite proxy handles it cleanly, with fallback to direct backend if needed
const BASE = '';

const handleRes = async (res) => {
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json();
};

export const api = {
  status:      () => fetch(`${BASE}/api/status`).then(handleRes),
  stations:    () => fetch(`${BASE}/api/stations`).then(handleRes),
  edges:       () => fetch(`${BASE}/api/network/edges`).then(handleRes),
  flowsDaily:  () => fetch(`${BASE}/api/flows/daily`).then(handleRes),
  flowsHeatmap:(lines) => fetch(`${BASE}/api/flows/heatmap${lines ? `?lines=${encodeURIComponent(lines)}` : ''}`).then(handleRes),
  flowsStation:(name) => fetch(`${BASE}/api/flows/station/${encodeURIComponent(name)}`).then(handleRes),
  closures:    () => fetch(`${BASE}/api/closures`).then(handleRes),
  energy:      () => fetch(`${BASE}/api/energy`).then(handleRes),
  centrality:  () => fetch(`${BASE}/api/centrality`).then(handleRes),
  cascade:     (body) => fetch(`${BASE}/api/cascade`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(handleRes),

  // Feature 2: Operator Feedback
  submitFeedback: (responseId, rating, feedbackText = '', queryText = '', queryType = '') =>
    fetch(`${BASE}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        response_id:   responseId,
        rating,
        feedback_text: feedbackText,
        query_text:    queryText,
        query_type:    queryType,
      })
    }).then(handleRes),

  getFeedbackSummary: () =>
    fetch(`${BASE}/api/feedback/summary`).then(handleRes),
};
