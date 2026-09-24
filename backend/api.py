import sys, os, time, json, asyncio
from pathlib import Path

# Add project root and src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from data_pipeline.loader import DataStore, flow_columns
from analytics.energy import compute_energy_efficiency
from analytics.cascade_sim import simulate_cascade

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if hasattr(obj, 'isoformat'): return obj.isoformat()
        return super().default(obj)

def sanitize_json(data):
    return json.loads(json.dumps(data, cls=NumpyEncoder))

app = FastAPI(title="Talk To My Train API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _D():
    """Return the DataStore dict."""
    return DataStore.get()

@app.on_event("startup")
async def startup_event():
    try:
        data_path = str(PROJECT_ROOT / "data" / "training dataset")
        DataStore.initialize(data_path)
        print("DataStore initialized.")
        try:
            from agent.graph import build_agent_graph
            app.state.agent = build_agent_graph()
            print("Agent graph built.")
        except Exception as e:
            print(f"Warning: agent graph failed: {e}")
            app.state.agent = None
    except Exception as e:
        print(f"Startup error: {e}")
    # Feature 2: initialise feedback SQLite database
    try:
        from analytics.feedback_store import init_feedback_db
        init_feedback_db(PROJECT_ROOT)
    except Exception as e:
        print(f"Warning: feedback DB init failed: {e}")

@app.get("/api/status")
async def get_status():
    try:
        D = _D()
        flows = D.get("flows", pd.DataFrame())
        stations = D.get("stations", pd.DataFrame())
        hist = D.get("historical_index")
        return sanitize_json({
            "status": "ok",
            "data_loaded": True,
            "stations_count": len(stations),
            "flows_rows": len(flows),
            "historical_index_rows": len(hist) if hist is not None else 0
        })
    except RuntimeError:
        return {"status": "loading", "data_loaded": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stations")
async def get_stations():
    try:
        D = _D()
        stations = D.get("stations", pd.DataFrame())
        flows = D.get("flows", pd.DataFrame())
        centrality = D.get("centrality", {}) or {}
        if stations.empty:
            return []
        daily_mean = {}
        if not flows.empty:
            f = flows.copy()
            f["date"] = pd.to_datetime(f["timestamp"]).dt.date
            cols = flow_columns(flows)
            daily_mean = f.groupby("date")[cols].sum().mean().to_dict()
        result = []
        for _, row in stations.iterrows():
            name = row.get("station_name", "")
            lat = float(row.get("latitude", 0) or row.get("lat", 0) or 0)
            lon = float(row.get("longitude", 0) or row.get("lon", 0) or 0)
            lines = [l.strip() for l in str(row.get("u_bahn_lines", "") or "").split(",") if l.strip()]
            cent = centrality.get(name, {})
            result.append({
                "name": name,
                "short_name": name.replace(" (Berlin)", "").replace("U ", ""),
                "lat": lat,
                "lon": lon,
                "lines": lines,
                "daily_flow": float(daily_mean.get(name, 0)),
                "betweenness": float(cent.get("betweenness", 0)),
                "degree": int(cent.get("degree", 0))
            })
        return sanitize_json(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/network/edges")
async def get_network_edges():
    try:
        D = _D()
        conns = D.get("connections", pd.DataFrame())
        stations = D.get("stations", pd.DataFrame())
        if conns.empty or stations.empty:
            return []
        id_to_info = {}
        for _, row in stations.iterrows():
            sid = row.get("station_id")
            if sid is not None:
                id_to_info[sid] = {
                    "lat": float(row.get("latitude", 0) or row.get("lat", 0) or 0),
                    "lon": float(row.get("longitude", 0) or row.get("lon", 0) or 0),
                    "name": str(row.get("station_name", "")),
                    "lines": [l.strip() for l in str(row.get("u_bahn_lines", "") or "").split(",") if l.strip()]
                }
        edges = []
        for _, row in conns.iterrows():
            s1 = row.get("station_id_1") or row.get("from_station_id")
            s2 = row.get("station_id_2") or row.get("to_station_id")
            if s1 in id_to_info and s2 in id_to_info:
                i1, i2 = id_to_info[s1], id_to_info[s2]
                common = set(i1["lines"]).intersection(set(i2["lines"]))
                line = list(common)[0] if common else (i1["lines"][0] if i1["lines"] else "U1")
                edges.append({
                    "from_lat": i1["lat"],
                    "from_lon": i1["lon"],
                    "to_lat": i2["lat"],
                    "to_lon": i2["lon"],
                    "line": line.strip(),
                    "from_name": i1["name"].replace(" (Berlin)", "").replace("U ", ""),
                    "to_name": i2["name"].replace(" (Berlin)", "").replace("U ", "")
                })
        return sanitize_json(edges)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flows/daily")
async def get_daily_flows():
    try:
        D = _D()
        flows = D.get("flows", pd.DataFrame())
        if flows.empty:
            return {"dates": [], "values": [], "rolling_mean": []}
        f = flows.copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        f["date"] = f["timestamp"].dt.date
        cols = flow_columns(f)
        counts = f.groupby("date")["timestamp"].count()
        valid = counts[counts >= 70].index
        f = f[f["date"].isin(valid)]
        f["network"] = f[cols].sum(axis=1)
        daily = f.groupby("date")["network"].sum().reset_index()
        daily["rolling"] = daily["network"].rolling(7, center=True, min_periods=3).mean()
        return sanitize_json({
            "dates": [str(d) for d in daily["date"]],
            "values": daily["network"].tolist(),
            "rolling_mean": daily["rolling"].fillna(0).tolist()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flows/heatmap")
async def get_flows_heatmap(lines: Optional[str] = None):
    try:
        D = _D()
        flows = D.get("flows", pd.DataFrame())
        stations_df = D.get("stations", pd.DataFrame())
        if flows.empty:
            return {"stations": [], "hours": list(range(24)), "values": [], "max_value": 0, "mode": "empty"}
        f = flows.copy()
        if not pd.api.types.is_datetime64_any_dtype(f["timestamp"]):
            f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce")
        f["hour"] = f["timestamp"].dt.hour
        cols = flow_columns(f)

        requested_lines = [l.strip() for l in lines.split(",") if l.strip()] if lines else []

        # If exactly 1 line is requested, return ALL stations on that line
        if len(requested_lines) == 1 and not stations_df.empty:
            single_line = requested_lines[0]
            line_stations = []
            for _, row in stations_df.iterrows():
                u_lines = [l.strip() for l in str(row.get("u_bahn_lines", "") or "").split(",") if l.strip()]
                if single_line in u_lines:
                    st_name = row.get("station_name")
                    if st_name in cols:
                        line_stations.append(st_name)
            target_cols = line_stations
            mode = f"line_{single_line}"
        elif len(requested_lines) > 1 and not stations_df.empty and len(requested_lines) < 9:
            # Multiple lines selected (subset of all lines): Top 20 across these lines
            matching_stations = []
            for _, row in stations_df.iterrows():
                u_lines = [l.strip() for l in str(row.get("u_bahn_lines", "") or "").split(",") if l.strip()]
                if any(rl in u_lines for rl in requested_lines):
                    st_name = row.get("station_name")
                    if st_name in cols:
                        matching_stations.append(st_name)
            totals = f[matching_stations].sum() if matching_stations else pd.Series(dtype=float)
            target_cols = totals.nlargest(20).index.tolist()
            mode = "top_20"
        else:
            # All lines or none specified: Top 20 overall
            totals = f[cols].sum()
            target_cols = totals.nlargest(20).index.tolist()
            mode = "top_20"

        if not target_cols:
            return {"stations": [], "hours": list(range(24)), "values": [], "max_value": 0, "mode": mode}

        hourly_avg = f.groupby("hour")[target_cols].mean()
        short = [s.replace(" (Berlin)", "").replace("U ", "")[:28] for s in target_cols]
        values = []
        for s in target_cols:
            row_vals = []
            for h in range(24):
                val = float(hourly_avg.loc[h, s]) if (h in hourly_avg.index and s in hourly_avg.columns) else 0.0
                row_vals.append(round(val, 1))
            values.append(row_vals)
        max_val = float(np.nanmax(values)) if values else 1000.0
        return sanitize_json({
            "stations": short,
            "full_names": target_cols,
            "hours": list(range(24)),
            "values": values,
            "max_value": max_val,
            "mode": mode,
            "count": len(target_cols)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flows/station/{station_name}")
async def get_station_flows(station_name: str):
    try:
        from urllib.parse import unquote
        station_name = unquote(station_name)
        D = _D()
        flows = D.get("flows", pd.DataFrame())
        if flows.empty or station_name not in flows.columns:
            raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")
        last = flows[["timestamp", station_name]].tail(30 * 96).copy()
        last["timestamp"] = pd.to_datetime(last["timestamp"])
        return sanitize_json({
            "name": station_name,
            "timestamps": last["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "values": last[station_name].tolist(),
            "baseline_mean": float(flows[station_name].mean()),
            "baseline_std": float(flows[station_name].std())
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/closures")
async def get_closures():
    try:
        D = _D()
        closures = D.get("closures", pd.DataFrame())
        if closures.empty:
            return []
        import re
        result = []
        for i, row in closures.iterrows():
            when_dt = pd.to_datetime(row.get("when"), errors="coerce")
            end_dt = pd.to_datetime(row.get("end"), errors="coerce")
            if pd.isna(when_dt):
                continue
            if pd.isna(end_dt):
                end_dt = when_dt + pd.Timedelta(hours=2)
            desc = str(row.get("description", ""))
            lm = re.search(r"Line (U\d)", desc)
            line = lm.group(1) if lm else str(row.get("_line", ""))
            result.append({
                "id": int(i),
                "when": when_dt.isoformat(),
                "end": end_dt.isoformat(),
                "description": desc[:100],
                "line": line,
                "closure_type": str(row.get("closure_type", "line_suspension")),
                "duration_hours": round((end_dt - when_dt).total_seconds() / 3600, 2)
            })
        return sanitize_json(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/energy")
async def get_energy():
    try:
        D = _D()
        energy = D.get("energy", pd.DataFrame())
        flows = D.get("flows", pd.DataFrame())
        stations = D.get("stations", pd.DataFrame())
        if energy.empty or flows.empty or stations.empty:
            return []
        return sanitize_json(compute_energy_efficiency(energy, flows, stations))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/centrality")
async def get_centrality():
    try:
        D = _D()
        centrality = D.get("centrality", {}) or {}
        flows = D.get("flows", pd.DataFrame())
        if not centrality:
            return []
        daily_mean = {}
        if not flows.empty:
            f = flows.copy()
            f["date"] = pd.to_datetime(f["timestamp"]).dt.date
            cols = flow_columns(flows)
            daily_mean = f.groupby("date")[cols].sum().mean().to_dict()
        cents = [{
            "station": s,
            "short_name": s.replace(" (Berlin)", "").replace("U ", ""),
            "betweenness": float(d.get("betweenness", 0)),
            "degree": int(d.get("degree", 0)),
            "daily_flow": float(daily_mean.get(s, 0))
        } for s, d in centrality.items()]
        cents.sort(key=lambda x: x["betweenness"], reverse=True)
        for i, c in enumerate(cents[:15]):
            c["rank"] = i + 1
        return sanitize_json(cents[:15])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CascadeRequest(BaseModel):
    closed_stations: List[str]
    timestamp: str
    closed_segment_from: Optional[str] = None
    closed_segment_to: Optional[str] = None

@app.post("/api/cascade")
async def api_cascade(req: CascadeRequest):
    try:
        D = _D()
        G = D.get("graph")
        flows = D.get("flows", pd.DataFrame())
        stn_df = D.get("stations", pd.DataFrame())
        if G is None or flows.empty:
            raise HTTPException(status_code=503, detail="Graph/flows not ready")
        seg = (req.closed_segment_from, req.closed_segment_to) if req.closed_segment_from and req.closed_segment_to else None
        results = simulate_cascade(
            G=G,
            closed_stations=req.closed_stations,
            closed_segment=seg,
            flows=flows,
            timestamp=req.timestamp,
            stations_df=stn_df
        )
        at_risk = [r for r in results if r.get("at_risk")]
        return sanitize_json({
            "all_stations": results,
            "at_risk_stations": at_risk[:10],
            "max_overflow_ratio": max((r.get("overflow_ratio", 1) for r in results), default=1)
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Feature 2: User Feedback Endpoints
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    response_id: str
    rating: int                      # +1 = helpful, -1 = not helpful
    feedback_text: Optional[str] = ""
    query_text: Optional[str] = ""
    response_text: Optional[str] = ""
    query_type: Optional[str] = ""

class OutcomeRequest(BaseModel):
    response_id: str
    outcome_text: str
    recorded_by: Optional[str] = ""

@app.post("/api/feedback")
async def post_feedback(req: FeedbackRequest):
    """Store operator rating (+1/-1) for a given response_id."""
    try:
        from analytics.feedback_store import store_feedback
        if req.rating not in (-1, 1):
            raise HTTPException(status_code=400, detail="rating must be +1 or -1")
        feedback_id = store_feedback(
            response_id=req.response_id,
            rating=req.rating,
            feedback_text=req.feedback_text or "",
            query_text=req.query_text or "",
            response_text=req.response_text or "",
            query_type=req.query_type or "",
        )
        return {"status": "stored", "feedback_id": feedback_id}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback/outcome")
async def post_outcome(req: OutcomeRequest):
    """Record what operationally happened after a recommendation (separate from rating)."""
    try:
        from analytics.feedback_store import store_actual_outcome
        outcome_id = store_actual_outcome(
            response_id=req.response_id,
            outcome_text=req.outcome_text,
            recorded_by=req.recorded_by or "",
        )
        return {"status": "stored", "outcome_id": outcome_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feedback/summary")
async def get_feedback_stats():
    """Return aggregate feedback statistics for analysis."""
    try:
        from analytics.feedback_store import get_feedback_summary
        return sanitize_json(get_feedback_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            question = data.get("text", "").strip()
            if not question:
                continue
            
            t0 = time.time()
            
            from agent.nodes import (
                intent_extractor_node, fingerprinter_node,
                tool_planner_node, tool_executor_node,
                synthesizer_node, response_formatter_node
            )
            
            try:
                from langchain_core.messages import HumanMessage
            except ImportError:
                class HumanMessage:
                    def __init__(self, content): self.content = content
            
            state = {"messages": [HumanMessage(content=question)]}
            
            # Steps 1-3: intent + fingerprint + plan (fast — keyword-first, no LLM for clear queries)
            await ws.send_json({"type": "thinking", "step": "Classifying query and building HCADE fingerprint...", "icon": "brain", "index": 0})
            
            # Run steps 1,2,3 sequentially but they're now FAST (no LLM if keyword matches)
            update = await loop.run_in_executor(None, intent_extractor_node, state)
            state.update(update)
            update = await loop.run_in_executor(None, fingerprinter_node, state)
            state.update(update)
            update = await loop.run_in_executor(None, tool_planner_node, state)
            state.update(update)
            
            qt   = state.get("query_type", "GENERAL")
            plan = state.get("tool_results", {}).get("_plan", [])
            fp   = state.get("fingerprint", {})
            await ws.send_json({"type": "thinking_done", "index": 0, "detail": f"Type: {qt} | Tools: {', '.join(plan[:3])} | Hour: {fp.get('hour_bucket','?')}"})
            
            # Step 4: Tool executor (parallelized internally via thread pool)
            await ws.send_json({"type": "thinking", "step": f"Running {len(plan)} analytics tools in parallel against dataset...", "icon": "execute", "index": 1})
            update = await loop.run_in_executor(None, tool_executor_node, state)
            state.update(update)
            await ws.send_json({"type": "thinking_done", "index": 1, "detail": "Dataset analysis complete."})
            
            # Step 5: Synthesizer + HCADE recommender (pure Python, fast)
            await ws.send_json({"type": "thinking", "step": "Searching historical analogues via HCADE...", "icon": "history", "index": 2})
            update = await loop.run_in_executor(None, synthesizer_node, state)
            state.update(update)
            rec = state.get("recommendation", {})
            await ws.send_json({"type": "thinking_done", "index": 2, "detail": f"{rec.get('num_historical_matches', 0)} analogues found. Confidence: {rec.get('confidence', 'N/A')}"})
            
            # Step 6: Response formatter (LLM call — main latency here)
            await ws.send_json({"type": "thinking", "step": "Generating evidence-grounded response with gpt-5.6-luna...", "icon": "llm", "index": 3})
            update = await loop.run_in_executor(None, response_formatter_node, state)
            state.update(update)
            await ws.send_json({"type": "thinking_done", "index": 3, "detail": "Response ready."})
            
            response_text = state.get("final_response", "No response generated.")
            latency = time.time() - t0
            
            # Stream response word by word at ChatGPT speed
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                await ws.send_json({"type": "chunk", "text": chunk})
                await asyncio.sleep(0.005)
            
            await ws.send_json({
                "type":        "done",
                "latency":     round(latency, 2),
                "response_id": state.get("response_id", ""),
                "query_type":  state.get("query_type", "GENERAL"),
            })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.api:app', host='0.0.0.0', port=8000, reload=True, app_dir=str(PROJECT_ROOT))
