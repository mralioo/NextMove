"""
Talk To My Train — InnoTrans 2026 Hackathon Dashboard
Premium 5-tab Streamlit app. Theme: teal / black / gold.

Run with:
    streamlit run src/dashboard/app.py
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
SRC  = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

matplotlib_backend_set = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib_backend_set = True
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Talk To My Train · InnoTrans 2026",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# THEME  ─  teal / obsidian / gold
# ═══════════════════════════════════════════════════════════════════════════
THEME = {
    "bg":        "#060F1E",
    "surface":   "#0D1F35",
    "surface2":  "#112640",
    "teal":      "#00D4C8",
    "teal_dim":  "#007A74",
    "gold":      "#F5C518",
    "gold_dim":  "#8A6E0A",
    "text":      "#E0EEF4",
    "muted":     "#5C8099",
    "border":    "#1A3A52",
    "danger":    "#FF5C5C",
    "success":   "#00D4C8",
}

# Berlin U-Bahn authentic line colours
LINE_COLORS = {
    "U1": "#55B09E", "U2": "#DA4A2F", "U3": "#006B35",
    "U4": "#FFCC00", "U5": "#7D4C3C", "U6": "#7B2481",
    "U7": "#009BD2", "U8": "#005FAD", "U9": "#F5A623",
}

PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["surface"],
        font=dict(color=THEME["text"], family="Inter, sans-serif", size=12),
        title=dict(font=dict(color=THEME["teal"], size=18, family="Space Grotesk, sans-serif")),
        xaxis=dict(gridcolor=THEME["border"], linecolor=THEME["border"], zerolinecolor=THEME["border"]),
        yaxis=dict(gridcolor=THEME["border"], linecolor=THEME["border"], zerolinecolor=THEME["border"]),
        legend=dict(bgcolor=THEME["surface2"], bordercolor=THEME["border"], borderwidth=1),
        colorway=[THEME["teal"], THEME["gold"], "#FF6B6B", "#A78BFA", "#34D399", "#F97316"],
        margin=dict(l=10, r=10, t=40, b=10),
    )
)

# ───────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ───────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root Reset ───────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {THEME["bg"]};
    color: {THEME["text"]};
}}

/* ─── App background ───────────────────────────── */
.stApp {{ background: {THEME["bg"]}; }}

/* ─── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {THEME["surface"]};
    border-right: 1px solid {THEME["border"]};
}}
section[data-testid="stSidebar"] * {{ color: {THEME["text"]}; }}

/* ─── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {THEME["surface"]};
    padding: 6px 8px;
    border-radius: 12px;
    border: 1px solid {THEME["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 13px;
    letter-spacing: 0.03em;
    color: {THEME["muted"]};
    border-radius: 8px;
    padding: 8px 18px;
    border: none;
    background: transparent;
    transition: all 0.2s ease;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {THEME["teal"]}22, {THEME["teal"]}44) !important;
    color: {THEME["teal"]} !important;
    border: 1px solid {THEME["teal"]}66 !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding: 16px 0;
}}

/* ─── Metric cards ──────────────────────────────── */
div[data-testid="metric-container"] {{
    background: {THEME["surface"]};
    border: 1px solid {THEME["border"]};
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color 0.2s ease;
}}
div[data-testid="metric-container"]:hover {{
    border-color: {THEME["teal"]}88;
}}
div[data-testid="metric-container"] label {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {THEME["muted"]};
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 28px;
    color: {THEME["teal"]};
}}

/* ─── Buttons ───────────────────────────────────── */
.stButton > button {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.05em;
    color: {THEME["bg"]};
    background: linear-gradient(135deg, {THEME["teal"]}, {THEME["teal_dim"]});
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {THEME["gold"]}, {THEME["gold_dim"]});
    transform: translateY(-1px);
    box-shadow: 0 4px 15px {THEME["teal"]}44;
}}

/* ─── Text input ────────────────────────────────── */
.stTextInput input, .stTextArea textarea {{
    font-family: 'Inter', sans-serif;
    background: {THEME["surface2"]};
    color: {THEME["text"]};
    border: 1px solid {THEME["border"]};
    border-radius: 10px;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {THEME["teal"]};
    box-shadow: 0 0 0 2px {THEME["teal"]}33;
}}

/* ─── Select box ────────────────────────────────── */
.stSelectbox [data-baseweb="select"] div {{
    background: {THEME["surface2"]};
    color: {THEME["text"]};
    border-color: {THEME["border"]};
    border-radius: 8px;
}}

/* ─── Chat messages ─────────────────────────────── */
.chat-user {{
    background: linear-gradient(135deg, {THEME["teal"]}22, {THEME["teal"]}11);
    border: 1px solid {THEME["teal"]}44;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 40px;
    font-size: 14px;
    color: {THEME["text"]};
    font-family: 'Inter', sans-serif;
}}
.chat-agent {{
    background: {THEME["surface"]};
    border: 1px solid {THEME["border"]};
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 40px 8px 0;
    font-size: 13.5px;
    color: {THEME["text"]};
    font-family: 'Inter', sans-serif;
    line-height: 1.65;
    white-space: pre-wrap;
}}
.chat-agent-label {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: {THEME["teal"]};
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.timing-badge {{
    display: inline-block;
    background: {THEME["gold"]}22;
    border: 1px solid {THEME["gold"]}55;
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {THEME["gold"]};
    margin-top: 8px;
}}

/* ─── Section headers ───────────────────────────── */
.section-header {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 22px;
    color: {THEME["text"]};
    margin-bottom: 4px;
}}
.section-sub {{
    font-size: 13px;
    color: {THEME["muted"]};
    margin-bottom: 20px;
    font-family: 'Inter', sans-serif;
}}
.teal {{ color: {THEME["teal"]}; }}
.gold  {{ color: {THEME["gold"]}; }}

/* ─── Divider ───────────────────────────────────── */
hr {{ border-color: {THEME["border"]}; margin: 20px 0; }}

/* ─── Expander ──────────────────────────────────── */
.streamlit-expanderHeader {{
    font-family: 'Space Grotesk', sans-serif;
    color: {THEME["text"]};
    background: {THEME["surface"]};
    border-radius: 8px;
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached — happens once on startup)
# ═══════════════════════════════════════════════════════════════════════════
DATA_DIR_DEFAULT = str(ROOT / "data" / "training dataset")

@st.cache_resource(show_spinner="Loading U-Bahn data...")
def load_data(data_dir: str):
    from data_pipeline.loader import DataStore
    DataStore.initialize(data_dir)
    return DataStore.get()

@st.cache_resource(show_spinner="Building agent...")
def load_agent():
    try:
        from agent.graph import build_agent_graph
        return build_agent_graph()
    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:12px 0 20px 0;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;
             color:{THEME['teal']};letter-spacing:-0.5px;">🚇 Talk To My Train</div>
        <div style="font-size:11px;color:{THEME['muted']};letter-spacing:0.08em;text-transform:uppercase;
             margin-top:4px;">InnoTrans 2026 · Berlin U-Bahn Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    data_dir = st.text_input(
        "Data directory",
        value=DATA_DIR_DEFAULT,
        help="Path to the folder containing all CSV files"
    )

    st.markdown("---")

    # Load data
    with st.spinner("Initializing data..."):
        try:
            D = load_data(data_dir)
            st.success("Data loaded")
        except Exception as ex:
            st.error(f"Data load failed: {ex}")
            D = None

    if D:
        flows    = D.get("flows", pd.DataFrame())
        stations = D.get("stations", pd.DataFrame())
        conns    = D.get("connections", pd.DataFrame())
        closures = D.get("closures", pd.DataFrame())
        events   = D.get("events_mapped", D.get("events", pd.DataFrame()))
        weather  = D.get("weather", pd.DataFrame())
        energy   = D.get("energy", pd.DataFrame())
        G        = D.get("graph")
        centrality = D.get("centrality", {})

        n_stations = len(stations)
        n_events   = len(events)
        n_closures = len(closures)
        total_flow = int(flows.drop(columns=["timestamp"], errors="ignore").sum().sum()) if not flows.empty else 0

        st.markdown("### Network Status")
        st.metric("Stations",   n_stations)
        st.metric("Events",     n_events)
        st.metric("Disruptions", n_closures)
        st.metric("Total trips", f"{total_flow/1_000_000:.1f}M")
    else:
        flows = stations = conns = closures = events = weather = energy = pd.DataFrame()
        G = None; centrality = {}

    st.markdown("---")
    st.markdown(f"<div style='font-size:11px;color:{THEME['muted']};'>Built for InnoTrans 2026 Hackathon<br/>Powered by gpt-5.6-luna · HCADE</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def apply_template(fig):
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig

def flow_cols(df: pd.DataFrame):
    return [c for c in df.columns if c != "timestamp"]

def station_to_lines(stations_df: pd.DataFrame) -> dict:
    """Returns station_name -> list of lines."""
    result = {}
    for _, row in stations_df.iterrows():
        name  = row.get("station_name", "")
        lines = str(row.get("u_bahn_lines", "")).split(",")
        result[name] = [l.strip() for l in lines if l.strip()]
    return result

def get_station_daily_mean(flows: pd.DataFrame) -> pd.Series:
    """Mean daily flow per station."""
    fdf = flows.copy()
    fdf["date"] = pd.to_datetime(fdf["timestamp"]).dt.date
    cols = flow_cols(fdf)
    daily = fdf.groupby("date")[cols].sum()
    return daily.mean()


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_network, tab_analytics, tab_chat, tab_disruptions, tab_energy = st.tabs([
    "🗺️  Network",
    "📊  Flow Analytics",
    "💬  Chat Agent",
    "⚠️  Disruptions",
    "⚡  Energy",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — NETWORK
# ═══════════════════════════════════════════════════════════════════════════
with tab_network:
    st.markdown('<div class="section-header">U-Bahn Network Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Interactive station map · Line selector · Flow intensity sizing</div>', unsafe_allow_html=True)

    if stations.empty:
        st.warning("No station data available.")
    else:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
        with col_ctrl1:
            all_lines = sorted(LINE_COLORS.keys())
            selected_lines = st.multiselect(
                "Filter lines", all_lines, default=all_lines,
                help="Show only selected U-Bahn lines"
            )
        with col_ctrl2:
            size_by = st.selectbox(
                "Size stations by",
                ["Total Flow", "Betweenness Centrality", "Uniform"],
            )
        with col_ctrl3:
            show_connections = st.toggle("Show connections", value=True)

        # ── Compute station attributes ──────────────────────────────────
        s2l = station_to_lines(stations)

        # Assign primary line colour to each station
        def primary_line(name):
            ls = s2l.get(name, [])
            for l in ls:
                if l in LINE_COLORS:
                    return l
            return "U1"

        stations["_primary_line"] = stations["station_name"].apply(primary_line)
        stations["_color"] = stations["_primary_line"].map(LINE_COLORS).fillna(THEME["muted"])

        # Filter by selected lines
        mask = stations["_primary_line"].isin(selected_lines)
        s_filt = stations[mask].copy()

        # Compute size
        if not flows.empty:
            daily_mean = get_station_daily_mean(flows)
        else:
            daily_mean = pd.Series(dtype=float)

        def get_size(name):
            if size_by == "Total Flow":
                v = daily_mean.get(name, 100)
                return max(6, min(28, float(v) / 400 + 6))
            elif size_by == "Betweenness Centrality":
                bc = centrality.get(name, {}).get("betweenness", 0)
                return max(6, min(28, float(bc) * 1500 + 8))
            return 12

        s_filt["_size"] = s_filt["station_name"].apply(get_size)
        s_filt["_flow"] = s_filt["station_name"].apply(lambda n: int(daily_mean.get(n, 0)))
        s_filt["_bc"]   = s_filt["station_name"].apply(lambda n: round(centrality.get(n, {}).get("betweenness", 0), 4))

        # ── Build figure ────────────────────────────────────────────────
        fig_net = go.Figure()

        # Draw connections
        if show_connections and not conns.empty and not stations.empty:
            id_to_coord = {
                row["station_id"]: (row["lon"], row["lat"])
                for _, row in stations.iterrows()
                if pd.notna(row.get("lat")) and pd.notna(row.get("lon"))
            }
            for _, edge in conns.iterrows():
                s1 = edge.get("station_id_1") or edge.get("from_station_id")
                s2 = edge.get("station_id_2") or edge.get("to_station_id")
                if s1 in id_to_coord and s2 in id_to_coord:
                    x0, y0 = id_to_coord[s1]
                    x1, y1 = id_to_coord[s2]
                    fig_net.add_trace(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode="lines",
                        line=dict(color=THEME["border"], width=1.2),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

        # Draw station bubbles per line (for legend grouping)
        for line in selected_lines:
            sl = s_filt[s_filt["_primary_line"] == line]
            if sl.empty:
                continue
            fig_net.add_trace(go.Scatter(
                x=sl["lon"], y=sl["lat"],
                mode="markers",
                name=line,
                marker=dict(
                    size=sl["_size"],
                    color=LINE_COLORS.get(line, THEME["teal"]),
                    opacity=0.9,
                    line=dict(width=1.5, color=THEME["bg"]),
                ),
                customdata=np.column_stack([
                    sl["station_name"], sl["_flow"].astype(str), sl["_bc"].astype(str)
                ]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Line: " + line + "<br>"
                    "Daily flow: %{customdata[1]}<br>"
                    "Betweenness: %{customdata[2]}"
                    "<extra></extra>"
                ),
            ))

        fig_net.update_layout(
            paper_bgcolor=THEME["bg"],
            plot_bgcolor=THEME["surface"],
            xaxis=dict(title="Longitude", gridcolor=THEME["border"], showgrid=True),
            yaxis=dict(title="Latitude",  gridcolor=THEME["border"], showgrid=True,
                       scaleanchor="x", scaleratio=1.8),
            legend=dict(
                orientation="h", y=-0.12,
                bgcolor=THEME["surface"], bordercolor=THEME["border"], borderwidth=1,
                font=dict(family="Space Grotesk", size=12),
            ),
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            hoverlabel=dict(
                bgcolor=THEME["surface2"], bordercolor=THEME["teal"],
                font=dict(family="Inter", size=13, color=THEME["text"]),
            ),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # ── Station Flow Heatmap (horizontal) ───────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header" style="font-size:18px;">Hourly Flow Heatmap</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Mean passenger flow per station × hour — sorted by peak hour flow</div>', unsafe_allow_html=True)

        if not flows.empty:
            n_top = st.slider("Number of stations shown", 10, 60, 30, key="heatmap_n")

            fdf = flows.copy()
            fdf["hour"] = pd.to_datetime(fdf["timestamp"]).dt.hour
            cols = flow_cols(fdf)
            hourly_mean = fdf.groupby("hour")[cols].mean()

            # Filter to stations visible in current line selection
            visible_stations = s_filt["station_name"].tolist()
            available_cols = [c for c in cols if c in visible_stations]
            if not available_cols:
                available_cols = cols

            # Pick top N by total flow
            top_stations = hourly_mean[available_cols].sum().nlargest(n_top).index.tolist()
            heat_data = hourly_mean[top_stations].T  # shape: (n_stations, 24)

            # Short display names
            short_names = [n.replace("U Spichernstr. (Berlin)", "Spichernstr.")
                            .replace(" (Berlin)", "").replace("U ", "") for n in heat_data.index]

            fig_heat = go.Figure(go.Heatmap(
                z=heat_data.values,
                x=[f"{h:02d}:00" for h in range(24)],
                y=short_names,
                colorscale=[
                    [0.0,  THEME["surface"]],
                    [0.3,  THEME["teal_dim"]],
                    [0.7,  THEME["teal"]],
                    [1.0,  THEME["gold"]],
                ],
                hoverongaps=False,
                hoverlabel=dict(bgcolor=THEME["surface2"], font=dict(color=THEME["text"], size=12)),
                hovertemplate="Station: %{y}<br>Hour: %{x}<br>Mean flow: %{z:.0f}<extra></extra>",
                colorbar=dict(
                    title=dict(text="Mean Passengers", font=dict(color=THEME["muted"], size=11)),
                    tickfont=dict(color=THEME["muted"]),
                    bgcolor=THEME["surface"],
                    bordercolor=THEME["border"],
                ),
            ))
            fig_heat.update_layout(
                paper_bgcolor=THEME["bg"],
                plot_bgcolor=THEME["surface"],
                height=max(350, n_top * 18),
                margin=dict(l=150, r=20, t=10, b=40),
                xaxis=dict(title="Hour of Day", tickfont=dict(size=11), gridcolor=THEME["border"]),
                yaxis=dict(tickfont=dict(size=11), gridcolor=THEME["border"]),
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — FLOW ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown('<div class="section-header">Flow Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Daily trends · Anomalies · Station comparison · Weekday vs weekend</div>', unsafe_allow_html=True)

    if flows.empty:
        st.warning("No flow data available.")
    else:
        cols_f = flow_cols(flows)
        fdf = flows.copy()
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"])
        fdf["date"]      = fdf["timestamp"].dt.date
        fdf["hour"]      = fdf["timestamp"].dt.hour
        fdf["weekday"]   = fdf["timestamp"].dt.weekday
        fdf["network"]   = fdf[cols_f].sum(axis=1)

        # ── Daily total + event/closure overlay ─────────────────────────
        daily_net = fdf.groupby("date")["network"].sum().reset_index()
        daily_net["timestamp"] = pd.to_datetime(daily_net["date"])

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_net["timestamp"], y=daily_net["network"],
            mode="lines",
            fill="tozeroy",
            fillcolor=f"{THEME['teal']}22",
            line=dict(color=THEME["teal"], width=2),
            name="Network flow",
        ))

        # Rolling 7-day baseline
        daily_net["rolling"] = daily_net["network"].rolling(7, center=True, min_periods=3).mean()
        fig_daily.add_trace(go.Scatter(
            x=daily_net["timestamp"], y=daily_net["rolling"],
            mode="lines",
            line=dict(color=THEME["gold"], width=1.5, dash="dot"),
            name="7-day rolling mean",
        ))

        # Closure markers
        if not closures.empty and "when" in closures.columns:
            for _, cl in closures.iterrows():
                ts = pd.to_datetime(cl.get("when"), errors="coerce")
                if pd.isna(ts): continue
                fig_daily.add_vline(
                    x=ts.timestamp() * 1000,
                    line_width=1.2, line_dash="dash",
                    line_color=THEME["danger"] + "88",
                    annotation_text=str(cl.get("description", ""))[:20],
                    annotation_font_color=THEME["danger"],
                    annotation_font_size=9,
                )

        fig_daily.update_layout(
            template=PLOTLY_TEMPLATE, height=320,
            title=dict(text="Network-Wide Daily Passenger Flow"),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # ── Top stations + Weekday vs Weekend ───────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f'<div style="font-family:Space Grotesk;font-size:16px;font-weight:600;color:{THEME["text"]};margin-bottom:8px;">Top 20 Stations by Total Flow</div>', unsafe_allow_html=True)
            daily_mean_s = get_station_daily_mean(fdf)
            top20 = daily_mean_s.nlargest(20)
            short = [n.replace(" (Berlin)", "").replace("U ", "")[:25] for n in top20.index]

            fig_top = go.Figure(go.Bar(
                x=top20.values,
                y=short,
                orientation="h",
                marker=dict(
                    color=top20.values,
                    colorscale=[
                        [0, THEME["teal_dim"]],
                        [0.5, THEME["teal"]],
                        [1, THEME["gold"]],
                    ],
                    showscale=False,
                ),
                hovertemplate="%{y}: %{x:.0f} pax/day<extra></extra>",
            ))
            fig_top.update_layout(
                template=PLOTLY_TEMPLATE, height=450,
                xaxis=dict(title="Mean daily pax"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=140, r=20, t=20, b=30),
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col_b:
            st.markdown(f'<div style="font-family:Space Grotesk;font-size:16px;font-weight:600;color:{THEME["text"]};margin-bottom:8px;">Weekday vs Weekend Hourly Profile</div>', unsafe_allow_html=True)
            fdf["is_weekend"] = fdf["weekday"] >= 5
            hourly_wk = fdf[~fdf["is_weekend"]].groupby("hour")["network"].mean()
            hourly_we = fdf[fdf["is_weekend"]].groupby("hour")["network"].mean()

            fig_wknd = go.Figure()
            fig_wknd.add_trace(go.Scatter(
                x=hourly_wk.index, y=hourly_wk.values,
                mode="lines+markers",
                name="Weekday",
                line=dict(color=THEME["teal"], width=2.5),
                marker=dict(size=5),
            ))
            fig_wknd.add_trace(go.Scatter(
                x=hourly_we.index, y=hourly_we.values,
                mode="lines+markers",
                name="Weekend",
                line=dict(color=THEME["gold"], width=2, dash="dash"),
                marker=dict(size=5),
            ))
            fig_wknd.update_layout(
                template=PLOTLY_TEMPLATE, height=240,
                title=dict(text="Hourly Ridership Pattern"),
                xaxis=dict(title="Hour", tickmode="linear", tick0=0, dtick=2),
                yaxis=dict(title="Mean passengers"),
                legend=dict(orientation="h", y=1.15),
                margin=dict(l=20, r=20, t=30, b=30),
            )
            st.plotly_chart(fig_wknd, use_container_width=True)

            # Weather correlation mini chart
            st.markdown(f'<div style="font-family:Space Grotesk;font-size:14px;font-weight:600;color:{THEME["muted"]};margin-top:12px;margin-bottom:4px;">Temperature vs Daily Flow</div>', unsafe_allow_html=True)
            if not weather.empty:
                wdf = weather.copy()
                wdf["date"] = pd.to_datetime(wdf["timestamp"]).dt.date
                temp_daily = wdf.groupby("date")["temp"].mean().reset_index()
                merged = temp_daily.merge(daily_net[["date","network"]], on="date", how="inner")
                if not merged.empty:
                    fig_wth = go.Figure(go.Scatter(
                        x=merged["temp"], y=merged["network"],
                        mode="markers",
                        marker=dict(
                            color=merged["temp"], colorscale="Teal",
                            size=7, opacity=0.7,
                            colorbar=dict(title="°C", tickfont=dict(color=THEME["muted"]))
                        ),
                        hovertemplate="Temp: %{x:.1f}°C<br>Network flow: %{y:.0f}<extra></extra>",
                    ))
                    fig_wth.update_layout(
                        template=PLOTLY_TEMPLATE, height=190,
                        xaxis=dict(title="Temperature (°C)"),
                        yaxis=dict(title="Daily passengers"),
                        margin=dict(l=20, r=20, t=20, b=30),
                    )
                    st.plotly_chart(fig_wth, use_container_width=True)

        # ── Station deep-dive ────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f'<div style="font-family:Space Grotesk;font-size:16px;font-weight:600;color:{THEME["text"]};margin-bottom:8px;">Station Deep-Dive</div>', unsafe_allow_html=True)
        all_station_names = sorted([c for c in cols_f if c in fdf.columns])
        chosen_station = st.selectbox("Select a station", all_station_names, key="station_deepdive",
                                      format_func=lambda n: n.replace(" (Berlin)", "").replace("U ", ""))
        if chosen_station:
            ts_station = fdf[["timestamp", chosen_station]].copy()
            ts_station["rolling_mean"] = ts_station[chosen_station].rolling(96, center=True, min_periods=48).mean()
            ts_station["z_score"] = (
                (ts_station[chosen_station] - ts_station["rolling_mean"]) /
                (ts_station[chosen_station].rolling(96, center=True, min_periods=48).std() + 1e-6)
            )

            fig_station = go.Figure()
            fig_station.add_trace(go.Scatter(
                x=ts_station["timestamp"], y=ts_station[chosen_station],
                mode="lines", name="Flow",
                line=dict(color=THEME["teal"], width=1.2),
            ))
            fig_station.add_trace(go.Scatter(
                x=ts_station["timestamp"], y=ts_station["rolling_mean"],
                mode="lines", name="28-day rolling mean",
                line=dict(color=THEME["gold"], width=1.5, dash="dot"),
            ))
            # Anomaly markers (|Z| > 2)
            anom = ts_station[ts_station["z_score"].abs() > 2.0]
            if not anom.empty:
                fig_station.add_trace(go.Scatter(
                    x=anom["timestamp"], y=anom[chosen_station],
                    mode="markers", name=f"Anomalies (Z>2)",
                    marker=dict(color=THEME["danger"], size=7, symbol="x"),
                ))

            fig_station.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                title=dict(text=f"Flow Timeline: {chosen_station.replace(' (Berlin)','').replace('U ','')}"),
                xaxis=dict(title="Date"),
                yaxis=dict(title="Passengers (15-min)"),
                legend=dict(orientation="h", y=1.15),
            )
            st.plotly_chart(fig_station, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CHAT AGENT
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown('<div class="section-header">Talk To My Train</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Ask any question about the Berlin U-Bahn network · Powered by <span class="teal">gpt-5.6-luna</span> + HCADE analytics</div>', unsafe_allow_html=True)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_loaded" not in st.session_state:
        st.session_state.agent_loaded = False
    if "agent_obj" not in st.session_state:
        st.session_state.agent_obj = None

    # Load agent once
    if not st.session_state.agent_loaded:
        with st.spinner("Building agent graph..."):
            try:
                from agent.graph import build_agent_graph
                st.session_state.agent_obj = build_agent_graph()
                st.session_state.agent_loaded = True
            except Exception as e:
                st.error(f"Agent build failed: {e}")

    # Quick-fire example queries
    st.markdown(f'<div style="font-size:12px;color:{THEME["muted"]};font-family:Space Grotesk;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:8px;">Example queries</div>', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    examples = [
        "Which line has the worst energy efficiency?",
        "U6 is suspended between Hallesches Tor and Kaiserin-Augusta-Str.",
        "Guns N' Roses concert June 23rd at Uber Arena. What should we prepare?",
    ]
    for i, (ec, eq) in enumerate(zip(ex_cols, examples)):
        with ec:
            if st.button(eq[:45] + "...", key=f"ex_{i}"):
                st.session_state["pending_query"] = eq

    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                latency_badge = f'<div class="timing-badge">Process time: {msg.get("latency","?"):.2f}s (LLM + analytics)</div>' if msg.get("latency") else ""
                st.markdown(
                    f'<div class="chat-agent-label">TRAIN INTELLIGENCE</div>'
                    f'<div class="chat-agent">{msg["content"]}</div>'
                    f'{latency_badge}',
                    unsafe_allow_html=True
                )

    # Input
    with st.form(key="chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            user_input = st.text_area(
                "Your question",
                value=st.session_state.pop("pending_query", ""),
                placeholder="Ask about disruptions, energy efficiency, passenger flow anomalies, event surges...",
                height=80, label_visibility="collapsed",
            )
        with col_btn:
            st.write("")
            submitted = st.form_submit_button("Ask", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

        with st.spinner("Analysing with HCADE + gpt-5.6-luna..."):
            import time as _time

            try:
                # Try full agent pipeline
                if st.session_state.agent_obj and D:
                    try:
                        from langchain_core.messages import HumanMessage
                    except ImportError:
                        class HumanMessage:
                            def __init__(self, content): self.content = content

                    t0 = _time.time()
                    result = st.session_state.agent_obj.invoke({
                        "messages": [HumanMessage(content=user_input.strip())]
                    })
                    latency = _time.time() - t0
                    response = result.get("final_response", "No response generated.")
                else:
                    # Fallback: direct ai_connection
                    from ai_connection import ask as ai_ask
                    from agent.prompts import SYSTEM_PROMPT
                    t0 = _time.time()
                    response = ai_ask(f"{SYSTEM_PROMPT}\n\nOperator query: {user_input.strip()}")
                    latency = _time.time() - t0
            except Exception as e:
                response = f"Error running agent: {e}\n\nPlease verify the data directory and API key in your .env file."
                latency = 0.0

        st.session_state.chat_history.append({
            "role": "agent",
            "content": response,
            "latency": latency,
        })
        st.rerun()

    if st.button("Clear chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — DISRUPTIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab_disruptions:
    st.markdown('<div class="section-header">Disruptions & Cascade Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Historical closures timeline · Cascade impact simulation</div>', unsafe_allow_html=True)

    if closures.empty:
        st.warning("No closures data available.")
    else:
        # ── Gantt chart of closures ─────────────────────────────────────
        cl_gantt = closures.copy()
        cl_gantt["when_dt"] = pd.to_datetime(cl_gantt.get("when", cl_gantt.get("start_date")), errors="coerce")

        if "end" in cl_gantt.columns:
            cl_gantt["end_dt"] = pd.to_datetime(cl_gantt["end"], errors="coerce")
        else:
            cl_gantt["end_dt"] = cl_gantt["when_dt"] + pd.Timedelta(hours=2)

        cl_gantt = cl_gantt.dropna(subset=["when_dt"])
        cl_gantt["label"] = cl_gantt.get("description", cl_gantt.get("closure_description", pd.Series())).astype(str).str[:40]
        cl_gantt["line"]  = cl_gantt.get("_line", pd.Series(dtype=str)).fillna("Unknown")
        cl_gantt["color"] = cl_gantt["line"].map(LINE_COLORS).fillna(THEME["muted"])

        fig_gantt = go.Figure()
        for i, row in cl_gantt.iterrows():
            fig_gantt.add_trace(go.Bar(
                x=[(row["end_dt"] - row["when_dt"]).total_seconds() / 3600],
                y=[f"{row['when_dt'].strftime('%b %d')} · {row['label']}"],
                base=[row["when_dt"].timestamp() * 1000],
                orientation="h",
                marker_color=row["color"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['label']}</b><br>"
                    f"Start: {row['when_dt']}<br>"
                    f"End: {row['end_dt']}<br>"
                    f"Line: {row['line']}"
                    "<extra></extra>"
                ),
            ))

        fig_gantt.update_layout(
            template=PLOTLY_TEMPLATE, height=max(300, len(cl_gantt) * 28),
            xaxis=dict(title="Date", type="date"),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
            title=dict(text="Disruption Timeline (All Closures)"),
            margin=dict(l=250, r=20, t=40, b=30),
            bargap=0.3,
        )
        st.plotly_chart(fig_gantt, use_container_width=True)

        # ── Cascade simulator ───────────────────────────────────────────
        st.markdown("---")
        st.markdown(f'<div style="font-family:Space Grotesk;font-size:16px;font-weight:600;color:{THEME["text"]};margin-bottom:8px;">Live Cascade Simulation</div>', unsafe_allow_html=True)

        col_casc1, col_casc2 = st.columns([2, 3])
        with col_casc1:
            all_station_names_cascade = sorted(station_to_lines(stations).keys()) if not stations.empty else []
            closed_stn = st.multiselect(
                "Select closed stations",
                all_station_names_cascade,
                help="Simulate what happens when these stations close",
                format_func=lambda n: n.replace(" (Berlin)", "").replace("U ", ""),
            )
            cascade_ts = st.text_input("Timestamp", "2026-07-17 08:00:00")
            run_cascade = st.button("Run Cascade Simulation")

        with col_casc2:
            if run_cascade and closed_stn and D:
                with st.spinner("Simulating cascade..."):
                    try:
                        from analytics.cascade_sim import simulate_cascade
                        results = simulate_cascade(
                            G=D["graph"],
                            closed_stations=closed_stn,
                            closed_segment=None,
                            flows=flows,
                            timestamp=cascade_ts,
                        )
                        if results:
                            res_df = pd.DataFrame(results).head(20)
                            res_df["short_name"] = res_df["station"].apply(
                                lambda n: n.replace(" (Berlin)","").replace("U ","")[:30])
                            res_df["color"] = res_df["at_risk"].map({True: THEME["danger"], False: THEME["teal"]})
                            
                            fig_casc = go.Figure(go.Bar(
                                x=res_df["overflow_ratio"],
                                y=res_df["short_name"],
                                orientation="h",
                                marker_color=res_df["color"],
                                hovertemplate="%{y}: overflow %{x:.2f}x baseline<extra></extra>",
                            ))
                            fig_casc.add_vline(x=1.0, line_color=THEME["muted"], line_dash="dot", annotation_text="Baseline")
                            fig_casc.add_vline(x=1.3, line_color=THEME["danger"], line_dash="dash", annotation_text="Risk threshold")
                            fig_casc.update_layout(
                                template=PLOTLY_TEMPLATE, height=380,
                                xaxis=dict(title="Overflow ratio (1.0 = normal)"),
                                yaxis=dict(autorange="reversed"),
                                title=dict(text="Predicted Station Overflow"),
                                margin=dict(l=180, r=20, t=40, b=30),
                            )
                            st.plotly_chart(fig_casc, use_container_width=True)
                    except Exception as e:
                        st.error(f"Cascade simulation error: {e}")
            elif not run_cascade:
                st.info("Select stations on the left and click 'Run Cascade Simulation' to see real-time overflow predictions.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — ENERGY
# ═══════════════════════════════════════════════════════════════════════════
with tab_energy:
    st.markdown('<div class="section-header">Energy & Efficiency Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">MWh per 1,000 passengers · Line-by-line efficiency ranking · Intervention recommendations</div>', unsafe_allow_html=True)

    if energy.empty:
        st.warning("No energy data available.")
    else:
        with st.spinner("Computing efficiency metrics..."):
            try:
                from analytics.energy import compute_energy_efficiency
                eff_results = compute_energy_efficiency(energy, flows, stations)
            except Exception as e:
                eff_results = []
                st.error(f"Energy computation error: {e}")

        if eff_results:
            eff_df = pd.DataFrame(eff_results).sort_values("mwh_per_1k_pax", ascending=False)

            # ── KPI cards ────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            worst = eff_df.iloc[0]
            best  = eff_df.iloc[-1]
            c1.metric("Worst efficiency line", worst.get("line","?"), help="Highest MWh per 1k passengers")
            c2.metric("Worst MWh/1k pax", f"{worst.get('mwh_per_1k_pax',0):.0f}")
            c3.metric("Best efficiency line",  best.get("line","?"))
            c4.metric("Best MWh/1k pax",   f"{best.get('mwh_per_1k_pax',0):.0f}")

            # ── Bar chart ────────────────────────────────────────────────
            colors = [
                THEME["danger"] if i == 0 else
                THEME["gold"]   if i == len(eff_df)-1 else
                THEME["teal"]
                for i in range(len(eff_df))
            ]
            fig_eff = go.Figure(go.Bar(
                x=eff_df["line"],
                y=eff_df["mwh_per_1k_pax"],
                marker_color=colors,
                text=[f"{v:.0f}" for v in eff_df["mwh_per_1k_pax"]],
                textposition="outside",
                textfont=dict(color=THEME["text"], size=12, family="Space Grotesk"),
                hovertemplate="<b>%{x}</b><br>%{y:.1f} MWh / 1k passengers<extra></extra>",
            ))
            fig_eff.update_layout(
                template=PLOTLY_TEMPLATE, height=360,
                title=dict(text="Energy Efficiency by Line (worst → best, left → right)"),
                xaxis=dict(title="U-Bahn Line"),
                yaxis=dict(title="MWh per 1,000 passengers"),
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_eff, use_container_width=True)

            # ── Per-line detail ──────────────────────────────────────────
            st.markdown("---")
            st.markdown(f'<div style="font-family:Space Grotesk;font-size:16px;font-weight:600;color:{THEME["text"]};margin-bottom:12px;">Line Details & Recommendations</div>', unsafe_allow_html=True)
            for _, row in eff_df.iterrows():
                with st.expander(f"{row.get('line','?')}  ·  {row.get('mwh_per_1k_pax',0):.0f} MWh/1k pax  ·  Rank #{row.get('efficiency_rank','?')}"):
                    st.markdown(f"""
                    <div style="color:{THEME['text']};font-family:Inter;font-size:14px;line-height:1.7;">
                    <b style="color:{THEME['teal']};">Efficiency:</b> {row.get('mwh_per_1k_pax',0):.1f} MWh per 1,000 passengers<br>
                    <b style="color:{THEME['teal']};">Total energy:</b> {row.get('total_mwh',0):.1f} MWh<br>
                    <b style="color:{THEME['teal']};">Total passengers:</b> {row.get('total_passengers',0):,.0f}<br><br>
                    <b style="color:{THEME['gold']};">Analysis:</b> {row.get('explanation', 'N/A')}<br><br>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Daily energy timeline ────────────────────────────────────────
        st.markdown("---")
        st.markdown(f'<div style="font-family:Space Grotesk;font-size:16px;font-weight:600;color:{THEME["text"]};margin-bottom:8px;">Daily Energy Consumption by Line</div>', unsafe_allow_html=True)
        e_cols = [c for c in energy.columns if c not in ["date", "Unnamed: 0"]]
        if e_cols:
            energy_plot = energy.copy()
            if "date" not in energy_plot.columns:
                energy_plot = energy_plot.rename(columns={energy_plot.columns[0]: "date"})
            energy_plot["date"] = pd.to_datetime(energy_plot["date"], errors="coerce")

            fig_energy_ts = go.Figure()
            for line in e_cols:
                if line in LINE_COLORS:
                    fig_energy_ts.add_trace(go.Scatter(
                        x=energy_plot["date"], y=energy_plot[line],
                        mode="lines", name=line,
                        line=dict(color=LINE_COLORS[line], width=1.8),
                    ))
            fig_energy_ts.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                title=dict(text="Daily MWh Consumption per Line"),
                xaxis=dict(title="Date"),
                yaxis=dict(title="MWh"),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_energy_ts, use_container_width=True)
