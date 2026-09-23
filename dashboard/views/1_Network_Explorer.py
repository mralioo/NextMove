import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    LINE_COLORS,
    load_connections,
    load_stations,
    network_resilience,
    station_avg_flow,
)
from utils.ui import page_header, sidebar_dataset_picker

st.set_page_config(page_title="Network Explorer", page_icon="🗺️", layout="wide")
folder = sidebar_dataset_picker()
page_header("Network Explorer", "Topology, line coverage, and station-fragmentation risk.")
st.caption(
    "Two views of the same 167-station graph: a map you can recolor by line/ridership/risk, "
    "and a table ranking stations by how much closing them would hurt the network."
)

stations = load_stations(folder)
connections = load_connections(folder)
avg_flow = station_avg_flow(folder)
resilience = network_resilience(folder)

stations = stations.merge(avg_flow[["station_name", "avg_daily_passengers"]], on="station_name", how="left")
stations = stations.merge(
    resilience[["station_name", "is_articulation_point", "betweenness_centrality", "fragmentation_score"]],
    on="station_name", how="left",
)

st.subheader("Station map")
c1, c2 = st.columns([1, 3])
with c1:
    st.caption(
        "**By line** — network layout. **By ridership** — bigger/darker dot = busier station. "
        "**By fragmentation risk** — red dots are articulation points (removing them splits the network); "
        "dot size tracks betweenness centrality (how many shortest paths pass through)."
    )
    highlight_mode = st.radio(
        "Highlight",
        ["By line", "By ridership", "By fragmentation risk"],
        index=0,
    )
    show_critical_only = st.checkbox("Show only critical (articulation-point) stations", value=False)

plot_stations = stations[stations["is_articulation_point"]] if show_critical_only else stations

id_pos = stations.set_index("station_id")[["longitude", "latitude"]]
lon_edges, lat_edges = [], []
for _, row in connections.iterrows():
    if row["station_id_1"] in id_pos.index and row["station_id_2"] in id_pos.index:
        p1, p2 = id_pos.loc[row["station_id_1"]], id_pos.loc[row["station_id_2"]]
        lon_edges += [p1["longitude"], p2["longitude"], None]
        lat_edges += [p1["latitude"], p2["latitude"], None]

fig = go.Figure()
fig.add_trace(go.Scattermapbox(
    lon=lon_edges, lat=lat_edges, mode="lines",
    line=dict(width=1.5, color="rgba(120,120,120,0.6)"),
    hoverinfo="skip", showlegend=False,
))

if highlight_mode == "By line":
    for line, color in LINE_COLORS.items():
        sub = plot_stations[plot_stations["primary_line"] == line]
        if sub.empty:
            continue
        fig.add_trace(go.Scattermapbox(
            lon=sub["longitude"], lat=sub["latitude"], mode="markers", name=line,
            marker=dict(size=9, color=color),
            text=sub["station_name"] + "<br>Lines: " + sub["u_bahn_lines"],
            hoverinfo="text",
        ))
elif highlight_mode == "By ridership":
    fig.add_trace(go.Scattermapbox(
        lon=plot_stations["longitude"], lat=plot_stations["latitude"], mode="markers",
        marker=dict(
            size=8 + 24 * plot_stations["avg_daily_passengers"].fillna(0) / stations["avg_daily_passengers"].max(),
            color=plot_stations["avg_daily_passengers"], colorscale="YlOrRd", showscale=True,
            colorbar=dict(title="Avg daily<br>passengers"),
        ),
        text=plot_stations["station_name"] + "<br>Avg daily: "
             + plot_stations["avg_daily_passengers"].round(0).astype("Int64").astype(str),
        hoverinfo="text", showlegend=False,
    ))
else:
    fig.add_trace(go.Scattermapbox(
        lon=plot_stations["longitude"], lat=plot_stations["latitude"], mode="markers",
        marker=dict(
            size=8 + 20 * plot_stations["betweenness_centrality"].fillna(0) / max(stations["betweenness_centrality"].max(), 1e-9),
            color=plot_stations["is_articulation_point"].map({True: "#d62728", False: "#1f77b4"}),
        ),
        text=plot_stations["station_name"] + "<br>Critical: "
             + plot_stations["is_articulation_point"].astype(str)
             + "<br>Betweenness: " + plot_stations["betweenness_centrality"].round(3).astype(str),
        hoverinfo="text", showlegend=False,
    ))

fig.update_layout(
    mapbox=dict(style="open-street-map", center=dict(lon=13.38, lat=52.51), zoom=10.3),
    height=650, margin=dict(t=0, b=0, l=0, r=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Fragmentation risk — which station closures hurt the network most?")
st.caption(
    "For every station, we simulate removing it from the graph (pure topology, no LLM). "
    "**Articulation point** = removing it actually splits the network into disconnected pieces — "
    "a true single point of failure. **Betweenness centrality** = the fraction of all shortest "
    "paths in the network that pass through this station — high even for non-articulation "
    "\"bridge\" stations. **Fragmentation score** = betweenness × average daily ridership, so a "
    "structurally central but low-traffic station doesn't outrank a high-traffic one. Sorted "
    "descending: the top rows are where a closure does the most combined structural + passenger damage."
)

top_n = st.slider("Top N stations", 5, 30, 10)
rank_tbl = resilience.head(top_n)[[
    "station_name", "is_articulation_point", "betweenness_centrality",
    "resulting_components", "largest_fragment_size", "avg_daily_passengers", "fragmentation_score",
]].rename(columns={
    "station_name": "Station", "is_articulation_point": "Critical (articulation pt.)",
    "betweenness_centrality": "Betweenness", "resulting_components": "Fragments if removed",
    "largest_fragment_size": "Largest remaining fragment", "avg_daily_passengers": "Avg daily passengers",
    "fragmentation_score": "Fragmentation score",
})
st.dataframe(rank_tbl, use_container_width=True, hide_index=True)

st.caption(
    "Mitigation cues: stations with high fragmentation score and multiple served lines are natural "
    "candidates for extra staffing, shuttle-bus bridging, or infrastructure redundancy investment."
)
