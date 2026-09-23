import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import (
    load_closures,
    load_energy,
    load_events,
    load_stations,
    load_weather,
    network_total_flow,
)
from utils.ui import page_header, sidebar_dataset_picker

st.set_page_config(page_title="Berlin U-Bahn Flow Explorer", page_icon="🚇", layout="wide")

folder = sidebar_dataset_picker()

page_header(
    "Berlin U-Bahn Passenger Flow Explorer",
    "InnoTrans 2026 Hackathon — exploratory dashboard for the network, flow, event, "
    "weather, closure, and energy datasets that back the conversational agent.",
)

stations = load_stations(folder)
flow_total = network_total_flow(folder)
events = load_events(folder)
closures = load_closures(folder)
weather = load_weather(folder)
energy = load_energy(folder)

start, end = flow_total["timestamp"].min(), flow_total["timestamp"].max()
n_days = flow_total["timestamp"].dt.date.nunique()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Stations", f"{stations.shape[0]}")
c2.metric("U-Bahn lines", f"{stations['u_bahn_lines'].str.split(',').explode().str.strip().nunique()}")
c3.metric("Days covered", f"{n_days}")
c4.metric("Avg daily passengers", f"{flow_total.set_index('timestamp')['total_passengers'].resample('D').sum().mean():,.0f}")
c5.metric("Events logged", f"{events.shape[0]}")
c6.metric("Closures logged", f"{closures.shape[0]}")

st.caption(f"Coverage: **{start:%Y-%m-%d %H:%M}** → **{end:%Y-%m-%d %H:%M}** (Berlin local time, 15-min grain)")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Network-wide passenger flow over time")
    granularity = st.radio("Resample", ["15 min (raw)", "Hourly", "Daily"], horizontal=True, index=2)
    freq = {"15 min (raw)": None, "Hourly": "h", "Daily": "D"}[granularity]
    plot_df = flow_total.copy()
    if freq:
        plot_df = plot_df.set_index("timestamp")["total_passengers"].resample(freq).sum().reset_index()
    fig = px.area(plot_df, x="timestamp", y="total_passengers",
                   labels={"timestamp": "", "total_passengers": "Passengers"})
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Line coverage")
    line_counts = stations["u_bahn_lines"].str.split(",").explode().str.strip().value_counts().reset_index()
    line_counts.columns = ["line", "stations_served"]
    fig2 = px.bar(line_counts.sort_values("stations_served"), x="stations_served", y="line",
                   orientation="h", labels={"stations_served": "Stations served", "line": ""})
    fig2.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Data quality & context snapshot")
q1, q2, q3 = st.columns(3)
with q1:
    st.markdown("**Weather**")
    st.write(f"Condition codes observed: {weather['coco'].nunique()}")
    st.write(f"Temp range: {weather['temp'].min():.1f}°C – {weather['temp'].max():.1f}°C")
    st.write(f"Days with precipitation: {(weather.set_index('timestamp')['prcp'].resample('D').sum() > 0).sum()}")
with q2:
    st.markdown("**Events**")
    st.write(f"Segments: {', '.join(events['segment'].value_counts().index[:4])}")
    st.write(f"Median estimated attendance: {events['estimated_attendance'].median():,.0f}")
    st.write(f"Largest event: {events.loc[events['estimated_attendance'].idxmax(), 'event_name']}")
with q3:
    st.markdown("**Closures**")
    st.write(f"Line suspensions: {(closures['closure_type'] == 'Line suspension').sum()}")
    st.write(f"Station closures: {(closures['closure_type'] == 'Station closure').sum()}")
    st.write(f"Avg duration: {closures['duration_hours'].mean():.1f}h")

st.divider()
st.info(
    "Use the sidebar to switch dataset folders (e.g. once the September 22-30 evaluation "
    "data is added), and the pages on the left to drill into network topology, passenger "
    "flow patterns, events, weather, closures, and energy efficiency.",
    icon="ℹ️",
)
