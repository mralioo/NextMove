import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_loader import load_events, network_total_flow
from utils.ui import page_header, sidebar_dataset_picker

st.set_page_config(page_title="Events", page_icon="🎫", layout="wide")
folder = sidebar_dataset_picker()
page_header("Events in Berlin", "Concerts, matches, shows, and conferences — and how their days compare to network flow.")

events = load_events(folder)
flow_total = network_total_flow(folder)

c1, c2, c3 = st.columns(3)
segments = sorted(events["segment"].dropna().unique())
sel_segments = c1.multiselect("Segment", segments, default=segments)
min_att = c2.number_input("Min. estimated attendance", min_value=0,
                           value=0, step=100)
date_range = c3.date_input(
    "Date range",
    (events["date"].min(), events["date"].max()),
    min_value=events["date"].min(), max_value=events["date"].max(),
)

filtered = events[
    events["segment"].isin(sel_segments)
    & (events["estimated_attendance"].fillna(0) >= min_att)
]
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[(filtered["date"] >= date_range[0]) & (filtered["date"] <= date_range[1])]

st.caption(f"{len(filtered)} of {len(events)} events match the current filters.")

st.subheader("Daily event load vs. network-wide passenger flow")
st.caption(
    "Qualitative correlation check: days with more/larger events tend to coincide with network flow "
    "spikes. Events aren't geocoded to specific stations in the source data, so use this to spot "
    "candidate high-impact days, then confirm the affected station via the Passenger Flow page."
)

daily_attendance = filtered.groupby("date")["estimated_attendance"].sum().reset_index()
daily_events_ct = filtered.groupby("date").size().reset_index(name="n_events")
daily = daily_attendance.merge(daily_events_ct, on="date")
daily_flow = flow_total.set_index("timestamp")["total_passengers"].resample("D").sum().reset_index()
daily_flow["date"] = daily_flow["timestamp"].dt.date

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=daily["date"], y=daily["estimated_attendance"], name="Est. event attendance (sum/day)",
                      marker_color="#f3791d", opacity=0.6), secondary_y=False)
fig.add_trace(go.Scatter(x=daily_flow["date"], y=daily_flow["total_passengers"], name="Network passengers/day",
                          line=dict(color="#224f86", width=2)), secondary_y=True)
fig.update_layout(height=430, legend=dict(orientation="h", yanchor="bottom", y=1.01))
fig.update_yaxes(title_text="Est. attendance", secondary_y=False)
fig.update_yaxes(title_text="Network passengers/day", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Events by segment")
    fig = px.pie(filtered, names="segment", hole=0.45)
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Attendance distribution")
    fig = px.histogram(filtered, x="estimated_attendance", nbins=30,
                        labels={"estimated_attendance": "Estimated attendance"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Largest events")
top_events = filtered.sort_values("estimated_attendance", ascending=False).head(20)
st.dataframe(
    top_events[["event_name", "began_local", "venue_name", "address", "segment", "genre", "estimated_attendance"]]
    .rename(columns={
        "event_name": "Event", "began_local": "Start", "venue_name": "Venue", "address": "Address",
        "segment": "Segment", "genre": "Genre", "estimated_attendance": "Est. attendance",
    }),
    use_container_width=True, hide_index=True,
)
