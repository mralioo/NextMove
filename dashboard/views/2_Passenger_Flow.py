import plotly.express as px
import streamlit as st

from utils.data_loader import (
    dow_hour_heatmap,
    flows_long,
    hourly_profile,
    load_flows,
    load_stations,
    station_avg_flow,
    station_cols,
)
from utils.ui import page_header, sidebar_dataset_picker

st.set_page_config(page_title="Passenger Flow", page_icon="📈", layout="wide")
folder = sidebar_dataset_picker()
page_header("Passenger Flow Analysis", "Time series, daily/weekly rhythm, and station rankings.")
st.caption(
    "Three ways to look at the same 15-minute flow series: who's busiest overall, how one or "
    "more stations behave over time/hour/weekday, and how any station's commute peak stacks "
    "up against the network."
)

stations = load_stations(folder)
flows = load_flows(folder)
avg_flow = station_avg_flow(folder)
all_stations = sorted(station_cols(flows))

st.subheader("Busiest & quietest stations")
st.caption("Ranked by average daily passengers (sum of a station's 15-min counts per day, averaged across the whole dataset window).")
c1, c2 = st.columns(2)
top_n = st.slider("How many stations to rank", 5, 30, 15)
ranked = avg_flow.sort_values("avg_daily_passengers", ascending=False)
with c1:
    fig = px.bar(ranked.head(top_n).sort_values("avg_daily_passengers"),
                 x="avg_daily_passengers", y="station_name", orientation="h",
                 title=f"Top {top_n} busiest stations (avg daily passengers)",
                 labels={"avg_daily_passengers": "Avg daily passengers", "station_name": ""})
    fig.update_layout(height=450, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.bar(ranked.tail(top_n).sort_values("avg_daily_passengers"),
                 x="avg_daily_passengers", y="station_name", orientation="h",
                 title=f"Bottom {top_n} quietest stations (avg daily passengers)",
                 labels={"avg_daily_passengers": "Avg daily passengers", "station_name": ""})
    fig.update_layout(height=450, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Station deep-dive")
st.caption("Pick one or more stations to compare directly — same underlying data, three different views (raw trend, typical hour-of-day shape, and day×hour intensity).")
default_sel = [ranked.iloc[0]["station_name"]]
selected = st.multiselect("Select one or more stations", all_stations, default=default_sel)
network_mean_peak = None

if selected:
    long = flows_long(folder)
    sel_long = long[long["station_name"].isin(selected)]

    tab1, tab2, tab3 = st.tabs(["Time series", "Hourly profile (weekday vs weekend)", "Day × hour heatmap"])

    with tab1:
        st.caption("Raw flow over time for the selected station(s) — resample to smooth out 15-min noise and spot trends or one-off spikes.")
        granularity = st.radio("Resample", ["15 min (raw)", "Hourly", "Daily"], horizontal=True, index=1, key="ts_gran")
        freq = {"15 min (raw)": None, "Hourly": "h", "Daily": "D"}[granularity]
        plot_df = sel_long.copy()
        if freq:
            plot_df = (plot_df.set_index("timestamp")
                       .groupby("station_name")["passengers"]
                       .resample(freq).sum()
                       .reset_index())
        fig = px.line(plot_df, x="timestamp", y="passengers", color="station_name",
                       labels={"timestamp": "", "passengers": "Passengers", "station_name": "Station"})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.caption("Average passengers by hour-of-day, split weekday vs weekend — this is what \"commute peak\" means throughout this dashboard.")
        prof = hourly_profile(folder, tuple(sorted(selected)))
        prof["Day type"] = prof["is_weekend"].map({True: "Weekend", False: "Weekday"})
        fig = px.line(prof, x="hour", y="passengers", color="Day type", markers=True,
                       labels={"hour": "Hour of day", "passengers": "Avg passengers"})
        network_prof = hourly_profile(folder)
        network_prof["Day type"] = network_prof["is_weekend"].map({True: "Weekend (network avg)", False: "Weekday (network avg)"})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        peak_row = prof.loc[prof["passengers"].idxmax()]
        st.caption(
            f"Peak average flow for the current selection: **{peak_row['passengers']:.0f} passengers** "
            f"around **{int(peak_row['hour']):02d}:00** on a "
            f"**{'weekend' if peak_row['is_weekend'] else 'weekday'}**."
        )

    with tab3:
        st.caption("Same average-flow data as the hourly profile, but every day of the week shown separately — darker cells are busier hour/day combinations.")
        heat = dow_hour_heatmap(folder, tuple(sorted(selected)))
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                         labels=dict(x="Hour of day", y="", color="Avg passengers"))
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one station above to see its time series and rhythm.")

st.divider()
st.subheader("Commute-peak benchmark")
st.caption("Compares a chosen station's peak weekday hourly average against the network-wide mean peak — "
           "directly answers questions like *'does station X's commute peak exceed the network average?'*")

bench_station = st.selectbox("Station", all_stations, index=all_stations.index(default_sel[0]) if default_sel[0] in all_stations else 0)
long = flows_long(folder)
station_weekday = long[(long["station_name"] == bench_station) & (~long["is_weekend"])]
station_hourly = station_weekday.groupby("hour")["passengers"].mean()
station_peak_hour = station_hourly.idxmax()
station_peak_val = station_hourly.max()

network_weekday = long[~long["is_weekend"]]
network_hourly_by_station = network_weekday.groupby(["station_name", "hour"])["passengers"].mean().reset_index()
network_station_peaks = network_hourly_by_station.groupby("station_name")["passengers"].max()
network_mean_peak = network_station_peaks.mean()

m1, m2, m3 = st.columns(3)
m1.metric(f"{bench_station} weekday peak", f"{station_peak_val:.0f} pax", f"at {station_peak_hour:02d}:00")
m2.metric("Network mean weekday peak (all stations)", f"{network_mean_peak:.0f} pax")
delta = station_peak_val - network_mean_peak
m3.metric("Difference vs network mean", f"{delta:+.0f} pax", f"{delta / network_mean_peak * 100:+.1f}%")
