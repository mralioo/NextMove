import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_weather, network_total_flow
from utils.ui import page_header, sidebar_dataset_picker

st.set_page_config(page_title="Weather", page_icon="🌦️", layout="wide")
folder = sidebar_dataset_picker()
page_header("Weather Impact", "Correlating temperature, precipitation, and conditions with ridership.")
st.caption("Same question at two granularities: a 15-min correlation heatmap (noisy — time-of-day dominates), then daily aggregates that isolate the weather signal better.")

weather = load_weather(folder)
flow_total = network_total_flow(folder)

joined = flow_total.merge(weather, on="timestamp", how="inner")

st.subheader("Correlation with network-wide 15-min passenger flow")
corr_cols = ["total_passengers", "temp", "rhum", "prcp", "wdir", "wspd", "pres", "cldc"]
corr = joined[corr_cols].corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                 labels=dict(color="Pearson r"))
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"`total_passengers` vs `temp`: r = {corr.loc['total_passengers', 'temp']:.2f} — "
    f"vs `prcp`: r = {corr.loc['total_passengers', 'prcp']:.2f}. "
    "Weak correlations are expected at 15-min granularity since time-of-day dominates; "
    "daily aggregation below isolates the weather signal better."
)

st.divider()
st.subheader("Daily aggregation: weather vs. ridership")
daily = joined.set_index("timestamp").resample("D").agg(
    total_passengers=("total_passengers", "sum"),
    temp=("temp", "mean"),
    prcp=("prcp", "sum"),
    wspd=("wspd", "mean"),
    cldc=("cldc", "mean"),
    coco=("coco", lambda s: s.mode().iat[0] if not s.mode().empty else None),
).reset_index()

x_var = st.selectbox("X-axis weather variable", ["temp", "prcp", "wspd", "cldc"], index=0,
                      format_func=lambda c: {"temp": "Temperature (°C)", "prcp": "Precipitation (sum/day)",
                                              "wspd": "Wind speed (kph)", "cldc": "Cloud cover (%)"}[c])
daily["weekday"] = daily["timestamp"].dt.day_name()
fig = px.scatter(daily, x=x_var, y="total_passengers", color="weekday",
                  labels={"total_passengers": "Daily network passengers"})
valid = daily[[x_var, "total_passengers"]].dropna()
if len(valid) >= 2:
    slope, intercept = np.polyfit(valid[x_var], valid["total_passengers"], 1)
    x_line = np.linspace(valid[x_var].min(), valid[x_var].max(), 50)
    fig.add_scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                     name="Trend (linear fit)", line=dict(color="black", dash="dash"))
    corr_xy = valid[x_var].corr(valid["total_passengers"])
    st.caption(f"Linear trend: passengers ≈ {slope:,.1f} × {x_var} + {intercept:,.0f} (r = {corr_xy:.2f})")
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Ridership by weather condition code")
    coco_map_note = "Condition codes follow the Meteostat `coco` scale (clear → severe weather)."
    st.caption(coco_map_note)
    by_coco = joined.groupby("coco")["total_passengers"].mean().reset_index().sort_values("coco")
    fig = px.bar(by_coco, x="coco", y="total_passengers",
                 labels={"coco": "Condition code", "total_passengers": "Avg 15-min passengers"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Temperature & precipitation over time")
    st.caption("The raw weather series itself — useful for spotting the specific day behind any peak found elsewhere on this page.")
    fig = px.line(daily, x="timestamp", y="temp", labels={"timestamp": "", "temp": "Temp (°C)"})
    fig.add_bar(x=daily["timestamp"], y=daily["prcp"], name="Precipitation", yaxis="y2", opacity=0.4)
    fig.update_layout(
        height=380,
        yaxis2=dict(overlaying="y", side="right", title="Precipitation"),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Weather-driven flow peaks (candidate anomalies)")
st.caption("Days in the top decile of precipitation or wind, ranked by same-day total ridership — "
           "useful starting point for 'peak caused by bad weather' style questions.")
threshold_prcp = daily["prcp"].quantile(0.9)
rough = daily[daily["prcp"] >= threshold_prcp].sort_values("total_passengers", ascending=False)
st.dataframe(
    rough[["timestamp", "temp", "prcp", "wspd", "total_passengers"]]
    .rename(columns={"timestamp": "Date", "temp": "Avg temp (°C)", "prcp": "Precip (sum)",
                      "wspd": "Avg wind (kph)", "total_passengers": "Total passengers"}),
    use_container_width=True, hide_index=True,
)
