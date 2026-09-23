import plotly.express as px
import streamlit as st

from utils.data_loader import load_closures
from utils.ui import page_header, sidebar_dataset_picker

st.set_page_config(page_title="Closures", page_icon="🚧", layout="wide")
folder = sidebar_dataset_picker()
page_header("Disruption & Closure Scenarios", "Simulated line suspensions and station closures.")

closures = load_closures(folder)

c1, c2, c3 = st.columns(3)
c1.metric("Total closures", closures.shape[0])
c2.metric("Line suspensions", int((closures["closure_type"] == "Line suspension").sum()))
c3.metric("Station closures", int((closures["closure_type"] == "Station closure").sum()))

st.subheader("Timeline")
gantt_df = closures.copy()
gantt_df["label"] = gantt_df["affected_line"].fillna(gantt_df["affected_segment"])
fig = px.timeline(gantt_df, x_start="when", x_end="end", y="label", color="closure_type",
                   hover_data=["reason", "duration"])
fig.update_yaxes(title="")
fig.update_layout(height=550)
st.plotly_chart(fig, use_container_width=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Closures by affected line")
    by_line = closures.dropna(subset=["affected_line"])["affected_line"].value_counts().reset_index()
    by_line.columns = ["line", "count"]
    fig = px.bar(by_line.sort_values("count"), x="count", y="line", orientation="h",
                 labels={"count": "Number of closures", "line": ""})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Reason breakdown")
    fig = px.pie(closures, names="reason", hole=0.45)
    fig.update_layout(height=350, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Duration distribution")
fig = px.histogram(closures, x="duration_hours", nbins=10, color="closure_type",
                    labels={"duration_hours": "Duration (hours)"})
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("All closures")
st.dataframe(
    closures[["when", "duration", "closure_type", "affected_line", "affected_segment", "reason"]]
    .rename(columns={"when": "Start", "duration": "Duration", "closure_type": "Type",
                      "affected_line": "Line", "affected_segment": "Segment / Station", "reason": "Reason"})
    .sort_values("Start"),
    use_container_width=True, hide_index=True,
)
