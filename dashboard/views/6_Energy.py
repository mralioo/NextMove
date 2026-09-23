import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import LINE_COLORS, line_flow_series, load_energy
from utils.ui import page_header, sidebar_dataset_picker

st.set_page_config(page_title="Energy", page_icon="⚡", layout="wide")
folder = sidebar_dataset_picker()
page_header("Energy Consumption & Efficiency", "Daily energy use per line, and energy-per-passenger efficiency.")
st.caption("Simulated daily energy per line, plus the efficiency ranking that answers training question 5 (worst Wh/passenger line).")

energy = load_energy(folder)
line_cols = [c for c in energy.columns if c != "timestamp"]

st.subheader("Daily energy consumption by line")
st.caption("Raw daily MWh per line — a busier line isn't necessarily a less efficient one; that's what the ratio further down checks.")
long_energy = energy.melt(id_vars="timestamp", value_vars=line_cols, var_name="line", value_name="MWh")
fig = px.line(long_energy, x="timestamp", y="MWh", color="line",
              color_discrete_map=LINE_COLORS, labels={"timestamp": ""})
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Total network energy over time")
    total_daily = energy.set_index("timestamp")[line_cols].sum(axis=1).reset_index(name="total_MWh")
    fig = px.area(total_daily, x="timestamp", y="total_MWh", labels={"timestamp": ""})
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Average daily energy share by line")
    st.caption("Which lines account for the biggest slice of total network energy, on an average day.")
    avg_share = energy[line_cols].mean().reset_index()
    avg_share.columns = ["line", "avg_MWh"]
    fig = px.pie(avg_share, names="line", values="avg_MWh", color="line", color_discrete_map=LINE_COLORS, hole=0.4)
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Energy-per-passenger efficiency by line")
st.caption(
    "Line-level passenger flow is approximated by summing flows at every station that serves the line "
    "(interchange stations are counted on each line they serve, so treat this as a directional signal, "
    "not an exact per-line passenger count)."
)

flow_by_line = line_flow_series(folder)
flow_daily = flow_by_line.set_index("timestamp").resample("D").sum()
energy_daily = energy.set_index("timestamp")

common_lines = [c for c in line_cols if c in flow_daily.columns]
common_dates = flow_daily.index.intersection(energy_daily.index)

eff_rows = []
for line in common_lines:
    e = energy_daily.loc[common_dates, line]
    p = flow_daily.loc[common_dates, line]
    total_energy_kwh = e.sum() * 1000  # MWh -> kWh
    total_passengers = p.sum()
    eff_rows.append({
        "line": line,
        "total_energy_MWh": e.sum(),
        "total_passengers": total_passengers,
        "wh_per_passenger": (total_energy_kwh * 1000 / total_passengers) if total_passengers else float("nan"),
    })
eff = pd.DataFrame(eff_rows).sort_values("wh_per_passenger", ascending=False)

fig = px.bar(eff, x="line", y="wh_per_passenger", color="line", color_discrete_map=LINE_COLORS,
             labels={"wh_per_passenger": "Wh per passenger", "line": ""})
fig.update_layout(height=400, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

worst = eff.iloc[0]
st.warning(
    f"**{worst['line']}** has the worst energy-per-passenger ratio in this dataset "
    f"(~{worst['wh_per_passenger']:.1f} Wh/passenger). Candidate explanations to investigate: "
    f"lower ridership relative to route length/rolling-stock size, off-peak running with fixed energy draw, "
    f"or longer inter-station distances. Cross-check against the Passenger Flow page for {worst['line']}-served "
    f"stations to see if low ridership (rather than high raw consumption) is the main driver.",
    icon="⚡",
)

st.dataframe(
    eff.rename(columns={"line": "Line", "total_energy_MWh": "Total energy (MWh)",
                         "total_passengers": "Total passengers (approx.)", "wh_per_passenger": "Wh / passenger"}),
    use_container_width=True, hide_index=True,
)
