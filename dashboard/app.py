"""Entrypoint / router. Uses st.navigation so the sidebar shows readable page
titles (Streamlit would otherwise label the home page after this file: "app").
Run with `streamlit run dashboard/app.py` (Makefile `make up`, Docker ENTRYPOINT)."""
import streamlit as st

pages = [
    st.Page("home.py", title="Overview", icon="🚇", default=True),
    st.Page("views/1_Network_Explorer.py", title="Network Explorer", icon="🗺️", url_path="network"),
    st.Page("views/2_Passenger_Flow.py", title="Passenger Flow", icon="👥", url_path="flow"),
    st.Page("views/3_Events.py", title="Events", icon="🎟️", url_path="events"),
    st.Page("views/4_Weather.py", title="Weather", icon="🌦️", url_path="weather"),
    st.Page("views/5_Closures.py", title="Closures", icon="🚧", url_path="closures"),
    st.Page("views/6_Energy.py", title="Energy", icon="⚡", url_path="energy"),
    st.Page("views/7_ML_Engine.py", title="ML Engine (TabPFN)", icon="🤖", url_path="ml"),
    st.Page("views/8_Observability.py", title="Observability", icon="🔭", url_path="observability"),
    st.Page("views/9_Evaluation.py", title="Evaluation", icon="✅", url_path="evaluation"),
    st.Page("views/10_Experiments.py", title="Experiments", icon="🧪", url_path="experiments"),
    st.Page("views/11_Agent_Workflow.py", title="Agent Workflow", icon="🧭", url_path="workflow"),
    st.Page("views/12_Resources.py", title="Resources", icon="🧰", url_path="resources"),
]
st.navigation(pages).run()
