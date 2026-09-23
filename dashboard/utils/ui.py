"""Shared sidebar / dataset-selection chrome used by every page."""
from __future__ import annotations

import streamlit as st

from utils.data_loader import DEFAULT_DATA_DIR, discover_dataset_dirs


def sidebar_dataset_picker() -> str:
    """Renders the dataset picker in the sidebar and returns the chosen
    folder path (as a string, so it stays hashable for st.cache_data).
    """
    st.sidebar.markdown("### 🚇 Dataset")
    datasets = discover_dataset_dirs(DEFAULT_DATA_DIR)
    if not datasets:
        st.sidebar.error(f"No dataset found under `{DEFAULT_DATA_DIR}/`.")
        st.stop()
    labels = list(datasets.keys())
    default_idx = 0
    choice = st.sidebar.selectbox("Active dataset folder", labels, index=default_idx)
    folder = str(datasets[choice])
    st.sidebar.caption(f"`{folder}`")
    st.sidebar.divider()
    return folder


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
