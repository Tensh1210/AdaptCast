"""Drift count gauge + alert banner component."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_drift_gauge(drift_status: dict) -> None:
    """Render a Plotly gauge for drift_count and summary metrics.

    Args:
        drift_status: Dict from ``fetch_drift_status()``.
                      Expected keys: drift_count, row_index, drift_detected.
    """
    drift_count: int = drift_status.get("drift_count", 0)
    row_index: int = drift_status.get("row_index", 0)

    # Alert banner
    if drift_count > 0:
        st.warning(
            f"**Drift detected!** {drift_count} drift event(s) recorded so far."
        )
    else:
        st.success("No drift detected.")

    # Gauge axis max: at least 10, or 2× drift_count when that's larger
    axis_max = max(10, drift_count * 2)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=drift_count,
            title={"text": "Drift Events", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, axis_max], "tickwidth": 1},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 2], "color": "#2ca02c"},       # green
                    {"range": [2, 10], "color": "#ff7f0e"},      # amber
                    {"range": [10, axis_max], "color": "#d62728"},  # red
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": drift_count,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col_a, col_b = st.columns(2)
    col_a.metric("Rows Processed", f"{row_index:,}")
    col_b.metric("Drift Events", drift_count)
