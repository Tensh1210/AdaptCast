"""Actual vs predicted line chart with drift event markers."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_prediction_chart(
    history: list[dict],
    drift_status: dict,
) -> None:
    """Render a Plotly chart of recent actual vs predicted values.

    Args:
        history: List of prediction point dicts from ``fetch_prediction_history()``.
                 Each dict has keys: row_index, prediction, actual.
        drift_status: Drift status dict from ``fetch_drift_status()``.
                      Used to mark the last drift row.
    """
    if not history:
        st.info("No predictions yet. Start the stream to see live forecasts.")
        return

    # Split into paired (actual known) and predict-only points
    paired = [p for p in history if p.get("actual") is not None]
    if not paired:
        st.info("Waiting for ground-truth values from the stream.")
        return

    xs = [p["row_index"] for p in paired]
    actuals = [p["actual"] for p in paired]
    preds = [p["prediction"] for p in paired]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=actuals,
            mode="lines",
            name="Actual",
            line=dict(color="#4C72B0", width=1.5),
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=preds,
            mode="lines",
            name="Predicted",
            line=dict(color="#DD8452", width=1.5, dash="dot"),
            opacity=0.85,
        )
    )

    # Mark last drift event with a vertical line
    last_drift_row = drift_status.get("last_drift_row")
    if last_drift_row is not None and xs[0] <= last_drift_row <= xs[-1]:
        fig.add_vline(
            x=last_drift_row,
            line_width=2,
            line_dash="dash",
            line_color="#d62728",
            annotation_text="Drift",
            annotation_position="top right",
            annotation_font_color="#d62728",
        )

    fig.update_layout(
        xaxis=dict(title="Stream Row"),
        yaxis=dict(title="OT (Oil Temperature)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    st.plotly_chart(fig, width="stretch")
