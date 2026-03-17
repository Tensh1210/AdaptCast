"""RMSE history line chart component."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_forecast_chart(runs: list[dict], model_versions: list[dict]) -> None:
    """Render a Plotly line chart of val_rmse across MLflow runs.

    Args:
        runs: List of run dicts from ``fetch_mlflow_runs()``.
              Each dict has keys: run_id, start_time, val_rmse, status.
        model_versions: List of version dicts from ``fetch_model_versions()``.
                        Used to identify the true champion run.
    """
    valid = [r for r in runs if r.get("val_rmse") is not None]

    if not valid:
        st.info("No model runs found.")
        return

    df = pd.DataFrame(valid).reset_index(drop=True)
    df["run_index"] = range(1, len(df) + 1)

    x_labels = [f"Run {i}" for i in df["run_index"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["run_index"],
            y=df["val_rmse"],
            mode="lines+markers",
            name="val_rmse",
            line=dict(color="#4C72B0", width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{text}</b><br>RMSE: %{y:.4f}<extra></extra>",
            text=[f"{label} ({rid})" for label, rid in zip(x_labels, df["run_id"])],
        )
    )

    # Find champion run_id from model_versions; fall back to last run
    champion_run_id = next(
        (v.get("run_id") for v in model_versions if v.get("is_champion")),
        None,
    )
    champion_mask = df["run_id"] == champion_run_id if champion_run_id else pd.Series(False, index=df.index)
    champion_rows = df[champion_mask]
    champ = champion_rows.iloc[0] if not champion_rows.empty else df.iloc[-1]

    fig.add_annotation(
        x=champ["run_index"],
        y=champ["val_rmse"],
        text="Champion",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        font=dict(color="#2ca02c", size=12),
        arrowcolor="#2ca02c",
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=df["run_index"].tolist(),
            ticktext=x_labels,
            title="Run",
        ),
        yaxis=dict(title="Validation RMSE"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    st.plotly_chart(fig, width="stretch")
