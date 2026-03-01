"""DriftRetrainer: dispatches online or full-retrain on DriftEvents, applies promotion gate."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.drift.monitor import DriftEvent
from src.models.baseline import evaluate, train_baseline
from src.models.online import OnlineForecaster
from src.models.registry import get_champion_rmse, register_champion

TARGET_COL = "OT"
MODEL_CONFIG = Path("configs/model.yaml")


@dataclass
class PromotionResult:
    """Outcome of a retraining attempt.

    Attributes:
        promoted: True if the challenger beat the champion by ≥ 5 %.
        challenger_rmse: Validation RMSE of the challenger model.
        champion_rmse: Validation RMSE of the current champion (None if no champion exists).
        mode: "online" (Mode A) or "full_retrain" (Mode B).
        run_id: MLflow run_id of the challenger (None for online mode if not logged).
    """

    promoted: bool
    challenger_rmse: float
    champion_rmse: float | None
    mode: str
    run_id: str | None = field(default=None)


class DriftRetrainer:
    """Reacts to DriftEvents by retraining and evaluating a challenger model.

    Two modes (selected by DriftEvent.severity):
    - Mode A  "online"       — 1 detector fired: incremental river update (fast).
    - Mode B  "full_retrain" — ≥ 2 detectors fired: full LightGBM retrain (thorough).

    Promotion gate: ``challenger_rmse < champion_rmse × 0.95`` (5 % improvement).

    Args:
        val_df: Validation DataFrame used for both champion and challenger evaluation.
            Sliced to the last ``val_window_size`` rows at construction time.
        model_name: Registered model name in the MLflow Model Registry.
        config_path: Path to ``configs/model.yaml``.
        experiment_name: MLflow experiment name for full-retrain runs.
    """

    def __init__(
        self,
        val_df: pd.DataFrame,
        model_name: str,
        config_path: Path = MODEL_CONFIG,
        experiment_name: str = "driftpilot",
    ) -> None:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)

        retrain_window = cfg["training"]["retrain_window_size"]
        val_window = cfg["training"]["val_window_size"]

        self._model_name = model_name
        self._experiment_name = experiment_name
        self._val_df = val_df.iloc[-val_window:]
        self._buffer: deque[dict] = deque(maxlen=retrain_window)
        self._online: OnlineForecaster | None = None

    def ingest(self, row: dict) -> None:
        """Buffer every incoming stream row. Call this for each row regardless of drift."""
        self._buffer.append(row)

    def handle(self, event: DriftEvent) -> PromotionResult:
        """Dispatch to Mode A or Mode B based on event severity.

        Args:
            event: The DriftEvent emitted by DriftMonitor.

        Returns:
            PromotionResult describing whether the challenger was promoted.
        """
        print(
            f"[retrainer] DriftEvent at row {event.row_index} | "
            f"detectors={event.triggered_detectors} | severity={event.severity}"
        )
        if event.severity == "high":
            return self._full_retrain()
        return self._online_update()

    @staticmethod
    def should_promote(challenger_rmse: float, champion_rmse: float) -> bool:
        """Return True iff challenger is ≥ 5 % better than champion."""
        return challenger_rmse < champion_rmse * 0.95

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_champion_rmse(self) -> float:
        """Return champion RMSE, or inf if no champion is registered yet."""
        rmse = get_champion_rmse(self._model_name)
        return rmse if rmse is not None else float("inf")

    def _online_update(self) -> PromotionResult:
        """Mode A: incremental HoeffdingAdaptiveTree update on buffered rows."""
        if self._online is None:
            self._online = OnlineForecaster()

        feature_cols = [c for c in self._val_df.columns if c != TARGET_COL]

        # Learn on every buffered row
        for row in self._buffer:
            x = {k: v for k, v in row.items() if k != TARGET_COL}
            y = row[TARGET_COL]
            self._online.learn_one(x, y)

        challenger_rmse = self._online.evaluate_on_df(self._val_df)["rmse"]
        champion_rmse = self._get_champion_rmse()
        promoted = False
        run_id: str | None = None

        if self.should_promote(challenger_rmse, champion_rmse):
            run_id = self._online.log_to_mlflow(
                experiment_name=self._experiment_name,
                metrics={"val_rmse": challenger_rmse},
            )
            register_champion(run_id=run_id, model_name=self._model_name)
            promoted = True
            print(
                f"[retrainer] Mode A — PROMOTED online model "
                f"(RMSE {challenger_rmse:.4f} vs champion {champion_rmse:.4f})"
            )
        else:
            print(
                f"[retrainer] Mode A — rejected (RMSE {challenger_rmse:.4f} "
                f"vs champion {champion_rmse:.4f}, need < {champion_rmse * 0.95:.4f})"
            )

        return PromotionResult(
            promoted=promoted,
            challenger_rmse=challenger_rmse,
            champion_rmse=champion_rmse if champion_rmse != float("inf") else None,
            mode="online",
            run_id=run_id,
        )

    def _full_retrain(self) -> PromotionResult:
        """Mode B: full LightGBM retrain on the rolling data buffer."""
        if not self._buffer:
            print("[retrainer] Mode B — buffer empty, skipping retrain.")
            champion_rmse = self._get_champion_rmse()
            return PromotionResult(
                promoted=False,
                challenger_rmse=float("inf"),
                champion_rmse=champion_rmse if champion_rmse != float("inf") else None,
                mode="full_retrain",
            )

        buffer_df = pd.DataFrame(list(self._buffer))

        # Ensure column types are numeric (dicts from stream may carry mixed types)
        for col in buffer_df.columns:
            buffer_df[col] = pd.to_numeric(buffer_df[col], errors="coerce")
        buffer_df = buffer_df.dropna()

        if buffer_df.empty:
            print("[retrainer] Mode B — buffer produced empty DataFrame after dropna.")
            champion_rmse = self._get_champion_rmse()
            return PromotionResult(
                promoted=False,
                challenger_rmse=float("inf"),
                champion_rmse=champion_rmse if champion_rmse != float("inf") else None,
                mode="full_retrain",
            )

        model, run_id = train_baseline(
            train_df=buffer_df,
            val_df=self._val_df,
            experiment_name=self._experiment_name,
        )

        # Retrieve val_rmse logged by train_baseline
        client = mlflow.MlflowClient()
        challenger_rmse = client.get_run(run_id).data.metrics.get("val_rmse", float("inf"))
        champion_rmse = self._get_champion_rmse()
        promoted = False

        if self.should_promote(challenger_rmse, champion_rmse):
            register_champion(run_id=run_id, model_name=self._model_name)
            promoted = True
            print(
                f"[retrainer] Mode B — PROMOTED LightGBM retrain "
                f"(RMSE {challenger_rmse:.4f} vs champion {champion_rmse:.4f})"
            )
        else:
            print(
                f"[retrainer] Mode B — rejected (RMSE {challenger_rmse:.4f} "
                f"vs champion {champion_rmse:.4f}, need < {champion_rmse * 0.95:.4f})"
            )

        return PromotionResult(
            promoted=promoted,
            challenger_rmse=challenger_rmse,
            champion_rmse=champion_rmse if champion_rmse != float("inf") else None,
            mode="full_retrain",
            run_id=run_id,
        )
