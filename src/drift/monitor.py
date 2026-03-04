"""Drift monitor: feeds prediction residuals to all detectors and emits DriftEvents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlflow

from src.drift.detectors import build_detectors


@dataclass
class DriftEvent:
    """Emitted by DriftMonitor when one or more detectors trigger.

    Attributes:
        row_index: Stream position (0-based) at which drift was detected.
        triggered_detectors: Names of detectors that fired (e.g. ["ADWIN"]).
        residual: Raw residual (y_true - y_pred) at the triggering row.
        severity: "low" if 1 detector fired; "high" if ≥ 2 fired.
    """

    row_index: int
    triggered_detectors: list[str]
    residual: float
    severity: str = field(init=False)

    def __post_init__(self) -> None:
        self.severity = "high" if len(self.triggered_detectors) >= 2 else "low"


class DriftMonitor:
    """Orchestrates ADWIN, PageHinkley, and KSWIN drift detectors in parallel.

    Feed one (y_pred, y_true) pair per streaming row via ``update()``.
    Receive a ``DriftEvent`` whenever any detector fires.

    Args:
        config_path: Path to ``configs/drift.yaml``.
        mlflow_run_id: Optional active MLflow run ID. When provided, each
            ``DriftEvent`` is logged as a metric step to that run.
    """

    # Detectors that receive abs(residual)
    _ABS_DETECTORS = {"ADWIN", "PageHinkley"}

    def __init__(
        self,
        config_path: Path = Path("configs/drift.yaml"),
        mlflow_run_id: str | None = None,
    ) -> None:
        self._detectors = build_detectors(config_path)
        self._row_index: int = 0
        self._drift_count: int = 0
        self._mlflow_run_id = mlflow_run_id

    @property
    def row_index(self) -> int:
        return self._row_index

    @property
    def drift_count(self) -> int:
        return self._drift_count

    def update(self, y_pred: float, y_true: float) -> DriftEvent | None:
        """Feed one prediction/truth pair to all detectors.

        Args:
            y_pred: Model prediction for this row.
            y_true: Ground-truth target value for this row.

        Returns:
            A ``DriftEvent`` if ≥ 1 detector fired, otherwise ``None``.
            All fired detectors are reset immediately after the event is emitted.
        """
        residual = float(y_true) - float(y_pred)
        abs_residual = abs(residual)

        triggered: list[str] = []
        for det in self._detectors:
            value = abs_residual if det.name in self._ABS_DETECTORS else residual
            if det.update(value):
                triggered.append(det.name)

        self._row_index += 1

        if not triggered:
            return None

        # Reset every detector that fired
        for det in self._detectors:
            if det.name in triggered:
                det.reset()

        self._drift_count += 1
        event = DriftEvent(
            row_index=self._row_index - 1,
            triggered_detectors=triggered,
            residual=residual,
        )

        self._log_to_mlflow(event)
        return event

    def reset(self) -> None:
        """Reset all detectors and clear row/drift counters."""
        for det in self._detectors:
            det.reset()
        self._row_index = 0
        self._drift_count = 0

    def _log_to_mlflow(self, event: DriftEvent) -> None:
        if self._mlflow_run_id is None:
            return
        with mlflow.start_run(run_id=self._mlflow_run_id):
            mlflow.log_metric("drift_detected", 1.0, step=event.row_index)
            mlflow.log_metric(
                "drift_severity",
                2.0 if event.severity == "high" else 1.0,
                step=event.row_index,
            )
