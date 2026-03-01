"""Tests for DriftRetrainer promotion gate, mode selection, and OnlineForecaster."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.drift.monitor import DriftEvent
from src.drift.retrainer import DriftRetrainer, PromotionResult
from src.models.online import OnlineForecaster

# ------------------------------------------------------------------
# Feature schema matching preprocess.py output
# ------------------------------------------------------------------
FEATURE_COLS = [
    "OT_lag_1", "OT_lag_24", "OT_lag_168",
    "OT_roll_mean_24", "OT_roll_std_24",
    "OT_roll_mean_168", "OT_roll_std_168",
    "hour", "dayofweek", "month",
]
TARGET_COL = "OT"
ALL_COLS = FEATURE_COLS + [TARGET_COL]

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def mlflow_uri(tmp_path_factory) -> str:
    db = tmp_path_factory.mktemp("mlruns") / "mlflow.db"
    uri = f"sqlite:///{db.as_posix()}"
    mlflow.set_tracking_uri(uri)
    return uri


@pytest.fixture(scope="module")
def synthetic_val_df() -> pd.DataFrame:
    """50-row DataFrame with all required feature + target columns."""
    n = 50
    data = {col: RNG.normal(0, 1, n) for col in FEATURE_COLS}
    # OT follows a simple linear combination for predictability
    data[TARGET_COL] = data["OT_lag_1"] * 0.8 + RNG.normal(0, 0.1, n)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def retrainer(synthetic_val_df, mlflow_uri) -> DriftRetrainer:
    return DriftRetrainer(
        val_df=synthetic_val_df,
        model_name="test-driftpilot-forecaster",
        experiment_name="test-retrainer",
    )


def _make_event(severity_detectors: list[str]) -> DriftEvent:
    """Helper: build a DriftEvent with given triggered detectors."""
    event = DriftEvent.__new__(DriftEvent)
    event.row_index = 100
    event.triggered_detectors = severity_detectors
    event.residual = 0.5
    event.__post_init__()
    return event


# ------------------------------------------------------------------
# Promotion gate (pure logic — no training needed)
# ------------------------------------------------------------------

def test_should_promote_true():
    """challenger 0.50 < champion 0.80 × 0.95 = 0.76 → promote."""
    assert DriftRetrainer.should_promote(0.50, 0.80) is True


def test_should_promote_false_marginal():
    """challenger 0.77 > 0.76 threshold → reject."""
    assert DriftRetrainer.should_promote(0.77, 0.80) is False


def test_should_promote_false_equal():
    """Equal RMSE does not meet the strict < condition."""
    assert DriftRetrainer.should_promote(0.80, 0.80) is False


# ------------------------------------------------------------------
# Mode selection
# ------------------------------------------------------------------

def test_mode_selection_low_severity(retrainer):
    """Single-detector DriftEvent → Mode A (online)."""
    event = _make_event(["ADWIN"])
    assert event.severity == "low"

    stub_result = PromotionResult(
        promoted=False, challenger_rmse=1.0, champion_rmse=None, mode="online"
    )
    with patch.object(retrainer, "_online_update", return_value=stub_result) as mock:
        result = retrainer.handle(event)
        mock.assert_called_once()

    assert result.mode == "online"


def test_mode_selection_high_severity(retrainer):
    """Two-detector DriftEvent → Mode B (full_retrain)."""
    event = _make_event(["ADWIN", "PageHinkley"])
    assert event.severity == "high"

    stub_result = PromotionResult(
        promoted=False, challenger_rmse=1.0, champion_rmse=None, mode="full_retrain"
    )
    with patch.object(retrainer, "_full_retrain", return_value=stub_result) as mock:
        result = retrainer.handle(event)
        mock.assert_called_once()

    assert result.mode == "full_retrain"


# ------------------------------------------------------------------
# OnlineForecaster
# ------------------------------------------------------------------

def test_online_forecaster_learns_and_predicts():
    """learn_one then predict_one must return a float."""
    forecaster = OnlineForecaster(grace_period=10)
    x = {col: float(RNG.normal()) for col in FEATURE_COLS}
    y = float(RNG.normal())

    for _ in range(50):
        forecaster.learn_one(x, y)

    pred = forecaster.predict_one(x)
    assert isinstance(pred, float), f"predict_one returned {type(pred)}, expected float"
    assert forecaster.rows_seen == 50


def test_online_forecaster_evaluate_returns_metrics(synthetic_val_df):
    """evaluate_on_df must return rmse/mae/r2 with rmse ≥ 0."""
    forecaster = OnlineForecaster(grace_period=10)

    # Warm up on 100 synthetic rows
    for _ in range(100):
        x = {col: float(RNG.normal()) for col in FEATURE_COLS}
        y = float(RNG.normal())
        forecaster.learn_one(x, y)

    metrics = forecaster.evaluate_on_df(synthetic_val_df, target_col=TARGET_COL)

    assert "rmse" in metrics, "evaluate_on_df missing 'rmse'"
    assert "mae" in metrics, "evaluate_on_df missing 'mae'"
    assert "r2" in metrics, "evaluate_on_df missing 'r2'"
    assert metrics["rmse"] >= 0, f"RMSE should be ≥ 0, got {metrics['rmse']}"
