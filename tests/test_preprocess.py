"""Tests for src/data/preprocess.py."""
from __future__ import annotations

import pytest
import pandas as pd

from src.data.download import download_data
from src.data.preprocess import (
    LAGS,
    ROLLING_WINDOWS,
    TARGET_COL,
    run_preprocessing,
)

EXPECTED_LAG_COLS = [f"{TARGET_COL}_lag_{lag}" for lag in LAGS]
EXPECTED_ROLLING_COLS = [
    f"{TARGET_COL}_roll_{stat}_{window}"
    for window in ROLLING_WINDOWS
    for stat in ("mean", "std")
]
EXPECTED_FEATURE_COLS = EXPECTED_LAG_COLS + EXPECTED_ROLLING_COLS


@pytest.fixture(scope="session")
def splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download raw data and run preprocessing once for the whole test session."""
    download_data()
    return run_preprocessing()


def test_feature_columns_present(splits):
    train_df, _, _ = splits
    for col in EXPECTED_FEATURE_COLS:
        assert col in train_df.columns, f"Missing expected feature column: {col}"


def test_no_nan_after_drop(splits):
    for name, df in zip(("train", "val", "test"), splits):
        nan_count = df.isna().sum().sum()
        assert nan_count == 0, f"{name} split has {nan_count} NaN values"


def test_split_ratios(splits):
    train_df, val_df, test_df = splits
    total = len(train_df) + len(val_df) + len(test_df)

    train_ratio = len(train_df) / total
    val_ratio = len(val_df) / total
    test_ratio = len(test_df) / total

    assert abs(train_ratio - 0.70) <= 0.01, (
        f"Train ratio {train_ratio:.3f} outside ±1% of 0.70"
    )
    assert abs(val_ratio - 0.15) <= 0.01, (
        f"Val ratio {val_ratio:.3f} outside ±1% of 0.15"
    )
    assert abs(test_ratio - 0.15) <= 0.01, (
        f"Test ratio {test_ratio:.3f} outside ±1% of 0.15"
    )


def test_split_no_overlap(splits):
    train_df, val_df, test_df = splits
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert train_idx.isdisjoint(val_idx), "Train and val index sets overlap"
    assert train_idx.isdisjoint(test_idx), "Train and test index sets overlap"
    assert val_idx.isdisjoint(test_idx), "Val and test index sets overlap"
