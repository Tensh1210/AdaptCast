"""Feature engineering and train/val/test split for ETTh1."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_PATH = _PROJECT_ROOT / "data/raw/ETTh1.csv"
PROCESSED_DIR = _PROJECT_ROOT / "data/processed"
_CONFIG_PATH = _PROJECT_ROOT / "configs/model.yaml"

# Module-level defaults (overridden at runtime from configs/model.yaml)
LAGS = [1, 24, 168]
ROLLING_WINDOWS = [24, 168]
TARGET_COL = "OT"


def _engineer_features(
    df: pd.DataFrame,
    lags: list[int] = LAGS,
    rolling_windows: list[int] = ROLLING_WINDOWS,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Add lag, rolling, and calendar features; drop NaN rows introduced by lag-168."""
    df = df.copy()

    # Lag features
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # Rolling mean and std (shift(1) prevents look-ahead leakage)
    for window in rolling_windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df[target_col].shift(1).rolling(window).mean()
        )
        df[f"{target_col}_roll_std_{window}"] = (
            df[target_col].shift(1).rolling(window).std()
        )

    # Calendar features (LightGBM handles ordinal encoding natively)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Drop the first max(lags) rows where the longest lag is NaN
    df = df.iloc[max(lags):].copy()

    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"[preprocess] Dropped {rows_dropped} additional rows with NaN after feature engineering.")

    return df


def _chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically without shuffling."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def run_preprocessing(
    raw_path: Path = RAW_PATH,
    out_dir: Path = PROCESSED_DIR,
    config_path: Path = _CONFIG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV, engineer features, split, and save Parquet files.

    Returns:
        (train_df, val_df, test_df) — DataFrames with features and target.
    """
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load feature config; fall back to module defaults if config is unavailable
    lags, rolling_windows, target_col = LAGS, ROLLING_WINDOWS, TARGET_COL
    if Path(config_path).exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        feat = cfg.get("features", {})
        lags = feat.get("lags", LAGS)
        rolling_windows = feat.get("rolling_windows", ROLLING_WINDOWS)
        target_col = feat.get("target_col", TARGET_COL)

    try:
        df = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"[preprocess] Failed to parse {raw_path}: {exc}. "
            "Ensure the CSV has a 'date' column."
        ) from exc

    if target_col not in df.columns:
        raise ValueError(
            f"[preprocess] Target column '{target_col}' not found in {raw_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.sort_index()

    df = _engineer_features(df, lags=lags, rolling_windows=rolling_windows, target_col=target_col)

    train_df, val_df, test_df = _chronological_split(df)

    train_df.to_parquet(out_dir / "train.parquet")
    val_df.to_parquet(out_dir / "val.parquet")
    test_df.to_parquet(out_dir / "test.parquet")

    print(
        f"[preprocess] Splits saved to {out_dir}/\n"
        f"  train: {len(train_df):,} rows\n"
        f"  val:   {len(val_df):,} rows\n"
        f"  test:  {len(test_df):,} rows"
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    run_preprocessing()
