"""Feature engineering and train/val/test split for ETTh1."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/ETTh1.csv")
PROCESSED_DIR = Path("data/processed")

LAGS = [1, 24, 168]
ROLLING_WINDOWS = [24, 168]
TARGET_COL = "OT"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features; drop NaN rows introduced by lag-168."""
    df = df.copy()

    # Lag features
    for lag in LAGS:
        df[f"{TARGET_COL}_lag_{lag}"] = df[TARGET_COL].shift(lag)

    # Rolling mean and std
    for window in ROLLING_WINDOWS:
        df[f"{TARGET_COL}_roll_mean_{window}"] = (
            df[TARGET_COL].shift(1).rolling(window).mean()
        )
        df[f"{TARGET_COL}_roll_std_{window}"] = (
            df[TARGET_COL].shift(1).rolling(window).std()
        )

    # Calendar features (LightGBM handles ordinal encoding natively)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Drop the first 168 rows where lag-168 is NaN
    df = df.iloc[168:].copy()
    df = df.dropna()

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV, engineer features, split, and save Parquet files.

    Returns:
        (train_df, val_df, test_df) — DataFrames with features and target.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")
    df = df.sort_index()

    df = _engineer_features(df)

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
