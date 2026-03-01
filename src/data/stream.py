"""Row-by-row generator that simulates a live data stream from the test split."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

import pandas as pd

TEST_PATH = Path("data/processed/test.parquet")


def stream_test_data(
    path: Path = TEST_PATH,
    delay_seconds: float = 0.0,
) -> Generator[dict, None, None]:
    """Yield one row at a time from the test Parquet file.

    Args:
        path: Path to the test Parquet file.
        delay_seconds: Seconds to sleep between rows.
                       Use 0.0 for unit tests; 0.1 for live demo mode.

    Yields:
        A dict representation of each row (column → value).
    """
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        yield row.to_dict()
        if delay_seconds > 0.0:
            time.sleep(delay_seconds)
