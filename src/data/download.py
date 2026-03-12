"""Download ETTh1.csv from GitHub if not already present."""
from __future__ import annotations

from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEST = _PROJECT_ROOT / "data/raw/ETTh1.csv"


def download_data(url: str = URL, dest: Path = DEST) -> Path:
    """Fetch the ETTh1 dataset; skip download if the file already exists.

    Args:
        url: Remote URL of the CSV file.
        dest: Local destination path.

    Returns:
        Path to the downloaded (or pre-existing) file.
    """
    dest = Path(dest)
    if dest.exists():
        print(f"[download] {dest} already exists — skipping download.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching {url} …")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)
    except Exception:
        # Remove partial file so the next run attempts a fresh download
        if dest.exists():
            dest.unlink()
        raise

    print(f"[download] Saved to {dest} ({dest.stat().st_size / 1_024:.1f} KiB).")
    return dest


if __name__ == "__main__":
    download_data()
