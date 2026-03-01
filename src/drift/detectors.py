"""Thin wrappers around river drift detectors with a uniform interface.

Each wrapper exposes:
  update(value: float) -> bool   — feed one observation; True if drift detected
  reset() -> None                — reinitialise the underlying river object
  name: str                      — detector identifier for logging
"""
from __future__ import annotations

from pathlib import Path

import yaml
from river.drift import ADWIN, KSWIN, PageHinkley


class ADWINDetector:
    """ADWIN — Adaptive Windowing drift detector (Bifet & Gavaldà, 2007).

    Feed absolute residuals. O(log n) time and space.
    """

    name: str = "ADWIN"

    def __init__(self, delta: float = 0.002) -> None:
        self._delta = delta
        self._detector = ADWIN(delta=delta)

    def update(self, value: float) -> bool:
        """Update detector with one observation. Returns True if drift detected."""
        self._detector.update(value)
        return self._detector.drift_detected

    def reset(self) -> None:
        """Reinitialise — river has no native reset; must reinstantiate."""
        self._detector = ADWIN(delta=self._delta)


class PageHinkleyDetector:
    """Page-Hinkley cumulative-sum drift detector (Page, 1954).

    Feed absolute residuals. Sensitive to gradual upward drift.
    """

    name: str = "PageHinkley"

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50,
        alpha: float = 0.9999,
    ) -> None:
        self._params = dict(
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha,
        )
        self._detector = PageHinkley(**self._params)

    def update(self, value: float) -> bool:
        self._detector.update(value)
        return self._detector.drift_detected

    def reset(self) -> None:
        self._detector = PageHinkley(**self._params)


class KSWINDetector:
    """KSWIN — Kolmogorov-Smirnov Windowed drift detector (Raab et al., 2020).

    Feed raw (signed) residuals. Non-parametric; two-tailed.
    """

    name: str = "KSWIN"

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
    ) -> None:
        self._params = dict(alpha=alpha, window_size=window_size, stat_size=stat_size)
        self._detector = KSWIN(**self._params)

    def update(self, value: float) -> bool:
        self._detector.update(value)
        return self._detector.drift_detected

    def reset(self) -> None:
        self._detector = KSWIN(**self._params)


def build_detectors(
    config_path: Path = Path("configs/drift.yaml"),
) -> list[ADWINDetector | PageHinkleyDetector | KSWINDetector]:
    """Load drift.yaml and return one initialised instance of each detector.

    Returns:
        [ADWINDetector, PageHinkleyDetector, KSWINDetector]
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    return [
        ADWINDetector(**cfg["adwin"]),
        PageHinkleyDetector(**cfg["page_hinkley"]),
        KSWINDetector(**cfg["kswin"]),
    ]
