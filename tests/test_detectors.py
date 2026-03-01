"""Tests for src/drift/detectors.py — synthetic stream injection, no real data."""
from __future__ import annotations

import numpy as np
import pytest

from src.drift.detectors import (
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
    build_detectors,
)

RNG = np.random.default_rng(42)


def test_adwin_detects_sudden_shift():
    """ADWIN must fire within 80 steps after a sharp mean shift."""
    detector = ADWINDetector(delta=0.002)

    # Stable phase: 300 samples from N(0, 1)
    for val in RNG.normal(0, 1, 300):
        detector.update(abs(val))

    # Shift phase: 300 samples from N(5, 1)
    detected_at: int | None = None
    for i, val in enumerate(RNG.normal(5, 1, 300)):
        if detector.update(abs(val)):
            detected_at = i
            break

    assert detected_at is not None, "ADWIN failed to detect the sudden shift"
    assert detected_at < 80, (
        f"ADWIN detected drift at step {detected_at} — expected < 80 after shift"
    )

    # Reset must clear the drift flag
    detector.reset()
    assert not detector.update(abs(float(RNG.normal(5, 1)))), (
        "After reset(), next update should not immediately report drift"
    )


def test_page_hinkley_detects_upward_drift():
    """PageHinkley must detect a linearly increasing residual stream."""
    detector = PageHinkleyDetector(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)

    # Warm-up: 50 stable samples
    for val in RNG.normal(0, 1, 50):
        detector.update(abs(val))

    # Gradual ramp: 0 → 5 over 200 steps
    ramp = np.linspace(0, 5, 200)
    detected = False
    for val in ramp:
        if detector.update(abs(val)):
            detected = True
            break

    assert detected, "PageHinkley failed to detect the gradual upward ramp"


def test_kswin_detects_distribution_change():
    """KSWIN must detect a distribution shift on raw residuals.

    The stable fill phase uses a constant (0.0) rather than random samples.
    A constant stream gives KS statistic = 0 on every comparison (both KS windows
    are identical), so no false alarm can fire and reset the reference pool during fill.
    The shifted phase uses a 5-sigma step (N(5,1) from baseline 0) — within ~30
    samples the recent window (stat_size=30) is all-shifted while the reference
    draws from the all-zero baseline, making KS detection certain.
    """
    detector = KSWINDetector(alpha=0.005, window_size=100, stat_size=30)

    # Constant fill: KS stat = 0 on every step → guaranteed no false alarms.
    for _ in range(150):
        detector.update(0.0)

    # Shifted phase: N(5, 1) is a 5-sigma step; detection expected within ~30 rows.
    detected = False
    for val in RNG.normal(5, 1, 200):
        if detector.update(float(val)):
            detected = True
            break

    assert detected, "KSWIN failed to detect the distribution shift 0 → N(5, 1)"


def test_no_false_alarm_stable_stream():
    """No detector should fire on a perfectly constant stream.

    A random stream cannot be used for KSWIN: with window_size=100 and stat_size=30,
    ~16 KS tests are run over 500 samples, giving a ~7.7 % chance of a false alarm at
    alpha=0.005 regardless of the random seed. A constant stream is deterministic:
    both KSWIN windows are identical → KS statistic = 0 → p-value = 1.0 → no alarm.
    ADWIN and PageHinkley also cannot fire on a constant stream (zero deviation).
    """
    adwin = ADWINDetector(delta=0.002)
    ph = PageHinkleyDetector(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)
    kswin = KSWINDetector(alpha=0.005, window_size=100, stat_size=30)

    # Constant stream: identical distributions in every window → no detector fires.
    constant = np.full(500, 0.5)
    for val in constant:
        assert not adwin.update(abs(float(val))), (
            "ADWIN fired a false alarm on a constant stream"
        )
        assert not ph.update(abs(float(val))), (
            "PageHinkley fired a false alarm on a constant stream"
        )
        assert not kswin.update(float(val)), (
            "KSWIN fired a false alarm on a constant stream"
        )


def test_build_detectors_returns_three():
    """build_detectors() must return exactly 3 correctly typed detectors."""
    detectors = build_detectors()

    assert len(detectors) == 3, f"Expected 3 detectors, got {len(detectors)}"
    assert isinstance(detectors[0], ADWINDetector), "First detector must be ADWINDetector"
    assert isinstance(detectors[1], PageHinkleyDetector), (
        "Second detector must be PageHinkleyDetector"
    )
    assert isinstance(detectors[2], KSWINDetector), "Third detector must be KSWINDetector"

    for det in detectors:
        assert hasattr(det, "update"), f"{det.name} missing .update()"
        assert hasattr(det, "reset"), f"{det.name} missing .reset()"
        assert hasattr(det, "name"), f"{type(det).__name__} missing .name"
