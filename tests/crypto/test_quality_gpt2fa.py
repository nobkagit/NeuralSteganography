"""Tests for quality policies applied to GPT2-fa style distributions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cryptography")

from neuralstego.codec.errors import QualityConfigError
from neuralstego.crypto.quality import apply_quality


def _entropy(probs: np.ndarray) -> float:
    mask = probs > 0.0
    values = probs[mask]
    return float(-(values * np.log2(values)).sum())


def test_apply_quality_with_top_k_restricts_support() -> None:
    probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    filtered = apply_quality(probs, top_k=2)

    assert filtered.sum() == pytest.approx(1.0)
    assert np.count_nonzero(filtered) == 2


def test_apply_quality_with_top_p_limits_mass() -> None:
    probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05], dtype=np.float64)
    filtered = apply_quality(probs, top_p=0.7)

    # The filtered distribution should only contain the minimal set of tokens
    # whose mass exceeds the nucleus threshold.
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    cutoff = np.searchsorted(cumulative, 0.7, side="left")
    expected_support = set(order[: cutoff + 1])
    actual_support = {index for index, value in enumerate(filtered) if value > 0.0}

    assert actual_support == expected_support


def test_temperature_controls_entropy() -> None:
    probs = np.array([0.05, 0.2, 0.25, 0.3, 0.2], dtype=np.float64)

    cold = apply_quality(probs, temperature=0.5)
    warm = apply_quality(probs, temperature=1.5)

    assert _entropy(cold) < _entropy(warm)


def test_invalid_configuration_raises() -> None:
    probs = np.array([0.6, 0.4], dtype=np.float64)

    with pytest.raises(QualityConfigError):
        apply_quality(probs, top_k=0)

    with pytest.raises(QualityConfigError):
        apply_quality(probs, temperature=0.0)
