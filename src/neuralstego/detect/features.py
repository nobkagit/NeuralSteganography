"""Feature extraction utilities for the detection guard."""

from __future__ import annotations

from typing import Dict

EXPECTED_FEATURES = (
    "ppl",
    "avg_nll",
    "avg_entropy",
    "ngram_repeat_ratio",
    "type_token_ratio",
    "avg_sentence_len",
)


def extract_features(metrics: Dict[str, float]) -> Dict[str, float]:
    """Extract a consistent feature mapping from collected metrics."""

    return {name: float(metrics.get(name, 0.0)) for name in EXPECTED_FEATURES}
