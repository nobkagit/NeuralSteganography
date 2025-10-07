"""Quality-control helpers for GPT2-fa steganographic pipelines."""

from __future__ import annotations

import math

import numpy as np

from ..codec.errors import QualityConfigError
from ..codec.types import ProbDist

__all__ = ["apply_quality"]


def apply_quality(
    dist: ProbDist,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float = 1.0,
) -> ProbDist:
    """Apply quality constraints to a language-model distribution.

    Parameters
    ----------
    dist:
        Probability distribution emitted by the base language model.
    top_k:
        Retain only the *k* most-likely tokens when provided.
    top_p:
        Retain the smallest prefix whose cumulative probability exceeds the
        specified threshold (also known as nucleus sampling).
    temperature:
        Soften or sharpen the distribution prior to filtering.  Values below
        ``1.0`` make the distribution more peaky while values above ``1.0``
        increase diversity.  The parameter must be strictly positive.

    Returns
    -------
    ProbDist
        A filtered and re-normalised distribution matching the input type.

    Raises
    ------
    QualityConfigError
        If the supplied parameters would eliminate all probability mass or if
        they violate their respective domains.
    """

    if temperature <= 0.0:
        raise QualityConfigError("temperature must be positive")

    tokens, probs = _dist_to_arrays(dist)
    probs = _normalise(probs)

    if not math.isclose(temperature, 1.0):
        logits = np.log(probs + 1e-12)
        scaled = logits / float(temperature)
        scaled -= float(np.max(scaled))
        probs = np.exp(scaled)
        probs = _normalise(probs)

    mask = np.ones_like(probs, dtype=bool)

    if top_k is not None:
        if top_k <= 0:
            raise QualityConfigError("top_k must be positive")
        order = np.argsort(probs)[::-1]
        keep = order[: min(top_k, probs.size)]
        mask = np.zeros_like(mask)
        mask[keep] = True

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise QualityConfigError("top_p must lie within (0, 1]")
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        cutoff = np.searchsorted(cumulative, top_p, side="left")
        nucleus = order[: cutoff + 1]
        nucleus_mask = np.zeros_like(mask)
        nucleus_mask[nucleus] = True
        mask &= nucleus_mask

    filtered = np.where(mask, probs, 0.0)
    if not np.any(filtered):
        raise QualityConfigError("Quality constraints removed all probability mass")

    filtered = _normalise(filtered)
    return _arrays_to_dist(tokens, filtered, dist)


def _dist_to_arrays(dist: ProbDist) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(dist, np.ndarray):
        probs = dist.astype(np.float64, copy=True)
        tokens = np.arange(probs.size, dtype=np.int64)
        return tokens, probs
    if isinstance(dist, dict):
        items = sorted(dist.items())
        tokens = np.array([int(token) for token, _ in items], dtype=np.int64)
        probs = np.array([float(prob) for _, prob in items], dtype=np.float64)
        return tokens, probs
    raise TypeError(f"Unsupported distribution type: {type(dist)!r}")


def _arrays_to_dist(
    tokens: np.ndarray,
    probs: np.ndarray,
    original: ProbDist,
) -> ProbDist:
    if isinstance(original, np.ndarray):
        result = np.zeros_like(original, dtype=np.float64)
        result[tokens] = probs
        return result
    mapping = {
        int(token): float(prob)
        for token, prob in zip(tokens.tolist(), probs.tolist())
        if prob > 0.0
    }
    return mapping


def _normalise(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise QualityConfigError("Probability mass vanished during normalisation")
    return values / total

