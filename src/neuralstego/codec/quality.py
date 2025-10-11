"""Quality and capacity policies for arithmetic steganography."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .errors import QualityConfigError
from .types import ProbDist


class QualityPolicy(Protocol):
    """Protocol for capacity-quality balancing policies."""

    def validate(self) -> None:
        """Validate the configuration."""


@dataclass
class TopKPolicy:
    """Policy constraining sampling to the top-k tokens."""

    k: int

    def validate(self) -> None:
        if self.k <= 0:
            raise QualityConfigError("k must be positive for TopKPolicy")


@dataclass
class TopPPolicy:
    """Policy constraining sampling to a probability mass threshold."""

    p: float

    def validate(self) -> None:
        if not 0 < self.p <= 1:
            raise QualityConfigError("p must be within (0, 1] for TopPPolicy")


@dataclass
class CapacityPerTokenPolicy:
    """Policy limiting the number of embedded bits per token."""

    max_bits: int

    def validate(self) -> None:
        if self.max_bits <= 0:
            raise QualityConfigError(
                "max_bits must be positive for CapacityPerTokenPolicy",
            )


def apply_quality(
    dist: ProbDist,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    min_prob: float | None = None,
) -> ProbDist:
    """Apply quality policies to a probability distribution.

    The policies progressively filter the tail of the distribution prior to
    renormalisation.  ``top_k`` keeps the *k* most likely tokens, ``top_p``
    retains the minimum set of tokens whose cumulative probability reaches the
    specified mass, and ``min_prob`` discards tokens whose probability falls
    below the threshold.  The function always returns a distribution of the
    same type as the input.
    """

    tokens, probs = _dist_to_arrays(dist)

    if top_k is not None:
        if top_k <= 0:
            raise QualityConfigError("top_k must be positive")
        keep_mask = np.zeros_like(probs, dtype=bool)
        order = np.argsort(probs)[::-1]
        keep_mask[order[: min(top_k, probs.size)]] = True
    else:
        keep_mask = np.ones_like(probs, dtype=bool)

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise QualityConfigError("top_p must be within (0, 1]")
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        cutoff = np.searchsorted(cumulative, top_p, side="left")
        keep_mask &= np.isin(np.arange(probs.size), order[: cutoff + 1])

    if min_prob is not None:
        if min_prob < 0:
            raise QualityConfigError("min_prob must be non-negative")
        keep_mask &= probs >= min_prob

    if not np.any(keep_mask):
        raise QualityConfigError("Quality policies removed all probability mass")

    filtered = np.zeros_like(probs)
    filtered[keep_mask] = probs[keep_mask]
    filtered = _normalise(filtered)

    return _arrays_to_dist(tokens, filtered, dist)


def cap_bits_per_token(dist: ProbDist, cap_per_token_bits: int) -> ProbDist:
    """Approximate capacity control by lowering entropy with temperature scaling.

    If the target capacity is greater than or equal to the current entropy the
    distribution is returned unchanged.  Otherwise an optimisation over the
    temperature parameter ``tau`` (``0 < tau ≤ 1``) sharpens the distribution to
    reduce entropy until it is at or below the desired threshold.  This
    procedure provides an approximate constraint; the resulting entropy will be
    close to ``cap_per_token_bits`` but not necessarily identical.
    """

    if cap_per_token_bits <= 0:
        raise QualityConfigError("cap_per_token_bits must be positive")

    tokens, probs = _dist_to_arrays(dist)
    probs = _normalise(probs)

    current_entropy = _entropy_bits(probs)
    if current_entropy <= cap_per_token_bits:
        return _arrays_to_dist(tokens, probs, dist)

    low, high = 1e-6, 1.0
    target = probs
    for _ in range(60):
        mid = (low + high) / 2.0
        candidate = _apply_temperature(probs, mid)
        cand_entropy = _entropy_bits(candidate)
        if cand_entropy > cap_per_token_bits:
            high = mid
        else:
            target = candidate
            low = mid

    return _arrays_to_dist(tokens, target, dist)


def _dist_to_arrays(dist: ProbDist) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    if isinstance(dist, np.ndarray):
        probs = np.asarray(dist, dtype=np.float64)
        tokens = np.arange(probs.size, dtype=np.int64)
    elif isinstance(dist, dict):
        items = sorted(dist.items())
        tokens = np.array([int(token) for token, _ in items], dtype=np.int64)
        probs = np.array([float(prob) for _, prob in items], dtype=np.float64)
    else:
        raise TypeError(f"Unsupported distribution type: {type(dist)!r}")

    if np.any(probs < 0.0):
        raise QualityConfigError("Probabilities must be non-negative")

    return tokens, probs


def _arrays_to_dist(
    tokens: NDArray[np.int_],
    probs: NDArray[np.float64],
    original: ProbDist,
) -> ProbDist:
    if isinstance(original, np.ndarray):
        result = np.zeros_like(original, dtype=np.float64)
        result[tokens] = probs
        return result
    mapping = {token: prob for token, prob in zip(tokens.tolist(), probs.tolist()) if prob > 0.0}
    return mapping


def _normalise(probs: NDArray[np.float64]) -> NDArray[np.float64]:
    total = probs.sum()
    if not math.isfinite(total) or total <= 0.0:
        raise QualityConfigError("Probability mass vanished after filtering")
    return probs / total


def _entropy_bits(probs: NDArray[np.float64]) -> float:
    mask = probs > 0.0
    if not np.any(mask):
        return 0.0
    values = probs[mask]
    return float(-(values * np.log2(values)).sum())


def _apply_temperature(probs: NDArray[np.float64], tau: float) -> NDArray[np.float64]:
    if tau <= 0.0:
        raise QualityConfigError("temperature must be positive")
    if math.isclose(tau, 1.0):
        return probs
    logits = np.log(probs + 1e-12)
    scaled = logits / tau
    scaled -= scaled.max()
    exp = np.exp(scaled)
    normalised = exp / exp.sum()
    return normalised


__all__ = [
    "QualityPolicy",
    "TopKPolicy",
    "TopPPolicy",
    "CapacityPerTokenPolicy",
    "apply_quality",
    "cap_bits_per_token",
]
