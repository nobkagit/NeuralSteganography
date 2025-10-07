"""Language-model distribution providers for arithmetic coding."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import LMProvider, ProbDist


@dataclass
class MockLM(LMProvider):
    """Deterministic mock language model with a Zipfian prior."""

    vocab_size: int = 32
    alpha: float = 1.2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        ranks = np.arange(1, self.vocab_size + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, self.alpha)
        self._base_probs = (weights / weights.sum()).astype(np.float64)

    def next_token_probs(self, context_ids: Sequence[int]) -> ProbDist:  # noqa: D401 - simple mapping
        """Return the fixed Zipfian probability distribution regardless of context."""

        _ = context_ids  # context ignored for deterministic mock
        return self._base_probs.copy()


class CachedLM(LMProvider):
    """Caching wrapper that memoizes context probability queries."""

    def __init__(self, lm: LMProvider, *, maxsize: int = 128) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._lm = lm
        self._maxsize = maxsize
        self._cache: "OrderedDict[tuple[int, ...], ProbDist]" = OrderedDict()

    def next_token_probs(self, context_ids: Sequence[int]) -> ProbDist:
        key = tuple(context_ids)
        if key in self._cache:
            self._cache.move_to_end(key)
            return _clone_distribution(self._cache[key])

        probs = self._lm.next_token_probs(list(context_ids))
        self._cache[key] = _clone_distribution(probs)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return _clone_distribution(probs)


class TransformersLM(LMProvider):
    """Adapter for HuggingFace Transformers models (planned)."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - placeholder signature
        raise NotImplementedError("Transformers integration will be implemented in phase 4")

    def next_token_probs(self, context_ids: Sequence[int]) -> ProbDist:
        """Return next-token probabilities from the wrapped transformer model."""

        raise NotImplementedError("Transformers integration will be implemented in phase 4")


def _clone_distribution(dist: ProbDist) -> ProbDist:
    """Return a safe copy of the probability distribution for cache reuse."""

    if isinstance(dist, np.ndarray):
        return dist.copy()
    if isinstance(dist, dict):
        return dict(dist)
    raise TypeError(f"Unsupported distribution type: {type(dist)!r}")


__all__ = ["MockLM", "CachedLM", "TransformersLM"]
