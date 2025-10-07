"""Type utilities and protocols for codec components."""

from __future__ import annotations

from typing import Dict, Protocol, Sequence, TypedDict, Union

import numpy as np

ProbDist = Union[Dict[int, float], np.ndarray]
"""Probability distribution over vocabulary token ids."""


class LogitVector(TypedDict, total=False):
    """Structure representing a set of token logits."""

    token_ids: Sequence[int]
    logits: Sequence[float]


class ProbabilityVector(TypedDict, total=False):
    """Structure representing a set of token probabilities."""

    token_ids: Sequence[int]
    probs: Sequence[float]


class VocabEntry(TypedDict):
    """A vocabulary entry mapping a token id to its textual form."""

    token_id: int
    token: str


class CodecState(TypedDict, total=False):
    """State snapshot for the arithmetic codec."""

    history: Sequence[int]
    residual_bits: bytes


class LMProvider(Protocol):
    """Protocol for language model probability providers."""

    def next_token_probs(self, context_ids: Sequence[int]) -> ProbDist:
        """Return the probability distribution over next tokens for the given context."""


__all__ = [
    "ProbDist",
    "LogitVector",
    "ProbabilityVector",
    "VocabEntry",
    "CodecState",
    "LMProvider",
]
