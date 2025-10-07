"""Tests for mock and cached language model providers."""

from __future__ import annotations

import numpy as np

from neuralstego.codec.distribution import CachedLM, MockLM


def test_mocklm_distribution_properties() -> None:
    lm = MockLM(vocab_size=48, alpha=1.3)
    probs = lm.next_token_probs([])

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (48,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)


def test_cachedlm_returns_copies() -> None:
    base = MockLM(vocab_size=32)
    cached = CachedLM(base, maxsize=2)

    context = [1, 2, 3]
    first = cached.next_token_probs(context)
    second = cached.next_token_probs(context)

    assert np.array_equal(first, second)

    second[0] = 0.0
    third = cached.next_token_probs(context)

    assert not np.array_equal(second, third)
    assert np.isclose(third.sum(), 1.0)


def test_cachedlm_eviction() -> None:
    base = MockLM(vocab_size=16)
    cached = CachedLM(base, maxsize=2)

    cached.next_token_probs([1])
    cached.next_token_probs([2])
    cached.next_token_probs([3])

    assert tuple([1]) not in cached._cache  # type: ignore[attr-defined]
    assert tuple([2]) in cached._cache  # type: ignore[attr-defined]
    assert tuple([3]) in cached._cache  # type: ignore[attr-defined]
