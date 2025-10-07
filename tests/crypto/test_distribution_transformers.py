"""Tests for the Transformers-based language model adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neuralstego.crypto.distribution import TransformersLM


class DummyGPT2Model:
    """Minimal stand-in for ``GPT2LMHeadModel`` used in unit tests."""

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(n_positions=32)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "DummyGPT2Model":
        if device is not None:
            self.device = torch.device(device)
        _ = dtype
        return self

    def eval(self) -> "DummyGPT2Model":
        return self

    def __call__(self, input_ids: torch.Tensor) -> SimpleNamespace:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a 2-D tensor")
        seq_len = input_ids.size(1)
        logits = torch.zeros((1, seq_len, self.vocab_size), dtype=torch.float32, device=self.device)
        last_token = int(input_ids[0, -1])
        base = torch.linspace(0.1, 1.0, steps=self.vocab_size, device=self.device)
        logits[0, -1] = base + (last_token * 0.01)
        return SimpleNamespace(logits=logits)


def test_transformers_lm_normalises_probabilities() -> None:
    model = DummyGPT2Model(vocab_size=6)
    lm = TransformersLM(model=model)
    probs = lm.next_token_probs([10, 20, 30])

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (6,)
    assert pytest.approx(1.0) == float(probs.sum())
    assert probs.argmax() == 5  # highest logit after softmax


def test_quality_policies_limit_distribution_support() -> None:
    context = [1, 2, 3]

    lm_topk = TransformersLM(model=DummyGPT2Model(vocab_size=8), top_k=2)
    probs_topk = lm_topk.next_token_probs(context)
    assert np.count_nonzero(probs_topk) == 2

    lm_topp = TransformersLM(model=DummyGPT2Model(vocab_size=8), top_p=0.5)
    probs_topp = lm_topp.next_token_probs(context)
    assert pytest.approx(1.0) == float(probs_topp.sum())
    assert np.count_nonzero(probs_topp) <= 4


def test_temperature_adjustment_changes_distribution_shape() -> None:
    context = [7, 8]
    cold = TransformersLM(model=DummyGPT2Model(vocab_size=5), temperature=0.5)
    warm = TransformersLM(model=DummyGPT2Model(vocab_size=5), temperature=1.5)

    cold_probs = cold.next_token_probs(context)
    warm_probs = warm.next_token_probs(context)

    assert not np.allclose(cold_probs, warm_probs)
    assert cold_probs.max() > warm_probs.max()
