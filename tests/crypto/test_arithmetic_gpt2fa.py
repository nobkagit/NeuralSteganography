"""Integration tests for arithmetic coding with Transformers-based LMs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neuralstego.crypto.arithmetic import decode_with_lm, encode_with_lm
from neuralstego.crypto.distribution import TransformersLM


class DummyGPT2Model:
    """Synthetic language model producing deterministic logits."""

    def __init__(self, vocab_size: int = 10) -> None:
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
        context_score = float(input_ids.sum()) * 0.01
        base = torch.linspace(0.0, 1.5, steps=self.vocab_size, device=self.device)
        logits[0, -1] = base + context_score
        return SimpleNamespace(logits=logits)


def test_arithmetic_roundtrip_with_transformers_lm() -> None:
    message = b"hidden payload"
    context = [3, 7, 11]

    lm = TransformersLM(model=DummyGPT2Model(vocab_size=12))
    encode_state: dict[str, object] = {}
    tokens = encode_with_lm(message, lm, context=context, state=encode_state)

    assert tokens
    assert "history" in encode_state

    decode_state = dict(encode_state)
    recovered = decode_with_lm(tokens, lm, context=context, state=decode_state)

    assert recovered == message
    assert decode_state.get("history") in (tuple(), None)


def test_quality_constraints_reduce_capacity() -> None:
    message = bytes(range(8))
    context = [1, 2, 3]

    baseline_lm = TransformersLM(model=DummyGPT2Model(vocab_size=16))
    baseline_state: dict[str, object] = {}
    baseline_tokens = encode_with_lm(message, baseline_lm, context=context, state=baseline_state)
    assert baseline_tokens

    constrained_lm = TransformersLM(model=DummyGPT2Model(vocab_size=16))
    constrained_state: dict[str, object] = {}
    constrained_tokens = encode_with_lm(
        message,
        constrained_lm,
        context=context,
        quality={"top_k": 2},
        state=constrained_state,
    )

    assert constrained_tokens
    baseline_history = np.array(baseline_state["history"], dtype=np.int32)
    constrained_history = np.array(constrained_state["history"], dtype=np.int32)

    avg_baseline = float(baseline_history.mean())
    avg_constrained = float(constrained_history.mean())

    assert avg_constrained <= avg_baseline + 1e-6
