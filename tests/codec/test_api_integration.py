"""Integration tests for the public codec API."""

from __future__ import annotations

import pytest

from neuralstego.api import decode_text, encode_text
from neuralstego.codec.distribution import MockLM


@pytest.mark.parametrize(
    ("message", "seed_text"),
    [
        (b"", ""),
        (b"hello stego", "context"),
        (bytes(range(32)), "mock"),
    ],
)
def test_encode_decode_roundtrip(message: bytes, seed_text: str) -> None:
    lm = MockLM()
    tokens = encode_text(message, lm, quality={}, seed_text=seed_text)
    recovered = decode_text(tokens, lm, quality={}, seed_text=seed_text)

    assert recovered == message
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) and 0 <= token < 16 for token in tokens)
