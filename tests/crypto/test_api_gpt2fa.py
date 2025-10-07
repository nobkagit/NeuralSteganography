"""High-level API tests for GPT2-fa steganographic encoding."""

from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from neuralstego.codec.distribution import MockLM
from neuralstego.crypto import api


def test_encode_decode_roundtrip_with_custom_lm() -> None:
    message = "پیام مخفی"
    password = "hunter2"
    seed_text = "سلام دنیا"
    lm = MockLM(vocab_size=64)

    quality = {"lm": lm, "top_k": 8, "temperature": 0.9}

    blob = api.encode_text(message, password, quality=quality, seed_text=seed_text)
    recovered = api.decode_text(blob, password, quality=quality, seed_text=seed_text)

    assert isinstance(blob, bytes)
    assert recovered == message


def test_fallback_to_mock_language_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken_transformers(*args: object, **kwargs: object) -> object:
        raise OSError("model unavailable")

    monkeypatch.setattr(api, "TransformersLM", _broken_transformers)

    message = "secret"
    password = "password123"

    blob = api.encode_text(message, password, quality={}, seed_text="")
    recovered = api.decode_text(blob, password, quality={}, seed_text="")

    assert recovered == message


def test_decode_with_wrong_seed_raises() -> None:
    quality = {"lm": MockLM(vocab_size=32)}
    blob = api.encode_text("hidden", "pw", quality=quality, seed_text="seed")

    with pytest.raises(ValueError):
        api.decode_text(blob, "pw", quality=quality, seed_text="different")
