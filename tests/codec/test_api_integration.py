"""Integration tests for the codec API (pending implementation)."""

import pytest

from neuralstego.codec.api import decode_arithmetic, encode_arithmetic
from neuralstego.codec.distribution import MockLM


@pytest.mark.xfail(reason="API integration not yet implemented", raises=NotImplementedError)
def test_api_encode_decode_roundtrip_pending() -> None:
    lm = MockLM()
    encode_arithmetic(b"secret", lm, quality={}, seed_text="")
    decode_arithmetic([], lm, quality={}, seed_text="")
