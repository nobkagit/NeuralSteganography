"""Integration tests for the high-level crypto API."""

from __future__ import annotations

import pytest


pytest.importorskip("cryptography.hazmat.primitives.ciphers.aead")

from neuralstego.crypto import api


def test_api_encrypt_decrypt_roundtrip() -> None:
    """The high-level API supports round-tripping payloads."""

    associated_data = b"metadata"
    envelope = api.encrypt_message(
        "password",
        b"payload",
        associated_data=associated_data,
    )
    assert api.decrypt_message(
        "password",
        envelope,
        associated_data=associated_data,
    ) == b"payload"
