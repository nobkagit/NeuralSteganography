"""Placeholder tests for the AEAD helpers."""

from __future__ import annotations

import pytest

from neuralstego.crypto import aead


@pytest.mark.xfail(reason="AEAD implementation pending", strict=False)
def test_encrypt_decrypt_roundtrip() -> None:
    """Encryption and decryption should eventually round-trip."""

    key = aead.generate_key()
    ciphertext = aead.encrypt(key, b"secret")
    assert aead.decrypt(key, ciphertext) == b"secret"
