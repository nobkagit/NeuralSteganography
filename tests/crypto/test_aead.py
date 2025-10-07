"""Tests for the AEAD helpers."""

from __future__ import annotations

import pytest

from neuralstego.crypto import aead


def test_encrypt_decrypt_roundtrip() -> None:
    """Encryption and decryption round-trip with authenticated data."""

    key = aead.generate_key()
    associated_data = b"neuralstego"
    plaintext = b"secret payload"
    ciphertext = aead.encrypt(key, plaintext, associated_data=associated_data)

    assert len(ciphertext.nonce) == aead.DEFAULT_NONCE_SIZE
    assert len(ciphertext.tag) == aead.DEFAULT_TAG_SIZE
    assert aead.decrypt(key, ciphertext, associated_data=associated_data) == plaintext


def test_encrypt_rejects_invalid_nonce_length() -> None:
    """Non-standard nonce sizes are rejected to avoid misuse."""

    key = aead.generate_key()
    with pytest.raises(aead.AEADError):
        aead.encrypt(key, b"data", nonce=b"short")


def test_decrypt_rejects_tampered_tag() -> None:
    """Tag tampering raises an :class:`AEADError`."""

    key = aead.generate_key()
    ciphertext = aead.encrypt(key, b"plaintext")
    corrupted = aead.AEADCiphertext(
        nonce=ciphertext.nonce,
        ciphertext=ciphertext.ciphertext,
        tag=b"\x00" * len(ciphertext.tag),
    )

    with pytest.raises(aead.AEADError):
        aead.decrypt(key, corrupted)
