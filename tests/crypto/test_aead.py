"""Tests for the AES-GCM helpers."""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidTag

from neuralstego.crypto.aead import (
    NONCE_SIZE,
    TAG_SIZE,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
)


def _fixed_key() -> bytes:
    return bytes(range(32))


def test_encrypt_decrypt_roundtrip_without_aad() -> None:
    plaintext = b"neuralstego-secret"
    key = _fixed_key()

    ciphertext, nonce, tag = aes_gcm_encrypt(key, plaintext)

    assert len(nonce) == NONCE_SIZE
    assert len(tag) == TAG_SIZE
    assert aes_gcm_decrypt(key, ciphertext, nonce, tag) == plaintext


def test_encrypt_decrypt_with_aad() -> None:
    plaintext = b"authenticated"
    aad = b"metadata"
    key = _fixed_key()

    ciphertext, nonce, tag = aes_gcm_encrypt(key, plaintext, aad=aad)
    recovered = aes_gcm_decrypt(key, ciphertext, nonce, tag, aad=aad)
    assert recovered == plaintext


def test_encrypt_with_supplied_nonce_is_deterministic() -> None:
    plaintext = b"deterministic"
    key = _fixed_key()
    nonce = b"\x01" * NONCE_SIZE

    first = aes_gcm_encrypt(key, plaintext, nonce=nonce)
    second = aes_gcm_encrypt(key, plaintext, nonce=nonce)
    assert first == second


def test_decrypt_raises_on_tag_tamper() -> None:
    key = _fixed_key()
    plaintext = b"tamper"
    ciphertext, nonce, tag = aes_gcm_encrypt(key, plaintext)
    bad_tag = bytes(~b & 0xFF for b in tag)

    with pytest.raises(InvalidTag):
        aes_gcm_decrypt(key, ciphertext, nonce, bad_tag)


def test_encrypt_rejects_invalid_nonce_length() -> None:
    key = _fixed_key()
    with pytest.raises(ValueError):
        aes_gcm_encrypt(key, b"data", nonce=b"short")

