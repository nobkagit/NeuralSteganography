"""Integration tests for the high-level crypto API."""

from __future__ import annotations

import pytest

from neuralstego.crypto.api import decrypt_message, encrypt_message
from neuralstego.crypto.errors import DecryptionError


def test_encrypt_decrypt_roundtrip_bytes() -> None:
    password = "correct horse"
    plaintext = b"hidden payload"
    aad = b"context"

    blob = encrypt_message(plaintext, password, aad=aad)
    recovered = decrypt_message(blob, password)

    assert recovered == plaintext


def test_encrypt_decrypt_roundtrip_text() -> None:
    password = "hunter2"
    plaintext = "پیام مخفی"

    blob = encrypt_message(plaintext, password)
    recovered = decrypt_message(blob, password)

    assert recovered == plaintext.encode("utf-8")


def test_decrypt_with_wrong_password_raises() -> None:
    blob = encrypt_message(b"secret", "right password")

    with pytest.raises(DecryptionError):
        decrypt_message(blob, "wrong password")

