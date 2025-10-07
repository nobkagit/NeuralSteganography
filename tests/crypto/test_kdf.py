"""Tests for the :mod:`neuralstego.crypto.kdf` module."""

from __future__ import annotations

from neuralstego.crypto.kdf import PBKDF2Params, derive_key


def test_derive_key_with_pbkdf2_backend() -> None:
    """PBKDF2 backend should derive a key of the requested length."""

    params = PBKDF2Params(iterations=1000)
    key = derive_key(
        b"password",
        salt=b"saltsalt",
        length=32,
        backend="pbkdf2",
        pbkdf2_params=params,
    )
    assert len(key) == 32
