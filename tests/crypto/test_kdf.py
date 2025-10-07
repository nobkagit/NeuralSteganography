"""Tests for the key-derivation helpers."""

from __future__ import annotations

import pytest

from neuralstego.crypto.kdf import (
    derive_key,
    derive_key_argon2id,
    derive_key_pbkdf2,
    gen_salt,
)


def test_derive_key_argon2id_is_deterministic_for_fixed_salt() -> None:
    """Argon2id derivation is deterministic for the same password and salt."""

    pytest.importorskip("argon2.low_level")
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    key_one = derive_key_argon2id("correct horse", salt=salt, time_cost=2)
    key_two = derive_key_argon2id("correct horse", salt=salt, time_cost=2)
    assert key_one == key_two
    assert isinstance(key_one, bytes)
    assert len(key_one) == 32


def test_derive_key_argon2id_changes_with_salt() -> None:
    """Changing the salt should alter the Argon2id derived key."""

    pytest.importorskip("argon2.low_level")
    base_salt = bytes.fromhex("00" * 16)
    other_salt = bytes.fromhex("11" * 16)
    base_key = derive_key_argon2id("battery staple", salt=base_salt, time_cost=2)
    other_key = derive_key_argon2id("battery staple", salt=other_salt, time_cost=2)
    assert base_key != other_key


def test_derive_key_pbkdf2_is_deterministic_for_fixed_salt() -> None:
    """PBKDF2 derivation returns consistent bytes for identical inputs."""

    salt = b"static-salt-value"
    key_one = derive_key_pbkdf2("tr0ub4dor", salt=salt, iterations=10_000)
    key_two = derive_key_pbkdf2("tr0ub4dor", salt=salt, iterations=10_000)
    assert key_one == key_two
    assert isinstance(key_one, bytes)
    assert len(key_one) == 32


def test_derive_key_pbkdf2_changes_with_salt() -> None:
    """PBKDF2 output must change when the salt changes."""

    salt_one = b"alpha-salt"
    salt_two = b"beta-salt-"
    key_one = derive_key_pbkdf2("correct horse", salt=salt_one, iterations=10_000)
    key_two = derive_key_pbkdf2("correct horse", salt=salt_two, iterations=10_000)
    assert key_one != key_two


def test_gen_salt_returns_unique_random_bytes() -> None:
    """``gen_salt`` should output unique byte strings of the requested size."""

    salt_one = gen_salt()
    salt_two = gen_salt()
    assert isinstance(salt_one, bytes)
    assert isinstance(salt_two, bytes)
    assert len(salt_one) == 16
    assert len(salt_two) == 16
    assert salt_one != salt_two


def test_derive_key_falls_back_to_pbkdf2_when_argon2_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dispatcher should fall back to PBKDF2 if Argon2 is unavailable."""

    monkeypatch.setattr("neuralstego.crypto.kdf._HASH_SECRET_RAW", None)
    monkeypatch.setattr("neuralstego.crypto.kdf._ARGON2_TYPE_ID", None)
    salt = b"salt-for-fallback"
    key = derive_key("secret", salt)
    assert isinstance(key, bytes)
    assert len(key) == 32
