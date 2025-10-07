"""Tests for JSON envelope packing and unpacking."""

from __future__ import annotations

import json

import pytest

from neuralstego.crypto.envelope import ENVELOPE_VERSION, pack_envelope, unpack_envelope
from neuralstego.crypto.errors import EnvelopeError


def _sample_kdf_meta() -> dict[str, object]:
    return {
        "name": "argon2id",
        "time_cost": 3,
        "memory_cost": 65536,
        "parallelism": 2,
        "salt": b"\x00" * 16,
    }


def test_pack_then_unpack_roundtrip() -> None:
    ciphertext = b"cipher"
    nonce = b"\x01" * 12
    tag = b"\x02" * 16
    aad = b"meta"

    blob = pack_envelope(ciphertext, nonce, tag, kdf_meta=_sample_kdf_meta(), aad=aad)
    result = unpack_envelope(blob)

    recovered_ct, recovered_nonce, recovered_tag, meta, recovered_aad, version = result

    assert recovered_ct == ciphertext
    assert recovered_nonce == nonce
    assert recovered_tag == tag
    assert recovered_aad == aad
    assert version == ENVELOPE_VERSION
    assert meta["name"] == "argon2id"
    assert isinstance(meta["salt"], bytes)


def test_unpack_missing_field_raises_error() -> None:
    blob = json.dumps({"v": 1, "kdf": {"name": "argon2id", "salt": "AA=="}}).encode("utf-8")

    with pytest.raises(EnvelopeError):
        unpack_envelope(blob)

