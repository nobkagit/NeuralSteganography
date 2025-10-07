"""Tests for the envelope orchestration layer."""

from __future__ import annotations

import pytest


pytest.importorskip("cryptography.hazmat.primitives.ciphers.aead")

from neuralstego.crypto import aead, envelope


def test_open_envelope_roundtrip() -> None:
    """Sealing and opening an envelope should be lossless."""

    key = aead.generate_key()
    salt = b"\x01" * envelope.DEFAULT_SALT_SIZE
    encrypted = aead.encrypt(key, b"payload")
    components = envelope.EnvelopeComponents(salt=salt, ciphertext=encrypted)
    sealed = envelope.seal_envelope(components)

    assert envelope.open_envelope(sealed) == components


def test_open_envelope_rejects_missing_components() -> None:
    """Opening an envelope with missing parts raises :class:`EnvelopeError`."""

    bad_envelope = envelope.Envelope(salt=b"", nonce=b"", ciphertext=b"", tag=b"")
    with pytest.raises(envelope.EnvelopeError):
        envelope.open_envelope(bad_envelope)
