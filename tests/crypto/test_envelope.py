"""Placeholder tests for the envelope orchestration layer."""

from __future__ import annotations

import pytest

from neuralstego.crypto import aead, envelope


@pytest.mark.xfail(reason="Envelope opening pending", strict=False)
def test_open_envelope_roundtrip() -> None:
    """Envelopes should support round-tripping once implemented."""

    ciphertext = envelope.EnvelopeComponents(
        salt=b"saltsalt",
        ciphertext=aead.AEADCiphertext(nonce=b"\x00" * 12, ciphertext=b"", tag=b"\x00" * 16),
    )
    sealed = envelope.seal_envelope(ciphertext)
    assert envelope.open_envelope(sealed) == ciphertext
