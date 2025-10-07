"""Placeholder tests for the high-level crypto API."""

from __future__ import annotations

import pytest

from neuralstego.crypto import api


@pytest.mark.xfail(reason="High-level API pending", strict=False)
def test_api_encrypt_decrypt_roundtrip() -> None:
    """The API should eventually support encryption round-trips."""

    envelope = api.encrypt_message(b"password", b"payload")
    assert api.decrypt_message(b"password", envelope) == b"payload"
