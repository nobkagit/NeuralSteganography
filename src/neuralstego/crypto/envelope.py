"""Envelope encryption orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .aead import AEADCiphertext
from .errors import EnvelopeError

__all__ = [
    "Envelope",
    "EnvelopeComponents",
    "DEFAULT_SALT_SIZE",
    "open_envelope",
    "seal_envelope",
]

DEFAULT_SALT_SIZE: Final[int] = 16


@dataclass(frozen=True)
class EnvelopeComponents:
    """Discrete building blocks used to construct an envelope."""

    salt: bytes
    ciphertext: AEADCiphertext


@dataclass(frozen=True)
class Envelope:
    """Serialized representation of an encrypted payload."""

    salt: bytes
    nonce: bytes
    ciphertext: bytes
    tag: bytes


def seal_envelope(components: EnvelopeComponents) -> Envelope:
    """Bundle derived materials into an :class:`Envelope` structure."""

    if not components.salt:
        raise EnvelopeError("Envelope salt must be non-empty.")
    if not components.ciphertext.nonce or not components.ciphertext.tag:
        raise EnvelopeError("Envelope ciphertext components must be populated.")
    return Envelope(
        salt=components.salt,
        nonce=components.ciphertext.nonce,
        ciphertext=components.ciphertext.ciphertext,
        tag=components.ciphertext.tag,
    )


def open_envelope(envelope: Envelope) -> EnvelopeComponents:
    """Invert :func:`seal_envelope` by recreating the component bundle."""

    if not envelope.salt:
        raise EnvelopeError("Envelope salt must be non-empty.")
    if not envelope.nonce or not envelope.tag:
        raise EnvelopeError("Envelope data is missing nonce or authentication tag.")
    ciphertext = AEADCiphertext(
        nonce=envelope.nonce,
        ciphertext=envelope.ciphertext,
        tag=envelope.tag,
    )
    return EnvelopeComponents(salt=envelope.salt, ciphertext=ciphertext)
