"""Authenticated encryption with associated data (AEAD) helpers."""

from __future__ import annotations

from dataclasses import dataclass
from secrets import token_bytes
from typing import Final

from .errors import AEADError

__all__ = [
    "AEADCiphertext",
    "DEFAULT_NONCE_SIZE",
    "DEFAULT_TAG_SIZE",
    "decrypt",
    "encrypt",
    "generate_key",
]

DEFAULT_KEY_SIZE: Final[int] = 32
DEFAULT_NONCE_SIZE: Final[int] = 12
DEFAULT_TAG_SIZE: Final[int] = 16


@dataclass(frozen=True)
class AEADCiphertext:
    """Container for AEAD ciphertext components."""

    nonce: bytes
    ciphertext: bytes
    tag: bytes


def generate_key(size: int = DEFAULT_KEY_SIZE) -> bytes:
    """Return a new random AEAD key."""

    if size <= 0:
        raise AEADError("AEAD key size must be positive.")
    return token_bytes(size)


def encrypt(
    key: bytes,
    plaintext: bytes,
    *,
    associated_data: bytes | None = None,
) -> AEADCiphertext:
    """Encrypt ``plaintext`` using an AEAD scheme.

    The concrete implementation is intentionally left for future work.
    """

    raise NotImplementedError("AEAD encryption has not been implemented yet.")


def decrypt(
    key: bytes,
    ciphertext: AEADCiphertext,
    *,
    associated_data: bytes | None = None,
) -> bytes:
    """Decrypt ``ciphertext`` previously produced by :func:`encrypt`."""

    raise NotImplementedError("AEAD decryption has not been implemented yet.")
