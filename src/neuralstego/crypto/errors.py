"""Exception types raised by :mod:`neuralstego.crypto` helpers."""

from __future__ import annotations

__all__ = [
    "EncryptionError",
    "DecryptionError",
    "EnvelopeError",
    "KDFError",
]


class EncryptionError(Exception):
    """Raised when a plaintext cannot be encrypted successfully."""


class DecryptionError(Exception):
    """Raised when a ciphertext blob cannot be decrypted."""


class EnvelopeError(Exception):
    """Raised when packing or unpacking an envelope fails."""


class KDFError(Exception):
    """Raised when key derivation parameters or execution are invalid."""
