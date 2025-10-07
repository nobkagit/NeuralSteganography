"""Exception hierarchy for the :mod:`neuralstego.crypto` package."""

from __future__ import annotations

__all__ = [
    "CryptoError",
    "KDFError",
    "AEADError",
    "EnvelopeError",
]


class CryptoError(Exception):
    """Base exception for all cryptographic failures within the project."""


class KDFError(CryptoError):
    """Raised when a key-derivation operation fails or misbehaves."""


class AEADError(CryptoError):
    """Raised for authenticated encryption and decryption errors."""


class EnvelopeError(CryptoError):
    """Raised when envelope encryption orchestration fails."""
