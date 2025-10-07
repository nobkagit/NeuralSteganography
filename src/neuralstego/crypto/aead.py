"""Authenticated encryption with associated data (AEAD) helpers."""

from __future__ import annotations

from dataclasses import dataclass
from secrets import token_bytes
from typing import Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import AEADError

__all__ = [
    "AEADCiphertext",
    "DEFAULT_KEY_SIZE",
    "DEFAULT_NONCE_SIZE",
    "DEFAULT_TAG_SIZE",
    "decrypt",
    "encrypt",
    "generate_key",
]

DEFAULT_KEY_SIZE: Final[int] = 32
DEFAULT_NONCE_SIZE: Final[int] = 12
DEFAULT_TAG_SIZE: Final[int] = 16
_VALID_KEY_SIZES: Final[tuple[int, ...]] = (16, 24, 32)


@dataclass(frozen=True)
class AEADCiphertext:
    """Container for AEAD ciphertext components."""

    nonce: bytes
    ciphertext: bytes
    tag: bytes


def _validate_key_size(size: int) -> None:
    """Ensure the provided key size is valid for AES-GCM."""

    if size not in _VALID_KEY_SIZES:
        allowed = ", ".join(str(value) for value in _VALID_KEY_SIZES)
        raise AEADError(f"AES-GCM key size must be one of: {allowed} bytes.")


def _ensure_lengths(ciphertext: AEADCiphertext) -> None:
    """Validate the nonce and authentication tag sizes."""

    if len(ciphertext.nonce) != DEFAULT_NONCE_SIZE:
        raise AEADError(
            f"AES-GCM nonce must be {DEFAULT_NONCE_SIZE} bytes, "
            f"got {len(ciphertext.nonce)}.",
        )
    if len(ciphertext.tag) != DEFAULT_TAG_SIZE:
        raise AEADError(
            f"AES-GCM tag must be {DEFAULT_TAG_SIZE} bytes, got {len(ciphertext.tag)}."
        )


def generate_key(size: int = DEFAULT_KEY_SIZE) -> bytes:
    """Return a new random AEAD key."""

    _validate_key_size(size)
    return token_bytes(size)


def _initialise_cipher(key: bytes) -> AESGCM:
    """Return an :class:`AESGCM` instance for ``key`` or raise :class:`AEADError`."""

    try:
        _validate_key_size(len(key))
        return AESGCM(key)
    except ValueError as exc:  # pragma: no cover - defensive, AESGCM already validates
        raise AEADError("Failed to initialise AES-GCM cipher.") from exc


def encrypt(
    key: bytes,
    plaintext: bytes,
    *,
    associated_data: bytes | None = None,
    nonce: bytes | None = None,
) -> AEADCiphertext:
    """Encrypt ``plaintext`` using AES-256-GCM."""

    nonce_bytes = nonce or token_bytes(DEFAULT_NONCE_SIZE)
    if len(nonce_bytes) != DEFAULT_NONCE_SIZE:
        raise AEADError(
            f"AES-GCM nonce must be {DEFAULT_NONCE_SIZE} bytes, got {len(nonce_bytes)}."
        )

    cipher = _initialise_cipher(key)
    ciphertext_with_tag = cipher.encrypt(nonce_bytes, plaintext, associated_data)
    ciphertext = ciphertext_with_tag[:-DEFAULT_TAG_SIZE]
    tag = ciphertext_with_tag[-DEFAULT_TAG_SIZE:]
    return AEADCiphertext(nonce=nonce_bytes, ciphertext=ciphertext, tag=tag)


def decrypt(
    key: bytes,
    ciphertext: AEADCiphertext,
    *,
    associated_data: bytes | None = None,
) -> bytes:
    """Decrypt ``ciphertext`` previously produced by :func:`encrypt`."""

    _ensure_lengths(ciphertext)
    cipher = _initialise_cipher(key)
    data = ciphertext.ciphertext + ciphertext.tag
    try:
        return cipher.decrypt(ciphertext.nonce, data, associated_data)
    except InvalidTag as exc:
        raise AEADError("Failed to authenticate ciphertext.") from exc
