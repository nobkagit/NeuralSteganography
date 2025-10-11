"""AES-GCM based authenticated encryption helpers."""

from __future__ import annotations

from os import urandom
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore[import-not-found]

__all__ = [
    "NONCE_SIZE",
    "TAG_SIZE",
    "aes_gcm_decrypt",
    "aes_gcm_encrypt",
]

NONCE_SIZE: Final[int] = 12
TAG_SIZE: Final[int] = 16


def _split_ciphertext(data: bytes) -> tuple[bytes, bytes]:
    """Split ``data`` into ciphertext and authentication tag segments."""

    if len(data) < TAG_SIZE:
        raise ValueError("Ciphertext is shorter than the authentication tag length.")
    return data[:-TAG_SIZE], data[-TAG_SIZE:]


def aes_gcm_encrypt(
    key: bytes,
    plaintext: bytes,
    *,
    aad: bytes = b"",
    nonce: bytes | None = None,
) -> tuple[bytes, bytes, bytes]:
    """Encrypt ``plaintext`` using AES-GCM.

    The nonce must be ``NONCE_SIZE`` (12) bytes long. If omitted, a secure random nonce
    is generated with :func:`os.urandom`. All inputs and outputs are raw ``bytes``.
    """

    nonce_bytes = nonce if nonce is not None else urandom(NONCE_SIZE)
    if len(nonce_bytes) != NONCE_SIZE:
        raise ValueError(f"AES-GCM nonce must be {NONCE_SIZE} bytes long.")

    cipher = AESGCM(key)
    ciphertext_with_tag = cipher.encrypt(nonce_bytes, plaintext, aad or None)
    ciphertext, tag = _split_ciphertext(ciphertext_with_tag)
    return ciphertext, nonce_bytes, tag


def aes_gcm_decrypt(
    key: bytes,
    ciphertext: bytes,
    nonce: bytes,
    tag: bytes,
    *,
    aad: bytes = b"",
) -> bytes:
    """Decrypt ``ciphertext`` produced by :func:`aes_gcm_encrypt`.

    ``nonce`` must be ``NONCE_SIZE`` bytes and ``tag`` must be ``TAG_SIZE`` bytes. The
    function returns the recovered plaintext and propagates exceptions from
    :class:`cryptography.hazmat.primitives.ciphers.aead.AESGCM` directly.
    """

    if len(nonce) != NONCE_SIZE:
        raise ValueError(f"AES-GCM nonce must be {NONCE_SIZE} bytes long.")
    if len(tag) != TAG_SIZE:
        raise ValueError(f"AES-GCM tag must be {TAG_SIZE} bytes long.")

    cipher = AESGCM(key)
    combined = ciphertext + tag
    return cipher.decrypt(nonce, combined, aad or None)

