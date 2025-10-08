"""Convenience exports for the cryptography helpers."""

from __future__ import annotations

from .aead import NONCE_SIZE, TAG_SIZE, aes_gcm_decrypt, aes_gcm_encrypt
from .api import decode_text, decrypt_message, encode_text, encrypt_message
from .arithmetic import decode_arithmetic, decode_with_lm, encode_arithmetic, encode_with_lm
from .distribution import TransformersLM
from .envelope import ENVELOPE_VERSION, pack_envelope, unpack_envelope
from .errors import DecryptionError, EncryptionError, EnvelopeError, KDFError
from .kdf import (
    KDFMethod,
    derive_key,
    derive_key_argon2id,
    derive_key_pbkdf2,
    gen_salt,
)

__all__ = [
    "NONCE_SIZE",
    "TAG_SIZE",
    "DecryptionError",
    "EncryptionError",
    "EnvelopeError",
    "KDFError",
    "KDFMethod",
    "aes_gcm_decrypt",
    "aes_gcm_encrypt",
    "decode_arithmetic",
    "decode_text",
    "decrypt_message",
    "derive_key",
    "derive_key_argon2id",
    "derive_key_pbkdf2",
    "encode_arithmetic",
    "encode_text",
    "encrypt_message",
    "gen_salt",
    "pack_envelope",
    "unpack_envelope",
    "ENVELOPE_VERSION",
    "TransformersLM",
    "encode_with_lm",
    "decode_with_lm",
]
