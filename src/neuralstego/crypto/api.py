"""High-level encryption API built on the crypto primitives."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Literal, cast

from .aead import aes_gcm_decrypt, aes_gcm_encrypt
from .envelope import ENVELOPE_VERSION, pack_envelope, unpack_envelope
from .errors import DecryptionError, EncryptionError, EnvelopeError, KDFError
from .kdf import KDFMethod, derive_key, gen_salt

__all__ = ["encrypt_message", "decrypt_message"]

try:
    _ARGON2_AVAILABLE = find_spec("argon2.low_level") is not None
except ModuleNotFoundError:  # pragma: no cover - depends on Python import behaviour
    _ARGON2_AVAILABLE = False


def _derive_key(password: str, salt: bytes, method: KDFMethod, params: dict[str, Any]) -> bytes:
    """Derive a symmetric key from ``password`` and ``salt`` using ``method``."""

    return derive_key(password, salt, method=method, **params)


def _prepare_kdf(
    method: KDFMethod,
    overrides: dict[str, Any],
) -> tuple[KDFMethod, dict[str, Any]]:
    params = dict(overrides)
    params.setdefault("length", 32)
    if method == "argon2id":
        params.setdefault("time_cost", 3)
        params.setdefault("memory_cost", 64 * 1024)
        params.setdefault("parallelism", 2)
        if not _ARGON2_AVAILABLE:
            fallback = {
                "length": params["length"],
                "iterations": params.get("iterations", 310_000),
                "hash_name": params.get("hash_name", "sha256"),
            }
            return "pbkdf2", fallback
        return method, params
    params.setdefault("iterations", 310_000)
    params.setdefault("hash_name", "sha256")
    return "pbkdf2", params


def encrypt_message(
    message: bytes | str,
    password: str,
    *,
    aad: bytes = b"",
    kdf_method: Literal["argon2id", "pbkdf2"] = "argon2id",
    kdf_params: dict[str, Any] | None = None,
) -> bytes:
    """Encrypt ``message`` with a password-derived AES-GCM key.

    Examples
    --------
    >>> blob = encrypt_message(b"secret", "hunter2")
    >>> isinstance(blob, bytes)
    True
    """

    if isinstance(message, str):
        plaintext = message.encode("utf-8")
    else:
        plaintext = message

    salt = gen_salt()
    effective_method, params = _prepare_kdf(kdf_method, kdf_params or {})

    try:
        key = _derive_key(password, salt, effective_method, params)
        ciphertext, nonce, tag = aes_gcm_encrypt(key, plaintext, aad=aad or b"")
        envelope = pack_envelope(
            ciphertext,
            nonce,
            tag,
            kdf_meta={"name": effective_method, "salt": salt, **params},
            aad=aad or None,
        )
    except KDFError as exc:
        raise EncryptionError(str(exc)) from exc
    except ValueError as exc:
        raise EncryptionError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise EncryptionError("Failed to encrypt message.") from exc

    return envelope


def decrypt_message(blob: bytes, password: str) -> bytes:
    """Decrypt an envelope produced by :func:`encrypt_message`.

    Examples
    --------
    >>> blob = encrypt_message("پیام", "p@ssw0rd")
    >>> decrypt_message(blob, "p@ssw0rd")
    b'\xd9\xbe\xdb\x8c\xd8\xa7\xd9\x85'
    """

    try:
        ciphertext, nonce, tag, kdf_meta, aad, version = unpack_envelope(blob)
    except EnvelopeError as exc:
        raise DecryptionError(str(exc)) from exc

    if version != ENVELOPE_VERSION:
        raise DecryptionError("Unsupported envelope version.")

    salt = kdf_meta.get("salt")
    if not isinstance(salt, (bytes, bytearray)):
        raise DecryptionError("Envelope is missing a salt value.")
    method_name = str(kdf_meta.get("name", "argon2id"))
    if method_name not in {"argon2id", "pbkdf2"}:
        raise DecryptionError("Unsupported KDF method in envelope.")
    params = {key: value for key, value in kdf_meta.items() if key not in {"salt", "name"}}
    params.setdefault("length", 32)

    try:
        key = _derive_key(password, bytes(salt), cast(KDFMethod, method_name), params)
        plaintext = aes_gcm_decrypt(key, ciphertext, nonce, tag, aad=aad or b"")
    except KDFError as exc:
        raise DecryptionError(str(exc)) from exc
    except ValueError as exc:
        raise DecryptionError(str(exc)) from exc
    except Exception as exc:
        raise DecryptionError("Failed to decrypt message.") from exc

    return plaintext

