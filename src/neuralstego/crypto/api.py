"""High-level crypto API including GPT2-fa steganography helpers."""

from __future__ import annotations

import base64
import json
from hashlib import sha256
from importlib.util import find_spec
from typing import Any, Literal, Mapping, Sequence, cast

from ..codec.distribution import MockLM
from ..codec.types import CodecState, LMProvider
from .aead import aes_gcm_decrypt, aes_gcm_encrypt
from .arithmetic import decode_arithmetic, encode_arithmetic
from .distribution import TransformersLM
from .envelope import ENVELOPE_VERSION, pack_envelope, unpack_envelope
from .errors import DecryptionError, EncryptionError, EnvelopeError, KDFError
from .kdf import KDFMethod, derive_key, gen_salt

__all__ = [
    "decrypt_message",
    "decode_text",
    "encode_text",
    "encrypt_message",
]

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


def encode_text(
    message: str,
    password: str,
    *,
    quality: Mapping[str, object] | None = None,
    seed_text: str = "",
) -> bytes:
    """Encrypt ``message`` and embed it into tokens using GPT2-fa or a fallback."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not isinstance(password, str):
        raise TypeError("password must be a string")

    quality = dict(quality or {})
    lm = _resolve_language_model(quality)

    ciphertext = encrypt_message(message.encode("utf-8"), password)
    def _encode_with_lm(lm_instance: LMProvider) -> tuple[list[int], list[int], CodecState]:
        context_ids_local = _seed_to_context(seed_text, lm_instance)
        tokens_local, state_local = encode_arithmetic(
            ciphertext,
            lm_instance,
            quality=quality,
            seed_text=context_ids_local,
        )
        return context_ids_local, [int(token) for token in tokens_local], state_local

    try:
        context_ids, tokens_list, state = _encode_with_lm(lm)
    except Exception:
        if isinstance(lm, MockLM):
            raise
        fallback_lm = MockLM()
        context_ids, tokens_list, state = _encode_with_lm(fallback_lm)
        lm = fallback_lm

    tokens = tokens_list

    history_values = cast(Sequence[int], state.get("history", ()) or ())
    residual_bytes_value = state.get("residual_bits", b"")
    if isinstance(residual_bytes_value, (bytes, bytearray)):
        residual_serial = bytes(residual_bytes_value)
    else:
        residual_sequence = cast(Sequence[int], residual_bytes_value or ())
        residual_serial = bytes(int(value) & 0xFF for value in residual_sequence)
    payload = {
        "tokens": list(map(int, tokens)),
        "history": [int(value) for value in history_values],
        "residual_bits": _encode_bytes(residual_serial),
        "seed_checksum": _seed_checksum(seed_text),
    }

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_text(
    token_stream: bytes | Sequence[int],
    password: str,
    *,
    quality: Mapping[str, object] | None = None,
    seed_text: str = "",
) -> str:
    """Recover an embedded message from ``token_stream`` using the supplied password."""

    if not isinstance(password, str):
        raise TypeError("password must be a string")

    quality = dict(quality or {})
    lm = _resolve_language_model(quality)

    if isinstance(token_stream, (bytes, bytearray)):
        tokens, state, checksum = _deserialize_stream(bytes(token_stream))
    else:
        raise TypeError("token_stream must be the serialised bytes produced by encode_text")

    if checksum != _seed_checksum(seed_text):
        raise ValueError("Seed text does not match the encoded payload")

    def _decode_with_lm(lm_instance: LMProvider) -> bytes:
        context_ids_local = _seed_to_context(seed_text, lm_instance)
        return decode_arithmetic(
            tokens,
            lm_instance,
            quality=quality,
            seed_text=context_ids_local,
            state=state,
        )

    try:
        plaintext_envelope = _decode_with_lm(lm)
    except Exception:
        if isinstance(lm, MockLM):
            raise
        fallback_lm = MockLM()
        plaintext_envelope = _decode_with_lm(fallback_lm)
        lm = fallback_lm

    plaintext = decrypt_message(plaintext_envelope, password)
    return plaintext.decode("utf-8")


def _resolve_language_model(quality: Mapping[str, object]) -> LMProvider:
    candidate = quality.get("lm")
    if hasattr(candidate, "next_token_probs"):
        return cast(LMProvider, candidate)
    if callable(candidate):
        lm = candidate()
        if not all(hasattr(lm, attr) for attr in ("encode_arithmetic", "decode_arithmetic", "encode_seed")):
            raise TypeError("Custom LM factory must return an LMProvider instance")
        return cast(LMProvider, lm)

    try:
        return TransformersLM()
    except Exception:  # pragma: no cover - fallback path
        return MockLM()


def _seed_to_context(seed_text: str, lm: LMProvider) -> list[int]:
    if not seed_text:
        return []

    tokenizer = getattr(lm, "ensure_tokenizer", None)
    if callable(tokenizer):
        tok = tokenizer()
    else:
        tok = getattr(lm, "tokenizer", None)

    if tok is None:
        return []

    if hasattr(tok, "encode"):
        tokens = tok.encode(seed_text, add_special_tokens=False)
        if not tokens:
            tokens = tok.encode(seed_text)
        return list(map(int, tokens))

    if hasattr(tok, "__call__"):
        result = tok(seed_text)
        if isinstance(result, Sequence):
            return [int(value) for value in result]

    return []


def _encode_bytes(data: bytes | None) -> str:
    if not data:
        return ""
    return base64.b64encode(bytes(data)).decode("ascii")


def _deserialize_stream(payload: bytes) -> tuple[list[int], CodecState, str]:
    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid encoded token stream") from exc

    tokens = [int(token) for token in data.get("tokens", [])]
    history = [int(value) for value in data.get("history", [])]
    residual = data.get("residual_bits", "")
    state: CodecState = {}
    if history:
        state["history"] = history
    if residual:
        state["residual_bits"] = base64.b64decode(residual)

    checksum = str(data.get("seed_checksum", ""))
    return tokens, state, checksum


def _seed_checksum(seed_text: str) -> str:
    return sha256(seed_text.encode("utf-8")).hexdigest()

