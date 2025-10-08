"""Utility helpers that mimic GPT2-fa encode/decode behaviour for the CLI.

The real project uses a GPT2-fa language model to perform the steganographic
encoding.  Shipping the full model is outside the scope of these kata tests, so
we provide a lightweight, deterministic stand-in that operates on UTF-8 text.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, MutableMapping, Sequence


@dataclass
class _CipherPayload:
    """Internal representation of the encoded payload."""

    tokens: bytes
    quality: Mapping[str, object]
    encoding: str

    def as_json(self) -> MutableMapping[str, object]:
        data: MutableMapping[str, object] = {
            "tokens": [int(b) for b in self.tokens],
            "encoding": self.encoding,
        }
        if self.quality:
            data["quality"] = dict(self.quality)
        return data


def _derive_key(password: str) -> bytes:
    if not password:
        raise ValueError("Password must not be empty.")
    # A deterministic salt keeps the output stable for tests while providing
    # a stable transformation that still depends on the password.
    salt = b"neuralstego-gpt2fa"
    hasher = hashlib.sha256()
    hasher.update(password.encode("utf-8"))
    hasher.update(salt)
    return hasher.digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    if not key:
        raise ValueError("Derived key must not be empty.")
    expanded = bytearray(len(data))
    for idx, value in enumerate(data):
        expanded[idx] = value ^ key[idx % len(key)]
    return bytes(expanded)


def _ensure_text(message: str) -> str:
    if not isinstance(message, str):
        raise TypeError("Message must be a string containing UTF-8 text.")
    return message


def _prepare_quality(quality: Mapping[str, object] | None) -> Mapping[str, object]:
    if quality is None:
        return {}
    if not isinstance(quality, Mapping):
        raise TypeError("quality must be a mapping of hyper-parameters.")
    return quality


def encode_text(
    message: str,
    password: str,
    *,
    quality: Mapping[str, object] | None = None,
    encoding: str = "utf-8",
) -> MutableMapping[str, object]:
    """Encode ``message`` using a password and the quality configuration.

    The behaviour mirrors the real GPT2-fa encoder only in its interface: it
    accepts a message and returns a JSON-serialisable payload with token data
    and metadata.  The transformation itself is a lightweight XOR cipher that
    allows the CLI to be exercised in tests without heavyweight dependencies.
    """

    text = _ensure_text(message)
    key = _derive_key(password)
    quality_cfg = _prepare_quality(quality)

    data_bytes = text.encode(encoding)
    cipher_bytes = _xor_bytes(data_bytes, key)
    payload = _CipherPayload(tokens=cipher_bytes, quality=quality_cfg, encoding=encoding)
    return payload.as_json()


def _normalise_tokens(tokens: Iterable[int]) -> bytes:
    values: list[int] = []
    for raw in tokens:
        if isinstance(raw, bool):  # bool is a subclass of int, skip special case
            raw = int(raw)
        if isinstance(raw, (int, float)):
            value = int(raw)
        else:
            raise TypeError("Tokens must be numbers that can be converted to integers.")
        values.append(value & 0xFF)
    return bytes(values)


def decode_text(payload: Mapping[str, object], password: str) -> str:
    """Decode ``payload`` back into the original message using ``password``."""

    if not isinstance(payload, Mapping):
        raise TypeError("Payload must be a mapping produced by encode_text.")

    encoding = payload.get("encoding", "utf-8")
    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, Sequence):
        raise ValueError("Payload does not contain token sequence.")

    key = _derive_key(password)
    cipher_bytes = _normalise_tokens(raw_tokens)
    plain_bytes = _xor_bytes(cipher_bytes, key)
    try:
        return plain_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError("Decoded message is not valid UTF-8 text.") from exc
