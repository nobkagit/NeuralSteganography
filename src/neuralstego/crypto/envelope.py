"""JSON envelope helpers for serialising encrypted payloads."""

from __future__ import annotations

import json
from base64 import b64decode, b64encode
from typing import Any, Final

from .aead import NONCE_SIZE, TAG_SIZE
from .errors import EnvelopeError

__all__ = [
    "ENVELOPE_VERSION",
    "pack_envelope",
    "unpack_envelope",
]

ENVELOPE_VERSION: Final[int] = 1
_AEAD_NAME: Final[str] = "aes-256-gcm"


def _b64encode(data: bytes) -> str:
    """Return standard Base64 encoding for ``data``."""

    return b64encode(data).decode("ascii")


def _b64decode(value: str, *, field: str) -> bytes:
    """Decode standard Base64 text, raising :class:`EnvelopeError` on failure."""

    try:
        return b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        raise EnvelopeError(f"Invalid base64 data for field '{field}'.") from exc


def _encode_kdf_meta(kdf_meta: dict[str, Any]) -> dict[str, Any]:
    """Prepare KDF metadata for JSON serialisation."""

    if "name" not in kdf_meta:
        raise EnvelopeError("KDF metadata must include a 'name'.")

    encoded: dict[str, Any] = {"name": kdf_meta["name"]}
    for key, value in kdf_meta.items():
        if key == "salt":
            if not isinstance(value, (bytes, bytearray)):
                raise EnvelopeError("KDF salt must be bytes.")
            encoded[key] = _b64encode(bytes(value))
        elif key != "name":
            encoded[key] = value
    if "salt" not in encoded:
        raise EnvelopeError("KDF metadata must include a salt value.")
    return encoded


def _decode_kdf_meta(meta: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    """Decode KDF metadata from JSON form."""

    if "name" not in meta or "salt" not in meta:
        raise EnvelopeError("Envelope is missing KDF metadata fields.")
    salt = _b64decode(str(meta["salt"]), field="kdf.salt")
    decoded = dict(meta)
    decoded["salt"] = salt
    return decoded, salt


def pack_envelope(
    ciphertext: bytes,
    nonce: bytes,
    tag: bytes,
    *,
    kdf_meta: dict[str, Any],
    aad: bytes | None = None,
) -> bytes:
    """Return a JSON envelope containing the encrypted payload.

    The blob uses standard Base64 encoding for binary fields.
    """

    if len(nonce) != NONCE_SIZE:
        raise EnvelopeError(f"Nonce must be {NONCE_SIZE} bytes long.")
    if len(tag) != TAG_SIZE:
        raise EnvelopeError(f"Authentication tag must be {TAG_SIZE} bytes long.")

    payload: dict[str, Any] = {
        "v": ENVELOPE_VERSION,
        "kdf": _encode_kdf_meta(kdf_meta),
        "aead": {
            "name": _AEAD_NAME,
            "nonce": _b64encode(nonce),
            "tag": _b64encode(tag),
        },
        "ct": _b64encode(ciphertext),
    }
    if aad is not None:
        payload["aad"] = _b64encode(aad)
    return json.dumps(payload, separators=(",", ":"), sort_keys=False).encode("utf-8")


def unpack_envelope(
    blob: bytes,
) -> tuple[bytes, bytes, bytes, dict[str, Any], bytes | None, int]:
    """Parse ``blob`` and return ciphertext components and metadata."""

    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvelopeError("Envelope payload is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise EnvelopeError("Envelope payload must be a JSON object.")

    version = payload.get("v")
    if not isinstance(version, int):
        raise EnvelopeError("Envelope is missing a valid version number.")

    kdf_raw = payload.get("kdf")
    if not isinstance(kdf_raw, dict):
        raise EnvelopeError("Envelope missing KDF metadata.")
    kdf_meta, _ = _decode_kdf_meta(kdf_raw)

    aead_section = payload.get("aead")
    if not isinstance(aead_section, dict):
        raise EnvelopeError("Envelope missing AEAD section.")
    try:
        nonce = _b64decode(str(aead_section["nonce"]), field="aead.nonce")
        tag = _b64decode(str(aead_section["tag"]), field="aead.tag")
    except KeyError as exc:  # pragma: no cover - defensive
        raise EnvelopeError("Envelope missing AEAD nonce or tag.") from exc

    if len(nonce) != NONCE_SIZE:
        raise EnvelopeError("Envelope nonce has an invalid length.")
    if len(tag) != TAG_SIZE:
        raise EnvelopeError("Envelope tag has an invalid length.")

    ct_value = payload.get("ct")
    if not isinstance(ct_value, str):
        raise EnvelopeError("Envelope missing ciphertext field.")
    ciphertext = _b64decode(ct_value, field="ct")

    aad_value = payload.get("aad")
    aad_bytes: bytes | None
    if aad_value is None:
        aad_bytes = None
    elif isinstance(aad_value, str):
        aad_bytes = _b64decode(aad_value, field="aad")
    else:  # pragma: no cover - defensive
        raise EnvelopeError("Envelope AAD field must be a string when present.")

    return ciphertext, nonce, tag, kdf_meta, aad_bytes, version

