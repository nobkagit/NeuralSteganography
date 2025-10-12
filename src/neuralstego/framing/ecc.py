"""Reed-Solomon error correction helpers."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Tuple

from .errors import NotAvailableError


def _load_reedsolo() -> ModuleType | None:
    try:
        return importlib.import_module("reedsolo")
    except ModuleNotFoundError:  # pragma: no cover - handled at runtime
        return None


reedsolo = _load_reedsolo()


def _require_reedsolo() -> ModuleType:
    if reedsolo is None:
        raise NotAvailableError(
            "reedsolo is not installed. Install the 'reedsolo' package to enable ECC support."
        )
    return reedsolo


def rs_encode(data: bytes, nsym: int = 10) -> bytes:
    """Encode *data* with Reed-Solomon error correction."""

    rs = _require_reedsolo().RSCodec(nsym)
    return bytes(rs.encode(bytearray(data)))


def rs_decode(codeword: bytes, nsym: int = 10) -> Tuple[bool, bytes]:
    """Decode a Reed-Solomon codeword.

    Returns ``(ok, data)`` where ``ok`` is ``True`` when decoding was successful
    and ``False`` otherwise.  When decoding fails the data component is an empty
    byte-string.
    """

    rs_module = _require_reedsolo()
    codec = rs_module.RSCodec(nsym)
    try:
        decoded = codec.decode(bytearray(codeword))
    except rs_module.ReedSolomonError:
        return False, b""

    if isinstance(decoded, tuple):
        data = decoded[0]
    else:
        data = decoded
    if isinstance(data, bytearray):
        data = bytes(data)
    return True, data
