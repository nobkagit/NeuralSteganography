"""Cryptographic primitives and helpers for :mod:`neuralstego`."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .errors import AEADError, CryptoError, EnvelopeError, KDFError
from .kdf import derive_key, derive_key_argon2id, derive_key_pbkdf2, gen_salt

if TYPE_CHECKING:
    from .aead import AEADCiphertext
    from .envelope import Envelope, EnvelopeComponents

__all__ = [
    "AEADCiphertext",
    "AEADError",
    "CryptoError",
    "Envelope",
    "EnvelopeComponents",
    "EnvelopeError",
    "KDFError",
    "api",
    "decrypt",
    "derive_key",
    "derive_key_argon2id",
    "derive_key_pbkdf2",
    "encrypt",
    "generate_key",
    "gen_salt",
    "open_envelope",
    "seal_envelope",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AEADCiphertext": (".aead", "AEADCiphertext"),
    "decrypt": (".aead", "decrypt"),
    "encrypt": (".aead", "encrypt"),
    "generate_key": (".aead", "generate_key"),
    "Envelope": (".envelope", "Envelope"),
    "EnvelopeComponents": (".envelope", "EnvelopeComponents"),
    "open_envelope": (".envelope", "open_envelope"),
    "seal_envelope": (".envelope", "seal_envelope"),
    "api": (".api", "api"),
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy crypto primitives on demand."""

    if name in _LAZY_IMPORTS:
        module_name, attribute = _LAZY_IMPORTS[name]
        module = import_module(f"{__name__}{module_name}")
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
