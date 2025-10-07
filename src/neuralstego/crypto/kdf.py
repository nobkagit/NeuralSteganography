"""Key-derivation helpers for transforming passwords into symmetric keys."""

from __future__ import annotations

from hashlib import pbkdf2_hmac
from importlib import import_module
from importlib.util import find_spec
from os import urandom
from typing import Any, Final, Literal, Protocol, runtime_checkable

from .errors import KDFError

__all__ = [
    "KDFMethod",
    "derive_key_argon2id",
    "derive_key_pbkdf2",
    "derive_key",
    "gen_salt",
]

DEFAULT_KEY_LENGTH: Final[int] = 32
DEFAULT_SALT_SIZE: Final[int] = 16
KDFMethod = Literal["argon2id", "pbkdf2"]


@runtime_checkable
class _Argon2HashFn(Protocol):
    """Runtime protocol describing :func:`argon2.low_level.hash_secret_raw`."""

    def __call__(
        self,
        secret: bytes,
        salt: bytes,
        time_cost: int,
        memory_cost: int,
        parallelism: int,
        hash_len: int,
        type: Any,
        version: int = 19,
    ) -> bytes:
        ...


def _load_argon2() -> tuple[_Argon2HashFn | None, Any]:
    """Return the Argon2 raw hash function and type if the dependency exists."""

    if find_spec("argon2") is None or find_spec("argon2.low_level") is None:
        return None, None

    module = import_module("argon2.low_level")
    hash_secret_raw = getattr(module, "hash_secret_raw")
    argon2_type = getattr(module, "Type")
    if isinstance(hash_secret_raw, _Argon2HashFn):
        return hash_secret_raw, getattr(argon2_type, "ID")
    return None, None


_HASH_SECRET_RAW, _ARGON2_TYPE_ID = _load_argon2()


def _ensure_positive(value: int, name: str) -> None:
    """Ensure ``value`` is positive, raising :class:`KDFError` otherwise."""

    if value <= 0:
        raise KDFError(f"{name} must be a positive integer.")


def _normalise_password(password: str) -> bytes:
    """Encode ``password`` into a UTF-8 byte string."""

    return password.encode("utf-8")


def gen_salt(size: int = DEFAULT_SALT_SIZE) -> bytes:
    """Return a cryptographically strong random salt of ``size`` bytes."""

    _ensure_positive(size, "Salt size")
    return urandom(size)


def derive_key_argon2id(
    password: str,
    salt: bytes,
    *,
    length: int = DEFAULT_KEY_LENGTH,
    time_cost: int = 3,
    memory_cost: int = 64 * 1024,
    parallelism: int = 2,
) -> bytes:
    """Derive a key using Argon2id with configurable but safe defaults."""

    if _HASH_SECRET_RAW is None or _ARGON2_TYPE_ID is None:
        raise KDFError("Argon2id backend requested but argon2-cffi is unavailable.")

    _ensure_positive(length, "Key length")
    _ensure_positive(time_cost, "time_cost")
    _ensure_positive(memory_cost, "memory_cost")
    _ensure_positive(parallelism, "parallelism")

    password_bytes = _normalise_password(password)
    return _HASH_SECRET_RAW(
        password_bytes,
        salt,
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=length,
        type=_ARGON2_TYPE_ID,
    )


def derive_key_pbkdf2(
    password: str,
    salt: bytes,
    *,
    length: int = DEFAULT_KEY_LENGTH,
    iterations: int = 310_000,
    hash_name: Literal["sha256", "sha512"] = "sha256",
) -> bytes:
    """Derive a key using PBKDF2-HMAC as a portable fallback mechanism."""

    _ensure_positive(length, "Key length")
    _ensure_positive(iterations, "iterations")

    password_bytes = _normalise_password(password)
    return pbkdf2_hmac(hash_name, password_bytes, salt, iterations, dklen=length)


def derive_key(
    password: str,
    salt: bytes,
    method: KDFMethod = "argon2id",
    **kwargs: Any,
) -> bytes:
    """Derive a key using the requested method, defaulting to Argon2id."""

    if method == "argon2id":
        if _HASH_SECRET_RAW is not None and _ARGON2_TYPE_ID is not None:
            return derive_key_argon2id(password, salt, **kwargs)
        return derive_key_pbkdf2(password, salt, **kwargs)
    if method == "pbkdf2":
        return derive_key_pbkdf2(password, salt, **kwargs)
    raise KDFError(f"Unsupported KDF method: {method}")
