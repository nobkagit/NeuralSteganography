"""Key-derivation functions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import pbkdf2_hmac
from importlib import import_module
from importlib.util import find_spec
from secrets import token_bytes
from typing import Any, Final, Literal, Protocol, runtime_checkable

from .errors import KDFError

__all__ = [
    "Argon2idParams",
    "PBKDF2Params",
    "KDFBackend",
    "derive_key",
    "generate_salt",
]

DEFAULT_SALT_SIZE: Final[int] = 16

KDFBackend = Literal["argon2id", "pbkdf2"]


@dataclass(frozen=True)
class Argon2idParams:
    """Tunable parameters for the Argon2id key derivation function."""

    time_cost: int = 4
    """Number of iterations (the ``t`` parameter)."""

    memory_cost: int = 102_400
    """Memory usage in kibibytes (the ``m`` parameter)."""

    parallelism: int = 8
    """Number of parallel lanes (the ``p`` parameter)."""


@dataclass(frozen=True)
class PBKDF2Params:
    """Parameters controlling PBKDF2 when Argon2id is unavailable."""

    iterations: int = 600_000
    """Number of iterations for PBKDF2."""

    hash_name: Literal["sha256", "sha512"] = "sha256"
    """Digest algorithm to use with PBKDF2."""


@runtime_checkable
class _Argon2HashFn(Protocol):
    """Runtime protocol describing the :mod:`argon2.low_level` API we consume."""

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
    """Load Argon2 primitives if available, otherwise return ``(None, None)``."""

    if find_spec("argon2") is None or find_spec("argon2.low_level") is None:
        return None, None

    module = import_module("argon2.low_level")
    hash_secret_raw = getattr(module, "hash_secret_raw")
    argon2_type = getattr(module, "Type")
    if isinstance(hash_secret_raw, _Argon2HashFn):
        return hash_secret_raw, getattr(argon2_type, "ID")
    return None, None


_HASH_SECRET_RAW, _ARGON2_TYPE_ID = _load_argon2()


def generate_salt(size: int = DEFAULT_SALT_SIZE) -> bytes:
    """Return a cryptographically secure random salt of ``size`` bytes."""

    if size <= 0:
        raise KDFError("Salt size must be a positive integer.")
    return token_bytes(size)


def derive_key(
    secret: bytes,
    salt: bytes,
    *,
    length: int,
    backend: KDFBackend | None = None,
    argon2_params: Argon2idParams | None = None,
    pbkdf2_params: PBKDF2Params | None = None,
) -> bytes:
    """Derive a symmetric key using the requested backend."""

    if length <= 0:
        raise KDFError("Derived key length must be positive.")
    chosen_backend: KDFBackend
    if backend is not None:
        chosen_backend = backend
    else:
        chosen_backend = "argon2id" if _HASH_SECRET_RAW is not None else "pbkdf2"

    if chosen_backend == "argon2id":
        if _HASH_SECRET_RAW is None or _ARGON2_TYPE_ID is None:
            raise KDFError("Argon2id backend requested but argon2-cffi is unavailable.")
        params = argon2_params or Argon2idParams()
        return _HASH_SECRET_RAW(
            secret,
            salt,
            time_cost=params.time_cost,
            memory_cost=params.memory_cost,
            parallelism=params.parallelism,
            hash_len=length,
            type=_ARGON2_TYPE_ID,
        )

    params = pbkdf2_params or PBKDF2Params()
    return pbkdf2_hmac(params.hash_name, secret, salt, params.iterations, dklen=length)
