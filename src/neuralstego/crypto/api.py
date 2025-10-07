"""High-level API for orchestrating cryptographic operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

from .aead import decrypt, encrypt
from .envelope import Envelope, EnvelopeComponents, open_envelope, seal_envelope
from .kdf import KDFMethod, derive_key, gen_salt

__all__ = [
    "Argon2idOptions",
    "PBKDF2Options",
    "KeyDerivationSpec",
    "encrypt_message",
    "decrypt_message",
]

DEFAULT_KEY_LENGTH: Final[int] = 32
DEFAULT_SALT_SIZE: Final[int] = 16


@dataclass(frozen=True)
class Argon2idOptions:
    """Parameters to control Argon2id key derivation."""

    time_cost: int = 3
    memory_cost: int = 64 * 1024
    parallelism: int = 2


@dataclass(frozen=True)
class PBKDF2Options:
    """Parameters to control PBKDF2 key derivation."""

    iterations: int = 310_000
    hash_name: Literal["sha256", "sha512"] = "sha256"


@dataclass(frozen=True)
class KeyDerivationSpec:
    """Configuration describing how to derive a key from a password."""

    method: KDFMethod = "argon2id"
    length: int = DEFAULT_KEY_LENGTH
    salt_size: int = DEFAULT_SALT_SIZE
    argon2: Argon2idOptions = field(default_factory=Argon2idOptions)
    pbkdf2: PBKDF2Options = field(default_factory=PBKDF2Options)


def _derive_password_key(password: str, salt: bytes, spec: KeyDerivationSpec) -> bytes:
    """Derive a symmetric key based on ``spec`` parameters."""

    kwargs: dict[str, Any]
    if spec.method == "argon2id":
        kwargs = {
            "length": spec.length,
            "time_cost": spec.argon2.time_cost,
            "memory_cost": spec.argon2.memory_cost,
            "parallelism": spec.argon2.parallelism,
        }
    else:
        kwargs = {
            "length": spec.length,
            "iterations": spec.pbkdf2.iterations,
            "hash_name": spec.pbkdf2.hash_name,
        }
    return derive_key(password, salt, method=spec.method, **kwargs)


def encrypt_message(
    password: str,
    plaintext: bytes,
    *,
    associated_data: bytes | None = None,
    kdf_spec: KeyDerivationSpec | None = None,
) -> Envelope:
    """Encrypt ``plaintext`` with a password-derived key."""

    spec = kdf_spec or KeyDerivationSpec()
    salt = gen_salt(spec.salt_size)
    key = _derive_password_key(password, salt, spec)
    ciphertext = encrypt(key, plaintext, associated_data=associated_data)
    components = EnvelopeComponents(salt=salt, ciphertext=ciphertext)
    return seal_envelope(components)


def decrypt_message(
    password: str,
    envelope: Envelope,
    *,
    associated_data: bytes | None = None,
    kdf_spec: KeyDerivationSpec | None = None,
) -> bytes:
    """Decrypt ``envelope`` with a password-derived key."""

    spec = kdf_spec or KeyDerivationSpec()
    components = open_envelope(envelope)
    key = _derive_password_key(password, components.salt, spec)
    return decrypt(key, components.ciphertext, associated_data=associated_data)
