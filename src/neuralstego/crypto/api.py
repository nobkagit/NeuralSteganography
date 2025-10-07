"""High-level API for orchestrating cryptographic operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .aead import decrypt, encrypt
from .envelope import Envelope, EnvelopeComponents, seal_envelope, open_envelope
from .kdf import (
    Argon2idParams,
    KDFBackend,
    PBKDF2Params,
    derive_key,
    generate_salt,
)

__all__ = [
    "KeyDerivationSpec",
    "encrypt_message",
    "decrypt_message",
]

DEFAULT_KEY_LENGTH: Final[int] = 32


@dataclass(frozen=True)
class KeyDerivationSpec:
    """Configuration controlling password-based key derivation."""

    backend: KDFBackend | None = None
    argon2_params: Argon2idParams | None = None
    pbkdf2_params: PBKDF2Params | None = None
    length: int = DEFAULT_KEY_LENGTH
    salt_size: int = 16


def encrypt_message(
    password: bytes,
    plaintext: bytes,
    *,
    associated_data: bytes | None = None,
    kdf_spec: KeyDerivationSpec | None = None,
) -> Envelope:
    """Encrypt ``plaintext`` with a password-derived key."""

    spec = kdf_spec or KeyDerivationSpec()
    salt = generate_salt(spec.salt_size)
    key = derive_key(
        password,
        salt,
        length=spec.length,
        backend=spec.backend,
        argon2_params=spec.argon2_params,
        pbkdf2_params=spec.pbkdf2_params,
    )
    ciphertext = encrypt(key, plaintext, associated_data=associated_data)
    components = EnvelopeComponents(salt=salt, ciphertext=ciphertext)
    return seal_envelope(components)


def decrypt_message(
    password: bytes,
    envelope: Envelope,
    *,
    associated_data: bytes | None = None,
    kdf_spec: KeyDerivationSpec | None = None,
) -> bytes:
    """Decrypt ``envelope`` with a password-derived key."""

    spec = kdf_spec or KeyDerivationSpec()
    components = open_envelope(envelope)
    key = derive_key(
        password,
        components.salt,
        length=spec.length,
        backend=spec.backend,
        argon2_params=spec.argon2_params,
        pbkdf2_params=spec.pbkdf2_params,
    )
    return decrypt(key, components.ciphertext, associated_data=associated_data)
