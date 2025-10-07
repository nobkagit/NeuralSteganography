"""High-level public API for neural steganography codecs."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .codec.types import LMProvider

_HEADER_BYTES = 4
_NIBBLE_BASE = 16


def encode_text(
    message_bits: bytes,
    lm: LMProvider,
    *,
    quality: dict,
    seed_text: str = "",
) -> list[int]:
    """Encode the provided bit payload into a list of token identifiers.

    This placeholder implementation does not perform arithmetic coding yet.
    Instead it converts the payload into hexadecimal nibbles which serve as
    stand-in token identifiers compatible with the mock language model used
    throughout the current development phase.
    """

    if not isinstance(message_bits, (bytes, bytearray)):
        raise TypeError("message_bits must be a bytes-like object")
    _validate_quality_mapping(quality)

    context_ids = seed_to_ids(seed_text, getattr(lm, "tokenizer", None))
    _probe_language_model(lm, context_ids)

    length_prefix = len(message_bits).to_bytes(_HEADER_BYTES, "big")
    header_tokens = _bytes_to_nibbles(length_prefix)
    payload_tokens = _bytes_to_nibbles(bytes(message_bits))

    return [*context_ids, *header_tokens, *payload_tokens]


def decode_text(
    token_ids: Sequence[int],
    lm: LMProvider,
    *,
    quality: dict,
    seed_text: str = "",
) -> bytes:
    """Recover the embedded payload from a list of token identifiers."""

    _validate_quality_mapping(quality)
    _ensure_token_sequence(token_ids)

    context_ids = seed_to_ids(seed_text, getattr(lm, "tokenizer", None))
    _probe_language_model(lm, context_ids)

    context_length = len(context_ids)
    if tuple(token_ids[:context_length]) != tuple(context_ids):
        raise ValueError("Token stream does not match the expected seed text context")

    header_start = context_length
    header_end = header_start + (_HEADER_BYTES * 2)
    if len(token_ids) < header_end:
        raise ValueError("Token stream is shorter than the required header")

    header_tokens = token_ids[header_start:header_end]
    header_bytes = _nibbles_to_bytes(header_tokens)
    payload_length = int.from_bytes(header_bytes, "big")

    payload_tokens = token_ids[header_end:]
    expected_payload_tokens = payload_length * 2
    if len(payload_tokens) != expected_payload_tokens:
        raise ValueError("Token stream length does not match the embedded payload size")

    return _nibbles_to_bytes(payload_tokens)


def seed_to_ids(seed_text: str, tokenizer: Any | None) -> list[int]:
    """Convert seed text into deterministic token identifiers (placeholder)."""

    _ = tokenizer  # tokenizer integration planned for a later phase
    seed_bytes = seed_text.encode("utf-8")
    return _bytes_to_nibbles(seed_bytes)


def _bytes_to_nibbles(data: bytes) -> list[int]:
    nibbles: list[int] = []
    for byte in data:
        high = (byte >> 4) & 0x0F
        low = byte & 0x0F
        nibbles.append(high)
        nibbles.append(low)
    return nibbles


def _nibbles_to_bytes(nibbles: Sequence[int]) -> bytes:
    if len(nibbles) % 2 != 0:
        raise ValueError("Nibble sequence must contain an even number of entries")
    bytes_out = bytearray(len(nibbles) // 2)
    for index in range(0, len(nibbles), 2):
        high, low = nibbles[index], nibbles[index + 1]
        _validate_nibble(high)
        _validate_nibble(low)
        bytes_out[index // 2] = (high << 4) | low
    return bytes(bytes_out)


def _validate_nibble(value: int) -> None:
    if not isinstance(value, int):
        raise TypeError("Token identifiers must be integers")
    if not 0 <= value < _NIBBLE_BASE:
        raise ValueError("Token identifiers must be in the range [0, 15]")


def _ensure_token_sequence(token_ids: Sequence[int]) -> None:
    if not isinstance(token_ids, Sequence):
        raise TypeError("token_ids must be a sequence of integers")
    for token in token_ids:
        _validate_nibble(token)


def _probe_language_model(lm: LMProvider, context_ids: Iterable[int]) -> None:
    """Exercise the language model interface to surface integration errors early."""

    context_tuple = tuple(context_ids)
    _ = lm.next_token_probs(context_tuple)


def _validate_quality_mapping(quality: dict) -> None:
    if not isinstance(quality, dict):
        raise TypeError("quality must be a dictionary of codec hints")
