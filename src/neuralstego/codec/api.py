"""High level encoding and decoding APIs for arithmetic steganography."""

from __future__ import annotations

from typing import Sequence

from .types import LMProvider


def encode_arithmetic(
    bits: bytes,
    lm: LMProvider,
    *,
    quality: dict,
    seed_text: str = "",
) -> list[int]:
    """Encode a bit payload into a sequence of token identifiers."""

    raise NotImplementedError("High-level encoding will be implemented in a later phase")


def decode_arithmetic(
    token_ids: Sequence[int],
    lm: LMProvider,
    *,
    quality: dict,
    seed_text: str = "",
) -> bytes:
    """Decode an embedded payload from a sequence of token identifiers."""

    raise NotImplementedError("High-level decoding will be implemented in a later phase")
