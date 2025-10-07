"""Core arithmetic coding algorithms for neural steganography."""

from __future__ import annotations

from typing import Iterable, Sequence

from .types import CodecState, ProbDist


def encode_bits(bits: bytes, probs: Iterable[ProbDist], *, state: CodecState | None = None) -> Sequence[int]:
    """Encode a bitstream into a sequence of token identifiers."""

    raise NotImplementedError("Arithmetic encoder will be implemented in a later phase")


def decode_bits(tokens: Sequence[int], probs: Iterable[ProbDist], *, state: CodecState | None = None) -> bytes:
    """Decode a sequence of token identifiers back into the embedded bitstream."""

    raise NotImplementedError("Arithmetic decoder will be implemented in a later phase")
