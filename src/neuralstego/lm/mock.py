"""Mock language model for testing the steganography pipeline."""
from __future__ import annotations

from typing import Dict, Iterable, List


class MockTokenizer:
    def encode(self, text: str) -> List[int]:  # pragma: no cover - trivial
        return list(text.encode("utf-8"))

    def decode(self, tokens: Iterable[int]) -> str:  # pragma: no cover - trivial
        return bytes(int(token) % 256 for token in tokens).decode("utf-8", errors="ignore")


def _bits_to_bytes(bits: Iterable[int]) -> bytes:
    data = list(bits)
    if len(data) % 8:
        raise ValueError("bit stream length must be a multiple of 8")
    out = bytearray()
    for i in range(0, len(data), 8):
        value = 0
        for offset, bit in enumerate(data[i : i + 8]):
            value |= (bit & 1) << offset
        out.append(value)
    return bytes(out)


def _bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        bits.extend((byte >> i) & 1 for i in range(8))
    return bits


class MockLM:
    """Deterministic mock used for unit tests."""

    def __init__(self) -> None:
        self.tokenizer = MockTokenizer()

    def encode_seed(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def encode_arithmetic(
        self, bits: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        if not bits:
            return []
        payload = _bits_to_bytes(bits)
        return [int(b) for b in payload]

    def decode_arithmetic(
        self, tokens: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        payload = bytes(int(t) % 256 for t in tokens)
        return _bytes_to_bits(payload)


__all__ = ["MockLM", "MockTokenizer"]
