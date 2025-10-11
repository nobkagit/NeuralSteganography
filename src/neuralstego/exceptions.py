"""Custom exception hierarchy for the neural steganography toolkit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


class NeuralStegoError(Exception):
    """Base class for all neural-steganography errors."""


class ConfigurationError(NeuralStegoError):
    """Raised when user-supplied configuration is invalid."""


class FramingError(NeuralStegoError):
    """Raised when packet framing or chunk assembly fails."""


class PacketECCError(FramingError):
    """Raised when ECC decoding fails irrecoverably."""


class PacketCRCError(FramingError):
    """Raised when CRC verification fails."""


@dataclass
class MissingChunksError(FramingError):
    missing_indices: List[int]
    partial_payload: bytes

    def __str__(self) -> str:  # pragma: no cover - trivial
        indices = ", ".join(str(i) for i in self.missing_indices)
        return f"Missing chunks at indices: {indices}"


@dataclass
class QualityGateError(NeuralStegoError):
    """Raised when the quality gate rejects all cover-generation attempts."""

    cover_text: str
    reasons: List[str]
    metrics: Dict[str, float]

    def __str__(self) -> str:  # pragma: no cover - human-friendly message
        if not self.reasons:
            return "quality gate rejected the generated cover"
        joined = "; ".join(self.reasons)
        return f"quality gate rejected the generated cover: {joined}"


__all__ = [
    "ConfigurationError",
    "FramingError",
    "MissingChunksError",
    "NeuralStegoError",
    "PacketCRCError",
    "PacketECCError",
    "QualityGateError",
]
