"""Quality and capacity policies for arithmetic steganography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .errors import QualityConfigError


class QualityPolicy(Protocol):
    """Protocol for capacity-quality balancing policies."""

    def validate(self) -> None:
        """Validate the configuration."""


@dataclass
class TopKPolicy:
    """Policy constraining sampling to the top-k tokens."""

    k: int

    def validate(self) -> None:
        if self.k <= 0:
            raise QualityConfigError("k must be positive for TopKPolicy")


@dataclass
class TopPPolicy:
    """Policy constraining sampling to a probability mass threshold."""

    p: float

    def validate(self) -> None:
        if not 0 < self.p <= 1:
            raise QualityConfigError("p must be within (0, 1] for TopPPolicy")


@dataclass
class CapacityPerTokenPolicy:
    """Policy limiting the number of embedded bits per token."""

    max_bits: int

    def validate(self) -> None:
        if self.max_bits <= 0:
            raise QualityConfigError("max_bits must be positive for CapacityPerTokenPolicy")
