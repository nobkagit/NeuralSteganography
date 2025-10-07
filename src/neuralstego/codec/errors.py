"""Custom exception hierarchy for the codec package."""

from __future__ import annotations


class CodecError(Exception):
    """Base class for codec-specific exceptions."""


class ArithmeticRangeError(CodecError):
    """Raised when arithmetic coder encounters an invalid probability range."""


class DecodeDivergenceError(CodecError):
    """Raised when decoding diverges from the expected arithmetic interval."""


class QualityConfigError(CodecError):
    """Raised when quality or capacity policies are misconfigured."""
