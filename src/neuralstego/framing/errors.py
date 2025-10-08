"""Exception types for the framing subsystem."""

from __future__ import annotations


class FramingError(RuntimeError):
    """Base class for framing related errors."""


class PacketValidationError(FramingError):
    """Raised when a packet fails schema validation."""


class PacketVersionError(PacketValidationError):
    """Raised when the packet version is unsupported."""


class PacketIntegrityError(PacketValidationError):
    """Raised when integrity checks fail (CRC/ECC)."""


class PacketConsistencyError(PacketValidationError):
    """Raised when packets of the same message disagree on metadata."""


class NotAvailableError(FramingError):
    """Raised when an optional dependency is unavailable."""
