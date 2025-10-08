"""Utilities for chunking byte payloads for independent encoding."""
from __future__ import annotations

from typing import Iterable, List
from uuid import uuid4


def make_msg_id() -> str:
    """Return a random message identifier suitable for packet headers."""

    return str(uuid4())


def chunk_bytes(data: bytes, *, chunk_size: int = 256) -> List[bytes]:
    """Split *data* into a list of chunks with maximum size *chunk_size*.

    Args:
        data: The raw payload to split.
        chunk_size: Maximum size in bytes for each chunk. Must be positive.

    Returns:
        A list of ``bytes`` objects containing the chunked payload.

    Raises:
        ValueError: If *chunk_size* is not a positive integer.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not data:
        return [b""]

    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def assemble_bytes(chunks: Iterable[bytes]) -> bytes:
    """Reassemble *chunks* that were previously produced by :func:`chunk_bytes`."""

    return b"".join(chunks)


__all__ = ["assemble_bytes", "chunk_bytes", "make_msg_id"]
