"""CRC helper functions."""

from __future__ import annotations

import struct
import zlib

CRC32_POLY = 0xEDB88320
CRC32_INITIAL = 0


def crc32(data: bytes) -> int:
    """Compute the CRC32 checksum of *data*.

    The checksum uses the same polynomial as :func:`zlib.crc32` and returns an
    unsigned 32-bit integer.
    """

    return zlib.crc32(data, CRC32_INITIAL) & 0xFFFFFFFF


def append_crc32(payload: bytes) -> bytes:
    """Return ``payload`` concatenated with a big-endian CRC32."""

    checksum = crc32(payload)
    return payload + struct.pack(">I", checksum)


def verify_crc32(blob: bytes) -> tuple[bool, bytes]:
    """Verify the CRC32 appended to *blob*.

    Returns a tuple ``(ok, payload_without_crc)``.  ``ok`` is ``True`` when the
    checksum matches, ``False`` otherwise.  The payload is returned without the
    trailing 4-byte checksum regardless of the outcome.  When *blob* is shorter
    than four bytes, the function treats the checksum as missing and returns
    ``False`` together with the unchanged data.
    """

    if len(blob) < 4:
        return False, blob

    payload, checksum_bytes = blob[:-4], blob[-4:]
    expected = struct.unpack(">I", checksum_bytes)[0]
    actual = crc32(payload)
    return actual == expected, payload
