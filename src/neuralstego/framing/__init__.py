"""Framing utilities for steganographic payload packets."""

from .errors import (
    FramingError,
    PacketValidationError,
    PacketVersionError,
    PacketIntegrityError,
    PacketConsistencyError,
    NotAvailableError,
)
from .packet import PacketCfg, ECCCfg, ParsedPacket, build_packet, parse_packet
from .crc import crc32, append_crc32, verify_crc32
from .ecc import rs_encode, rs_decode
from .chunker import chunk_payload, reassemble_packets

__all__ = [
    "FramingError",
    "PacketValidationError",
    "PacketVersionError",
    "PacketIntegrityError",
    "PacketConsistencyError",
    "NotAvailableError",
    "PacketCfg",
    "ECCCfg",
    "ParsedPacket",
    "build_packet",
    "parse_packet",
    "crc32",
    "append_crc32",
    "verify_crc32",
    "rs_encode",
    "rs_decode",
    "chunk_payload",
    "reassemble_packets",
]

