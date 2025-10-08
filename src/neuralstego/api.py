"""High level neural steganography API for chunked framing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol

from .codec import assemble_bytes, build_packet, chunk_bytes, make_msg_id, parse_packet
from .exceptions import ConfigurationError, MissingChunksError


class LMProvider(Protocol):
    """Protocol implemented by language model wrappers used for arithmetic coding."""

    def encode_arithmetic(
        self, bits: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        ...

    def decode_arithmetic(
        self, tokens: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        ...

    def encode_seed(self, text: str) -> List[int]:
        ...


@dataclass
class EncodeMetadata:
    msg_id: str
    total: int
    cfg: Dict[str, object]


class EncodeResult(list):
    """List-like container with attached metadata for encode results."""

    def __init__(self, spans: Iterable[List[int]], metadata: EncodeMetadata) -> None:
        super().__init__(spans)
        self.metadata = metadata


_DEFAULT_QUALITY = {
    "temp": 1.0,
    "precision": 16,
    "topk": 50000,
    "finish_sent": True,
}


def _normalise_ecc(ecc: Optional[str]) -> str:
    if not ecc:
        return "none"
    ecc_norm = ecc.lower()
    if ecc_norm not in {"none", "rs"}:
        raise ConfigurationError(f"unsupported ecc mode: {ecc}")
    return ecc_norm


def _bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        bits.extend((byte >> i) & 1 for i in range(8))
    return bits


def _bits_to_bytes(bits: Iterable[int]) -> bytes:
    bits_list = list(bits)
    if len(bits_list) % 8:
        raise ConfigurationError("decoded bit stream is not byte aligned")
    out = bytearray()
    for i in range(0, len(bits_list), 8):
        value = 0
        for offset, bit in enumerate(bits_list[i : i + 8]):
            value |= (bit & 1) << offset
        out.append(value)
    return bytes(out)


def _context_tokens(lm: LMProvider, seed_text: str) -> List[int]:
    return lm.encode_seed(seed_text)


def stego_encode(
    message: bytes,
    *,
    chunk_bytes: int = 256,
    use_crc: bool = True,
    ecc: str | None = "rs",
    nsym: int = 10,
    quality: Optional[Dict[str, float]] = None,
    seed_text: str = "",
    lm: LMProvider,
) -> EncodeResult:
    """Encode *message* into token spans using arithmetic coding."""

    ecc_mode = _normalise_ecc(ecc)
    cfg = {
        "chunk_bytes": int(chunk_bytes),
        "crc": bool(use_crc),
        "ecc": ecc_mode,
        "nsym": int(nsym if ecc_mode == "rs" else 0),
    }
    quality_args = {**_DEFAULT_QUALITY, **(quality or {})}

    chunks = chunk_bytes_func(message, chunk_size=cfg["chunk_bytes"])
    msg_id = make_msg_id()
    total = len(chunks)

    metadata = EncodeMetadata(msg_id=msg_id, total=total, cfg=cfg)
    spans: List[List[int]] = []

    for seq, chunk in enumerate(chunks):
        packet_bytes = build_packet(
            chunk,
            msg_id=msg_id,
            seq=seq,
            total=total,
            cfg=cfg,
        )
        bits = _bytes_to_bits(packet_bytes)
        context = _context_tokens(lm, seed_text)
        tokens = lm.encode_arithmetic(bits, context, quality=quality_args)
        spans.append(tokens)

    return EncodeResult(spans, metadata)


def stego_decode(
    spans: Iterable[List[int]],
    *,
    use_crc: bool = True,
    ecc: str | None = "rs",
    nsym: int = 10,
    quality: Optional[Dict[str, float]] = None,
    seed_text: str = "",
    lm: LMProvider,
) -> bytes:
    """Decode arithmetic coded *spans* back into the original message."""

    ecc_mode = _normalise_ecc(ecc)
    expected_cfg = {
        "crc": bool(use_crc),
        "ecc": ecc_mode,
        "nsym": int(nsym if ecc_mode == "rs" else 0),
    }
    quality_args = {**_DEFAULT_QUALITY, **(quality or {})}

    payload_by_seq: Dict[int, bytes] = {}
    msg_id: Optional[str] = None
    total: Optional[int] = None

    for span in spans:
        span_list = list(span)
        context = _context_tokens(lm, seed_text)
        bits = lm.decode_arithmetic(span_list, context, quality=quality_args)
        packet_bytes = _bits_to_bytes(bits)
        packet = parse_packet(packet_bytes, expected_cfg=expected_cfg)

        if msg_id is None:
            msg_id = packet.msg_id
            total = packet.total
        else:
            if packet.msg_id != msg_id:
                raise ConfigurationError("decoded packet msg_id mismatch")
            if packet.total != total:
                raise ConfigurationError("decoded packet total mismatch")

        if packet.seq in payload_by_seq:
            raise ConfigurationError(f"duplicate packet sequence {packet.seq}")
        payload_by_seq[packet.seq] = packet.payload

    if total is None:
        return b""

    present_indices = sorted(payload_by_seq.keys())
    missing = sorted(set(range(total)) - payload_by_seq.keys())
    ordered_payloads = [payload_by_seq[i] for i in present_indices]
    assembled = assemble_bytes(ordered_payloads)

    if missing:
        raise MissingChunksError(missing_indices=missing, partial_payload=assembled)

    return assembled


# Provide aliases compatible with previous module layout
chunk_bytes_func = chunk_bytes

__all__ = [
    "EncodeMetadata",
    "EncodeResult",
    "LMProvider",
    "chunk_bytes_func",
    "stego_decode",
    "stego_encode",
]
