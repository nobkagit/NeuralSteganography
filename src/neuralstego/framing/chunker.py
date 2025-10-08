"""Helpers for chunking payloads into packets and reassembling them."""

from __future__ import annotations

import uuid
from typing import List, Optional, Sequence, Tuple

from .crc import append_crc32, verify_crc32
from .ecc import rs_decode, rs_encode
from .errors import PacketConsistencyError, PacketIntegrityError, PacketValidationError
from .packet import PacketCfg, ParsedPacket, build_packet, parse_packet


def _normalise_payload(payload: bytes) -> bytes:
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    raise PacketValidationError("payload must be bytes")


def _apply_ecc(cfg: PacketCfg, data: bytes) -> bytes:
    if not cfg.ecc.enabled:
        return data
    if cfg.ecc.name != "rs":
        raise PacketValidationError(f"Unsupported ECC codec: {cfg.ecc.name}")
    nsym = cfg.ecc.nsym or 10
    return rs_encode(data, nsym=nsym)


def _remove_ecc(cfg: PacketCfg, data: bytes) -> Tuple[bool, bytes]:
    if not cfg.ecc.enabled:
        return True, data
    if cfg.ecc.name != "rs":
        raise PacketValidationError(f"Unsupported ECC codec: {cfg.ecc.name}")
    nsym = cfg.ecc.nsym or 10
    return rs_decode(data, nsym=nsym)


def chunk_payload(
    payload: bytes,
    *,
    chunk_size: int,
    cfg: PacketCfg,
    meta: Optional[dict] = None,
    msg_id: Optional[str] = None,
    store_plain: bool = False,
) -> List[bytes]:
    """Split *payload* into framed packet blobs."""

    if chunk_size <= 0:
        raise PacketValidationError("chunk_size must be positive")
    payload = _normalise_payload(payload)
    msg_uuid = msg_id or str(uuid.uuid4())

    chunks = [payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)]
    if not chunks:
        chunks = [b""]
    total = len(chunks)

    packets: List[bytes] = []
    for seq, chunk in enumerate(chunks):
        plain_chunk = bytes(chunk)
        processed = plain_chunk
        if cfg.crc_enabled:
            processed = append_crc32(processed)
        processed = _apply_ecc(cfg, processed)
        packet = build_packet(
            processed,
            seq=seq,
            total=total,
            msg_id=msg_uuid,
            cfg=cfg,
            meta=meta,
            plain_payload=plain_chunk if store_plain else None,
        )
        packets.append(packet)
    return packets


def reassemble_packets(blobs: Sequence[bytes]) -> Tuple[bytes, PacketCfg, Optional[dict], str]:
    """Reconstruct the original payload from a sequence of packet blobs."""

    if not blobs:
        raise PacketValidationError("No packets supplied")

    packets: List[ParsedPacket] = [parse_packet(blob) for blob in blobs]
    packets.sort(key=lambda pkt: pkt.seq)

    first = packets[0]
    total = first.total
    if len(packets) != total:
        raise PacketConsistencyError("Missing packets for reconstruction")

    for idx, pkt in enumerate(packets):
        if pkt.seq != idx:
            raise PacketConsistencyError("Packet sequence numbers are not contiguous")
        if pkt.total != total:
            raise PacketConsistencyError("Packet totals differ")
        if pkt.msg_id != first.msg_id:
            raise PacketConsistencyError("Packets belong to different messages")
        if pkt.cfg != first.cfg:
            raise PacketConsistencyError("Packet configurations differ")
        if pkt.meta != first.meta:
            raise PacketConsistencyError("Packet metadata differs")

    cfg = first.cfg
    recovered: List[bytes] = []
    for pkt in packets:
        data = pkt.payload
        ok, data = _remove_ecc(cfg, data)
        if not ok:
            raise PacketIntegrityError("ECC decoding failed")
        if cfg.crc_enabled:
            ok, data = verify_crc32(data)
            if not ok:
                raise PacketIntegrityError("CRC mismatch detected")
        recovered.append(data)

    return b"".join(recovered), cfg, first.meta, first.msg_id
