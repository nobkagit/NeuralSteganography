"""Packet framing helpers for chunked payloads."""
from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..exceptions import ConfigurationError, PacketCRCError, PacketECCError

try:
    from reedsolo import RSCodec, ReedSolomonError  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    RSCodec = None
    ReedSolomonError = Exception


@dataclass(frozen=True)
class Packet:
    msg_id: str
    seq: int
    total: int
    cfg: Dict[str, Any]
    payload: bytes


def _ensure_rs_codec(nsym: int) -> RSCodec:
    if RSCodec is None:
        raise ConfigurationError(
            "reedsolo is required for Reed-Solomon ECC but is not installed."
        )
    if nsym <= 0:
        raise ValueError("nsym must be positive when ecc='rs'")
    return RSCodec(nsym)


def _append_crc32(data: bytes) -> bytes:
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return data + struct.pack(">I", crc)


def _verify_crc32(data: bytes) -> bytes:
    if len(data) < 4:
        raise PacketCRCError("payload too small to contain CRC32")
    payload, crc_bytes = data[:-4], data[-4:]
    expected = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    if expected != crc_bytes:
        raise PacketCRCError("CRC32 mismatch detected")
    return payload


def _rs_encode(data: bytes, nsym: int) -> bytes:
    codec = _ensure_rs_codec(nsym)
    return bytes(codec.encode(data))


def _rs_decode(data: bytes, nsym: int) -> bytes:
    codec = _ensure_rs_codec(nsym)
    try:
        message, _, _ = codec.decode(data)
    except ReedSolomonError as exc:  # pragma: no cover - requires corruption
        raise PacketECCError("Reed-Solomon decoding failed") from exc
    return bytes(message)


def build_packet(
    payload: bytes,
    *,
    msg_id: str,
    seq: int,
    total: int,
    cfg: Dict[str, Any],
) -> bytes:
    """Build a binary JSON packet for *payload* with framing metadata."""

    if seq < 0 or total <= 0 or seq >= total:
        raise ValueError("invalid sequence/total combination")

    cfg_norm = {
        "chunk_bytes": cfg.get("chunk_bytes"),
        "crc": bool(cfg.get("crc", False)),
        "ecc": cfg.get("ecc", "none"),
        "nsym": int(cfg.get("nsym", 0)),
    }

    framed = payload
    if cfg_norm["crc"]:
        framed = _append_crc32(framed)

    if cfg_norm["ecc"] == "rs":
        framed = _rs_encode(framed, cfg_norm["nsym"])
    elif cfg_norm["ecc"] not in {"none", None}:
        raise ConfigurationError(f"unsupported ecc mode: {cfg_norm['ecc']}")

    packet_dict = {
        "version": 1,
        "msg_id": msg_id,
        "seq": seq,
        "total": total,
        "cfg": cfg_norm,
        "payload": base64.b64encode(framed).decode("ascii"),
    }

    return json.dumps(packet_dict, separators=(",", ":"), sort_keys=True).encode("utf-8")


def parse_packet(
    packet: bytes,
    *,
    expected_cfg: Optional[Dict[str, Any]] = None,
) -> Packet:
    """Parse *packet* back into a :class:`Packet` instance."""

    try:
        obj = json.loads(packet.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PacketECCError("invalid packet encoding") from exc

    required = {"msg_id", "seq", "total", "cfg", "payload"}
    if not required.issubset(obj):
        missing = ", ".join(sorted(required - obj.keys()))
        raise PacketECCError(f"missing packet keys: {missing}")

    cfg = obj["cfg"]
    cfg_norm = {
        "chunk_bytes": cfg.get("chunk_bytes"),
        "crc": bool(cfg.get("crc", False)),
        "ecc": cfg.get("ecc", "none"),
        "nsym": int(cfg.get("nsym", 0)),
    }

    if expected_cfg:
        for key, value in expected_cfg.items():
            if key in cfg_norm and value is not None and cfg_norm[key] != value:
                raise ConfigurationError(
                    f"packet cfg mismatch for {key}: expected {value}, got {cfg_norm[key]}"
                )

    try:
        framed = base64.b64decode(obj["payload"], validate=True)
    except (ValueError, TypeError) as exc:
        raise PacketECCError("payload is not valid base64") from exc

    if cfg_norm["ecc"] == "rs":
        framed = _rs_decode(framed, cfg_norm["nsym"])
    elif cfg_norm["ecc"] not in {"none", None}:
        raise ConfigurationError(f"unsupported ecc mode: {cfg_norm['ecc']}")

    if cfg_norm["crc"]:
        framed = _verify_crc32(framed)

    return Packet(
        msg_id=obj["msg_id"],
        seq=int(obj["seq"]),
        total=int(obj["total"]),
        cfg=cfg_norm,
        payload=framed,
    )


__all__ = ["Packet", "build_packet", "parse_packet"]
