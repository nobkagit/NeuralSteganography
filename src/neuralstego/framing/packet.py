"""Packet building and parsing utilities."""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .errors import PacketValidationError, PacketVersionError

SUPPORTED_VERSION = 1


@dataclass(frozen=True)
class ECCCfg:
    """Configuration for optional ECC processing."""

    name: str = "none"
    nsym: Optional[int] = None

    def to_dict(self) -> Optional[Dict[str, Any]]:
        if self.name == "none":
            return None
        data: Dict[str, Any] = {"name": self.name}
        if self.nsym is not None:
            data["nsym"] = self.nsym
        return data

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ECCCfg":
        if not data:
            return cls()
        if not isinstance(data, dict):
            raise PacketValidationError("'ecc' must be an object when provided")
        name = data.get("name", "none")
        nsym = data.get("nsym")
        if nsym is not None and (not isinstance(nsym, int) or nsym <= 0):
            raise PacketValidationError("'ecc.nsym' must be a positive integer")
        return cls(name=name, nsym=nsym)

    @property
    def enabled(self) -> bool:
        return self.name != "none"


@dataclass(frozen=True)
class PacketCfg:
    """Packet level integrity configuration."""

    crc: str = "none"
    ecc: ECCCfg = field(default_factory=ECCCfg)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "ecc", self.ecc if isinstance(self.ecc, ECCCfg) else ECCCfg.from_dict(self.ecc))
        if self.crc not in {"none", "crc32"}:
            raise PacketValidationError("Unsupported CRC mode")
        if not isinstance(self.ecc, ECCCfg):
            raise PacketValidationError("Invalid ECC configuration")

    def to_dict(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"crc": self.crc}
        ecc_dict = self.ecc.to_dict()
        if ecc_dict is not None:
            cfg["ecc"] = ecc_dict
        return cfg

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "PacketCfg":
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise PacketValidationError("'cfg' must be an object")
        crc = data.get("crc", "none")
        ecc = ECCCfg.from_dict(data.get("ecc"))
        return cls(crc=crc, ecc=ecc)

    @property
    def crc_enabled(self) -> bool:
        return self.crc == "crc32"


@dataclass(frozen=True)
class ParsedPacket:
    """Representation of a parsed packet."""

    version: int
    msg_id: str
    seq: int
    total: int
    cfg: PacketCfg
    meta: Optional[Dict[str, Any]]
    payload: bytes
    plain_payload: Optional[bytes]


def _ensure_uuid(msg_id: str) -> str:
    try:
        uuid.UUID(msg_id)
    except (ValueError, AttributeError):
        raise PacketValidationError("'id' must be a valid UUID string") from None
    return msg_id


def build_packet(
    payload: bytes,
    *,
    seq: int,
    total: int,
    msg_id: str,
    cfg: PacketCfg,
    meta: Optional[Dict[str, Any]] = None,
    plain_payload: Optional[bytes] = None,
) -> bytes:
    """Build a serialised packet blob."""

    if not isinstance(payload, (bytes, bytearray)):
        raise PacketValidationError("payload must be bytes")
    if seq < 0:
        raise PacketValidationError("'seq' must be non-negative")
    if total <= 0 or seq >= total:
        raise PacketValidationError("'total' must be positive and seq < total")
    if not isinstance(msg_id, str):
        raise PacketValidationError("'id' must be a string")

    msg_id = _ensure_uuid(msg_id)
    cfg_dict = cfg.to_dict()

    packet: Dict[str, Any] = {
        "v": SUPPORTED_VERSION,
        "id": msg_id,
        "seq": seq,
        "total": total,
        "cfg": cfg_dict,
    }
    if meta is not None:
        if not isinstance(meta, dict):
            raise PacketValidationError("'meta' must be a mapping when provided")
        packet["meta"] = meta
    if plain_payload is not None:
        packet["pt"] = base64.b64encode(bytes(plain_payload)).decode("ascii")
    packet["ct"] = base64.b64encode(bytes(payload)).decode("ascii")

    return json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")


def parse_packet(blob: bytes) -> ParsedPacket:
    """Parse a packet created by :func:`build_packet`."""

    if not isinstance(blob, (bytes, bytearray)):
        raise PacketValidationError("Packet blob must be bytes")

    try:
        data = json.loads(bytes(blob).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        raise PacketValidationError("Invalid packet encoding") from exc

    if not isinstance(data, dict):
        raise PacketValidationError("Packet must decode to an object")

    version = data.get("v")
    if version != SUPPORTED_VERSION:
        raise PacketVersionError(f"Unsupported packet version: {version!r}")

    msg_id = data.get("id")
    seq = data.get("seq")
    total = data.get("total")
    cfg_dict = data.get("cfg")
    meta = data.get("meta") if "meta" in data else None

    if not isinstance(seq, int) or seq < 0:
        raise PacketValidationError("'seq' must be a non-negative integer")
    if not isinstance(total, int) or total <= 0 or seq >= total:
        raise PacketValidationError("'total' must be a positive integer with seq < total")
    if not isinstance(msg_id, str):
        raise PacketValidationError("'id' must be a string")
    msg_id = _ensure_uuid(msg_id)

    cfg = PacketCfg.from_dict(cfg_dict)

    payload_b64 = data.get("ct")
    if not isinstance(payload_b64, str):
        raise PacketValidationError("'ct' must be a base64 string")
    try:
        payload = base64.b64decode(payload_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise PacketValidationError("'ct' is not valid base64") from exc

    plain_payload_b64 = data.get("pt")
    plain_payload = None
    if plain_payload_b64 is not None:
        if not isinstance(plain_payload_b64, str):
            raise PacketValidationError("'pt' must be a base64 string")
        try:
            plain_payload = base64.b64decode(plain_payload_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise PacketValidationError("'pt' is not valid base64") from exc

    if meta is not None and not isinstance(meta, dict):
        raise PacketValidationError("'meta' must be an object when provided")

    return ParsedPacket(
        version=version,
        msg_id=msg_id,
        seq=seq,
        total=total,
        cfg=cfg,
        meta=meta,
        payload=payload,
        plain_payload=plain_payload,
    )
