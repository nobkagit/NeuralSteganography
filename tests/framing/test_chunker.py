import base64
import json
import uuid

import pytest

from neuralstego.framing import (
    ECCCfg,
    PacketCfg,
    PacketConsistencyError,
    PacketIntegrityError,
    chunk_payload,
    reassemble_packets,
)
from neuralstego.framing.errors import NotAvailableError


def _corrupt_packet(blob: bytes, *, flip_positions: list[int]) -> bytes:
    data = json.loads(blob.decode("utf-8"))
    ct = bytearray(base64.b64decode(data["ct"]))
    for pos in flip_positions:
        ct[pos] ^= 0xFF
    data["ct"] = base64.b64encode(bytes(ct)).decode("ascii")
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


def test_chunker_roundtrip_crc_only():
    cfg = PacketCfg(crc="crc32")
    payload = bytes(range(100))
    meta = {"mime": "application/octet-stream"}

    packets = chunk_payload(payload, chunk_size=16, cfg=cfg, meta=meta, store_plain=True)
    assert len(packets) == (len(payload) + 15) // 16

    recovered, recovered_cfg, recovered_meta, msg_id = reassemble_packets(packets)
    assert recovered == payload
    assert recovered_cfg == cfg
    assert recovered_meta == meta
    uuid.UUID(msg_id)  # Should not raise.

    with pytest.raises(PacketConsistencyError):
        reassemble_packets(packets[:-1])


def test_chunker_with_ecc_recovers_corruption():
    cfg = PacketCfg(crc="crc32", ecc=ECCCfg(name="rs", nsym=8))
    payload = b"error correction demo payload"

    try:
        packets = chunk_payload(payload, chunk_size=10, cfg=cfg, store_plain=False)
    except NotAvailableError:
        pytest.skip("reedsolo not installed")

    corrupted = list(packets)
    corrupted[0] = _corrupt_packet(corrupted[0], flip_positions=[0, 1])

    recovered, _, _, _ = reassemble_packets(corrupted)
    assert recovered == payload


def test_chunker_detects_unrecoverable_corruption():
    cfg = PacketCfg(crc="crc32", ecc=ECCCfg(name="rs", nsym=4))
    payload = b"short"

    try:
        packets = chunk_payload(payload, chunk_size=5, cfg=cfg, store_plain=False)
    except NotAvailableError:
        pytest.skip("reedsolo not installed")

    corrupted = list(packets)
    corrupted[0] = _corrupt_packet(corrupted[0], flip_positions=[0, 1, 2])

    with pytest.raises(PacketIntegrityError):
        reassemble_packets(corrupted)
