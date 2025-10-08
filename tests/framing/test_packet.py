import uuid

import pytest

from neuralstego.framing.packet import ECCCfg, PacketCfg, build_packet, parse_packet
from neuralstego.framing.errors import PacketValidationError


def test_packet_roundtrip():
    payload = b"hello world"
    msg_id = str(uuid.uuid4())
    cfg = PacketCfg(crc="crc32", ecc=ECCCfg(name="rs", nsym=10))
    meta = {"mime": "text/plain", "note": "demo"}

    blob = build_packet(
        payload,
        seq=0,
        total=1,
        msg_id=msg_id,
        cfg=cfg,
        meta=meta,
        plain_payload=payload,
    )

    parsed = parse_packet(blob)
    assert parsed.msg_id == msg_id
    assert parsed.seq == 0
    assert parsed.total == 1
    assert parsed.payload == payload
    assert parsed.plain_payload == payload
    assert parsed.cfg == cfg
    assert parsed.meta == meta


def test_packet_validation_errors():
    cfg = PacketCfg()
    msg_id = str(uuid.uuid4())

    with pytest.raises(PacketValidationError):
        build_packet(b"data", seq=-1, total=1, msg_id=msg_id, cfg=cfg)

    with pytest.raises(PacketValidationError):
        blob = build_packet(b"data", seq=0, total=1, msg_id=msg_id, cfg=cfg)
        parse_packet(blob + b"garbage")
