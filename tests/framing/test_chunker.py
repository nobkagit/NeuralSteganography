import os

import pytest

from neuralstego.codec import assemble_bytes, build_packet, chunk_bytes, make_msg_id, parse_packet


@pytest.mark.parametrize(
    "cfg",
    [
        {"chunk_bytes": 128, "crc": False, "ecc": "none", "nsym": 0},
        {"chunk_bytes": 256, "crc": True, "ecc": "none", "nsym": 0},
        pytest.param(
            {"chunk_bytes": 256, "crc": True, "ecc": "rs", "nsym": 10},
            marks=pytest.mark.requires_reedsolo,
        ),
    ],
)
def test_chunk_build_parse_roundtrip(cfg):
    if cfg["ecc"] == "rs":
        pytest.importorskip("reedsolo")

    message = os.urandom(4096)
    chunks = chunk_bytes(message, chunk_size=cfg["chunk_bytes"])
    msg_id = make_msg_id()
    total = len(chunks)

    packets = [
        build_packet(chunk, msg_id=msg_id, seq=idx, total=total, cfg=cfg)
        for idx, chunk in enumerate(chunks)
    ]

    recovered = [
        parse_packet(packet, expected_cfg={"crc": cfg["crc"], "ecc": cfg["ecc"], "nsym": cfg["nsym"]})
        for packet in packets
    ]
    assembled = assemble_bytes(packet.payload for packet in recovered)
    assert assembled == message
