from neuralstego.framing.crc import append_crc32, crc32, verify_crc32


def test_crc32_known_value():
    payload = b"hello"
    assert crc32(payload) == 0x3610A686


def test_crc_roundtrip_and_detection():
    payload = b"payload"
    blob = append_crc32(payload)
    ok, recovered = verify_crc32(blob)
    assert ok
    assert recovered == payload

    corrupted = bytearray(blob)
    corrupted[0] ^= 0xFF
    ok, recovered = verify_crc32(bytes(corrupted))
    assert not ok
    assert recovered != payload
