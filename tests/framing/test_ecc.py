import pytest

from neuralstego.framing import NotAvailableError, rs_decode, rs_encode


def test_rs_roundtrip():
    data = b"neural stego"
    try:
        encoded = rs_encode(data, nsym=8)
    except NotAvailableError:
        pytest.skip("reedsolo not installed")
    ok, decoded = rs_decode(encoded, nsym=8)
    assert ok
    assert decoded == data


def test_rs_corrects_errors():
    data = b"0123456789abcdef"
    try:
        encoded = bytearray(rs_encode(data, nsym=8))
    except NotAvailableError:
        pytest.skip("reedsolo not installed")

    # Corrupt up to nsym/2 bytes (here 4)
    encoded[0] ^= 0x01
    encoded[3] ^= 0x01
    encoded[5] ^= 0x02

    ok, decoded = rs_decode(bytes(encoded), nsym=8)
    assert ok
    assert decoded == data


def test_rs_failure_on_excess_errors():
    data = b"another block"
    try:
        encoded = bytearray(rs_encode(data, nsym=4))
    except NotAvailableError:
        pytest.skip("reedsolo not installed")

    # Corrupt more than nsym/2 (=2) bytes.
    encoded[0] ^= 0x01
    encoded[1] ^= 0x02
    encoded[2] ^= 0x04

    ok, decoded = rs_decode(bytes(encoded), nsym=4)
    assert not ok
    assert decoded == b""
