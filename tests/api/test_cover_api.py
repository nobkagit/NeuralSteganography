import json

from neuralstego.api import cover_generate, cover_reveal, stego_encode
from neuralstego.lm.mock import MockLM


def test_cover_generate_returns_text_with_mock_lm():
    lm = MockLM()
    secret = "راز مخفی"
    seed = "این یک متن پایه است"
    cover_text = cover_generate(
        secret,
        seed_text=seed,
        quality={},
        use_crc=False,
        ecc="none",
        nsym=0,
        lm=lm,
        quality_gate=False,
    )
    assert isinstance(cover_text, str)
    assert cover_text


def test_cover_reveal_roundtrip_from_spans_json():
    lm = MockLM()
    secret = "پیام"
    seed = "متن کاور"
    encode_result = stego_encode(
        secret.encode("utf-8"),
        chunk_bytes=256,
        use_crc=False,
        ecc="none",
        nsym=0,
        quality={},
        seed_text=seed,
        lm=lm,
    )
    spans_json = json.dumps([list(span) for span in encode_result])
    recovered = cover_reveal(
        spans_json,
        seed_text=seed,
        quality={},
        use_crc=False,
        ecc="none",
        nsym=0,
        lm=lm,
    )
    assert recovered == secret.encode("utf-8")
