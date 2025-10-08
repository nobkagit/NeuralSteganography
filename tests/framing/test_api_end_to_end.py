import os

import pytest

from neuralstego.api import stego_decode, stego_encode
from neuralstego.lm.mock import MockLM


@pytest.mark.parametrize(
    "cfg",
    [
        {"ecc": "none", "use_crc": False, "nsym": 0},
        {"ecc": "none", "use_crc": True, "nsym": 0},
        pytest.param({"ecc": "rs", "use_crc": True, "nsym": 10}, marks=pytest.mark.requires_reedsolo),
    ],
)
def test_end_to_end_mock_lm(cfg):
    if cfg["ecc"] == "rs":
        pytest.importorskip("reedsolo")

    data = os.urandom(4096)
    lm = MockLM()

    result = stego_encode(
        data,
        chunk_bytes=256,
        use_crc=cfg["use_crc"],
        ecc=cfg["ecc"],
        nsym=cfg["nsym"],
        quality={"precision": 8},
        seed_text="متن نمونه.",
        lm=lm,
    )

    assert result.metadata.total == len(result)

    recovered = stego_decode(
        list(result),
        use_crc=cfg["use_crc"],
        ecc=cfg["ecc"],
        nsym=cfg["nsym"],
        quality={"precision": 8},
        seed_text="متن نمونه.",
        lm=lm,
    )
    assert recovered == data
