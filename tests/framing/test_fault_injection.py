import json
import os
from copy import deepcopy
from typing import List

import pytest

from neuralstego.api import stego_decode, stego_encode
from neuralstego.codec import chunk_bytes
from neuralstego.exceptions import MissingChunksError
from neuralstego.lm.mock import MockLM


def _encode_message(data: bytes, **kwargs):
    lm = MockLM()
    return stego_encode(
        data,
        chunk_bytes=kwargs.get("chunk_bytes", 256),
        use_crc=kwargs.get("use_crc", True),
        ecc=kwargs.get("ecc", "rs"),
        nsym=kwargs.get("nsym", 10),
        quality=kwargs.get("quality", {"precision": 8}),
        seed_text=kwargs.get("seed_text", "جملهٔ آغازین."),
        lm=lm,
    ), lm


def _corrupt_payload_symbol(span: List[int]) -> List[int]:
    packet = json.loads(bytes(span).decode("utf-8"))
    payload = bytearray(packet["payload"].encode("ascii"))
    if not payload:
        return span
    payload[0] = ord("B") if payload[0] != ord("B") else ord("A")
    packet["payload"] = payload.decode("ascii")
    mutated = json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return list(mutated)


@pytest.mark.requires_reedsolo
def test_symbol_flip_corrected_with_rs():
    pytest.importorskip("reedsolo")
    data = os.urandom(4096)
    result, lm = _encode_message(data)

    corrupted_spans = deepcopy(list(result))
    assert corrupted_spans, "expected at least one span"
    corrupted_spans[0] = _corrupt_payload_symbol(corrupted_spans[0])

    recovered = stego_decode(
        corrupted_spans,
        use_crc=True,
        ecc="rs",
        nsym=10,
        quality={"precision": 8},
        seed_text="جملهٔ آغازین.",
        lm=lm,
    )
    assert recovered == data


def test_symbol_flip_without_protection_changes_payload():
    data = os.urandom(4096)
    result, lm = _encode_message(data, use_crc=False, ecc="none", nsym=0)

    corrupted_spans = deepcopy(list(result))
    corrupted_spans[0] = _corrupt_payload_symbol(corrupted_spans[0])

    recovered = stego_decode(
        corrupted_spans,
        use_crc=False,
        ecc="none",
        nsym=0,
        quality={"precision": 8},
        seed_text="جملهٔ آغازین.",
        lm=lm,
    )
    assert recovered != data


@pytest.mark.requires_reedsolo
def test_missing_span_detected_and_partial_returned():
    pytest.importorskip("reedsolo")
    data = os.urandom(4096)
    result, lm = _encode_message(data)

    spans = list(result)
    missing_idx = len(spans) // 2
    removed = spans.pop(missing_idx)
    assert removed is not None

    with pytest.raises(MissingChunksError) as excinfo:
        stego_decode(
            spans,
            use_crc=True,
            ecc="rs",
            nsym=10,
            quality={"precision": 8},
            seed_text="جملهٔ آغازین.",
            lm=lm,
        )

    err = excinfo.value
    assert err.missing_indices == [missing_idx]

    chunk_size = result.metadata.cfg["chunk_bytes"]
    expected_chunks = chunk_bytes(data, chunk_size=chunk_size)
    expected_partial = b"".join(
        chunk for idx, chunk in enumerate(expected_chunks) if idx != missing_idx
    )
    assert err.partial_payload == expected_partial
