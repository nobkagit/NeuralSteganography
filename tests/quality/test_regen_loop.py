from __future__ import annotations

import pytest

from neuralstego.api import cover_generate
from neuralstego.detect.guard import GuardResult
from neuralstego.exceptions import QualityGateError
from neuralstego.lm.mock import MockLM


class _ToggleGuard:
    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    def evaluate(self, text: str, thresholds):
        self.calls += 1
        self.texts.append(text)
        if self.calls == 1:
            return GuardResult(False, ["ppl too high"], {"ppl": 420.0, "type_token_ratio": 0.10})
        return GuardResult(True, [], {"ppl": 48.0, "type_token_ratio": 0.36})


class _FailingGuard:
    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    def evaluate(self, text: str, thresholds):
        self.calls += 1
        self.texts.append(text)
        return GuardResult(False, ["ngram repeat"], {"ppl": 510.0, "type_token_ratio": 0.12})


def test_cover_generate_regenerates_until_guard_passes():
    guard = _ToggleGuard()
    lm = MockLM()

    cover_text = cover_generate(
        "داده محرمانه",
        seed_text="مکالمه پایه",
        quality={},
        use_crc=False,
        ecc="none",
        nsym=0,
        lm=lm,
        regen_attempts=2,
        regen_strategy={"seed_pool": ["مکالمه جایگزین"]},
        quality_guard=guard,
    )

    assert guard.calls == 2
    assert cover_text.startswith("مکالمه جایگزین")


def test_cover_generate_raises_quality_error_after_exhausting_attempts():
    guard = _FailingGuard()
    lm = MockLM()

    with pytest.raises(QualityGateError) as excinfo:
        cover_generate(
            "داده محرمانه",
            seed_text="گفت‌وگوی اولیه",
            quality={},
            use_crc=False,
            ecc="none",
            nsym=0,
            lm=lm,
            regen_attempts=1,
            regen_strategy={"seed_pool": ["گفت‌وگوی دوم"]},
            quality_guard=guard,
        )

    assert guard.calls == 2
    error = excinfo.value
    assert error.cover_text.startswith("گفت‌وگوی دوم")
    assert "ngram" in " ".join(error.reasons)
    assert error.metrics["ppl"] == pytest.approx(510.0)
