from neuralstego.detect.guard import QualityGuard
from neuralstego.metrics import LMScorer


SAMPLE_TEXT = """
سلام دنیا. این متن آزمایشی برای بررسی نگهبان کیفیت است.
""".strip()


def test_quality_guard_passes_when_within_thresholds():
    guard = QualityGuard(lm_scorer=LMScorer(prefer_transformers=False))
    thresholds = {
        "max_ppl": 100.0,
        "max_ngram_repeat": 0.5,
        "min_ttr": 0.2,
        "max_avg_entropy": 5.0,
        "min_avg_sentence_len": 2.0,
    }
    result = guard.evaluate(SAMPLE_TEXT, thresholds)
    assert result.passed
    assert not result.reasons
    assert result.metrics["ppl"] > 0


def test_quality_guard_flags_threshold_breach():
    guard = QualityGuard(lm_scorer=LMScorer(prefer_transformers=False))
    thresholds = {"max_ppl": 0.1}
    result = guard.evaluate(SAMPLE_TEXT, thresholds)
    assert not result.passed
    assert result.reasons
