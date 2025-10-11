from neuralstego.metrics import LMScorer


def test_lm_scorer_returns_basic_metrics():
    scorer = LMScorer(prefer_transformers=False)
    result = scorer.score("این یک متن آزمایشی است")
    assert set(result.keys()) == {"ppl", "avg_nll", "token_count"}
    assert result["token_count"] > 0
    assert result["ppl"] > 0
    assert result["avg_nll"] >= 0
