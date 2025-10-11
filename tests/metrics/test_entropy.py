from neuralstego.metrics import LMScorer
from neuralstego.metrics.entropy import avg_entropy


def test_entropy_positive_for_sample_text():
    scorer = LMScorer(prefer_transformers=False)
    entropy_value = avg_entropy("این یک متن کوتاه است", lm_scorer=scorer)
    assert entropy_value >= 0.0
