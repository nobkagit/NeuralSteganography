from neuralstego.metrics import avg_sentence_len, ngram_repeat_ratio, type_token_ratio


SAMPLE_TEXT = """
سلام دنیا. این یک نمونه ساده برای آزمایش است. این نمونه شامل چند جمله کوتاه است.
""".strip()


def test_ngram_repeat_ratio_bounds():
    ratio = ngram_repeat_ratio(SAMPLE_TEXT, n=2)
    assert 0.0 <= ratio <= 1.0


def test_type_token_ratio_bounds():
    ttr = type_token_ratio(SAMPLE_TEXT)
    assert 0.0 <= ttr <= 1.0


def test_average_sentence_length_positive():
    avg_len = avg_sentence_len(SAMPLE_TEXT)
    assert avg_len > 0
