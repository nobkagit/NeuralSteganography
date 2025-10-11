from neuralstego.detect import EXPECTED_FEATURES, extract_features


def test_extract_features_returns_all_keys():
    metrics = {
        "ppl": 10.0,
        "avg_nll": 2.0,
        "avg_entropy": 1.5,
        "ngram_repeat_ratio": 0.1,
        "type_token_ratio": 0.6,
        "avg_sentence_len": 12.0,
    }
    features = extract_features(metrics)
    assert tuple(features.keys()) == EXPECTED_FEATURES
