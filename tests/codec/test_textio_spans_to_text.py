import pytest

from neuralstego.codec.textio import seed_to_ids, spans_to_text, text_to_spans


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


def test_seed_to_ids_roundtrip():
    tok = DummyTokenizer()
    seed = "سلام"
    ids = seed_to_ids(seed, tok)
    assert ids == [ord(ch) for ch in seed]


def test_spans_to_text_joins_seed_and_tokens():
    tok = DummyTokenizer()
    seed = "پایه"
    seed_ids = seed_to_ids(seed, tok)
    spans = [[ord(" "), ord("ک"), ord("ا"), ord("و")], [ord("ر")]]
    text = spans_to_text(spans, seed_ids, tok)
    assert tok.encode(text) == seed_ids + [item for span in spans for item in span]


def test_text_to_spans_not_implemented():
    tok = DummyTokenizer()
    with pytest.raises(NotImplementedError):
        text_to_spans("متن", [1, 2], tok)
