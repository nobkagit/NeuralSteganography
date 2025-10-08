import math

import math

import pytest

from neuralstego.lm.transformers_adapter import TransformersLM


class DummyTensor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        result = self.data
        if isinstance(item, tuple):
            for idx in item:
                result = result[idx]
        else:
            result = result[item]
        if isinstance(result, list):
            return DummyTensor(result)
        return result

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.data)


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class DummyModel:
    def __init__(self, vocab_size=4):
        self.vocab_size = vocab_size
        self.is_eval = False

    def to(self, _device):
        return self

    def eval(self):
        self.is_eval = True

    def __call__(self, input_ids):
        seq_len = len(input_ids.data[0])
        logits = []
        for _ in range(seq_len):
            logits.append([float(i) for i in range(self.vocab_size)])
        return DummyOutput(DummyTensor([logits]))


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


class DummyFunctional:
    @staticmethod
    def softmax(tensor, dim=-1):
        values = tensor.data
        exp_values = [math.exp(v) for v in values]
        total = sum(exp_values)
        return DummyTensor([v / total for v in exp_values])


class DummyNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class DummyTorch:
    long = int

    class cuda:
        @staticmethod
        def is_available():
            return False

    class nn:
        functional = DummyFunctional()

    @staticmethod
    def tensor(data, dtype=None, device=None):
        return DummyTensor(data)

    @staticmethod
    def no_grad():
        return DummyNoGrad()


class DummyAutoTokenizer:
    @staticmethod
    def from_pretrained(model_name):
        return DummyTokenizer()


class DummyAutoModel:
    @staticmethod
    def from_pretrained(model_name):
        return DummyModel()


@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    monkeypatch.setattr("neuralstego.lm.transformers_adapter.AutoTokenizer", DummyAutoTokenizer)
    monkeypatch.setattr("neuralstego.lm.transformers_adapter.AutoModelForCausalLM", DummyAutoModel)
    monkeypatch.setattr("neuralstego.lm.transformers_adapter.torch", DummyTorch)
    monkeypatch.setattr("neuralstego.lm.transformers_adapter._IMPORT_ERROR", None)


def test_tokenize_detokenize_roundtrip():
    lm = TransformersLM(model_name="dummy")
    text = "سلام دنیا"
    token_ids = lm.tokenize(text)
    assert token_ids == [ord(ch) for ch in text]
    reconstructed = lm.detokenize(token_ids)
    assert reconstructed == text


def test_next_token_probs_returns_distribution():
    lm = TransformersLM(model_name="dummy")
    context = lm.tokenize("نمونه")
    probs = lm.next_token_probs(context)
    assert pytest.approx(sum(probs.values()), rel=1e-6) == 1.0
    assert set(probs.keys()) == set(range(len(probs)))
