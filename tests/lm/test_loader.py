import types

import pytest

import utils
from neuralstego.lm import load_lm


class _DummyTokenizer:
    def encode(self, text):
        return [len(text)]

    def decode(self, tokens):
        return ""


class _DummyModel:
    device = "cpu"

    def to(self, device):
        return self

    def eval(self):
        return self


def test_load_lm_alias_for_gpt2_fa(monkeypatch):
    captured = {}

    def fake_get_model(*, model_name):
        captured["model_name"] = model_name
        return _DummyTokenizer(), _DummyModel()

    monkeypatch.setattr(utils, "get_model", fake_get_model)

    lm = load_lm("gpt2-fa")

    assert captured["model_name"] == "HooshvareLab/gpt2-fa"
    assert hasattr(lm, "encode_arithmetic")


def test_load_pretrained_offline_hint(monkeypatch):
    calls = []

    def fake_from_pretrained(model_name, local_files_only=None):
        calls.append(local_files_only)
        if local_files_only:
            raise OSError("offline cache miss")
        raise RuntimeError("network disabled")

    factory = types.SimpleNamespace(from_pretrained=fake_from_pretrained)

    with pytest.raises(RuntimeError) as excinfo:
        utils._load_pretrained(factory, "demo-model")

    assert calls == [True, None]
    message = str(excinfo.value)
    assert "scripts/download_models.py --model demo-model" in message
