"""Language model provider utilities."""
from __future__ import annotations

from typing import Optional

from .arithmetic import ArithmeticLM
from .mock import MockLM
from .transformers_adapter import TransformersLM
from ..exceptions import ConfigurationError


def load_lm(name: str, *, device: Optional[str] = None):
    name_norm = name.lower()
    if name_norm == "mock":
        return MockLM()
    if name_norm in {"gpt2", "gpt2-fa"}:
        from utils import get_model  # type: ignore

        enc, model = get_model(model_name=name_norm)
        return ArithmeticLM(model, enc, device=device)
    raise ConfigurationError(f"unknown language model provider: {name}")


__all__ = ["ArithmeticLM", "MockLM", "TransformersLM", "load_lm"]
