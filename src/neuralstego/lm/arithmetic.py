"""Adapters around the reference arithmetic coder implementation."""
from __future__ import annotations

from typing import Dict, List, Optional

from arithmetic import decode_arithmetic, encode_arithmetic  # type: ignore

from ..exceptions import ConfigurationError
from utils import encode_context  # type: ignore


class ArithmeticLM:
    """Wraps the reference GPT-style arithmetic codec behind :class:`LMProvider`."""

    def __init__(self, model, tokenizer, *, device: Optional[str] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        model_device = getattr(model, "device", None)
        if hasattr(model_device, "type"):
            model_device = model_device.type
        self.device = device or model_device or "cpu"

    def encode_seed(self, text: str) -> List[int]:
        return encode_context(text, self.tokenizer)

    def encode_arithmetic(
        self, bits: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        spans, *_ = encode_arithmetic(
            self.model,
            self.tokenizer,
            bits,
            context,
            device=self.device,
            temp=quality.get("temp", 1.0),
            precision=int(quality.get("precision", 16)),
            topk=int(quality.get("topk", 50000)),
            finish_sent=bool(quality.get("finish_sent", True)),
        )
        return spans

    def decode_arithmetic(
        self, tokens: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        text = self.tokenizer.decode(tokens)
        bits = decode_arithmetic(
            self.model,
            self.tokenizer,
            text,
            context,
            device=self.device,
            temp=quality.get("temp", 1.0),
            precision=int(quality.get("precision", 16)),
            topk=int(quality.get("topk", 50000)),
        )
        if len(bits) % 8:
            raise ConfigurationError("decoded bit stream is not byte aligned")
        return bits


__all__ = ["ArithmeticLM"]
