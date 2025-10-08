"""Adapters for Hugging Face Transformers language models."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency import guard
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:  # pragma: no cover - defer failure until runtime use
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:  # pragma: no cover - executed when dependencies are available
    _IMPORT_ERROR = None


class TransformersLM:
    """Lightweight wrapper exposing next-token probabilities for Transformers."""

    def __init__(
        self,
        model_name: str = "HooshvareLab/gpt2-fa",
        *,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._tokenizer = None
        self._model = None

    # -- Internal helpers -------------------------------------------------
    def _require_dependencies(self) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover - import guard
            raise RuntimeError(
                "transformers and torch are required for TransformersLM"
            ) from _IMPORT_ERROR

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._require_dependencies()
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    @property
    def tokenizer(self):  # pragma: no cover - simple delegation
        return self._ensure_tokenizer()

    def _ensure_model(self):
        if self._model is None:
            self._require_dependencies()
            model = AutoModelForCausalLM.from_pretrained(self._model_name)
            if self._device:
                model = model.to(self._device)
            elif torch is not None and torch.cuda.is_available():  # pragma: no branch
                model = model.to("cuda")
                self._device = "cuda"
            else:
                self._device = "cpu"
            model.eval()
            self._model = model
        return self._model

    # -- Public helpers ---------------------------------------------------
    def tokenize(self, text: str) -> List[int]:
        tokenizer = self._ensure_tokenizer()
        if hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                tokens = tokenizer.encode(text)
            return [int(tok) for tok in tokens]
        raise RuntimeError("tokenizer does not support encode")

    def detokenize(self, ids: Iterable[int]) -> str:
        tokenizer = self._ensure_tokenizer()
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(list(ids), skip_special_tokens=True).strip()
        raise RuntimeError("tokenizer does not support decode")

    # -- Interface expected by arithmetic coder --------------------------
    def encode_seed(self, text: str) -> List[int]:
        return self.tokenize(text)

    def next_token_probs(self, context_ids: List[int]) -> Dict[int, float]:
        model = self._ensure_model()
        if torch is None:
            raise RuntimeError("torch is required for TransformersLM")

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]
            probs = torch.nn.functional.softmax(logits, dim=-1)

        prob_values = probs.detach().cpu().tolist()
        return {int(token_id): float(prob_values[token_id]) for token_id in range(len(prob_values))}

    def decode_arithmetic(self, *args, **kwargs):  # pragma: no cover - compatibility shim
        raise NotImplementedError("TransformersLM does not implement arithmetic decoding")

    def encode_arithmetic(self, *args, **kwargs):  # pragma: no cover - compatibility shim
        raise NotImplementedError("TransformersLM does not implement arithmetic encoding")


__all__ = ["TransformersLM"]
