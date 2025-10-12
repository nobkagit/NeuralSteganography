"""Adapters for Hugging Face Transformers language models."""
from __future__ import annotations

from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover - typing hints only
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLMType
    from transformers import AutoTokenizer as _AutoTokenizerType
else:  # pragma: no cover - fallback types when transformers is unavailable
    _AutoModelForCausalLMType = Any
    _AutoTokenizerType = Any

torch_module: ModuleType | None
AutoModelForCausalLMImpl: Any = None
AutoTokenizerImpl: Any = None

try:  # pragma: no cover - optional dependency import guard
    import torch as torch_module  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM as AutoModelForCausalLMImpl
    from transformers import AutoTokenizer as AutoTokenizerImpl
except ModuleNotFoundError as exc:  # pragma: no cover - defer failure until runtime use
    torch_module = None
    AutoModelForCausalLMImpl = None
    AutoTokenizerImpl = None
    _IMPORT_ERROR: ModuleNotFoundError | None = exc
else:  # pragma: no cover - executed when dependencies are available
    _IMPORT_ERROR = None

torch: Any = torch_module
AutoModelForCausalLM = cast(Any, AutoModelForCausalLMImpl)
AutoTokenizer = cast(Any, AutoTokenizerImpl)


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
        self._tokenizer: Any = None
        self._model: Any = None

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
