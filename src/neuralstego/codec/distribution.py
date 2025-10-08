"""Language-model distribution providers for arithmetic coding."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from .quality import apply_quality
from .types import LMProvider, ProbDist


@dataclass
class MockLM(LMProvider):
    """Deterministic mock language model with a Zipfian prior."""

    vocab_size: int = 32
    alpha: float = 1.2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        ranks = np.arange(1, self.vocab_size + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, self.alpha)
        self._base_probs = (weights / weights.sum()).astype(np.float64)

    def next_token_probs(self, context_ids: Sequence[int]) -> ProbDist:  # noqa: D401 - simple mapping
        """Return the fixed Zipfian probability distribution regardless of context."""

        _ = context_ids  # context ignored for deterministic mock
        return self._base_probs.copy()


class CachedLM(LMProvider):
    """Caching wrapper that memoizes context probability queries."""

    def __init__(self, lm: LMProvider, *, maxsize: int = 128) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._lm = lm
        self._maxsize = maxsize
        self._cache: "OrderedDict[tuple[int, ...], ProbDist]" = OrderedDict()

    def next_token_probs(self, context_ids: Sequence[int]) -> ProbDist:
        key = tuple(context_ids)
        if key in self._cache:
            self._cache.move_to_end(key)
            return _clone_distribution(self._cache[key])

        probs = self._lm.next_token_probs(list(context_ids))
        self._cache[key] = _clone_distribution(probs)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return _clone_distribution(probs)


class TransformersLM(LMProvider):
    """Adapter exposing HuggingFace language models via :class:`LMProvider`."""

    def __init__(
        self,
        model_name: str = "HooshvareLab/gpt2-fa",
        *,
        tokenizer_name: str | None = None,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | str | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_prob: float | None = None,
        temperature: float = 1.0,
        max_context: int | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        model_loader: Callable[[], GPT2LMHeadModel] | None = None,
        tokenizer_loader: Callable[[], Any] | None = None,
        model: GPT2LMHeadModel | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self._model_name = model_name
        self._tokenizer_name = tokenizer_name or model_name
        self._explicit_device = torch.device(device) if device is not None else None
        if isinstance(torch_dtype, str):
            try:
                torch_dtype = getattr(torch, torch_dtype)
            except AttributeError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Unknown torch dtype: {torch_dtype}") from exc
        self._torch_dtype = torch_dtype
        self._top_k = top_k
        self._top_p = top_p
        self._min_prob = min_prob
        self._temperature = float(temperature)
        self._max_context = max_context
        self._model_kwargs = dict(model_kwargs or {})
        self._model_loader = model_loader
        self._tokenizer_loader = tokenizer_loader
        self._model: GPT2LMHeadModel | None = model
        self.tokenizer = tokenizer

    def next_token_probs(self, context_ids: Sequence[int]) -> ProbDist:
        """Return next-token probabilities for ``context_ids``."""

        model = self._ensure_model()
        context = tuple(context_ids)
        if not context:
            context = self._default_context(model)
            if not context:
                raise ValueError("context_ids must contain at least one token")
        if self._max_context is None:
            context_window = getattr(model.config, "n_positions", None)
        else:
            context_window = self._max_context
        if context_window is not None and len(context) > context_window:
            context = context[-context_window:]

        device = model.device
        input_tensor = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)

        logits = outputs.logits[0, -1, :].detach().to(dtype=torch.float64).cpu().numpy()
        probs = _softmax(logits / self._temperature)

        if any(param is not None for param in (self._top_k, self._top_p, self._min_prob)):
            probs = apply_quality(
                probs,
                top_k=self._top_k,
                top_p=self._top_p,
                min_prob=self._min_prob,
            )
        else:
            probs = probs / probs.sum()

        return probs

    def _ensure_model(self) -> GPT2LMHeadModel:
        if self._model is None:
            if self._model_loader is not None:
                model = self._model_loader()
            else:
                model = GPT2LMHeadModel.from_pretrained(
                    self._model_name,
                    **self._model_kwargs,
                )
            if self._torch_dtype is not None:
                model = model.to(dtype=self._torch_dtype)
            if self._explicit_device is not None:
                model = model.to(self._explicit_device)
            model.eval()
            self._model = model
        return self._model

    def _load_tokenizer(self) -> Any | None:
        if self._tokenizer_loader is not None:
            return self._tokenizer_loader()
        try:
            return AutoTokenizer.from_pretrained(self._tokenizer_name)
        except Exception:  # pragma: no cover - optional convenience
            return None

    def ensure_tokenizer(self) -> Any | None:
        """Lazily load and cache the tokenizer associated with the model."""

        if self.tokenizer is None:
            self.tokenizer = self._load_tokenizer()
        return self.tokenizer

    def _default_context(self, model: GPT2LMHeadModel) -> tuple[int, ...] | None:
        """Return a fallback context tuple when none is provided."""

        config = getattr(model, "config", None)
        for attr in ("bos_token_id", "decoder_start_token_id", "eos_token_id"):
            token_id = getattr(config, attr, None) if config is not None else None
            if isinstance(token_id, int) and token_id >= 0:
                return (int(token_id),)
            if isinstance(token_id, (list, tuple)) and token_id:
                return (int(token_id[0]),)

        tokenizer = None
        if self.tokenizer is not None:
            tokenizer = self.tokenizer
        elif self._tokenizer_loader is not None:
            tokenizer = self.ensure_tokenizer()
        if tokenizer is not None:
            for attr in ("bos_token_id", "cls_token_id", "eos_token_id"):
                token_id = getattr(tokenizer, attr, None)
                if isinstance(token_id, int) and token_id >= 0:
                    return (int(token_id),)
                if isinstance(token_id, (list, tuple)) and token_id:
                    return (int(token_id[0]),)

            for attr in ("bos_token", "cls_token", "eos_token"):
                token = getattr(tokenizer, attr, None)
                if token:
                    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
                    if callable(converter):
                        converted = converter(token)
                        if isinstance(converted, int) and converted >= 0:
                            return (int(converted),)
                        if isinstance(converted, (list, tuple)) and converted:
                            return (int(converted[0]),)

            encoder = getattr(tokenizer, "encode", None)
            if callable(encoder):
                try:
                    encoded = encoder("", add_special_tokens=True)
                except TypeError:
                    encoded = encoder("")
                if isinstance(encoded, Sequence):
                    for value in encoded:
                        try:
                            token_id = int(value)
                        except (TypeError, ValueError):
                            continue
                        if token_id >= 0:
                            return (token_id,)

        return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=True)
    logits -= np.max(logits)
    exp = np.exp(logits)
    total = exp.sum()
    if total <= 0.0:
        raise ValueError("Logits do not produce a valid probability distribution")
    return exp / total


def _clone_distribution(dist: ProbDist) -> ProbDist:
    """Return a safe copy of the probability distribution for cache reuse."""

    if isinstance(dist, np.ndarray):
        return dist.copy()
    if isinstance(dist, dict):
        return dict(dist)
    raise TypeError(f"Unsupported distribution type: {type(dist)!r}")


__all__ = ["MockLM", "CachedLM", "TransformersLM"]
