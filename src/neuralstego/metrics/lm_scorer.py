"""Language model scoring utilities for cover-text analysis."""

from __future__ import annotations

import importlib
import importlib.util
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class _FallbackDistribution:
    """Light-weight empirical distribution used when transformers are unavailable."""

    counts: Counter[str]
    total: int

    def nll(self, sequence: Iterable[str]) -> float:
        """Compute the total negative log likelihood for the provided sequence."""

        nll_value = 0.0
        for token in sequence:
            probability = self.counts[token] / self.total
            nll_value -= math.log(probability)
        return nll_value

    def entropy(self) -> float:
        """Return the Shannon entropy of the empirical token distribution."""

        entropy_sum = 0.0
        for count in self.counts.values():
            probability = count / self.total
            entropy_sum -= probability * math.log(probability)
        return entropy_sum


class LMScorer:
    """Score text using GPT2-fa when available, otherwise a deterministic fallback."""

    _model = None
    _tokenizer = None

    def __init__(
        self,
        model_name: str = "HooshvareLab/gpt2-fa",
        prefer_transformers: bool = True,
    ) -> None:
        self.model_name = model_name
        self.prefer_transformers = prefer_transformers
        self._backend = "transformers" if prefer_transformers else "fallback"

    def _transformers_available(self) -> bool:
        if self._backend != "transformers":
            return False
        return importlib.util.find_spec("transformers") is not None

    def _ensure_transformers_model(self) -> None:
        if not self._transformers_available():
            raise RuntimeError("transformers is not available in the current environment")
        if self.__class__._model is not None:
            return
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        model.eval()
        self.__class__._tokenizer = tokenizer
        self.__class__._model = model

    @staticmethod
    def _tokenize(text: str) -> Iterable[str]:
        return [token for token in text.strip().split() if token]

    @staticmethod
    def _fallback_score(tokens: Iterable[str]) -> Dict[str, float]:
        token_list = list(tokens)
        if not token_list:
            return {"ppl": 0.0, "avg_nll": 0.0, "token_count": 0}
        distribution = _FallbackDistribution(Counter(token_list), len(token_list))
        total_nll = distribution.nll(token_list)
        token_count = len(token_list)
        avg_nll = total_nll / token_count
        ppl = math.exp(avg_nll)
        return {"ppl": ppl, "avg_nll": avg_nll, "token_count": token_count}

    def score(self, text: str) -> Dict[str, float]:
        """Compute perplexity, average negative log likelihood, and token count."""

        tokens = self._tokenize(text)
        if not tokens:
            return {"ppl": 0.0, "avg_nll": 0.0, "token_count": 0}
        if not self._transformers_available():
            return self._fallback_score(tokens)

        self._ensure_transformers_model()
        tokenizer = self.__class__._tokenizer
        model = self.__class__._model
        assert tokenizer is not None and model is not None
        inputs = tokenizer(text, return_tensors="pt")
        torch = importlib.import_module("torch")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = float(outputs.loss)
        avg_nll = loss
        ppl = math.exp(avg_nll)
        token_count = int(inputs["input_ids"].shape[1])
        return {"ppl": ppl, "avg_nll": avg_nll, "token_count": token_count}
