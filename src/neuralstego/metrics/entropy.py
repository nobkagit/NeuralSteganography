"""Entropy computations for token distributions."""

from __future__ import annotations

import importlib
import math
from collections import Counter
from typing import Iterable

from .lm_scorer import LMScorer


def _tokenize(text: str) -> Iterable[str]:
    return [token for token in text.strip().split() if token]


def _fallback_entropy(text: str) -> float:
    tokens = list(_tokenize(text))
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def avg_entropy(text: str, lm_scorer: LMScorer | None = None) -> float:
    """Compute the average next-token entropy for the provided text."""

    tokens = list(_tokenize(text))
    if not tokens:
        return 0.0
    scorer = lm_scorer or LMScorer()
    if not scorer._transformers_available():
        return _fallback_entropy(text)
    scorer._ensure_transformers_model()
    tokenizer = scorer._tokenizer
    model = scorer._model
    assert tokenizer is not None and model is not None
    inputs = tokenizer(text, return_tensors="pt")
    torch = importlib.import_module("torch")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits[:, :-1, :], dim=-1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12), dim=-1)
    mean_entropy = float(torch.mean(entropy))
    return mean_entropy
