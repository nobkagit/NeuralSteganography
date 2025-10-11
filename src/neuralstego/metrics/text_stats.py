"""Text-level statistics helpers for quick heuristics."""

from __future__ import annotations

import re
from collections import Counter
from typing import List

_SENTENCE_PATTERN = re.compile(r"[.!؟?\n]+")


def _tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"\s+", text.strip()) if token]


def _sentences(text: str) -> List[str]:
    raw = [segment.strip() for segment in _SENTENCE_PATTERN.split(text) if segment.strip()]
    return raw or ([text.strip()] if text.strip() else [])


def ngram_repeat_ratio(text: str, n: int = 3) -> float:
    """Return the ratio of repeated n-grams in the text."""

    tokens = _tokenize(text)
    if len(tokens) < n or n <= 0:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / total if total else 0.0


def type_token_ratio(text: str) -> float:
    """Compute the type-token ratio for the provided text."""

    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    unique = {token.lower() for token in tokens}
    return len(unique) / len(tokens)


def avg_sentence_len(text: str) -> float:
    """Return the average number of tokens per sentence."""

    sentences = _sentences(text)
    if not sentences:
        return 0.0
    token_counts = [len(_tokenize(sentence)) for sentence in sentences]
    return sum(token_counts) / len(token_counts)
