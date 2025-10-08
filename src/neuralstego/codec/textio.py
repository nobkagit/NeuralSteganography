"""Utilities for converting between token spans and cover text."""
from __future__ import annotations

from typing import List, Sequence


def _ensure_tokenizer(tok) -> object:
    if tok is None:
        raise ValueError("tokenizer instance is required")
    return tok


def seed_to_ids(seed: str, tok) -> List[int]:
    """Encode a seed string using the provided tokenizer."""

    tokenizer = _ensure_tokenizer(tok)

    if hasattr(tokenizer, "encode"):
        try:
            tokens = tokenizer.encode(seed, add_special_tokens=False)
        except TypeError:
            tokens = tokenizer.encode(seed)
        else:
            if not tokens:
                tokens = tokenizer.encode(seed)
        return [int(token) for token in tokens]

    if hasattr(tokenizer, "tokenize") and hasattr(tokenizer, "convert_tokens_to_ids"):
        pieces = tokenizer.tokenize(seed)
        ids = tokenizer.convert_tokens_to_ids(pieces)
        return [int(token) for token in ids]

    raise TypeError("tokenizer does not provide an encode method")


def spans_to_text(spans: Sequence[Sequence[int]], seed_ids: Sequence[int], tok) -> str:
    """Render *spans* into a single cover text string."""

    tokenizer = _ensure_tokenizer(tok)

    all_ids: List[int] = list(seed_ids)
    for span in spans:
        all_ids.extend(int(token) for token in span)

    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(all_ids, skip_special_tokens=True).strip()
        except TypeError:
            return tokenizer.decode(all_ids).strip()

    if hasattr(tokenizer, "convert_ids_to_tokens") and hasattr(tokenizer, "convert_tokens_to_string"):
        tokens = tokenizer.convert_ids_to_tokens(all_ids)
        return tokenizer.convert_tokens_to_string(tokens).strip()

    raise TypeError("tokenizer does not provide a decode method")


def text_to_spans(text: str, seed_ids: Sequence[int], tok) -> List[List[int]]:
    """Parse *text* back into spans. Placeholder until cover parsing is implemented."""

    raise NotImplementedError(
        "text_to_spans is not yet implemented; provide spans JSON payloads for decoding"
    )


__all__ = ["seed_to_ids", "spans_to_text", "text_to_spans"]
