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


def _encode_text(text: str, tok) -> List[int]:
    tokenizer = _ensure_tokenizer(tok)

    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
        if not encoded:
            encoded = tokenizer.encode(text)
        return [int(token) for token in encoded]

    if hasattr(tokenizer, "tokenize") and hasattr(tokenizer, "convert_tokens_to_ids"):
        pieces = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(pieces)
        return [int(token) for token in ids]

    raise TypeError("tokenizer does not provide an encode method")


def _find_seed_prefix(tokens: Sequence[int], seed_ids: Sequence[int]) -> int:
    if not seed_ids:
        return 0
    needle = list(int(token) for token in seed_ids)
    haystack = list(int(token) for token in tokens)
    limit = len(haystack) - len(needle) + 1
    for start in range(max(limit, 0)):
        if haystack[start : start + len(needle)] == needle:
            return start
    raise ValueError("seed tokens not present in encoded text")


def text_to_spans(
    text: str,
    seed_ids: Sequence[int],
    tok,
    *,
    span_lengths: Sequence[int] | None = None,
) -> List[List[int]]:
    """Parse *text* back into spans.

    Args:
        text: The cover text containing the embedded payload.
        seed_ids: Token identifiers representing the seed prefix.
        tok: Tokenizer instance compatible with :func:`seed_to_ids`.
        span_lengths: Optional explicit lengths for each encoded span.  When
            provided, the function will deterministically reconstruct the spans
            from the tokenised cover text.  If omitted, a
            :class:`NotImplementedError` is raised to signal that additional
            framing information is required.
    """

    if span_lengths is None:
        raise NotImplementedError(
            "span lengths are required to recover cover spans from natural text"
        )

    encoded = _encode_text(text, tok)
    prefix = _find_seed_prefix(encoded, seed_ids)
    payload_start = prefix + len(seed_ids)
    payload_tokens = encoded[payload_start:]

    spans: List[List[int]] = []
    offset = 0
    total_required = sum(int(length) for length in span_lengths)
    if total_required > len(payload_tokens):
        raise ValueError("encoded text is shorter than the expected span lengths")

    for raw_length in span_lengths:
        length = int(raw_length)
        if length < 0:
            raise ValueError("span length must be non-negative")
        if offset + length > len(payload_tokens):
            raise ValueError("span length exceeds available tokenised payload")
        span_slice = payload_tokens[offset : offset + length]
        spans.append([int(token) for token in span_slice])
        offset += length

    if offset != len(payload_tokens):
        raise ValueError("encoded text contains trailing tokens beyond span lengths")

    return spans


__all__ = ["seed_to_ids", "spans_to_text", "text_to_spans"]
