"""Compatibility helpers for arithmetic coding tests."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def _select_cutoff_k(probs: torch.Tensor, threshold: float, topk: int) -> int:
    """Return the number of tokens to retain after thresholding."""

    cutoff_indices = torch.nonzero(probs < threshold, as_tuple=False)
    if cutoff_indices.numel() == 0:
        candidate = probs.size(0)
    else:
        candidate = int(cutoff_indices[0].item())
    return min(max(2, candidate), int(topk))


def encode_arithmetic(
    model: object,
    enc: object,
    message: Sequence[int],
    context: Sequence[int],
    finish_sent: bool = False,
    device: str = "cpu",
    temp: float = 1.0,
    precision: int = 16,
    topk: int = 50000,
) -> Tuple[List[int], float, float, float, float]:
    """Legacy shim retained for backwards compatibility."""

    del model, enc, context, finish_sent, device, temp, precision, topk
    raise NotImplementedError("Legacy encode_arithmetic is no longer supported.")


def decode_arithmetic(
    model: object,
    enc: object,
    text: Sequence[int] | str,
    context: Sequence[int],
    temp: float = 1.0,
    precision: int = 16,
    topk: int = 50000,
) -> List[int]:
    """Legacy shim retained for backwards compatibility."""

    del model, enc, text, context, temp, precision, topk
    raise NotImplementedError("Legacy decode_arithmetic is no longer supported.")


__all__ = ["_select_cutoff_k", "encode_arithmetic", "decode_arithmetic"]
