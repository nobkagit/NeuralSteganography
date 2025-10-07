"""Tests for arithmetic coding threshold handling."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path
from typing import Callable

import torch


def _load_arithmetic_helper() -> Callable[[torch.Tensor, float, int], int]:
    """Load the private helper from ``arithmetic.py`` without package installs."""

    module_path = Path(__file__).resolve().parents[2] / "arithmetic.py"
    spec = importlib.util.spec_from_file_location("arithmetic", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load arithmetic module for testing")

    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert isinstance(loader, importlib.abc.Loader)

    repo_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, repo_root)
    try:
        loader.exec_module(module)
    finally:
        sys.path.remove(repo_root)

    helper = getattr(module, "_select_cutoff_k")
    if not callable(helper):
        raise AttributeError("_select_cutoff_k is not callable")

    return helper


_select_cutoff_k = _load_arithmetic_helper()


def test_select_cutoff_k_handles_absent_threshold() -> None:
    """When no probabilities are below the threshold, fall back to full vocab."""

    probs = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    k = _select_cutoff_k(probs, threshold=0.1, topk=50)

    assert k == 3


def test_select_cutoff_k_respects_topk_limit() -> None:
    """Ensure the top-k cap is still honoured when a fallback occurs."""

    probs = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    k = _select_cutoff_k(probs, threshold=0.1, topk=2)

    assert k == 2
