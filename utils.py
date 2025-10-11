"""Utility helpers for loading pretrained language models."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing hints only
    from transformers import AutoModelForCausalLM, AutoTokenizer

TokenizerModelPair = Tuple["AutoTokenizer", "AutoModelForCausalLM"]


def _load_pretrained(factory, model_name: str):
    """Load a pretrained resource with an offline-first strategy."""

    try:
        return factory.from_pretrained(model_name, local_files_only=True)
    except OSError:
        try:
            return factory.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - network issues
            hint = (
                "failed to load pretrained weights for "
                f"'{model_name}'. Download the model with `python scripts/download_models.py --model "
                f"{model_name}` before running offline."
            )
            raise RuntimeError(hint) from exc


def get_model(seed: int = 1234, model_name: str = "gpt2") -> TokenizerModelPair:
    """Return a tokenizer/model pair seeded for deterministic tests."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no branch - optional GPU path
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = _load_pretrained(AutoTokenizer, model_name)
    tokenizer.unk_token = None
    tokenizer.bos_token = None
    tokenizer.eos_token = None

    model = _load_pretrained(AutoModelForCausalLM, model_name)
    model.to(device)
    model.eval()

    return tokenizer, model


__all__ = ["_load_pretrained", "get_model", "TokenizerModelPair"]
