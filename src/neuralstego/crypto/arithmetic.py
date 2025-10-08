"""Arithmetic coding helpers integrating crypto quality policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from ..codec.arithmetic import decode_with_lm, encode_with_lm
from ..codec.types import CodecState, LMProvider, ProbDist
from .quality import apply_quality

__all__ = [
    "decode_with_lm",
    "encode_with_lm",
    "decode_arithmetic",
    "encode_arithmetic",
]


@dataclass
class _QualityControlledLM(LMProvider):
    """Adapter applying quality policies to an underlying language model."""

    base: LMProvider
    top_k: int | None = None
    top_p: float | None = None
    temperature: float = 1.0

    def next_token_probs(self, context_ids: Sequence[int]) -> ProbDist:  # noqa: D401
        """Return filtered next-token probabilities for ``context_ids``."""

        dist = self.base.next_token_probs(context_ids)
        if self.top_k is None and self.top_p is None and self.temperature == 1.0:
            return dist
        return apply_quality(
            dist,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
        )


def encode_arithmetic(
    payload: bytes,
    lm: LMProvider,
    *,
    quality: Mapping[str, object] | None = None,
    seed_text: Sequence[int] | None = None,
    state: MutableMapping[str, object] | None = None,
) -> tuple[list[int], CodecState]:
    """Encode ``payload`` bits using ``lm`` with the provided quality policies."""

    policies = _extract_quality(quality)
    controlled = _QualityControlledLM(lm, **policies)

    encode_state: CodecState = {}
    tokens = encode_with_lm(payload, controlled, context=tuple(seed_text or ()), state=encode_state)

    if state is not None:
        state.update(encode_state)

    return tokens, encode_state


def decode_arithmetic(
    token_ids: Sequence[int],
    lm: LMProvider,
    *,
    quality: Mapping[str, object] | None = None,
    seed_text: Sequence[int] | None = None,
    state: MutableMapping[str, object] | None = None,
) -> bytes:
    """Decode ``token_ids`` into the embedded payload bits."""

    if state is None:
        raise ValueError("state with bit consumption history is required for decoding")

    policies = _extract_quality(quality)
    controlled = _QualityControlledLM(lm, **policies)

    decode_state: CodecState = {"history": tuple(), "residual_bits": b""}
    decode_state.update(state)  # type: ignore[arg-type]

    return decode_with_lm(
        token_ids,
        controlled,
        context=tuple(seed_text or ()),
        state=decode_state,
    )


def _extract_quality(quality: Mapping[str, object] | None) -> dict[str, object]:
    if not quality:
        return {"top_k": None, "top_p": None, "temperature": 1.0}

    policies: dict[str, object] = {}
    if "top_k" in quality:
        policies["top_k"] = int(quality["top_k"]) if quality["top_k"] is not None else None
    else:
        policies["top_k"] = None

    if "top_p" in quality:
        policies["top_p"] = float(quality["top_p"]) if quality["top_p"] is not None else None
    else:
        policies["top_p"] = None

    if "temperature" in quality:
        policies["temperature"] = float(quality["temperature"])
    else:
        policies["temperature"] = 1.0

    return policies
