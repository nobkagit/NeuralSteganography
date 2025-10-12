"""Arithmetic coding helpers integrating crypto quality policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, cast

from ..codec.arithmetic import (
    _coerce_optional_float,
    _coerce_optional_int,
    decode_with_lm,
    encode_with_lm,
)
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
    state: CodecState | None = None,
) -> tuple[list[int], CodecState]:
    """Encode ``payload`` bits using ``lm`` with the provided quality policies."""

    policies = _extract_quality(quality)
    controlled = _QualityControlledLM(
        lm,
        top_k=policies.top_k,
        top_p=policies.top_p,
        temperature=policies.temperature,
    )

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
    state: CodecState | None = None,
) -> bytes:
    """Decode ``token_ids`` into the embedded payload bits."""

    if state is None:
        raise ValueError("state with bit consumption history is required for decoding")

    policies = _extract_quality(quality)
    controlled = _QualityControlledLM(
        lm,
        top_k=policies.top_k,
        top_p=policies.top_p,
        temperature=policies.temperature,
    )

    decode_state: CodecState = {"history": tuple(), "residual_bits": b""}
    if "history" in state:
        decode_state["history"] = cast(Sequence[int], state["history"])
    if "residual_bits" in state:
        decode_state["residual_bits"] = cast(bytes, state["residual_bits"])

    return decode_with_lm(
        token_ids,
        controlled,
        context=tuple(seed_text or ()),
        state=decode_state,
    )


@dataclass(frozen=True)
class _QualityPolicies:
    top_k: int | None
    top_p: float | None
    temperature: float


def _extract_quality(quality: Mapping[str, object] | None) -> _QualityPolicies:
    if not quality:
        return _QualityPolicies(top_k=None, top_p=None, temperature=1.0)

    top_k = quality.get("top_k")
    top_p = quality.get("top_p")
    temperature = quality.get("temperature", 1.0)

    coerced_top_k = _coerce_optional_int(top_k, "top_k")
    coerced_top_p = _coerce_optional_float(top_p, "top_p")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    coerced_temperature = float(temperature)

    return _QualityPolicies(top_k=coerced_top_k, top_p=coerced_top_p, temperature=coerced_temperature)
