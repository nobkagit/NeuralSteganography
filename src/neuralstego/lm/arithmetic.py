"""Adapters integrating the arithmetic codec with HuggingFace models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch

from ..codec.arithmetic import decode_with_lm, encode_with_lm
from ..codec.types import CodecState, LMProvider
from ..exceptions import ConfigurationError


def _bits_to_bytes(bits: Iterable[int]) -> bytes:
    data = list(bits)
    if len(data) % 8 != 0:
        raise ConfigurationError("bit stream length must be a multiple of 8")

    out = bytearray()
    for index in range(0, len(data), 8):
        value = 0
        for offset, bit in enumerate(data[index : index + 8]):
            value |= (int(bit) & 1) << offset
        out.append(value)
    return bytes(out)


def _bytes_to_bits(payload: bytes) -> List[int]:
    bits: List[int] = []
    for byte in payload:
        bits.extend((byte >> offset) & 1 for offset in range(8))
    return bits


@dataclass
class _ModelAdapter(LMProvider):
    """Expose ``next_token_probs`` for preloaded causal language models."""

    model: torch.nn.Module
    temperature: float
    max_context: Optional[int]

    def next_token_probs(self, context_ids: Iterable[int]):
        ids = list(int(token) for token in context_ids)
        if not ids:
            raise ConfigurationError("context must contain at least one token")

        if self.max_context is not None and len(ids) > self.max_context:
            ids = ids[-self.max_context :]

        device_attr = getattr(self.model, "device", None)
        if device_attr is not None:
            device = torch.device(device_attr)
        else:
            try:
                first_param = next(self.model.parameters())
            except (StopIteration, TypeError):
                device = torch.device("cpu")
            else:
                device = first_param.device

        input_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        logits = outputs.logits[0, -1, :].to(dtype=torch.float64)
        if self.temperature <= 0.0:
            raise ConfigurationError("temperature must be positive")
        logits = logits / self.temperature
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        return probs


def _normalise_quality(quality: Mapping[str, object] | None) -> Dict[str, float]:
    if not quality:
        return {}

    normalised: Dict[str, float] = {}
    for key, value in quality.items():
        if value is None:
            continue
        key_norm = key.replace("-", "_").lower()
        if key_norm in {"topk", "top_k"}:
            normalised["top_k"] = float(int(value))
        elif key_norm in {"topp", "top_p"}:
            normalised["top_p"] = float(value)
        elif key_norm in {"minprob", "min_prob"}:
            normalised["min_prob"] = float(value)
        elif key_norm in {"cap_per_token_bits", "cap_bits_per_token"}:
            normalised["cap_per_token_bits"] = float(int(value))
    return normalised


def _quality_temperature(quality: Mapping[str, object] | None) -> float:
    if not quality:
        return 1.0
    for key in ("temp", "temperature"):
        if key in quality and quality[key] is not None:
            return float(quality[key])
    return 1.0


def _quality_max_context(quality: Mapping[str, object] | None) -> Optional[int]:
    if not quality:
        return None
    for key in ("max_context", "maxContext"):
        if key in quality and quality[key] is not None:
            return int(quality[key])
    return None


class ArithmeticLM:
    """Wrap a loaded GPT-style model with arithmetic encode/decode helpers."""

    def __init__(self, model, tokenizer, *, device: Optional[str] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer

        model_device = getattr(model, "device", None)
        if hasattr(model_device, "type"):
            model_device = model_device.type
        resolved_device = device or model_device or "cpu"

        if isinstance(resolved_device, str):
            target_device = torch.device(resolved_device)
        else:
            target_device = resolved_device

        try:
            self.model = self.model.to(target_device)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ConfigurationError("model does not support device placement") from exc

        self.model.eval()
        self.device = target_device
        self._encode_states: List[CodecState] = []
        self._decode_states: Deque[CodecState] = deque()

    # ------------------------------------------------------------------
    def encode_seed(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            bos_tokens: List[int] = []
            try:
                bos_tokens = list(self.tokenizer.encode("<|endoftext|>", add_special_tokens=False))
            except TypeError:
                bos_tokens = list(self.tokenizer.encode("<|endoftext|>"))
            except Exception:
                bos_tokens = []

            try:
                tokens = list(self.tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                tokens = list(self.tokenizer.encode(text))
            if not tokens:
                tokens = list(self.tokenizer.encode(text))
            return [int(token) for token in bos_tokens + tokens]
        return list(text.encode("utf-8"))

    def encode_arithmetic(
        self,
        bits: List[int],
        context: List[int],
        *,
        quality: Mapping[str, object],
    ) -> List[int]:
        payload = _bits_to_bytes(bits)
        quality_args = _normalise_quality(quality)
        temperature = _quality_temperature(quality)
        max_context = _quality_max_context(quality)

        provider = _ModelAdapter(self.model, temperature=temperature, max_context=max_context)
        state: MutableMapping[str, object] = {}
        tokens = encode_with_lm(
            payload,
            provider,
            context=context,
            quality=quality_args,
            state=state,
            max_context=max_context,
        )

        codec_state: CodecState = {
            "history": tuple(int(value) for value in state.get("history", ())),
            "residual_bits": bytes(state.get("residual_bits", b"")),
        }
        self._encode_states.append(dict(codec_state))
        self._decode_states.append(dict(codec_state))
        return [int(token) for token in tokens]

    def decode_arithmetic(
        self,
        tokens: List[int],
        context: List[int],
        *,
        quality: Mapping[str, object],
    ) -> List[int]:
        if not tokens:
            if self._decode_states:
                self._decode_states.popleft()
            return []
        if not self._decode_states:
            raise ConfigurationError("decode state unavailable for ArithmeticLM")

        state_snapshot = self._decode_states.popleft()
        decode_state: MutableMapping[str, object] = {
            "history": tuple(int(value) for value in state_snapshot.get("history", ())),
            "residual_bits": bytes(state_snapshot.get("residual_bits", b"")),
        }

        quality_args = _normalise_quality(quality)
        temperature = _quality_temperature(quality)
        max_context = _quality_max_context(quality)
        provider = _ModelAdapter(self.model, temperature=temperature, max_context=max_context)

        payload = decode_with_lm(
            tokens,
            provider,
            context=context,
            quality=quality_args,
            state=decode_state,
            max_context=max_context,
        )
        return _bytes_to_bits(payload)

    # ------------------------------------------------------------------
    def drain_states(self) -> List[CodecState]:
        states = [dict(state) for state in self._encode_states]
        self._encode_states.clear()
        return states

    def load_states(self, states: Iterable[CodecState]) -> None:
        self._decode_states = deque(dict(state) for state in states)


__all__ = ["ArithmeticLM"]
