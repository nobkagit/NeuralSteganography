"""High level neural steganography API for chunked framing."""
from __future__ import annotations

import base64
import binascii
import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

try:  # pragma: no cover - optional dependency for pretty logging
    from rich.console import Console  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - graceful fallback when rich is absent
    Console = None

from .codec import assemble_bytes, build_packet, chunk_bytes, make_msg_id, parse_packet
from .codec.packet import RSCodec as _RSCodec  # type: ignore[attr-defined]
from .codec.arithmetic import (
    _coerce_optional_float,
    _coerce_optional_int,
    decode_with_lm as codec_decode_with_lm,
    encode_with_lm as codec_encode_with_lm,
)
from .codec.textio import seed_to_ids, spans_to_text, text_to_spans
from .lm import load_lm
from .codec.types import CodecState, LMProvider as CodecLMBase
from .exceptions import ConfigurationError, MissingChunksError, QualityGateError
from .detect.guard import GuardResult, QualityGuard
from .metrics import LMScorer


class LMProvider(Protocol):
    """Protocol implemented by language model wrappers used for arithmetic coding."""

    def encode_arithmetic(
        self, bits: List[int], context: List[int], *, quality: Mapping[str, object]
    ) -> List[int]:
        ...

    def decode_arithmetic(
        self, tokens: List[int], context: List[int], *, quality: Mapping[str, object]
    ) -> List[int]:
        ...

    def encode_seed(self, text: str) -> List[int]:
        ...


@dataclass
class EncodeMetadata:
    msg_id: str
    total: int
    cfg: Dict[str, object]


class EncodeResult(list):
    """List-like container with attached metadata for encode results."""

    def __init__(self, spans: Iterable[List[int]], metadata: EncodeMetadata) -> None:
        super().__init__(spans)
        self.metadata = metadata


@dataclass
class _AttemptConfig:
    seed_text: str
    overrides: Dict[str, Any]
    seed_variant: str


_DEFAULT_QUALITY = {
    "temp": 1.0,
    "precision": 16,
    "topk": 50000,
    "finish_sent": True,
}


DEFAULT_GATE_THRESHOLDS: Dict[str, float] = {
    "max_ppl": 120.0,
    "max_ngram_repeat": 0.35,
    "min_ttr": 0.25,
    "max_avg_entropy": 5.5,
}


DEFAULT_REGEN_STRATEGY: Dict[str, Any] = {
    "seed_pool": [
        "در یک گفت‌وگوی کوتاه درباره‌ی فناوری صحبت می‌کنیم.",
        "در یک گفت‌وگوی دوستانه درباره‌ی فرهنگ و هنر صحبت می‌کنیم.",
    ],
    "top_k_steps": [80, 70, 60],
    "temperature_steps": [0.8, 0.7],
}


_QUALITY_CONSOLE = Console(stderr=True) if Console is not None else None


def _quality_log(message: str) -> None:
    if _QUALITY_CONSOLE is not None:
        _QUALITY_CONSOLE.log(message)


_DEFAULT_QUALITY_GUARD: QualityGuard | None = None


def _ensure_quality_guard(candidate: QualityGuard | None = None) -> QualityGuard:
    if candidate is not None:
        return candidate

    global _DEFAULT_QUALITY_GUARD
    if _DEFAULT_QUALITY_GUARD is None:
        _DEFAULT_QUALITY_GUARD = QualityGuard(
            lm_scorer=LMScorer(prefer_transformers=False)
        )
    return _DEFAULT_QUALITY_GUARD


_QUALITY_KEY_ALIASES = {
    "temperature": "temp",
    "top-k": "top_k",
    "topk": "top_k",
    "top_p": "top_p",
    "top-p": "top_p",
    "cap-per-token-bits": "cap_per_token_bits",
    "cap_bits_per_token": "cap_per_token_bits",
    "cap-bits-per-token": "cap_per_token_bits",
    "max-context": "max_context",
    "maxContext": "max_context",
}


def _normalise_ecc(ecc: Optional[str]) -> str:
    if not ecc:
        return "none"
    ecc_norm = ecc.lower()
    if ecc_norm not in {"none", "rs"}:
        raise ConfigurationError(f"unsupported ecc mode: {ecc}")
    return ecc_norm


def _bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        bits.extend((byte >> i) & 1 for i in range(8))
    return bits


def _bits_to_bytes(bits: Iterable[int]) -> bytes:
    bits_list = list(bits)
    if len(bits_list) % 8:
        raise ConfigurationError("decoded bit stream is not byte aligned")
    out = bytearray()
    for i in range(0, len(bits_list), 8):
        value = 0
        for offset, bit in enumerate(bits_list[i : i + 8]):
            value |= (bit & 1) << offset
        out.append(value)
    return bytes(out)


def _context_tokens(lm: LMProvider, seed_text: str) -> List[int]:
    return lm.encode_seed(seed_text)


def _normalise_quality_dict(quality: Mapping[str, object] | None) -> Dict[str, Any]:
    if not quality:
        return {}

    normalised: Dict[str, Any] = {}
    for key, value in quality.items():
        canonical = _QUALITY_KEY_ALIASES.get(key, key)
        normalised[canonical] = value
    return normalised


def _codec_quality_arguments(
    quality: Mapping[str, object] | None,
) -> Tuple[Dict[str, Any], Optional[int]]:
    if not quality:
        return {}, None

    quality = _normalise_quality_dict(quality)

    policies: Dict[str, Any] = {}
    max_context: Optional[int] = None

    if "top_k" in quality and quality["top_k"] is not None:
        top_k = _coerce_optional_int(quality["top_k"], "top_k")
        if top_k is not None:
            policies["top_k"] = top_k

    if "top_p" in quality and quality["top_p"] is not None:
        top_p = _coerce_optional_float(quality["top_p"], "top_p")
        if top_p is not None:
            policies["top_p"] = top_p

    if "min_prob" in quality and quality["min_prob"] is not None:
        min_prob = _coerce_optional_float(quality["min_prob"], "min_prob")
        if min_prob is not None:
            policies["min_prob"] = min_prob

    if "cap_per_token_bits" in quality and quality["cap_per_token_bits"] is not None:
        cap_bits = _coerce_optional_int(quality["cap_per_token_bits"], "cap_per_token_bits")
        if cap_bits is not None:
            policies["cap_per_token_bits"] = cap_bits

    if "max_context" in quality and quality["max_context"] is not None:
        max_context = _coerce_optional_int(quality["max_context"], "max_context")

    filtered = {key: value for key, value in policies.items() if value is not None}
    return filtered, max_context


def _extract_codec_state(state: Mapping[str, object]) -> CodecState:
    history_raw = state.get("history", ())
    if isinstance(history_raw, Sequence):
        history = [int(value) for value in history_raw]
    else:
        history = []

    residual_raw = state.get("residual_bits", b"")
    if isinstance(residual_raw, (bytes, bytearray)):
        residual = bytes(residual_raw)
    elif isinstance(residual_raw, Sequence):
        residual = bytes(int(value) & 0xFF for value in residual_raw)
    else:
        residual = bytes()

    return {"history": history, "residual_bits": residual}


def _serialise_state_payload(state: CodecState) -> Dict[str, Any]:
    history = [int(value) for value in state.get("history", ())]
    residual_bits = state.get("residual_bits", b"")
    if isinstance(residual_bits, (bytes, bytearray)):
        residual_bytes = bytes(residual_bits)
    elif isinstance(residual_bits, Sequence):
        residual_bytes = bytes(int(value) & 0xFF for value in residual_bits)
    else:
        residual_bytes = bytes()

    encoded_residual = base64.b64encode(residual_bytes).decode("ascii")
    return {"history": history, "residual": encoded_residual}


def _deserialise_state_payload(payload: Mapping[str, Any]) -> CodecState:
    history_raw = payload.get("history", [])
    if not isinstance(history_raw, Sequence):
        raise ConfigurationError("state history must be a sequence of integers")
    history = [int(value) for value in history_raw]

    residual_encoded = payload.get("residual", "")
    if not isinstance(residual_encoded, str):
        raise ConfigurationError("state residual must be a base64 string")
    try:
        residual_bytes = base64.b64decode(residual_encoded.encode("ascii"), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ConfigurationError("invalid residual_bits encoding") from exc

    return {"history": history, "residual_bits": residual_bytes}


def _bytes_to_nibbles(data: bytes) -> List[int]:
    tokens: List[int] = []
    for byte in data:
        tokens.append((byte >> 4) & 0x0F)
        tokens.append(byte & 0x0F)
    return tokens


def _nibbles_to_bytes(tokens: Sequence[int]) -> bytes:
    if len(tokens) % 2 != 0:
        raise ConfigurationError("encoded token stream is truncated")

    output = bytearray()
    for high, low in zip(tokens[0::2], tokens[1::2]):
        if not (0 <= int(high) <= 0x0F and 0 <= int(low) <= 0x0F):
            raise ConfigurationError("encoded tokens must be 4-bit values")
        output.append((int(high) << 4) | int(low))
    return bytes(output)


def _serialise_envelope(envelope: Mapping[str, Any]) -> List[int]:
    payload = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return _bytes_to_nibbles(payload.encode("utf-8"))


def _deserialise_envelope(tokens: Sequence[int]) -> Mapping[str, Any]:
    raw = _nibbles_to_bytes(tokens)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ConfigurationError("encoded token stream is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ConfigurationError("encoded payload must be a JSON object")
    return decoded


class _CodecLMAdapter(LMProvider):
    """Adapter bridging probability-only language models to :class:`LMProvider`."""

    def __init__(self, base: CodecLMBase) -> None:
        self._base = base
        self._encode_states: List[CodecState] = []
        self._decode_states: Deque[CodecState] = deque()

    @property
    def tokenizer(self) -> Any:
        return getattr(self._base, "tokenizer", None)

    def encode_seed(self, text: str) -> List[int]:
        encoder = getattr(self._base, "encode_seed", None)
        if callable(encoder):
            tokens = encoder(text)
            return [int(token) for token in tokens]

        tokenizer = self.tokenizer
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if not encoded and hasattr(tokenizer, "encode"):
                encoded = tokenizer.encode(text)
            return [int(token) for token in encoded]

        return list(text.encode("utf-8"))

    def encode_arithmetic(
        self, bits: List[int], context: List[int], *, quality: Mapping[str, object]
    ) -> List[int]:
        payload = _bits_to_bytes(bits)
        quality_args, max_context = _codec_quality_arguments(quality)
        state: CodecState = {}
        tokens = codec_encode_with_lm(
            payload,
            self._base,
            context=tuple(context),
            quality=quality_args,
            state=state,
            max_context=max_context,
        )
        self._encode_states.append(_extract_codec_state(state))
        return [int(token) for token in tokens]

    def decode_arithmetic(
        self, tokens: List[int], context: List[int], *, quality: Mapping[str, object]
    ) -> List[int]:
        if not self._decode_states:
            raise ConfigurationError("decode state unavailable for codec language model")

        state_payload = self._decode_states.popleft()
        extracted = _extract_codec_state(state_payload)
        history_seq = cast(Sequence[int], extracted.get("history", ()))
        residual_bits = extracted.get("residual_bits", b"")
        decode_state: CodecState = {
            "history": tuple(history_seq),
            "residual_bits": bytes(residual_bits),
        }
        quality_args, max_context = _codec_quality_arguments(quality)
        payload = codec_decode_with_lm(
            tokens,
            self._base,
            context=tuple(context),
            quality=quality_args,
            state=decode_state,
            max_context=max_context,
        )
        return _bytes_to_bits(payload)

    def drain_states(self) -> List[CodecState]:
        states = list(self._encode_states)
        self._encode_states.clear()
        return states

    def load_states(self, states: Sequence[CodecState]) -> None:
        self._decode_states = deque(states)


def _ensure_lm(candidate: Any) -> LMProvider:
    if all(hasattr(candidate, attr) for attr in ("encode_arithmetic", "decode_arithmetic", "encode_seed")):
        return cast(LMProvider, candidate)

    if hasattr(candidate, "next_token_probs"):
        return _CodecLMAdapter(cast(CodecLMBase, candidate))

    raise TypeError("language model must implement arithmetic encode/decode or next_token_probs")


def _default_cover_lm() -> LMProvider:
    try:
        return cast(LMProvider, load_lm("gpt2-fa"))
    except Exception as exc:  # pragma: no cover - propagates loader errors
        raise ConfigurationError("failed to load default language model") from exc


def _ensure_cover_lm(lm: LMProvider | None) -> LMProvider:
    if lm is None:
        return _default_cover_lm()
    return _ensure_lm(lm)


def _resolve_tokenizer(candidate: Any):
    tokenizer = getattr(candidate, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    try:
        from .lm.transformers_adapter import TransformersLM
    except Exception as exc:  # pragma: no cover - dependency missing path
        raise ConfigurationError("language model tokenizer unavailable") from exc

    helper = TransformersLM()
    return helper.tokenizer


def _normalise_secret(secret: bytes | str) -> bytes:
    if isinstance(secret, bytes):
        return secret
    if isinstance(secret, str):
        return secret.encode("utf-8")
    raise TypeError("secret must be bytes or string")


def _parse_spans_payload(payload: str) -> List[List[int]]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            "cover text parsing not implemented; provide spans JSON input"
        ) from exc

    if isinstance(decoded, Mapping):
        spans_obj = decoded.get("spans")
    else:
        spans_obj = decoded

    if not isinstance(spans_obj, Sequence):
        raise ConfigurationError("spans payload must be a sequence")

    spans: List[List[int]] = []
    for entry in spans_obj:
        if not isinstance(entry, Sequence):
            raise ConfigurationError("span entry must be a sequence of integers")
        spans.append([int(value) for value in entry])

    return spans


def _prepare_gate_thresholds(overrides: Mapping[str, float] | None) -> Dict[str, float]:
    thresholds = dict(DEFAULT_GATE_THRESHOLDS)
    if not overrides:
        return thresholds

    for key, value in overrides.items():
        if value is None:
            continue
        try:
            thresholds[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"invalid threshold value for {key!s}: {value!r}"
            ) from exc
    return thresholds


def _prepare_regen_strategy(strategy: Mapping[str, Any] | None) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_REGEN_STRATEGY)
    if strategy:
        for key, value in strategy.items():
            if value is None:
                continue
            merged[str(key)] = value

    merged["seed_pool"] = list(merged.get("seed_pool", []))
    merged["top_k_steps"] = list(merged.get("top_k_steps", []))
    merged["temperature_steps"] = list(merged.get("temperature_steps", []))
    return merged


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(float(value))


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigurationError(f"invalid numeric value for regeneration strategy: {value!r}")


def _iter_attempts(
    seed_text: str,
    regen_attempts: int,
    strategy: Mapping[str, Any],
) -> Iterable[_AttemptConfig]:
    config = _prepare_regen_strategy(strategy)
    total = max(regen_attempts, 0) + 1

    seed_pool = deque(str(seed) for seed in config.get("seed_pool", []))
    top_k_steps = deque(config.get("top_k_steps", []))
    temperature_steps = deque(config.get("temperature_steps", []))

    for index in range(total):
        if index == 0:
            candidate_seed = seed_text
            seed_variant = "base"
        else:
            candidate_seed = seed_text
            if seed_pool:
                candidate_seed = str(seed_pool.popleft())
            seed_variant = f"alt-{index}"

        overrides: Dict[str, Any] = {}
        if index > 0:
            if top_k_steps:
                overrides["top_k"] = _coerce_int(top_k_steps.popleft())
            if temperature_steps:
                overrides["temp"] = _coerce_float(temperature_steps.popleft())

        yield _AttemptConfig(candidate_seed, overrides, seed_variant)


def _generate_cover_once(
    payload: bytes,
    *,
    seed_text: str,
    quality: Mapping[str, Any],
    chunk_bytes: int,
    use_crc: bool,
    ecc: str,
    nsym: int,
    lm: LMProvider,
) -> str:
    encode_result = stego_encode(
        payload,
        chunk_bytes=chunk_bytes,
        use_crc=use_crc,
        ecc=ecc,
        nsym=nsym,
        quality=quality,
        seed_text=seed_text,
        lm=lm,
    )

    spans = [list(map(int, span)) for span in encode_result]

    try:
        tokenizer = _resolve_tokenizer(lm)
    except ConfigurationError:
        tokenizer = None

    if tokenizer is None:
        raise ConfigurationError("language model tokenizer unavailable for cover rendering")

    seed_ids = seed_to_ids(seed_text, tokenizer)
    cover_text = spans_to_text(spans, seed_ids, tokenizer)
    return cover_text


def cover_generate(
    secret: bytes | str,
    *,
    seed_text: str,
    quality: Mapping[str, object] | None = None,
    chunk_bytes: int = 256,
    use_crc: bool = True,
    ecc: str = "rs",
    nsym: int = 10,
    lm: LMProvider | None = None,
    quality_gate: bool = True,
    gate_thresholds: Mapping[str, float] | None = None,
    regen_attempts: int = 2,
    regen_strategy: Mapping[str, Any] | None = None,
    quality_guard: QualityGuard | None = None,
) -> str:
    """Generate a natural-language cover string embedding *secret* information."""

    payload = _normalise_secret(secret)
    lm_provider = _ensure_cover_lm(lm)

    quality_args = _normalise_quality_dict(quality)
    if not quality_gate:
        return _generate_cover_once(
            payload,
            seed_text=seed_text,
            quality=quality_args,
            chunk_bytes=chunk_bytes,
            use_crc=use_crc,
            ecc=ecc,
            nsym=nsym,
            lm=lm_provider,
        )

    thresholds = _prepare_gate_thresholds(gate_thresholds)
    guard = _ensure_quality_guard(quality_guard)
    strategy = regen_strategy or {}

    total_attempts = max(regen_attempts, 0) + 1
    last_result: GuardResult | None = None
    last_cover = ""

    for attempt_index, attempt in enumerate(
        _iter_attempts(seed_text, regen_attempts, strategy), start=1
    ):
        attempt_quality = dict(quality_args)
        attempt_quality.update(attempt.overrides)

        cover_text = _generate_cover_once(
            payload,
            seed_text=attempt.seed_text,
            quality=attempt_quality,
            chunk_bytes=chunk_bytes,
            use_crc=use_crc,
            ecc=ecc,
            nsym=nsym,
            lm=lm_provider,
        )

        last_cover = cover_text
        guard_result = guard.evaluate(cover_text, thresholds)
        last_result = guard_result

        status = "PASS" if guard_result.passed else "REJECT"
        ppl = guard_result.metrics.get("ppl")
        ttr = guard_result.metrics.get("type_token_ratio")
        metric_parts = []
        if ppl is not None:
            metric_parts.append(f"ppl={ppl:.2f}")
        if ttr is not None:
            metric_parts.append(f"ttr={ttr:.2f}")
        adjustments = ", ".join(f"{key}={value}" for key, value in attempt.overrides.items())
        adjustment_desc = adjustments if adjustments else "default"
        metrics_desc = ", ".join(metric_parts) if metric_parts else "no-metrics"
        _quality_log(
            (
                f"[cyan]quality attempt {attempt_index}/{total_attempts} ({attempt.seed_variant})[/cyan] "
                f"{metrics_desc} → {status} ({adjustment_desc})"
            )
        )
        if not guard_result.passed and guard_result.reasons:
            for reason in guard_result.reasons:
                _quality_log(f"[yellow]- {reason}[/yellow]")

        if guard_result.passed:
            return cover_text

        if attempt_index >= total_attempts:
            break

    if last_result is None:
        raise QualityGateError(last_cover, ["quality evaluation failed"], {})

    raise QualityGateError(
        last_cover,
        list(last_result.reasons),
        dict(last_result.metrics),
    )


def cover_reveal(
    cover_text: str,
    *,
    seed_text: str,
    quality: Mapping[str, object] | None = None,
    use_crc: bool = True,
    ecc: str = "rs",
    nsym: int = 10,
    lm: LMProvider | None = None,
) -> bytes:
    """Recover the secret payload embedded within ``cover_text``."""

    lm_provider = _ensure_cover_lm(lm)
    quality_args = _normalise_quality_dict(quality)

    try:
        tokenizer = _resolve_tokenizer(lm_provider)
    except ConfigurationError:
        tokenizer = None

    spans: List[List[int]]
    if tokenizer is not None:
        try:
            seed_ids = seed_to_ids(seed_text, tokenizer)
            spans = text_to_spans(cover_text, seed_ids, tokenizer)
        except NotImplementedError:
            spans = _parse_spans_payload(cover_text)
    else:
        spans = _parse_spans_payload(cover_text)

    result = stego_decode(
        spans,
        use_crc=use_crc,
        ecc=ecc,
        nsym=nsym,
        quality=quality_args,
        seed_text=seed_text,
        lm=lm_provider,
    )
    return result


def stego_encode(
    message: bytes,
    *,
    chunk_bytes: int = 256,
    use_crc: bool = True,
    ecc: str | None = "rs",
    nsym: int = 10,
    quality: Mapping[str, object] | None = None,
    seed_text: str = "",
    lm: LMProvider,
) -> EncodeResult:
    """Encode *message* into token spans using arithmetic coding."""

    ecc_mode = _normalise_ecc(ecc)
    cfg = {
        "chunk_bytes": int(chunk_bytes),
        "crc": bool(use_crc),
        "ecc": ecc_mode,
        "nsym": int(nsym if ecc_mode == "rs" else 0),
    }
    quality_args = {**_DEFAULT_QUALITY, **_normalise_quality_dict(quality)}

    chunk_size = cast(int, cfg["chunk_bytes"])
    chunks = chunk_bytes_func(message, chunk_size=chunk_size)
    msg_id = make_msg_id()
    total = len(chunks)

    metadata = EncodeMetadata(msg_id=msg_id, total=total, cfg=cfg)
    spans: List[List[int]] = []

    for seq, chunk in enumerate(chunks):
        packet_bytes = build_packet(
            chunk,
            msg_id=msg_id,
            seq=seq,
            total=total,
            cfg=cfg,
        )
        bits = _bytes_to_bits(packet_bytes)
        context = _context_tokens(lm, seed_text)
        tokens = lm.encode_arithmetic(bits, context, quality=quality_args)
        spans.append(tokens)

    return EncodeResult(spans, metadata)


def stego_decode(
    spans: Iterable[List[int]],
    *,
    use_crc: bool = True,
    ecc: str | None = "rs",
    nsym: int = 10,
    quality: Mapping[str, object] | None = None,
    seed_text: str = "",
    lm: LMProvider,
) -> bytes:
    """Decode arithmetic coded *spans* back into the original message."""

    ecc_mode = _normalise_ecc(ecc)
    expected_cfg = {
        "crc": bool(use_crc),
        "ecc": ecc_mode,
        "nsym": int(nsym if ecc_mode == "rs" else 0),
    }
    quality_args = {**_DEFAULT_QUALITY, **_normalise_quality_dict(quality)}

    payload_by_seq: Dict[int, bytes] = {}
    msg_id: Optional[str] = None
    total: Optional[int] = None

    for span in spans:
        span_list = list(span)
        context = _context_tokens(lm, seed_text)
        bits = lm.decode_arithmetic(span_list, context, quality=quality_args)
        packet_bytes = _bits_to_bytes(bits)
        packet = parse_packet(packet_bytes, expected_cfg=expected_cfg)

        if msg_id is None:
            msg_id = packet.msg_id
            total = packet.total
        else:
            if packet.msg_id != msg_id:
                raise ConfigurationError("decoded packet msg_id mismatch")
            if packet.total != total:
                raise ConfigurationError("decoded packet total mismatch")

        if packet.seq in payload_by_seq:
            raise ConfigurationError(f"duplicate packet sequence {packet.seq}")
        payload_by_seq[packet.seq] = packet.payload

    if total is None:
        return b""

    present_indices = sorted(payload_by_seq.keys())
    missing = sorted(set(range(total)) - payload_by_seq.keys())
    ordered_payloads = [payload_by_seq[i] for i in present_indices]
    assembled = assemble_bytes(ordered_payloads)

    if missing:
        raise MissingChunksError(missing_indices=missing, partial_payload=assembled)

    return assembled


def encode_text(
    message: bytes | bytearray | memoryview,
    lm: Any,
    *,
    chunk_bytes: int = 256,
    use_crc: bool = True,
    ecc: str | None = "rs",
    nsym: int = 10,
    quality: Optional[Mapping[str, object]] = None,
    seed_text: str = "",
) -> List[int]:
    """Encode ``message`` using ``lm`` and return a serialised token stream."""

    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")

    payload = bytes(message)
    quality_args = _normalise_quality_dict(quality)
    lm_adapter = _ensure_lm(lm)

    ecc_mode = _normalise_ecc(ecc)
    if ecc_mode == "rs" and _RSCodec is None:
        ecc_mode = "none"
    effective_nsym = int(nsym if ecc_mode == "rs" else 0)

    encode_result = stego_encode(
        payload,
        chunk_bytes=chunk_bytes,
        use_crc=use_crc,
        ecc=ecc_mode,
        nsym=effective_nsym,
        quality=quality_args,
        seed_text=seed_text,
        lm=lm_adapter,
    )

    spans = [list(map(int, span)) for span in encode_result]
    chunk_entries: List[Dict[str, Any]] = []

    adapter_states: List[CodecState] = []
    state_drain = getattr(lm_adapter, "drain_states", None)
    if callable(state_drain):
        adapter_states = list(state_drain())
        if adapter_states and len(adapter_states) != len(spans):
            raise ConfigurationError("language model state mismatch during encoding")

    for index, tokens in enumerate(spans):
        entry: Dict[str, Any] = {
            "seq": index,
            "msg_id": encode_result.metadata.msg_id,
            "tokens": tokens,
        }
        if adapter_states:
            entry["state"] = _serialise_state_payload(adapter_states[index])
        chunk_entries.append(entry)

    cfg = encode_result.metadata.cfg
    cfg_ecc_raw = cfg.get("ecc", ecc_mode)
    ecc_payload = (
        _normalise_ecc(cfg_ecc_raw) if isinstance(cfg_ecc_raw, str) or cfg_ecc_raw is None else str(cfg_ecc_raw)
    )
    chunk_bytes_value = cfg.get("chunk_bytes", chunk_bytes)
    chunk_bytes_serial = _coerce_optional_int(chunk_bytes_value, "chunk_bytes")
    if chunk_bytes_serial is None:
        chunk_bytes_serial = int(chunk_bytes)

    nsym_value = cfg.get("nsym", effective_nsym)
    nsym_serial = _coerce_optional_int(nsym_value, "nsym")
    if nsym_serial is None:
        nsym_serial = int(effective_nsym)

    cfg_payload = {
        "chunk_bytes": chunk_bytes_serial,
        "crc": bool(cfg.get("crc", use_crc)),
        "ecc": ecc_payload,
        "nsym": nsym_serial,
    }

    envelope = {
        "version": 1,
        "metadata": {
            "msg_id": encode_result.metadata.msg_id,
            "total": encode_result.metadata.total,
            "cfg": cfg_payload,
        },
        "chunks": chunk_entries,
    }

    return _serialise_envelope(envelope)


def decode_text(
    token_stream: Iterable[int],
    lm: Any,
    *,
    quality: Optional[Mapping[str, object]] = None,
    seed_text: str = "",
    chunk_bytes: int | None = None,
    use_crc: bool | None = None,
    ecc: str | None = None,
    nsym: int | None = None,
) -> bytes:
    """Decode ``token_stream`` produced by :func:`encode_text`."""

    if not isinstance(token_stream, Iterable):
        raise TypeError("token_stream must be an iterable of integers")

    tokens = [int(token) for token in token_stream]
    envelope = _deserialise_envelope(tokens)

    version = envelope.get("version")
    if version != 1:
        raise ConfigurationError(f"unsupported payload version: {version}")

    metadata = envelope.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("encoded payload missing metadata")

    msg_id = metadata.get("msg_id")
    total = metadata.get("total")
    if not isinstance(msg_id, str) or not isinstance(total, int):
        raise ConfigurationError("metadata must include msg_id and total")

    cfg = metadata.get("cfg")
    if not isinstance(cfg, Mapping):
        raise ConfigurationError("metadata missing cfg")

    encoded_chunk_bytes = int(cfg.get("chunk_bytes", 256))
    if chunk_bytes is not None and int(chunk_bytes) != encoded_chunk_bytes:
        raise ConfigurationError("chunk_bytes override does not match encoded payload")

    use_crc_flag = bool(use_crc) if use_crc is not None else bool(cfg.get("crc", False))
    stored_ecc_value = cfg.get("ecc", "none")
    if ecc is not None:
        ecc_mode = _normalise_ecc(ecc)
    else:
        ecc_mode = _normalise_ecc(stored_ecc_value if isinstance(stored_ecc_value, str) else "none")
    if ecc_mode == "rs" and _RSCodec is None:
        ecc_mode = "none"
    nsym_value = int(nsym) if nsym is not None else int(cfg.get("nsym", 0))
    if ecc_mode != "rs":
        nsym_value = 0

    chunks = envelope.get("chunks")
    if not isinstance(chunks, Sequence):
        raise ConfigurationError("encoded payload must include chunk entries")

    spans_map: Dict[int, List[int]] = {}
    state_map: Dict[int, CodecState | None] = {}
    auto_seq = 0
    for entry in chunks:
        if isinstance(entry, Mapping):
            if "seq" not in entry:
                raise ConfigurationError("chunk entry missing seq")
            seq_value = entry["seq"]
            try:
                seq = int(seq_value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("chunk seq must be an integer") from exc
            entry_msg_id = entry.get("msg_id")
            if entry_msg_id is not None and entry_msg_id != msg_id:
                raise ConfigurationError("chunk msg_id does not match payload metadata")
            tokens_raw = entry.get("tokens")
            if not isinstance(tokens_raw, Sequence):
                raise ConfigurationError("chunk tokens must be a sequence")
            span = [int(token) for token in tokens_raw]
            spans_map[seq] = span
            state_obj = entry.get("state")
            if state_obj is not None:
                if not isinstance(state_obj, Mapping):
                    raise ConfigurationError("chunk state must be a mapping")
                state_map[seq] = _deserialise_state_payload(state_obj)
            else:
                state_map[seq] = None
        elif isinstance(entry, Sequence):
            seq = auto_seq
            auto_seq += 1
            spans_map[seq] = [int(token) for token in entry]
            state_map[seq] = None
        else:
            raise ConfigurationError("chunk entry must be a mapping or sequence")

    if len(spans_map) != total:
        raise ConfigurationError("chunk count does not match metadata total")

    spans: List[List[int]] = []
    state_payloads: List[CodecState | None] = []
    for seq in range(total):
        if seq not in spans_map:
            raise ConfigurationError(f"missing chunk sequence {seq}")
        spans.append(spans_map[seq])
        state_payloads.append(state_map.get(seq))

    lm_adapter = _ensure_lm(lm)
    quality_args = _normalise_quality_dict(quality)

    state_loader = getattr(lm_adapter, "load_states", None)
    if callable(state_loader):
        if any(state is None for state in state_payloads):
            raise ConfigurationError("codec language model state missing from payload")
        state_loader([cast(CodecState, state) for state in state_payloads])

    result = stego_decode(
        spans,
        use_crc=use_crc_flag,
        ecc=ecc_mode,
        nsym=nsym_value,
        quality=quality_args,
        seed_text=seed_text,
        lm=lm_adapter,
    )

    return result


# Provide aliases compatible with previous module layout
chunk_bytes_func = chunk_bytes

__all__ = [
    "EncodeMetadata",
    "EncodeResult",
    "LMProvider",
    "chunk_bytes_func",
    "cover_generate",
    "cover_reveal",
    "decode_text",
    "encode_text",
    "stego_decode",
    "stego_encode",
]
