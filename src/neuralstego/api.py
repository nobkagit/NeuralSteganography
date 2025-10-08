"""High level neural steganography API for chunked framing."""
from __future__ import annotations

import base64
import binascii
import json
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

from .codec import assemble_bytes, build_packet, chunk_bytes, make_msg_id, parse_packet
from .codec.arithmetic import decode_with_lm as codec_decode_with_lm
from .codec.arithmetic import encode_with_lm as codec_encode_with_lm
from .codec.types import LMProvider as CodecLMBase
from .exceptions import ConfigurationError, MissingChunksError


class LMProvider(Protocol):
    """Protocol implemented by language model wrappers used for arithmetic coding."""

    def encode_arithmetic(
        self, bits: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        ...

    def decode_arithmetic(
        self, tokens: List[int], context: List[int], *, quality: Dict[str, float]
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


_DEFAULT_QUALITY = {
    "temp": 1.0,
    "precision": 16,
    "topk": 50000,
    "finish_sent": True,
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


CodecState = Dict[str, Any]


def _codec_quality_arguments(
    quality: Mapping[str, object] | None,
) -> Tuple[Dict[str, Any], Optional[int]]:
    if not quality:
        return {}, None

    policies: Dict[str, Any] = {}
    max_context: Optional[int] = None

    if "top_k" in quality and quality["top_k"] is not None:
        policies["top_k"] = int(quality["top_k"])
    if "topk" in quality and "top_k" not in policies and quality["topk"] is not None:
        policies["top_k"] = int(quality["topk"])

    if "top_p" in quality and quality["top_p"] is not None:
        policies["top_p"] = float(quality["top_p"])

    if "min_prob" in quality and quality["min_prob"] is not None:
        policies["min_prob"] = float(quality["min_prob"])

    if "cap_per_token_bits" in quality and quality["cap_per_token_bits"] is not None:
        policies["cap_per_token_bits"] = int(quality["cap_per_token_bits"])
    if (
        "cap_bits_per_token" in quality
        and policies.get("cap_per_token_bits") is None
        and quality["cap_bits_per_token"] is not None
    ):
        policies["cap_per_token_bits"] = int(quality["cap_bits_per_token"])

    if "max_context" in quality and quality["max_context"] is not None:
        max_context = int(quality["max_context"])

    filtered = {key: value for key, value in policies.items() if value is not None}
    return filtered, max_context


def _extract_codec_state(state: MutableMapping[str, object]) -> CodecState:
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
        self, bits: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        payload = _bits_to_bytes(bits)
        quality_args, max_context = _codec_quality_arguments(quality)
        state: MutableMapping[str, object] = {}
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
        self, tokens: List[int], context: List[int], *, quality: Dict[str, float]
    ) -> List[int]:
        if not self._decode_states:
            raise ConfigurationError("decode state unavailable for codec language model")

        state_payload = self._decode_states.popleft()
        decode_state: MutableMapping[str, object] = {
            "history": tuple(int(value) for value in state_payload.get("history", ())),
            "residual_bits": state_payload.get("residual_bits", b""),
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


def stego_encode(
    message: bytes,
    *,
    chunk_bytes: int = 256,
    use_crc: bool = True,
    ecc: str | None = "rs",
    nsym: int = 10,
    quality: Optional[Dict[str, float]] = None,
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
    quality_args = {**_DEFAULT_QUALITY, **(quality or {})}

    chunks = chunk_bytes_func(message, chunk_size=cfg["chunk_bytes"])
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
    quality: Optional[Dict[str, float]] = None,
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
    quality_args = {**_DEFAULT_QUALITY, **(quality or {})}

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
    quality_args = dict(quality or {})
    lm_adapter = _ensure_lm(lm)

    encode_result = stego_encode(
        payload,
        chunk_bytes=chunk_bytes,
        use_crc=use_crc,
        ecc=ecc,
        nsym=nsym,
        quality=quality_args,
        seed_text=seed_text,
        lm=lm_adapter,
    )

    spans = [list(map(int, span)) for span in encode_result]
    chunk_entries: List[Dict[str, Any]] = []

    adapter_states: List[CodecState] = []
    if isinstance(lm_adapter, _CodecLMAdapter):
        adapter_states = lm_adapter.drain_states()
        if len(adapter_states) != len(spans):
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
    cfg_ecc_raw = cfg.get("ecc", _normalise_ecc(ecc))
    ecc_payload = (
        _normalise_ecc(cfg_ecc_raw) if isinstance(cfg_ecc_raw, str) or cfg_ecc_raw is None else str(cfg_ecc_raw)
    )
    cfg_payload = {
        "chunk_bytes": int(cfg.get("chunk_bytes", chunk_bytes)),
        "crc": bool(cfg.get("crc", use_crc)),
        "ecc": ecc_payload,
        "nsym": int(cfg.get("nsym", nsym)),
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
    ecc_mode = (
        _normalise_ecc(ecc)
        if ecc is not None
        else _normalise_ecc(stored_ecc_value if isinstance(stored_ecc_value, str) else "none")
    )
    nsym_value = int(nsym) if nsym is not None else int(cfg.get("nsym", 0))

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
    quality_args = dict(quality or {})

    if isinstance(lm_adapter, _CodecLMAdapter):
        if any(state is None for state in state_payloads):
            raise ConfigurationError("codec language model state missing from payload")
        lm_adapter.load_states([cast(CodecState, state) for state in state_payloads])

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
    "decode_text",
    "encode_text",
    "stego_decode",
    "stego_encode",
]
