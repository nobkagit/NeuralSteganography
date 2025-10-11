"""Core arithmetic coding algorithms for neural steganography."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Mapping, Sequence, Tuple, cast

import math
import numpy as np
from numpy.typing import NDArray

from .errors import ArithmeticRangeError, DecodeDivergenceError
from .quality import apply_quality, cap_bits_per_token
from .types import CodecState, LMProvider, ProbDist

_FRACTION_LIMIT = 1 << 30
"""Maximum denominator when converting floating probabilities to fractions."""


@dataclass
class BitReader:
    """Sequential bit reader with zero-padding for exhausted input."""

    payload: bytes

    def __post_init__(self) -> None:
        self._bits: Tuple[int, ...] = _bytes_to_bits(self.payload)
        self._length = len(self._bits)
        self._position = 0
        self._padding_consumed = 0

    @property
    def total_bits(self) -> int:
        """Return the number of actual payload bits."""

        return self._length

    @property
    def consumed_bits(self) -> int:
        """Return the number of payload bits that have been consumed."""

        return self._position

    @property
    def padding_bits(self) -> int:
        """Return the number of zero bits consumed beyond the payload length."""

        return self._padding_consumed

    def exhausted(self) -> bool:
        """Return ``True`` when the payload bits have been fully consumed."""

        return self._position >= self._length

    def peek(self, count: int) -> Tuple[int, ...]:
        """Return the next ``count`` bits without consuming them."""

        if count < 0:
            raise ValueError("count must be non-negative")
        return tuple(self._bit_at(self._position + offset) for offset in range(count))

    def read(self, count: int) -> Tuple[int, ...]:
        """Consume and return ``count`` bits, padding with zeros if required."""

        bits = self.peek(count)
        if count == 0:
            return bits

        actual = min(count, max(self._length - self._position, 0))
        self._position += actual
        self._padding_consumed += count - actual
        return bits

    def _bit_at(self, index: int) -> int:
        if index < self._length:
            return self._bits[index]
        return 0


class BitWriter:
    """Helper accumulating bits and exposing them as bytes."""

    def __init__(self) -> None:
        self._bits: List[int] = []

    @property
    def bit_length(self) -> int:
        """Return the total number of bits written so far."""

        return len(self._bits)

    def write_bits(self, bits: Sequence[int]) -> None:
        """Append a sequence of bits to the internal buffer."""

        for bit in bits:
            if bit not in (0, 1):
                raise ValueError(f"Invalid bit value: {bit}")
            self._bits.append(bit)

    def to_bytes(self, *, bit_length: int | None = None) -> bytes:
        """Return the written bits as a ``bytes`` object (MSB first)."""

        if bit_length is None:
            bit_length = len(self._bits)
        if bit_length < 0:
            raise ValueError("bit_length must be non-negative")

        bits = self._bits[:bit_length]
        if not bits:
            return b""
        padding = (-len(bits)) % 8
        bits_padded = bits + [0] * padding
        output = bytearray()
        for idx in range(0, len(bits_padded), 8):
            byte = 0
            for bit in bits_padded[idx : idx + 8]:
                byte = (byte << 1) | bit
            output.append(byte)
        return bytes(output)


def encode_with_lm(
    bits: bytes,
    lm: LMProvider,
    *,
    context: Sequence[int] | None = None,
    quality: Mapping[str, object] | None = None,
    state: CodecState | None = None,
    max_context: int | None = None,
) -> list[int]:
    """Encode a payload using probability distributions from ``lm``."""

    reader = BitReader(bits)
    total_bits = reader.total_bits
    if total_bits == 0:
        if state is not None:
            state["history"] = tuple()
            state["residual_bits"] = (0).to_bytes(8, byteorder="big", signed=False)
        return []

    tokens: List[int] = []
    consumption: List[int] = []
    context_ids = list(context or [])
    context_window = _resolve_context_window(lm, max_context)

    while reader.consumed_bits < total_bits:
        dist = _next_distribution(lm, context_ids, quality, context_window)
        ranked_tokens, capacity = _rank_tokens(dist)
        if capacity <= 0:
            raise ArithmeticRangeError("Language model distribution provides no capacity")

        before = reader.consumed_bits
        bits_chunk = reader.read(capacity)
        consumed = min(capacity, reader.consumed_bits - before)
        index = _bits_to_int(bits_chunk)
        limit = 1 << capacity
        if index >= limit:
            index = limit - 1

        token_id = ranked_tokens[index]
        tokens.append(token_id)
        consumption.append(consumed)
        context_ids.append(token_id)

    if state is not None:
        state["history"] = tuple(consumption)
        state["residual_bits"] = total_bits.to_bytes(8, byteorder="big", signed=False)

    return tokens


def decode_with_lm(
    tokens: Sequence[int],
    lm: LMProvider,
    *,
    context: Sequence[int] | None = None,
    quality: Mapping[str, object] | None = None,
    state: CodecState | None = None,
    max_context: int | None = None,
) -> bytes:
    """Decode tokens produced by :func:`encode_with_lm`."""

    if not tokens:
        return b""

    history: Sequence[int] | None = None
    total_bits: int | None = None
    if state is not None:
        history = cast(Sequence[int] | None, state.get("history"))
        residual = state.get("residual_bits")
        if residual:
            total_bits = int.from_bytes(residual, byteorder="big", signed=False)

    if history is None or len(history) < len(tokens):
        raise DecodeDivergenceError("Bit consumption history is required for decoding")

    consumption = list(history[: len(tokens)])
    remaining_history = list(history[len(tokens) :])

    writer = BitWriter()
    context_ids = list(context or [])
    context_window = _resolve_context_window(lm, max_context)

    for index, token_id in enumerate(tokens):
        dist = _next_distribution(lm, context_ids, quality, context_window)
        ranked_tokens, capacity = _rank_tokens(dist)
        if capacity <= 0:
            raise DecodeDivergenceError("Language model distribution provides no capacity")

        try:
            token_index = ranked_tokens.index(token_id)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise DecodeDivergenceError(f"Token {token_id} not present in distribution") from exc

        emitted = _int_to_bits(token_index, capacity)
        writer.write_bits(emitted[: consumption[index]])
        context_ids.append(token_id)

    if state is not None:
        if remaining_history:
            state["history"] = tuple(remaining_history)
        else:
            state.pop("history", None)

    if total_bits is None:
        total_bits = writer.bit_length

    if total_bits > writer.bit_length:
        raise DecodeDivergenceError("Decoded bitstream shorter than expected")

    return writer.to_bytes(bit_length=total_bits)


def encode_bits(
    bits: bytes,
    probs: Iterable[ProbDist],
    *,
    state: CodecState | None = None,
) -> Sequence[int]:
    """Encode a bitstream into a sequence of token identifiers."""

    reader = BitReader(bits)
    total_bits = reader.total_bits
    if total_bits == 0:
        if state is not None:
            state["history"] = tuple()
            state["residual_bits"] = (0).to_bytes(8, byteorder="big", signed=False)
        return []

    low = Fraction(0, 1)
    high = Fraction(1, 1)
    tokens: List[int] = []
    consumption: List[int] = []

    prob_iter = iter(probs)
    while reader.consumed_bits < total_bits:
        try:
            dist = next(prob_iter)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise ArithmeticRangeError("Insufficient probability distributions for encoding") from exc

        token_id, consumed, low, high = _encode_step(reader, dist, low, high)
        tokens.append(token_id)
        consumption.append(consumed)

    if state is not None:
        state["history"] = tuple(consumption)
        state["residual_bits"] = total_bits.to_bytes(8, byteorder="big", signed=False)

    return tokens


def decode_bits(
    tokens: Sequence[int],
    probs: Iterable[ProbDist],
    *,
    state: CodecState | None = None,
) -> bytes:
    """Decode a sequence of token identifiers back into the embedded bitstream."""

    if not tokens:
        return b""

    history: Sequence[int] | None = None
    total_bits: int | None = None
    if state is not None:
        history = cast(Sequence[int] | None, state.get("history"))
        residual = state.get("residual_bits")
        if residual:
            total_bits = int.from_bytes(residual, byteorder="big", signed=False)

    if history is None or len(history) < len(tokens):
        raise DecodeDivergenceError("Bit consumption history is required for decoding")

    consumption = list(history[: len(tokens)])
    remaining_history = list(history[len(tokens) :])

    low = Fraction(0, 1)
    high = Fraction(1, 1)
    writer = BitWriter()

    prob_iter = iter(probs)
    for index, token_id in enumerate(tokens):
        try:
            dist = next(prob_iter)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise ArithmeticRangeError("Insufficient probability distributions for decoding") from exc

        consumed = consumption[index]
        low, high, emitted = _decode_step(token_id, dist, low, high, consumed)
        writer.write_bits(emitted)

    if state is not None:
        if remaining_history:
            state["history"] = tuple(remaining_history)
        else:
            state.pop("history", None)

    if total_bits is None:
        total_bits = writer.bit_length

    if total_bits > writer.bit_length:
        raise DecodeDivergenceError("Decoded bitstream shorter than expected")

    return writer.to_bytes(bit_length=total_bits)


def _resolve_context_window(lm: LMProvider, override: int | None) -> int | None:
    if override is not None and override > 0:
        return override
    window = getattr(lm, "context_window", None)
    if isinstance(window, int) and window > 0:
        return window
    return None


def _next_distribution(
    lm: LMProvider,
    context_ids: List[int],
    quality: Mapping[str, object] | None,
    context_window: int | None,
) -> ProbDist:
    if context_window is not None and len(context_ids) > context_window:
        trimmed = context_ids[-context_window:]
    else:
        trimmed = context_ids
    dist = lm.next_token_probs(tuple(trimmed))
    return _apply_quality(dist, quality)


def _apply_quality(dist: ProbDist, quality: Mapping[str, object] | None) -> ProbDist:
    if not quality:
        return dist

    filtered = dist
    if isinstance(quality, Mapping):
        top_k = _coerce_optional_int(quality.get("top_k"), "top_k")
        top_p = _coerce_optional_float(quality.get("top_p"), "top_p")
        min_prob = _coerce_optional_float(quality.get("min_prob"), "min_prob")
        if any(param is not None for param in (top_k, top_p, min_prob)):
            filtered = apply_quality(filtered, top_k=top_k, top_p=top_p, min_prob=min_prob)

        cap_bits_raw = quality.get("cap_per_token_bits")
        cap_bits = _coerce_optional_int(cap_bits_raw, "cap_per_token_bits")
        if cap_bits is not None:
            filtered = cap_bits_per_token(filtered, cap_bits)

    return filtered


def _coerce_optional_int(value: object | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer when provided")
    return value


def _coerce_optional_float(value: object | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number when provided")
    return float(value)


def _rank_tokens(dist: ProbDist) -> Tuple[List[int], int]:
    tokens, probs = _dist_to_arrays(dist)
    mask = probs > 0
    tokens = tokens[mask]
    probs = probs[mask]
    if tokens.size == 0:
        raise ArithmeticRangeError("Probability distribution must contain positive mass")

    order = np.argsort(probs)[::-1]
    sorted_tokens = tokens[order]
    capacity = int(math.floor(math.log2(sorted_tokens.size)))
    if capacity <= 0:
        return sorted_tokens.tolist(), 0

    limit = 1 << capacity
    return sorted_tokens[:limit].tolist(), capacity


def _dist_to_arrays(dist: ProbDist) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    if isinstance(dist, np.ndarray):
        tokens = np.arange(dist.size, dtype=np.int64)
        probs = np.asarray(dist, dtype=np.float64)
        return tokens, probs
    if isinstance(dist, dict):
        items = sorted(dist.items())
        tokens = np.array([int(token) for token, _ in items], dtype=np.int64)
        probs = np.array([float(prob) for _, prob in items], dtype=np.float64)
        return tokens, probs
    raise TypeError(f"Unsupported probability distribution type: {type(dist)!r}")


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _encode_step(
    reader: BitReader,
    dist: ProbDist,
    low: Fraction,
    high: Fraction,
) -> Tuple[int, int, Fraction, Fraction]:
    """Encode a single token and return the updated interval."""

    cdf = _cumulative_distribution(dist)
    if not cdf:
        raise ArithmeticRangeError("Distribution contains no probability mass")

    interval_width = high - low
    depth = 1
    while True:
        prefix_bits = reader.peek(depth)
        prefix_low, prefix_high = _prefix_interval(prefix_bits)

        for token_id, cum_low, cum_high in cdf:
            token_low = low + interval_width * cum_low
            token_high = low + interval_width * cum_high
            if token_low <= prefix_low and prefix_high <= token_high:
                reader.read(depth)
                return token_id, depth, token_low, token_high

        depth += 1
        if depth > reader.total_bits + 64:
            raise ArithmeticRangeError("Unable to resolve token interval with available bits")


def _decode_step(
    token_id: int,
    dist: ProbDist,
    low: Fraction,
    high: Fraction,
    bits_consumed: int,
) -> Tuple[Fraction, Fraction, Tuple[int, ...]]:
    """Decode a single token and return emitted bits with the new interval."""

    cdf = _cumulative_distribution(dist)
    interval_width = high - low
    target_interval: Tuple[Fraction, Fraction] | None = None
    for candidate_id, cum_low, cum_high in cdf:
        if candidate_id != token_id:
            continue
        token_low = low + interval_width * cum_low
        token_high = low + interval_width * cum_high
        if token_high <= token_low:
            raise ArithmeticRangeError("Degenerate token interval encountered during decoding")
        target_interval = (token_low, token_high)
        break

    if target_interval is None:
        raise DecodeDivergenceError(f"Token {token_id} not present in distribution")

    token_low, token_high = target_interval
    bits: Tuple[int, ...] = ()
    if bits_consumed > 0:
        bits = _prefix_from_interval(token_low, token_high, bits_consumed)

    return token_low, token_high, bits


def _cumulative_distribution(dist: ProbDist) -> List[Tuple[int, Fraction, Fraction]]:
    tokens, probs = _dist_to_sequences(dist)
    total = sum(probs)
    if total <= 0:
        raise ArithmeticRangeError("Probability distribution must have positive mass")

    norm_probs = [prob / total for prob in probs if prob > 0]
    norm_tokens = [token for token, prob in zip(tokens, probs) if prob > 0]

    cumulative = Fraction(0, 1)
    result: List[Tuple[int, Fraction, Fraction]] = []
    for token, prob in zip(norm_tokens, norm_probs):
        next_cumulative = cumulative + prob
        result.append((token, cumulative, next_cumulative))
        cumulative = next_cumulative

    return result


def _dist_to_sequences(dist: ProbDist) -> Tuple[List[int], List[Fraction]]:
    if isinstance(dist, np.ndarray):
        tokens = list(range(dist.size))
        probs = [_to_fraction(float(prob)) for prob in dist.tolist()]
        return tokens, probs
    if isinstance(dist, dict):
        items = sorted(dist.items())
        tokens = [token for token, _ in items]
        probs = [_to_fraction(float(prob)) for _, prob in items]
        return tokens, probs
    raise TypeError(f"Unsupported probability distribution type: {type(dist)!r}")


def _prefix_interval(bits: Sequence[int]) -> Tuple[Fraction, Fraction]:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    length = len(bits)
    if length == 0:
        return Fraction(0, 1), Fraction(1, 1)
    denominator = 1 << length
    low = Fraction(value, denominator)
    high = Fraction(value + 1, denominator)
    return low, high


def _prefix_from_interval(low: Fraction, high: Fraction, length: int) -> Tuple[int, ...]:
    scale = 1 << length
    min_index = _ceil_fraction(low * scale)
    high_scaled = high * scale
    max_index = (high_scaled.numerator - 1) // high_scaled.denominator
    candidate = min_index
    if candidate > max_index:
        candidate = max_index

    prefix_low = Fraction(candidate, scale)
    prefix_high = Fraction(candidate + 1, scale)
    if prefix_low < low or prefix_high > high:
        candidate = max_index
        prefix_low = Fraction(candidate, scale)
        prefix_high = Fraction(candidate + 1, scale)
        if prefix_low < low or prefix_high > high:
            raise DecodeDivergenceError("No binary prefix fits within the interval")

    return _int_to_bits(candidate, length)


def _int_to_bits(value: int, length: int) -> Tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in reversed(range(length)))


def _ceil_fraction(frac: Fraction) -> int:
    num, den = frac.numerator, frac.denominator
    return -(-num // den)


def _to_fraction(value: float) -> Fraction:
    if value < 0.0:
        raise ArithmeticRangeError("Probabilities must be non-negative")
    frac = Fraction.from_float(value).limit_denominator(_FRACTION_LIMIT)
    return frac


def _bytes_to_bits(payload: bytes) -> Tuple[int, ...]:
    bits: List[int] = []
    for byte in payload:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return tuple(bits)


__all__ = [
    "BitReader",
    "BitWriter",
    "encode_bits",
    "decode_bits",
    "encode_with_lm",
    "decode_with_lm",
]
