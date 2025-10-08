"""Command line interface for NeuralSteganography."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, MutableMapping, Sequence

from .api import decode_text as codec_decode_text, encode_text as codec_encode_text
from .codec.distribution import MockLM
from .crypto.api import decrypt_message, encrypt_message
from .crypto.envelope import EnvelopeError, unpack_envelope
from .crypto.errors import DecryptionError, EncryptionError
from .crypto.gpt2fa import decode_text as gpt2fa_decode_text, encode_text as gpt2fa_encode_text

# Backwards compatibility for tests monkeypatching these helpers.
encode_text = gpt2fa_encode_text
decode_text = gpt2fa_decode_text


class CLIError(RuntimeError):
    """Domain specific exception raised for CLI errors."""


def _convert_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_quality(pairs: Iterable[Sequence[str]] | None) -> Mapping[str, object]:
    quality: MutableMapping[str, object] = {}
    if not pairs:
        return quality
    for pair in pairs:
        if len(pair) != 2:
            raise CLIError("Each --quality requires a NAME and VALUE pair.")
        name, raw_value = pair
        if not name:
            raise CLIError("Quality name must not be empty.")
        if name in quality:
            raise CLIError(f"Quality parameter '{name}' specified multiple times.")
        quality[name] = _convert_scalar(raw_value)
    return quality


def _normalise_quality_flags(arguments: Sequence[str]) -> list[str]:
    """Expand ``--quality.NAME value`` flags into ``--quality NAME value`` pairs."""

    normalised: list[str] = []
    for token in arguments:
        if token.startswith("--quality."):
            option, eq, value = token.partition("=")
            key = option[len("--quality.") :]
            normalised.append("--quality")
            normalised.append(key)
            if eq:
                normalised.append(value)
        else:
            normalised.append(token)
    return normalised


def _read_text(path: str | Path) -> str:
    data: bytes
    if str(path) == "-":
        data = sys.stdin.buffer.read()
    else:
        try:
            data = Path(path).read_bytes()
        except FileNotFoundError as exc:
            raise CLIError(f"Input file '{path}' was not found.") from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CLIError("Input data is not valid UTF-8 text.") from exc


def _read_bytes(path: str | Path) -> bytes:
    if str(path) == "-":
        return sys.stdin.buffer.read()
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise CLIError(f"Input file '{path}' was not found.") from exc
    except OSError as exc:
        raise CLIError(f"Failed to read input '{path}': {exc}.") from exc


def _write_text(path: str | Path, message: str) -> None:
    encoded = message.encode("utf-8")
    if str(path) == "-":
        sys.stdout.buffer.write(encoded)
        if not message.endswith("\n"):
            sys.stdout.buffer.write(b"\n")
        sys.stdout.flush()
        return
    Path(path).write_bytes(encoded)


def _write_bytes(path: str | Path, data: bytes) -> None:
    if str(path) == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
        return
    Path(path).write_bytes(data)


def _load_tokens(path: str | Path) -> MutableMapping[str, object]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CLIError(f"Tokens file '{path}' was not found.") from exc
    except OSError as exc:
        raise CLIError(f"Failed to read tokens file '{path}': {exc}.") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CLIError(f"Tokens file '{path}' is not valid JSON: {exc}.") from exc
    if not isinstance(data, MutableMapping):
        raise CLIError("Tokens file must contain a JSON object.")
    return data


def _dump_tokens(path: str | Path, payload: Mapping[str, object]) -> None:
    formatted = json.dumps(payload, ensure_ascii=False, indent=2)
    Path(path).write_text(formatted + "\n", encoding="utf-8")


def _handle_encode(args: argparse.Namespace) -> int:
    quality = _parse_quality(args.quality)
    message = _read_text(args.input)
    try:
        payload = encode_text(message, args.password, quality=quality)
    except Exception as exc:  # pragma: no cover - defensive barrier
        raise CLIError(f"Failed to encode message: {exc}") from exc
    _dump_tokens(args.output, payload)
    return 0


def _handle_decode(args: argparse.Namespace) -> int:
    payload = _load_tokens(args.input)
    try:
        message = decode_text(payload, args.password)
    except Exception as exc:  # pragma: no cover - defensive barrier
        raise CLIError(f"Failed to decode tokens: {exc}") from exc
    _write_text(args.output, message)
    return 0


def _handle_codec_encode(args: argparse.Namespace) -> int:
    quality = _parse_quality(args.quality)
    message_bits = _read_bytes(args.input)
    lm = MockLM()
    try:
        tokens = codec_encode_text(message_bits, lm, quality=dict(quality), seed_text=args.seed)
    except Exception as exc:  # pragma: no cover - defensive barrier
        raise CLIError(f"Failed to encode payload: {exc}") from exc

    payload: dict[str, object] = {
        "tokens": list(tokens),
        "quality": dict(quality),
    }
    if args.seed:
        payload["seed_text"] = args.seed
    _dump_tokens(args.output, payload)
    return 0


def _handle_codec_decode(args: argparse.Namespace) -> int:
    payload = _load_tokens(args.input)
    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, Sequence):
        raise CLIError("Token payload must contain a 'tokens' sequence.")

    if args.seed:
        seed_text = args.seed
    else:
        seed_field = payload.get("seed_text", "")
        if not isinstance(seed_field, str):
            raise CLIError("Seed text in payload must be a string if present.")
        seed_text = seed_field

    if args.quality:
        quality = dict(_parse_quality(args.quality))
    else:
        stored_quality = payload.get("quality", {})
        if stored_quality is None:
            quality = {}
        elif isinstance(stored_quality, Mapping):
            quality = dict(stored_quality)
        else:
            raise CLIError("Quality metadata in payload must be a mapping.")

    tokens = [int(token) for token in raw_tokens]
    lm = MockLM()
    try:
        message = codec_decode_text(tokens, lm, quality=quality, seed_text=seed_text)
    except Exception as exc:  # pragma: no cover - defensive barrier
        raise CLIError(f"Failed to decode payload: {exc}") from exc
    _write_bytes(args.output, message)
    return 0


def _handle_encrypt(args: argparse.Namespace) -> int:
    plaintext = _read_bytes(args.input)
    aad = args.aad.encode("utf-8") if args.aad else b""
    try:
        envelope = encrypt_message(plaintext, args.password, aad=aad)
    except EncryptionError as exc:
        raise CLIError(f"Encryption failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive barrier
        raise CLIError("Encryption failed due to an unexpected error.") from exc
    _write_bytes(args.output, envelope)
    return 0


def _handle_decrypt(args: argparse.Namespace) -> int:
    blob = _read_bytes(args.input)
    if args.aad is not None:
        expected_aad = args.aad.encode("utf-8")
        try:
            _, _, _, _, stored_aad, _ = unpack_envelope(blob)
        except EnvelopeError as exc:
            raise CLIError(f"Failed to inspect envelope: {exc}") from exc
        actual_aad = stored_aad or b""
        if actual_aad != expected_aad:
            raise CLIError("Provided AAD does not match the envelope metadata.")
    try:
        plaintext = decrypt_message(blob, args.password)
    except DecryptionError as exc:
        raise CLIError(f"Decryption failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive barrier
        raise CLIError("Decryption failed due to an unexpected error.") from exc
    _write_bytes(args.output, plaintext)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralstego",
        description="Command line utilities for NeuralSteganography.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode a plaintext message into GPT2-fa tokens.",
    )
    encode_parser.add_argument(
        "-p",
        "--password",
        required=True,
        help="Password used to seed the encoder/decoder.",
    )
    encode_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to a UTF-8 text file containing the message (use '-' for stdin).",
    )
    encode_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Where to write the resulting JSON tokens.",
    )
    encode_parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        default=[],
        help="Quality parameters forwarded to the GPT2-fa encoder (repeatable).",
    )
    encode_parser.set_defaults(handler=_handle_encode)

    decode_parser = subparsers.add_parser(
        "decode",
        help="Decode GPT2-fa tokens back into the original message.",
    )
    decode_parser.add_argument(
        "-p",
        "--password",
        required=True,
        help="Password that was used during encoding.",
    )
    decode_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the JSON tokens file produced by the encoder.",
    )
    decode_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Where to write the recovered UTF-8 message (use '-' for stdout).",
    )
    decode_parser.set_defaults(handler=_handle_decode)

    codec_encode_parser = subparsers.add_parser(
        "codec-encode",
        help="Encode raw bytes using the arithmetic codec mock.",
    )
    codec_encode_parser.add_argument(
        "--in",
        dest="input",
        required=True,
        help="Path to the binary payload to embed (use '-' for stdin).",
    )
    codec_encode_parser.add_argument(
        "--out",
        dest="output",
        required=True,
        help="Where to write the resulting token JSON (use '-' for stdout).",
    )
    codec_encode_parser.add_argument(
        "--seed",
        default="",
        help="Optional seed text used to warm up the language model.",
    )
    codec_encode_parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        default=[],
        help="Quality hints forwarded to the arithmetic codec (repeatable).",
    )
    codec_encode_parser.set_defaults(handler=_handle_codec_encode)

    codec_decode_parser = subparsers.add_parser(
        "codec-decode",
        help="Decode tokens produced by codec-encode back into the original bytes.",
    )
    codec_decode_parser.add_argument(
        "--in",
        dest="input",
        required=True,
        help="Path to the JSON tokens file (use '-' for stdin).",
    )
    codec_decode_parser.add_argument(
        "--out",
        dest="output",
        required=True,
        help="Where to write the recovered payload (use '-' for stdout).",
    )
    codec_decode_parser.add_argument(
        "--seed",
        default="",
        help="Seed text to use during decoding (overrides payload metadata if supplied).",
    )
    codec_decode_parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        default=[],
        help="Quality overrides for decoding (repeatable).",
    )
    codec_decode_parser.set_defaults(handler=_handle_codec_decode)

    encrypt_parser = subparsers.add_parser(
        "encrypt",
        help="Encrypt a message into a password-protected envelope.",
    )
    encrypt_parser.add_argument(
        "-p",
        "--password",
        required=True,
        help="Password used to derive the encryption key.",
    )
    encrypt_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the plaintext input (use '-' for stdin).",
    )
    encrypt_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Where to write the encrypted envelope (use '-' for stdout).",
    )
    encrypt_parser.add_argument(
        "--aad",
        help="Additional authenticated data (treated as UTF-8).",
    )
    encrypt_parser.set_defaults(handler=_handle_encrypt)

    decrypt_parser = subparsers.add_parser(
        "decrypt",
        help="Decrypt an envelope produced by the encrypt command.",
    )
    decrypt_parser.add_argument(
        "-p",
        "--password",
        required=True,
        help="Password that was used during encryption.",
    )
    decrypt_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the encrypted envelope (use '-' for stdin).",
    )
    decrypt_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Where to write the recovered plaintext (use '-' for stdout).",
    )
    decrypt_parser.add_argument(
        "--aad",
        help="Expected additional authenticated data for validation (UTF-8).",
    )
    decrypt_parser.set_defaults(handler=_handle_decrypt)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    normalised_args = _normalise_quality_flags(raw_args)
    parser = _build_parser()
    args = parser.parse_args(normalised_args)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
