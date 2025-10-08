"""Command line interface for NeuralSteganography."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, MutableMapping, Sequence

from .crypto.gpt2fa import decode_text, encode_text


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


def _parse_quality(pairs: Iterable[Sequence[str]]) -> Mapping[str, object]:
    quality: MutableMapping[str, object] = {}
    for pair in pairs:
        if len(pair) != 2:
            raise CLIError("Each --quality requires a NAME and VALUE pair.")
        name, raw_value = pair
        if name in quality:
            raise CLIError(f"Quality parameter '{name}' specified multiple times.")
        quality[name] = _convert_scalar(raw_value)
    return quality


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


def _write_text(path: str | Path, message: str) -> None:
    encoded = message.encode("utf-8")
    if str(path) == "-":
        sys.stdout.buffer.write(encoded)
        if not message.endswith("\n"):
            sys.stdout.buffer.write(b"\n")
        sys.stdout.flush()
        return
    Path(path).write_bytes(encoded)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralstego",
        description="Encode and decode messages using the GPT2-fa steganography pipeline.",
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
