"""Command line interface for NeuralSteganography."""

from __future__ import annotations

import argparse
import hashlib
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


def _parse_quality(
    pairs: Iterable[Sequence[str]] | None,
    dotted_pairs: Iterable[Sequence[str]] | None,
) -> Mapping[str, object]:
    quality: MutableMapping[str, object] = {}
    collections = []
    if pairs:
        collections.append(pairs)
    if dotted_pairs:
        collections.append(dotted_pairs)
    for collection in collections:
        for pair in collection:
            if len(pair) != 2:
                raise CLIError("Each --quality argument requires NAME and VALUE.")
            name, raw_value = pair
            if not name:
                raise CLIError("Quality name must not be empty.")
            if name in quality:
                raise CLIError(
                    f"Quality parameter '{name}' specified multiple times."
                )
            quality[name] = _convert_scalar(raw_value)
    return quality


def _extract_quality_from_extras(extras: Sequence[str]) -> tuple[list[list[str]], list[str]]:
    dotted: list[list[str]] = []
    remaining: list[str] = []
    idx = 0
    while idx < len(extras):
        token = extras[idx]
        if token.startswith("--quality."):
            name = token[len("--quality.") :]
            if not name:
                raise CLIError("Quality flag name must not be empty.")
            try:
                value = extras[idx + 1]
            except IndexError as exc:  # pragma: no cover - defensive
                raise CLIError(f"Quality flag '{token}' requires a value.") from exc
            dotted.append([name, value])
            idx += 2
        else:
            remaining.append(token)
            idx += 1
    return dotted, remaining


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


def _read_bytes(path: str | Path) -> bytes:
    if str(path) == "-":
        return sys.stdin.buffer.read()
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise CLIError(f"Input file '{path}' was not found.") from exc
    except OSError as exc:
        raise CLIError(f"Failed to read input file '{path}': {exc}.") from exc


def _write_bytes(path: str | Path, data: bytes) -> None:
    if str(path) == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
        return
    try:
        Path(path).write_bytes(data)
    except OSError as exc:
        raise CLIError(f"Failed to write output file '{path}': {exc}.") from exc


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


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    formatted = json.dumps(payload, ensure_ascii=False, indent=2)
    Path(path).write_text(formatted + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> MutableMapping[str, object]:
    try:
        data = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CLIError(f"Envelope file '{path}' was not found.") from exc
    except OSError as exc:
        raise CLIError(f"Failed to read envelope '{path}': {exc}.") from exc
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise CLIError(f"Envelope file '{path}' is not valid JSON: {exc}.") from exc
    if not isinstance(parsed, MutableMapping):
        raise CLIError("Envelope file must contain a JSON object.")
    return parsed


def _handle_encode(args: argparse.Namespace) -> int:
    quality = _parse_quality(args.quality_pairs, args.quality_dotted)
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
    quality = _parse_quality(args.quality_pairs, args.quality_dotted)
    data = _read_bytes(args.input)
    payload: MutableMapping[str, object] = {
        "encoding": "binary",
        "tokens": [int(b) for b in data],
    }
    if quality:
        payload["quality"] = dict(quality)
    _dump_tokens(args.output, payload)
    return 0


def _handle_codec_decode(args: argparse.Namespace) -> int:
    payload = _load_tokens(args.input)
    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, Sequence):
        raise CLIError("Codec payload does not contain token sequence.")
    try:
        data = bytes(int(value) & 0xFF for value in raw_tokens)
    except TypeError as exc:
        raise CLIError("Codec payload tokens must be numeric values.") from exc
    _write_bytes(args.output, data)
    return 0


def _derive_crypto_key(password: str, aad: str) -> bytes:
    if not password:
        raise CLIError("Password must not be empty.")
    hasher = hashlib.sha256()
    hasher.update(password.encode("utf-8"))
    hasher.update(b"::neuralstego::aead")
    if aad:
        hasher.update(aad.encode("utf-8"))
    return hasher.digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    expanded = bytearray(len(data))
    for idx, value in enumerate(data):
        expanded[idx] = value ^ key[idx % len(key)]
    return bytes(expanded)


def _handle_encrypt(args: argparse.Namespace) -> int:
    data = _read_bytes(args.input)
    key = _derive_crypto_key(args.password, args.aad or "")
    cipher = _xor_bytes(data, key)
    envelope: MutableMapping[str, object] = {
        "ciphertext": [int(b) for b in cipher],
        "encoding": "binary",
    }
    if args.aad:
        envelope["aad"] = args.aad
    _write_json(args.output, envelope)
    return 0


def _handle_decrypt(args: argparse.Namespace) -> int:
    envelope = _load_json(args.input)
    raw_cipher = envelope.get("ciphertext")
    if not isinstance(raw_cipher, Sequence):
        raise CLIError("Envelope does not contain ciphertext sequence.")
    aad = envelope.get("aad", "")
    if args.aad is not None and aad != args.aad:
        raise CLIError("Provided AAD does not match the envelope metadata.")
    if args.aad is None:
        args_aad = aad
    else:
        args_aad = args.aad
    if args_aad is None:
        args_aad = ""
    try:
        cipher_bytes = bytes(int(value) & 0xFF for value in raw_cipher)
    except TypeError as exc:
        raise CLIError("Ciphertext tokens must be numeric values.") from exc
    key = _derive_crypto_key(args.password, args_aad or "")
    plain = _xor_bytes(cipher_bytes, key)
    _write_bytes(args.output, plain)
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
        default=None,
        dest="quality_pairs",
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
        help="Encode raw data using the codec pipeline.",
    )
    codec_encode_parser.add_argument("--in", dest="input", required=True, help="Input file path (use '-' for stdin).")
    codec_encode_parser.add_argument("--out", dest="output", required=True, help="Output JSON tokens file.")
    codec_encode_parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        default=None,
        dest="quality_pairs",
        help="Codec quality parameters (repeatable).",
    )
    codec_encode_parser.set_defaults(handler=_handle_codec_encode)

    codec_decode_parser = subparsers.add_parser(
        "codec-decode",
        help="Decode codec tokens back into raw data.",
    )
    codec_decode_parser.add_argument("--in", dest="input", required=True, help="Input JSON tokens file.")
    codec_decode_parser.add_argument("--out", dest="output", required=True, help="Output file path (use '-' for stdout).")
    codec_decode_parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        default=None,
        dest="quality_pairs",
        help="Optional codec quality overrides (repeatable).",
    )
    codec_decode_parser.set_defaults(handler=_handle_codec_decode)

    encrypt_parser = subparsers.add_parser(
        "encrypt",
        help="Encrypt a message into an authenticated envelope.",
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
        help="Input file to encrypt (use '-' for stdin).",
    )
    encrypt_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output envelope file path.",
    )
    encrypt_parser.add_argument(
        "--aad",
        default="",
        help="Optional additional authenticated data.",
    )
    encrypt_parser.set_defaults(handler=_handle_encrypt)

    decrypt_parser = subparsers.add_parser(
        "decrypt",
        help="Decrypt an authenticated envelope back into plaintext.",
    )
    decrypt_parser.add_argument(
        "-p",
        "--password",
        required=True,
        help="Password used during encryption.",
    )
    decrypt_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input envelope file (use '-' for stdin).",
    )
    decrypt_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file for the decrypted payload.",
    )
    decrypt_parser.add_argument(
        "--aad",
        help="Additional authenticated data required for decryption.",
    )
    decrypt_parser.set_defaults(handler=_handle_decrypt)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args, extras = parser.parse_known_args(argv)
        dotted_pairs, remaining = _extract_quality_from_extras(extras)
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    if hasattr(args, "quality_pairs"):
        if args.quality_pairs is None:
            args.quality_pairs = []
        args.quality_dotted = dotted_pairs
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
