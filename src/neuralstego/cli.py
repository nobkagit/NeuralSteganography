"""Command line interface for the neural steganography toolkit."""

from __future__ import annotations

import argparse
import binascii
import base64
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from rich.console import Console
except ModuleNotFoundError:  # pragma: no cover - fallback when rich is unavailable
    class Console:  # type: ignore[override]
        def print(self, *objects: object, **_: object) -> None:
            text = " ".join(str(obj) for obj in objects)
            print(text)

from .api import (
    cover_generate as api_cover_generate,
    cover_reveal as api_cover_reveal,
    decode_text as codec_decode_text,
    encode_text as codec_encode_text,
)
from .codec.distribution import MockLM
from .codec.packet import RSCodec as _RSCodec  # type: ignore[attr-defined]
from .crypto.errors import DecryptionError, EncryptionError
from .exceptions import ConfigurationError, MissingChunksError, NeuralStegoError

console = Console()


def _get_crypto_api():
    try:
        from .crypto import api as crypto_api
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        return None, exc
    return crypto_api, None


def _fallback_warning(exc: ModuleNotFoundError) -> None:
    missing = exc.name or "cryptography"
    console.print(
        f"[yellow]Optional dependency '{missing}' is unavailable. Falling back to a lightweight implementation.[/yellow]"
    )


def _derive_fallback_key(password: str, extra: bytes = b"") -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode("utf-8"))
    hasher.update(extra)
    return hasher.digest()


def _fallback_encrypt_message(message: bytes, password: str, aad: bytes) -> bytes:
    key = _derive_fallback_key(password, aad)
    mask = key * ((len(message) // len(key)) + 1)
    ciphertext = bytes(m ^ b for m, b in zip(mask, message))
    envelope = {
        "version": "fallback-1",
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "aad": base64.b64encode(aad).decode("ascii") if aad else None,
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def _fallback_decrypt_message(blob: bytes, password: str) -> bytes:
    try:
        envelope = json.loads(blob.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError("invalid fallback envelope") from exc

    cipher_b64 = envelope.get("ciphertext")
    if not isinstance(cipher_b64, str):
        raise ConfigurationError("fallback envelope missing ciphertext")
    try:
        ciphertext = base64.b64decode(cipher_b64)
    except (ValueError, binascii.Error) as exc:
        raise ConfigurationError("invalid ciphertext encoding") from exc

    aad_b64 = envelope.get("aad")
    if isinstance(aad_b64, str):
        try:
            aad = base64.b64decode(aad_b64)
        except (ValueError, binascii.Error) as exc:
            raise ConfigurationError("invalid aad encoding") from exc
    else:
        aad = b""

    key = _derive_fallback_key(password, aad)
    mask = key * ((len(ciphertext) // len(key)) + 1)
    plaintext = bytes(m ^ c for m, c in zip(mask, ciphertext))
    return plaintext


def _fallback_encode_text(
    message: str,
    password: str,
    *,
    quality: Mapping[str, object],
    seed_text: str,
) -> Dict[str, Any]:
    encoding = "utf-8"
    payload = message.encode(encoding)
    seed_bytes = seed_text.encode("utf-8") if seed_text else b""
    key = _derive_fallback_key(password, seed_bytes)
    mask = key * ((len(payload) // len(key)) + 1)
    tokens = [int(p ^ mask[index]) for index, p in enumerate(payload)]
    result: Dict[str, Any] = {
        "encoding": encoding,
        "quality": dict(quality),
        "tokens": tokens,
    }
    if seed_text:
        result["seed"] = seed_text
    return result


def _fallback_decode_text(
    payload: Mapping[str, Any] | Sequence[int],
    password: str,
    *,
    seed_text: str,
) -> str:
    if isinstance(payload, Mapping):
        tokens_obj = payload.get("tokens")
        encoding = payload.get("encoding", "utf-8")
    else:
        tokens_obj = payload
        encoding = "utf-8"

    if not isinstance(tokens_obj, Sequence):
        raise ConfigurationError("encoded payload missing token sequence")

    token_values = [int(value) & 0xFF for value in tokens_obj]
    seed_bytes = seed_text.encode("utf-8") if seed_text else b""
    key = _derive_fallback_key(password, seed_bytes)
    mask = key * ((len(token_values) // len(key)) + 1)
    decoded = bytes(mask[index] ^ value for index, value in enumerate(token_values))
    try:
        return decoded.decode(str(encoding))
    except UnicodeDecodeError as exc:
        raise ConfigurationError("failed to decode fallback payload") from exc


def encode_text(
    message: str,
    password: str,
    *,
    quality: Mapping[str, object],
    seed_text: str = "",
) -> Dict[str, Any]:
    """Wrapper around :func:`neuralstego.crypto.api.encode_text` for the CLI."""

    crypto_api, import_error = _get_crypto_api()
    if crypto_api is None:
        _fallback_warning(import_error)
        payload_dict = _fallback_encode_text(
            message,
            password,
            quality=quality,
            seed_text=seed_text,
        )
        return payload_dict

    payload = crypto_api.encode_text(message, password, quality=quality, seed_text=seed_text)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ConfigurationError("encoder returned invalid JSON payload") from exc

    if not isinstance(decoded, MutableMapping):
        raise ConfigurationError("encoder returned an unexpected payload type")
    return dict(decoded)


def decode_text(
    payload: Mapping[str, Any] | Sequence[int],
    password: str,
    *,
    quality: Mapping[str, object] | None = None,
    seed_text: str = "",
) -> str:
    """Wrapper around :func:`neuralstego.crypto.api.decode_text` for the CLI."""

    if isinstance(payload, Mapping):
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    elif isinstance(payload, Sequence):
        blob = json.dumps(list(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    else:  # pragma: no cover - defensive guard
        raise ConfigurationError("encoded payload must be a mapping or sequence")

    quality_map = dict(quality or {})

    crypto_api, import_error = _get_crypto_api()
    if crypto_api is None:
        _fallback_warning(import_error)
        if isinstance(payload, Mapping):
            payload_mapping = payload
        else:
            payload_mapping = {"tokens": list(payload)}
        return _fallback_decode_text(payload_mapping, password, seed_text=seed_text)

    return crypto_api.decode_text(blob, password, quality=quality_map, seed_text=seed_text)


def _read_bytes(path: str | None) -> bytes:
    if not path or path == "-":
        return sys.stdin.buffer.read()
    return Path(path).read_bytes()


def _write_bytes(path: str | None, data: bytes) -> None:
    if not path or path == "-":
        sys.stdout.buffer.write(data)
        return
    Path(path).write_bytes(data)


def _read_text(path: str | None, *, encoding: str = "utf-8") -> str:
    if not path or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding=encoding)


def _write_text(path: str | None, data: str, *, encoding: str = "utf-8") -> None:
    if not path or path == "-":
        sys.stdout.write(data)
        return
    Path(path).write_text(data, encoding=encoding)


def _write_json(path: str | None, payload: Mapping[str, Any]) -> None:
    serialised = json.dumps(payload, ensure_ascii=False, indent=2)
    if not path or path == "-":
        console.print(serialised)
    else:
        Path(path).write_text(serialised, encoding="utf-8")


def _read_json(path: str | None) -> Any:
    raw = _read_text(path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("failed to parse JSON input") from exc


def _coerce_quality_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if any(ch in raw for ch in ".eE"):
        try:
            return float(raw)
        except ValueError:
            return raw
    try:
        return int(raw)
    except ValueError:
        return raw


def _quality_from_pairs(pairs: Sequence[Sequence[str]] | None) -> Dict[str, object]:
    quality: Dict[str, object] = {}
    if not pairs:
        return quality
    for key, value in pairs:
        quality[str(key)] = _coerce_quality_value(str(value))
    return quality


def _quality_from_prefixed(args: Sequence[str], prefix: str) -> Tuple[Dict[str, object], List[str]]:
    quality: Dict[str, object] = {}
    remaining: List[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token.startswith(prefix):
            key = token[len(prefix) :]
            if not key:
                raise ValueError("quality flag must include a key name")
            index += 1
            if index >= len(args):
                raise ValueError(f"missing value for {token}")
            value = args[index]
            quality[key] = _coerce_quality_value(value)
        else:
            remaining.append(token)
        index += 1
    return quality, remaining


def _load_codec_lm(name: str) -> Any:
    if name.lower() == "mock":
        return MockLM()
    raise ConfigurationError(f"unsupported language model '{name}' for codec commands")


def _resolve_ecc(choice: str) -> Tuple[str, bool]:
    ecc = choice.lower()
    if ecc not in {"auto", "none", "rs"}:
        raise ConfigurationError(f"unsupported ecc mode '{choice}'")
    if ecc == "auto":
        if _RSCodec is None:
            return "none", True
        return "rs", False
    if ecc == "rs" and _RSCodec is None:
        return "none", True
    return ecc, False


def _handle_encrypt(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="neuralstego encrypt", description="Encrypt a message with a password.")
    parser.add_argument("-p", "--password", required=True, help="Password used for encryption")
    parser.add_argument("-i", "--input", dest="input_path", required=True, help="Input file path")
    parser.add_argument("-o", "--output", dest="output_path", required=True, help="Output envelope path")
    parser.add_argument("--aad", dest="aad", help="Associated authenticated data (UTF-8)")
    args = parser.parse_args(list(argv))

    plaintext = _read_bytes(args.input_path)
    aad_bytes = args.aad.encode("utf-8") if args.aad is not None else b""

    crypto_api, import_error = _get_crypto_api()
    if crypto_api is None:
        _fallback_warning(import_error)
        blob = _fallback_encrypt_message(plaintext, args.password, aad_bytes)
    else:
        try:
            blob = crypto_api.encrypt_message(plaintext, args.password, aad=aad_bytes)
        except EncryptionError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return 1

    _write_bytes(args.output_path, blob)
    return 0


def _handle_decrypt(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="neuralstego decrypt", description="Decrypt an envelope with a password.")
    parser.add_argument("-p", "--password", required=True, help="Password used for decryption")
    parser.add_argument("-i", "--input", dest="input_path", required=True, help="Input envelope path")
    parser.add_argument("-o", "--output", dest="output_path", required=True, help="Output file path")
    parser.add_argument("--aad", dest="aad", help="Associated authenticated data (UTF-8)")
    args = parser.parse_args(list(argv))

    blob = _read_bytes(args.input_path)
    crypto_api, import_error = _get_crypto_api()
    if crypto_api is None:
        _fallback_warning(import_error)
        try:
            plaintext = _fallback_decrypt_message(blob, args.password)
        except ConfigurationError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return 1
    else:
        try:
            plaintext = crypto_api.decrypt_message(blob, args.password)
        except DecryptionError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return 1

    _write_bytes(args.output_path, plaintext)
    return 0


def _handle_encode(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="neuralstego encode", description="Encode text into steganographic tokens.")
    parser.add_argument("-p", "--password", required=True, help="Password used for encryption")
    parser.add_argument("-i", "--input", dest="input_path", required=True, help="Input text file")
    parser.add_argument("-o", "--output", dest="output_path", required=True, help="Output JSON file")
    parser.add_argument("--seed-text", default="", help="Optional seed text for the language model")
    parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Quality parameter override as key/value pairs",
    )

    args = parser.parse_args(list(argv))
    quality = _quality_from_pairs(args.quality)

    message = _read_text(args.input_path)

    encode_kwargs: Dict[str, object] = {"quality": quality}
    if args.seed_text:
        encode_kwargs["seed_text"] = args.seed_text

    try:
        payload = encode_text(message, args.password, **encode_kwargs)
    except (ConfigurationError, NeuralStegoError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    payload.setdefault("quality", dict(quality))
    payload.setdefault("encoding", "utf-8")
    _write_json(args.output_path, payload)
    return 0


def _handle_decode(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="neuralstego decode", description="Decode steganographic tokens back into text.")
    parser.add_argument("-p", "--password", required=True, help="Password used for decryption")
    parser.add_argument("-i", "--input", dest="input_path", required=True, help="Input JSON file")
    parser.add_argument("-o", "--output", dest="output_path", required=True, help="Output text file")
    parser.add_argument("--seed-text", default="", help="Optional seed text for the language model")
    parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Quality parameter override as key/value pairs",
    )

    args = parser.parse_args(list(argv))
    quality = _quality_from_pairs(args.quality)

    decode_kwargs: Dict[str, object] = {}
    if quality:
        decode_kwargs["quality"] = quality
    if args.seed_text:
        decode_kwargs["seed_text"] = args.seed_text

    try:
        payload = _read_json(args.input_path)
        if not isinstance(payload, Mapping):
            raise ConfigurationError("encoded payload must be a JSON object")
        message = decode_text(payload, args.password, **decode_kwargs)
    except (ConfigurationError, NeuralStegoError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    _write_text(args.output_path, message)
    return 0


def _handle_cover_generate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="neuralstego cover-generate",
        description="Generate a Farsi cover text embedding secret data.",
    )
    parser.add_argument("-p", "--password", help="Optional password for encryption (not yet supported)")
    parser.add_argument("-i", "--in", dest="input_path", default="-", help="Secret input file (default: stdin)")
    parser.add_argument("-o", "--out", dest="output_path", default="-", help="Cover text output (default: stdout)")
    parser.add_argument("--seed", dest="seed_text", default="", help="Seed text to prime the language model")
    parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Quality parameter override as key/value pairs",
    )
    parser.add_argument("--chunk-bytes", type=int, default=256, help="Bytes per chunk for framing")
    parser.add_argument("--crc", choices=["on", "off"], default="on", help="Enable or disable CRC32")
    parser.add_argument(
        "--ecc",
        choices=["auto", "none", "rs"],
        default="rs",
        help="Error correction coding mode",
    )
    parser.add_argument("--nsym", type=int, default=10, help="Reed-Solomon parity symbols")

    args = parser.parse_args(list(argv))
    if args.password:
        console.print("[yellow]Password-based encryption is not yet supported for cover generation.[/yellow]")

    quality = _quality_from_pairs(args.quality)

    secret = _read_bytes(args.input_path)

    ecc_mode, degraded = _resolve_ecc(args.ecc)
    if degraded:
        console.print("[yellow]Reed-Solomon ECC unavailable; continuing without ECC.[/yellow]")

    try:
        cover_text = api_cover_generate(
            secret,
            seed_text=args.seed_text,
            quality=quality,
            chunk_bytes=args.chunk_bytes,
            use_crc=args.crc == "on",
            ecc=ecc_mode,
            nsym=args.nsym,
        )
    except (ConfigurationError, NeuralStegoError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    _write_text(args.output_path, cover_text)
    return 0


def _handle_cover_reveal(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="neuralstego cover-reveal",
        description="Reveal embedded secret data from a cover text.",
    )
    parser.add_argument("-p", "--password", help="Optional password for decryption (not yet supported)")
    parser.add_argument("-i", "--in", dest="input_path", default="-", help="Cover text input (default: stdin)")
    parser.add_argument("-o", "--out", dest="output_path", default="-", help="Recovered secret output (default: stdout)")
    parser.add_argument("--seed", dest="seed_text", default="", help="Seed text used during encoding")
    parser.add_argument(
        "--quality",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Quality parameter override as key/value pairs",
    )
    parser.add_argument("--crc", choices=["on", "off"], default="on", help="Enable or disable CRC32")
    parser.add_argument(
        "--ecc",
        choices=["auto", "none", "rs"],
        default="rs",
        help="Error correction coding mode",
    )
    parser.add_argument("--nsym", type=int, default=10, help="Reed-Solomon parity symbols")

    args = parser.parse_args(list(argv))
    if args.password:
        console.print("[yellow]Password-based decryption is not yet supported for cover reveal.[/yellow]")

    quality = _quality_from_pairs(args.quality)

    ecc_mode, degraded = _resolve_ecc(args.ecc)
    if degraded:
        console.print("[yellow]Reed-Solomon ECC unavailable; continuing without ECC.[/yellow]")

    cover_text = _read_text(args.input_path)

    try:
        secret_bytes = api_cover_reveal(
            cover_text,
            seed_text=args.seed_text,
            quality=quality,
            use_crc=args.crc == "on",
            ecc=ecc_mode,
            nsym=args.nsym,
        )
    except MissingChunksError as exc:
        console.print("[red]Error:[/red] Missing chunk indices: " + ", ".join(map(str, exc.missing_indices)))
        if exc.partial_payload:
            console.print("[yellow]Writing partial payload despite missing chunks.[/yellow]")
            try:
                text = exc.partial_payload.decode("utf-8")
            except UnicodeDecodeError:
                console.print("[red]Error:[/red] Partial payload is not valid UTF-8.")
                return 1
            _write_text(args.output_path, text)
        return 1
    except (ConfigurationError, NeuralStegoError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    try:
        secret_text = secret_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        console.print(f"[red]Error:[/red] Failed to decode secret as UTF-8: {exc}")
        return 1

    _write_text(args.output_path, secret_text)
    return 0


def _handle_codec_encode(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="neuralstego codec-encode",
        description="Encode binary data into arithmetic codec tokens.",
        allow_abbrev=False,
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Input file path")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSON file")
    parser.add_argument("--lm", default="mock", help="Language model backend (default: mock)")
    parser.add_argument("--chunk-bytes", type=int, default=256, help="Number of bytes per chunk")
    parser.add_argument("--crc", choices=["on", "off"], default="on", help="Enable or disable CRC32")
    parser.add_argument(
        "--ecc",
        choices=["auto", "none", "rs"],
        default="auto",
        help="Select ECC mode (auto chooses RS when available)",
    )
    parser.add_argument("--nsym", type=int, default=10, help="Reed-Solomon parity symbols")
    parser.add_argument("--seed-text", default="", help="Optional seed text for the language model")

    known, extra = parser.parse_known_args(list(argv))
    try:
        quality, leftover = _quality_from_prefixed(extra, "--quality.")
    except ValueError as exc:
        parser.error(str(exc))
    if leftover:
        parser.error(f"unrecognized arguments: {' '.join(leftover)}")

    ecc_mode, degraded = _resolve_ecc(known.ecc)
    if degraded:
        console.print(
            "[yellow]Reed-Solomon ECC is unavailable. Encoding will continue without ECC protection.[/yellow]"
        )

    message = _read_bytes(known.input_path)

    try:
        lm = _load_codec_lm(known.lm)
        tokens = codec_encode_text(
            message,
            lm,
            chunk_bytes=known.chunk_bytes,
            use_crc=known.crc == "on",
            ecc=ecc_mode,
            nsym=known.nsym,
            quality=quality,
            seed_text=known.seed_text,
        )
    except (ConfigurationError, NeuralStegoError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    Path(known.output_path).write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    return 0


def _handle_codec_decode(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="neuralstego codec-decode",
        description="Decode arithmetic codec tokens back into binary data.",
        allow_abbrev=False,
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSON file")
    parser.add_argument("--out", dest="output_path", required=True, help="Output file path")
    parser.add_argument("--lm", default="mock", help="Language model backend (default: mock)")
    parser.add_argument("--chunk-bytes", type=int, default=256, help="Expected chunk size")
    parser.add_argument("--crc", choices=["on", "off"], default="on", help="Enable or disable CRC32")
    parser.add_argument(
        "--ecc",
        choices=["auto", "none", "rs"],
        default="auto",
        help="Select ECC mode (auto chooses RS when available)",
    )
    parser.add_argument("--nsym", type=int, default=10, help="Reed-Solomon parity symbols")
    parser.add_argument("--seed-text", default="", help="Optional seed text for the language model")

    known, extra = parser.parse_known_args(list(argv))
    try:
        quality, leftover = _quality_from_prefixed(extra, "--quality.")
    except ValueError as exc:
        parser.error(str(exc))
    if leftover:
        parser.error(f"unrecognized arguments: {' '.join(leftover)}")

    ecc_mode, degraded = _resolve_ecc(known.ecc)
    if degraded:
        console.print(
            "[yellow]Reed-Solomon ECC is unavailable. Decoding will proceed without ECC verification.[/yellow]"
        )

    try:
        raw = _read_text(known.input_path)
        try:
            tokens_obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("failed to parse JSON token stream") from exc
        if isinstance(tokens_obj, Mapping):
            token_stream = tokens_obj.get("tokens")
            if not isinstance(token_stream, Sequence):
                raise ConfigurationError("JSON payload missing 'tokens' array")
            spans = [int(token) for token in token_stream]
        elif isinstance(tokens_obj, Sequence):
            spans = [int(token) for token in tokens_obj]
        else:
            raise ConfigurationError("token payload must be a JSON array or object")

        lm = _load_codec_lm(known.lm)
        data = codec_decode_text(
            spans,
            lm,
            quality=quality,
            seed_text=known.seed_text,
            chunk_bytes=known.chunk_bytes,
            use_crc=known.crc == "on",
            ecc=ecc_mode,
            nsym=known.nsym,
        )
    except MissingChunksError as exc:
        console.print("[red]Error:[/red] Missing chunk indices: " + ", ".join(map(str, exc.missing_indices)))
        if exc.partial_payload:
            console.print("[yellow]Writing partial payload despite missing chunks.[/yellow]")
            _write_bytes(known.output_path, exc.partial_payload)
        return 1
    except (ConfigurationError, NeuralStegoError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    _write_bytes(known.output_path, data)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralstego",
        description="Command line interface for the neural steganography toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in [
        "encrypt",
        "decrypt",
        "encode",
        "decode",
        "cover-generate",
        "cover-reveal",
        "codec-encode",
        "codec-decode",
    ]:
        subparsers.add_parser(command)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        build_parser().print_help()
        return 0

    command, rest = args[0], args[1:]

    if command == "encrypt":
        return _handle_encrypt(rest)
    if command == "decrypt":
        return _handle_decrypt(rest)
    if command == "encode":
        return _handle_encode(rest)
    if command == "decode":
        return _handle_decode(rest)
    if command == "cover-generate":
        return _handle_cover_generate(rest)
    if command == "cover-reveal":
        return _handle_cover_reveal(rest)
    if command == "codec-encode":
        return _handle_codec_encode(rest)
    if command == "codec-decode":
        return _handle_codec_decode(rest)

    console.print(f"[red]Error:[/red] unknown command '{command}'")
    build_parser().print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())

