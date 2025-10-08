"""Command line interface for the neural steganography toolkit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rich.console import Console
from rich.table import Table

from .api import stego_decode, stego_encode
from .exceptions import ConfigurationError, MissingChunksError, NeuralStegoError
from .lm import load_lm

console = Console()


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return ivalue


def _read_bytes(path: Optional[str]) -> bytes:
    if not path or path == "-":
        return sys.stdin.buffer.read()
    return Path(path).read_bytes()


def _write_bytes(path: Optional[str], data: bytes) -> None:
    if not path or path == "-":
        sys.stdout.buffer.write(data)
        return
    Path(path).write_bytes(data)


def _write_json(path: Optional[str], data: Dict[str, Any]) -> None:
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    if not path or path == "-":
        console.print(serialized)
    else:
        Path(path).write_text(serialized, encoding="utf-8")


def _quality_from_args(args: argparse.Namespace) -> Dict[str, float]:
    quality: Dict[str, float] = {}
    if args.temp is not None:
        quality["temp"] = float(args.temp)
    if args.precision is not None:
        quality["precision"] = float(args.precision)
    if args.topk is not None:
        quality["topk"] = float(args.topk)
    return quality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuralstego", description=__doc__)
    parser.add_argument("--lm", default="mock", help="Language model backend (mock|gpt2-fa)")
    parser.add_argument("--device", default=None, help="Torch device override")

    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser("encode", help="Encode a message into token spans")
    encode.add_argument("--input", "-i", help="Input file (default: stdin)")
    encode.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    encode.add_argument("--chunk-bytes", type=_positive_int, default=256)
    encode.add_argument("--crc", choices=["on", "off"], default="on")
    encode.add_argument("--ecc", choices=["none", "rs"], default="rs")
    encode.add_argument("--nsym", type=_positive_int, default=10)
    encode.add_argument("--seed-text", default="")
    encode.add_argument("--temp", type=float, help="Softmax temperature override")
    encode.add_argument("--precision", type=float, help="Arithmetic precision override")
    encode.add_argument("--topk", type=float, help="Top-k cutoff override")

    decode = subparsers.add_parser("decode", help="Decode token spans back into bytes")
    decode.add_argument("--input", "-i", help="Input JSON file (default: stdin)")
    decode.add_argument("--output", "-o", help="Output file for decoded bytes (default: stdout)")
    decode.add_argument("--chunk-bytes", type=_positive_int, default=256)
    decode.add_argument("--crc", choices=["on", "off"], default="on")
    decode.add_argument("--ecc", choices=["none", "rs"], default="rs")
    decode.add_argument("--nsym", type=_positive_int, default=10)
    decode.add_argument("--seed-text", default="")
    decode.add_argument("--temp", type=float, help="Softmax temperature override")
    decode.add_argument("--precision", type=float, help="Arithmetic precision override")
    decode.add_argument("--topk", type=float, help="Top-k cutoff override")

    return parser


def handle_encode(args: argparse.Namespace) -> int:
    message = _read_bytes(args.input)
    if not isinstance(message, (bytes, bytearray)):
        raise ConfigurationError("input message must be binary")

    lm = load_lm(args.lm, device=args.device)
    quality = _quality_from_args(args)

    result = stego_encode(
        bytes(message),
        chunk_bytes=args.chunk_bytes,
        use_crc=args.crc == "on",
        ecc=args.ecc,
        nsym=args.nsym,
        quality=quality,
        seed_text=args.seed_text,
        lm=lm,
    )

    payload = {
        "version": 1,
        "msg_id": result.metadata.msg_id,
        "total": result.metadata.total,
        "cfg": result.metadata.cfg,
        "spans": result,
    }
    _write_json(args.output, payload)
    return 0


def handle_decode(args: argparse.Namespace) -> int:
    raw = _read_bytes(args.input)
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - json errors
        raise ConfigurationError("failed to parse JSON input") from exc

    spans = envelope.get("spans")
    if not isinstance(spans, list):
        raise ConfigurationError("JSON payload missing 'spans' array")

    cfg = envelope.get("cfg", {})
    if cfg:
        cfg_chunk = int(cfg.get("chunk_bytes", args.chunk_bytes))
        if cfg_chunk != args.chunk_bytes:
            raise ConfigurationError(
                f"chunk size mismatch: CLI={args.chunk_bytes} json={cfg_chunk}"
            )
        cfg_crc = bool(cfg.get("crc", args.crc == "on"))
        if cfg_crc != (args.crc == "on"):
            raise ConfigurationError("CRC setting mismatch between CLI and payload")
        cfg_ecc = cfg.get("ecc", args.ecc)
        if str(cfg_ecc).lower() != args.ecc:
            raise ConfigurationError("ECC setting mismatch between CLI and payload")
        cfg_nsym = int(cfg.get("nsym", args.nsym))
        if cfg_nsym != args.nsym and args.ecc == "rs":
            raise ConfigurationError("nsym mismatch between CLI and payload")

    lm = load_lm(args.lm, device=args.device)
    quality = _quality_from_args(args)

    try:
        decoded = stego_decode(
            spans,
            use_crc=args.crc == "on",
            ecc=args.ecc,
            nsym=args.nsym,
            quality=quality,
            seed_text=args.seed_text,
            lm=lm,
        )
    except MissingChunksError as exc:
        table = Table(title="Decoding incomplete")
        table.add_column("Missing indices", justify="left")
        table.add_row(", ".join(str(i) for i in exc.missing_indices))
        console.print(table)
        if exc.partial_payload:
            console.print("Writing partial payload despite errors", style="yellow")
            _write_bytes(args.output, exc.partial_payload)
        return 1

    _write_bytes(args.output, decoded)
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "encode":
            return handle_encode(args)
        if args.command == "decode":
            return handle_decode(args)
        raise ConfigurationError(f"unknown command {args.command}")
    except NeuralStegoError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
