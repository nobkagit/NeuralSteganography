"""Tests for the GPT2-fa CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralstego import cli as cli_module  # noqa: E402


def _run_cli(*args: str) -> int:
    return cli_module.main(list(args))


def test_roundtrip(tmp_path):
    message_path = tmp_path / "msg.txt"
    message = "پیام آزمایشی با GPT2-fa"
    message_path.write_text(message, encoding="utf-8")
    tokens_path = tmp_path / "tokens.json"
    output_path = tmp_path / "out.txt"

    exit_code = _run_cli(
        "encode",
        "-p",
        "Pa$$w0rd",
        "-i",
        str(message_path),
        "-o",
        str(tokens_path),
        "--quality",
        "top-k",
        "50",
        "--quality",
        "temperature",
        "0.7",
    )
    assert exit_code == 0

    encoded_payload = json.loads(tokens_path.read_text(encoding="utf-8"))
    assert encoded_payload["quality"] == {"top-k": 50, "temperature": 0.7}

    exit_code = _run_cli(
        "decode",
        "-p",
        "Pa$$w0rd",
        "-i",
        str(tokens_path),
        "-o",
        str(output_path),
    )
    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == message


def test_cli_forwards_quality_and_uses_encoder(monkeypatch, tmp_path):
    message_path = tmp_path / "msg.txt"
    message_path.write_text("secret", encoding="utf-8")
    tokens_path = tmp_path / "tokens.json"
    output_path = tmp_path / "out.txt"

    calls: dict[str, object] = {}

    def fake_encode(message: str, password: str, *, quality):
        calls["encode"] = {
            "message": message,
            "password": password,
            "quality": quality,
        }
        return {
            "tokens": [1, 2, 3],
            "quality": dict(quality),
            "encoding": "utf-8",
        }

    def fake_decode(payload, password: str):
        calls["decode"] = {
            "payload": payload,
            "password": password,
        }
        return "secret"

    monkeypatch.setattr(cli_module, "encode_text", fake_encode)
    monkeypatch.setattr(cli_module, "decode_text", fake_decode)

    exit_code = _run_cli(
        "encode",
        "-p",
        "hunter2",
        "-i",
        str(message_path),
        "-o",
        str(tokens_path),
        "--quality",
        "top-k",
        "50",
        "--quality",
        "temperature",
        "0.7",
        "--quality",
        "nucleus",
        "true",
    )
    assert exit_code == 0

    assert calls["encode"]["message"] == "secret"
    assert calls["encode"]["password"] == "hunter2"
    assert calls["encode"]["quality"] == {"top-k": 50, "temperature": 0.7, "nucleus": True}

    stored_tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
    assert stored_tokens["tokens"] == [1, 2, 3]

    exit_code = _run_cli(
        "decode",
        "-p",
        "hunter2",
        "-i",
        str(tokens_path),
        "-o",
        str(output_path),
    )
    assert exit_code == 0

    assert calls["decode"]["password"] == "hunter2"
    assert calls["decode"]["payload"]["tokens"] == [1, 2, 3]
    assert output_path.read_text(encoding="utf-8") == "secret"
