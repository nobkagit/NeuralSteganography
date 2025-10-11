"""Tests for the ``neuralstego cover-walkthrough`` command."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralstego import cli as cli_module  # noqa: E402
from neuralstego.exceptions import QualityGateError  # noqa: E402


def _run_cli(*args: str) -> int:
    return cli_module.main(list(args))


def test_cover_walkthrough_happy_path(monkeypatch, capsys):
    calls: Dict[str, Any] = {}

    def fake_cover_generate(secret: bytes, **kwargs: Any) -> str:
        calls["secret"] = secret
        calls["generate_kwargs"] = kwargs
        return "متن کاور نمونه"

    def fake_cover_reveal(cover_text: str, **kwargs: Any) -> bytes:
        calls["cover_text"] = cover_text
        calls["reveal_kwargs"] = kwargs
        return "راز".encode("utf-8")

    monkeypatch.setattr(cli_module, "api_cover_generate", fake_cover_generate)
    monkeypatch.setattr(cli_module, "api_cover_reveal", fake_cover_reveal)

    exit_code = _run_cli(
        "cover-walkthrough",
        "--message",
        "راز",
        "--seed-text",
        "گفتگو",
        "--chunk-bytes",
        "128",
        "--nsym",
        "12",
        "--quality",
        "temp",
        "0.75",
        "--quality.top_p",
        "0.90",
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert "مرحله ۱" in captured.out
    assert "مرحله ۲" in captured.out
    assert "مرحله ۳" in captured.out
    assert "نتیجه" in captured.out

    assert calls["secret"] == "راز".encode("utf-8")
    generate_kwargs = calls["generate_kwargs"]
    assert generate_kwargs["seed_text"] == "گفتگو"
    assert generate_kwargs["chunk_bytes"] == 128
    assert generate_kwargs["nsym"] == 12
    assert generate_kwargs["quality"] == {"temp": 0.75, "top_p": 0.9}
    assert generate_kwargs["quality_guard"] is cli_module.QUALITY_GUARD

    assert calls["cover_text"] == "متن کاور نمونه"
    reveal_kwargs = calls["reveal_kwargs"]
    assert reveal_kwargs["seed_text"] == "گفتگو"
    assert reveal_kwargs["quality"] == {"temp": 0.75, "top_p": 0.9}
    assert reveal_kwargs["use_crc"] is True


def test_cover_walkthrough_reports_gate_failure(monkeypatch, capsys):
    def fake_cover_generate(*_: Any, **__: Any) -> str:
        raise QualityGateError(
            "کاور رد شده",
            ["ppl 210.0 > max 120.0"],
            {"ppl": 210.0},
        )

    revealed: Dict[str, Any] = {}

    def fake_cover_reveal(cover_text: str, **kwargs: Any) -> bytes:
        revealed["cover_text"] = cover_text
        revealed["kwargs"] = kwargs
        return b"secret"

    monkeypatch.setattr(cli_module, "api_cover_generate", fake_cover_generate)
    monkeypatch.setattr(cli_module, "api_cover_reveal", fake_cover_reveal)

    exit_code = _run_cli(
        "cover-walkthrough",
        "--message",
        "متن",
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Quality gate rejected" in captured.out
    assert "Continuing with the rejected cover" in captured.out
    assert revealed["cover_text"] == "کاور رد شده"
    assert revealed["kwargs"]["quality"] is None
