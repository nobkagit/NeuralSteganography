from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuralstego import cli as cli_module  # noqa: E402
from neuralstego.detect.guard import GuardResult  # noqa: E402
from neuralstego.exceptions import QualityGateError  # noqa: E402


def _run_cli(*args: str) -> int:
    return cli_module.main(list(args))


def test_cover_generate_forwards_quality_gate_options(monkeypatch, tmp_path, capsys):
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("راز", encoding="utf-8")
    output_path = tmp_path / "cover.txt"
    seed_pool_path = tmp_path / "seeds.txt"
    seed_pool_path.write_text("بذر دوم\nبذر سوم\n", encoding="utf-8")

    calls: dict[str, object] = {}

    def fake_cover_generate(secret: bytes, **kwargs):
        calls["secret"] = secret
        calls["kwargs"] = kwargs
        return "متن کاور"

    monkeypatch.setattr(cli_module, "api_cover_generate", fake_cover_generate)

    exit_code = _run_cli(
        "cover-generate",
        "-i",
        str(secret_path),
        "-o",
        str(output_path),
        "--seed",
        "بذر اصلی",
        "--quality-gate",
        "on",
        "--max-ppl",
        "90",
        "--max-detector-score",
        "0.4",
        "--max-ngram-repeat",
        "0.2",
        "--min-ttr",
        "0.3",
        "--regen-attempts",
        "3",
        "--regen-seed-pool",
        str(seed_pool_path),
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    assert "classifier is not configured" in captured.out
    assert output_path.read_text(encoding="utf-8") == "متن کاور"

    kwargs = calls["kwargs"]
    assert kwargs["quality_gate"] is True
    assert kwargs["gate_thresholds"] == {
        "max_ppl": 90.0,
        "max_detector_score": 0.4,
        "max_ngram_repeat": 0.2,
        "min_ttr": 0.3,
    }
    assert kwargs["regen_attempts"] == 3
    assert kwargs["regen_strategy"] == {"seed_pool": ["بذر دوم", "بذر سوم"]}
    assert kwargs["quality_guard"] is cli_module.QUALITY_GUARD
    assert kwargs["seed_text"] == "بذر اصلی"
    assert calls["secret"] == secret_path.read_bytes()


def test_cover_generate_reports_quality_gate_failure(monkeypatch, tmp_path, capsys):
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("راز", encoding="utf-8")
    output_path = tmp_path / "cover.txt"

    def fake_cover_generate(*_, **__):
        raise QualityGateError(
            "متن ضعیف",
            ["ppl 200.0 exceeds max 90.0"],
            {"ppl": 200.0, "type_token_ratio": 0.15},
        )

    monkeypatch.setattr(cli_module, "api_cover_generate", fake_cover_generate)

    exit_code = _run_cli(
        "cover-generate",
        "-i",
        str(secret_path),
        "-o",
        str(output_path),
        "--seed",
        "بذر اصلی",
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert output_path.read_text(encoding="utf-8") == "متن ضعیف"
    assert "Quality gate rejected" in captured.out
    assert "ppl" in captured.out
    assert "- ppl 200.0 exceeds max 90.0" in captured.out


def test_quality_audit_reports_metrics(monkeypatch, tmp_path, capsys):
    cover_path = tmp_path / "cover.txt"
    cover_path.write_text("متن کاور", encoding="utf-8")

    class StubGuard:
        classifier = None

        def __init__(self, result: GuardResult) -> None:
            self.result = result
            self.calls = 0
            self.thresholds = None
            self.text = ""

        def evaluate(self, text, thresholds):
            self.calls += 1
            self.text = text
            self.thresholds = thresholds
            return self.result

    result = GuardResult(True, [], {"ppl": 32.0, "type_token_ratio": 0.41, "avg_entropy": 2.3})
    guard = StubGuard(result)
    monkeypatch.setattr(cli_module, "QUALITY_GUARD", guard)

    exit_code = _run_cli(
        "quality-audit",
        "-i",
        str(cover_path),
        "--max-ppl",
        "80",
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert guard.calls == 1
    assert guard.thresholds["max_ppl"] == 80.0
    assert "Quality metrics" in captured.out
    assert "PASS" in captured.out


def test_quality_audit_exits_nonzero_on_failure(monkeypatch, tmp_path, capsys):
    cover_path = tmp_path / "cover.txt"
    cover_path.write_text("متن تکراری", encoding="utf-8")

    class StubGuard:
        classifier = None

        def __init__(self, result: GuardResult) -> None:
            self.result = result

        def evaluate(self, text, thresholds):
            return self.result

    result = GuardResult(False, ["ppl 150.0 exceeds max 90.0"], {"ppl": 150.0})
    guard = StubGuard(result)
    monkeypatch.setattr(cli_module, "QUALITY_GUARD", guard)

    exit_code = _run_cli(
        "quality-audit",
        "-i",
        str(cover_path),
        "--max-ppl",
        "90",
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.out
    assert "Reasons" in captured.out
