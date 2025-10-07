"""End-to-end tests for the CLI crypto commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str) -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    pythonpath = str(root / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = pythonpath
    subprocess.run([sys.executable, "-m", "neuralstego.cli", *args], check=True, env=env)


def test_cli_encrypt_decrypt_roundtrip(tmp_path: Path) -> None:
    input_file = tmp_path / "message.txt"
    envelope_file = tmp_path / "message.enc"
    output_file = tmp_path / "message.out"

    sample_text = "سلام دنیا!\nHello NeuralStego."
    input_file.write_text(sample_text, encoding="utf-8")

    _run_cli(
        "encrypt",
        "-p",
        "Pa$$w0rd",
        "-i",
        str(input_file),
        "-o",
        str(envelope_file),
        "--aad",
        "یادداشت",
    )

    _run_cli(
        "decrypt",
        "-p",
        "Pa$$w0rd",
        "-i",
        str(envelope_file),
        "-o",
        str(output_file),
        "--aad",
        "یادداشت",
    )

    assert output_file.read_text(encoding="utf-8") == sample_text

