"""CLI smoke test for arithmetic codec round-trip."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rich")


def test_cli_codec_roundtrip(tmp_path: Path) -> None:
    message_path = tmp_path / "message.bin"
    tokens_path = tmp_path / "tokens.json"
    output_path = tmp_path / "output.bin"

    payload = "پیام آزمایشی — sample bits\n".encode("utf-8")
    message_path.write_bytes(payload)

    base_cmd = [sys.executable, "-m", "neuralstego"]
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    env = os.environ.copy()
    python_path_entries = [str(src_path)]
    if existing := env.get("PYTHONPATH"):
        python_path_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    encode_cmd = [
        *base_cmd,
        "codec-encode",
        "--in",
        str(message_path),
        "--out",
        str(tokens_path),
        "--quality.cap-per-token-bits",
        "4",
        "--quality.top-k",
        "64",
    ]
    subprocess.run(encode_cmd, check=True, env=env)

    decode_cmd = [
        *base_cmd,
        "codec-decode",
        "--in",
        str(tokens_path),
        "--out",
        str(output_path),
    ]
    subprocess.run(decode_cmd, check=True, env=env)

    assert output_path.read_bytes() == payload
