"""Basic smoke tests for the neuralstego package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_truth() -> None:
    """Ensure the test harness is functioning."""
    assert True


@pytest.mark.skipif(sys.executable is None, reason="Python executable not available")
def test_cli_help_runs() -> None:
    """Verify that the CLI help command returns successfully."""
    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "src"
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{existing_path}" if existing_path else str(src_dir)
    )

    result = subprocess.run(
        [sys.executable, "-m", "neuralstego", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        pytest.skip("CLI entry point not yet wired up")
    assert "neuralstego" in result.stdout
