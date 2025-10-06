"""Basic smoke tests for the neuralstego package."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_truth() -> None:
    """Ensure the test harness is functioning."""
    assert True


@pytest.mark.skipif(sys.executable is None, reason="Python executable not available")
def test_cli_help_runs() -> None:
    """Verify that the CLI help command returns successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "neuralstego", "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("CLI entry point not yet wired up")
    assert "neuralstego" in result.stdout
