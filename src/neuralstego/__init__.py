"""Public package interface for :mod:`neuralstego`."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time hinting only
    from .cli import main as _cli_main


def main(*args: Any, **kwargs: Any) -> Any:
    """Entrypoint wrapper that defers CLI imports until invocation."""

    from .cli import main as _cli_main

    return _cli_main(*args, **kwargs)


__all__ = ["main"]
