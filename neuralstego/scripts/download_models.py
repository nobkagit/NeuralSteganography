#!/usr/bin/env python3
"""Placeholder script for downloading neuralstego models."""

from __future__ import annotations

import pathlib


def main() -> None:
    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"Created placeholder models directory at {models_dir.resolve()}")


if __name__ == "__main__":
    main()
