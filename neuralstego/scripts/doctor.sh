#!/usr/bin/env bash
set -euo pipefail

echo "Checking Python version..."
python --version

echo "Checking pip version..."
pip --version

echo "Checking pytest availability..."
if python -m pytest --version >/dev/null 2>&1; then
  echo "pytest is available."
else
  echo "pytest is not installed. Install with: python -m pip install -e '.[dev]'" >&2
fi
