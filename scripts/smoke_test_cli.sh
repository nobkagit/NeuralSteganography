#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
else
  echo "[smoke_test_cli] هشدار: محیط مجازی .venv پیدا نشد؛ از Python سیستم استفاده می‌شود." >&2
fi

cd "$ROOT_DIR"

neuralstego --version
neuralstego doctor
pytest -q

echo "[smoke_test_cli] تمام دستورات موردنیاز با موفقیت اجرا شدند."
