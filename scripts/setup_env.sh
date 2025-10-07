#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[setup_env] هیچ مفسر Pythonی در PATH یافت نشد." >&2
    exit 1
  fi
fi

if [[ -x "$VENV_PATH/bin/python" ]]; then
  echo "[setup_env] استفاده از محیط مجازی موجود در $VENV_PATH"
else
  echo "[setup_env] ایجاد محیط مجازی جدید در $VENV_PATH"
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
PYTHON="$VENV_PATH/bin/python"

echo "[setup_env] ارتقای pip، setuptools و wheel..."
"$PYTHON" -m pip install --upgrade pip setuptools wheel

pushd "$ROOT_DIR" >/dev/null

echo "[setup_env] نصب بسته‌های پروژه به صورت editable با extras توسعه..."
"$PYTHON" -m pip install -e ".[dev]"

if [[ -f "requirements.txt" ]]; then
  echo "[setup_env] نصب وابستگی‌های اضافه از requirements.txt..."
  "$PYTHON" -m pip install -r requirements.txt
fi

popd >/dev/null

ENV_FILE="$ROOT_DIR/.env"
EXAMPLE_FILE="$ROOT_DIR/.env.example"
if [[ -f "$EXAMPLE_FILE" && ! -f "$ENV_FILE" ]]; then
  cp "$EXAMPLE_FILE" "$ENV_FILE"
  echo "[setup_env] فایل .env از روی نمونه ساخته شد."
fi

mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/models"

echo "[setup_env] محیط توسعه آماده است. برای فعال‌سازی: source $VENV_PATH/bin/activate"
