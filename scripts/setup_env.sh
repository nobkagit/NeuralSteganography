#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
EXAMPLE_FILE="$ROOT_DIR/.env.example"

if [[ ! -f "$EXAMPLE_FILE" ]]; then
  echo "[setup_env] فایل $EXAMPLE_FILE یافت نشد." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$EXAMPLE_FILE" "$ENV_FILE"
  echo "[setup_env] فایل .env جدید از روی نمونه ساخته شد."
else
  echo "[setup_env] فایل .env موجود است؛ تغییری اعمال نشد."
fi

python - <<'PY'
import pathlib
root = pathlib.Path(__file__).resolve().parents[1]
(root / "data").mkdir(exist_ok=True)
(root / "models").mkdir(exist_ok=True)
print("[setup_env] پوشه‌های data/ و models/ آماده شدند.")
PY

echo "[setup_env] آماده‌سازی محیط اولیه به پایان رسید."
