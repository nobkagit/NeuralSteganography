#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
  PY_LABEL="(.venv)"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[doctor] ❌ هیچ مفسر Pythonی در PATH یافت نشد." >&2
    exit 1
  fi
  PY_LABEL="(system)"
fi

echo "[doctor] استفاده از Python: $PYTHON_BIN $PY_LABEL"

status=0
missing_modules=()

ok() {
  echo "[doctor] ✅ $1"
}

fail() {
  echo "[doctor] ❌ $1"
  status=1
}

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if "$PYTHON_BIN" - <<'PY'
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  ok "نسخهٔ Python $PY_VERSION (>= 3.10)"
else
  fail "نسخهٔ Python $PY_VERSION شناسایی شد؛ لطفاً Python 3.10 یا جدیدتر نصب کنید."
fi

if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  ok "pip در دسترس است."
else
  fail "pip نصب نیست یا کار نمی‌کند. اجرای \"$PYTHON_BIN -m ensurepip --upgrade\" را بررسی کنید."
fi

if "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
  ok "ماژول venv فعال است."
else
  fail "ماژول venv غیرفعال است؛ لطفاً Python را با پشتیبانی از venv نصب کنید."
fi

if command -v git >/dev/null 2>&1; then
  ok "git در PATH یافت شد."
else
  fail "git نصب نشده است؛ لطفاً https://git-scm.com را نصب کنید."
fi

check_module() {
  local module="$1"
  local hint="$2"
  if "$PYTHON_BIN" - <<PY
import importlib
import importlib.util
mod = importlib.util.find_spec("$module")
raise SystemExit(0 if mod is not None else 1)
PY
  then
    ok "پکیج ${module} نصب است."
  else
    echo "[doctor] ⚠️ پکیج ${module} یافت نشد. دستور پیشنهادی: ${hint}"
    missing_modules+=("${module}|${hint}")
  fi
}

check_module "transformers" "pip install transformers"
check_module "torch" "pip install torch --extra-index-url https://download.pytorch.org/whl/cpu"
check_module "cryptography" "pip install cryptography"
check_module "bitarray" "pip install bitarray"

if [[ $status -eq 0 ]]; then
  echo "[doctor] چک ابزارهای پایه با موفقیت انجام شد."
else
  echo "[doctor] برخی چک‌های سیستمی شکست خورد. لطفاً پیام‌های بالا را بررسی کنید." >&2
fi

if [[ ${#missing_modules[@]} -gt 0 ]]; then
  echo "[doctor] بسته‌های Python زیر هنوز نصب نشده‌اند:"
  for entry in "${missing_modules[@]}"; do
    IFS='|' read -r module hint <<<"$entry"
    echo "[doctor]   - ${module}: ${hint}"
  done
else
  echo "[doctor] تمام پکیج‌های Python موردنیاز در دسترس هستند."
fi

if [[ $status -ne 0 ]]; then
  exit $status
fi

exit 0
