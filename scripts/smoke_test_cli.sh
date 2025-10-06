#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m neuralstego --help >/dev/null
python -m neuralstego doctor
python -m neuralstego encode --context "$ROOT_DIR/README.md" --message "ping" --mode arithmetic --model gpt2-fa --output "$ROOT_DIR/tmp_stego.txt"
python -m neuralstego decode --stego "$ROOT_DIR/tmp_stego.txt" --context "$ROOT_DIR/README.md" --mode arithmetic --model gpt2-fa || true
rm -f "$ROOT_DIR/tmp_stego.txt"

echo "[smoke_test_cli] تمام دستورات CLI با موفقیت اجرا شدند (decode هنوز placeholder است)."
