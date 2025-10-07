#!/usr/bin/env bash
set -euo pipefail

TMP_DIR="$(mktemp -d)"
MSG="$TMP_DIR/msg.bin"
TOKENS="$TMP_DIR/tokens.json"
OUT="$TMP_DIR/out.bin"

# نمونه بیت‌ها (بهمراه یونیکد برای تست عبور بایت‌های UTF-8)
printf "پیام آزمایشی — sample bits\n" > "$MSG"

# Encode با MockLM (فعلاً)
neuralstego codec-encode \
  --in "$MSG" \
  --out "$TOKENS" \
  --quality.cap-per-token-bits 4 \
  --quality.top-k 64

# Decode
neuralstego codec-decode \
  --in "$TOKENS" \
  --out "$OUT"

diff -u "$MSG" "$OUT" && echo "[OK] Arithmetic codec roundtrip passed with MockLM."
