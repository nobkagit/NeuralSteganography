#!/usr/bin/env bash
set -euo pipefail

TMP_DIR="$(mktemp -d)"
MSG="$TMP_DIR/msg.txt"
TOKENS="$TMP_DIR/tokens.json"
OUT="$TMP_DIR/out.txt"

# نمونه پیام
echo "پیام آزمایشی با GPT2-fa" > "$MSG"

# Encode با GPT2-fa
neuralstego encode -p "Pa$$w0rd" -i "$MSG" -o "$TOKENS" --quality top-k 50 --quality temperature 0.7

# Decode
neuralstego decode -p "Pa$$w0rd" -i "$TOKENS" -o "$OUT"

# چک کردن تفاوت
diff -u "$MSG" "$OUT" && echo "[OK] GPT2-fa roundtrip passed."
