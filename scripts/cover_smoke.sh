#!/usr/bin/env bash
set -euo pipefail
TMP="$(mktemp -d)"
echo "گزارش سری: جلسه اصلی فردا ساعت ۱۵ برگزار می‌شود." > "$TMP/secret.txt"

neuralstego cover-generate \
  -p "Pa$$w0rd" \
  -i "$TMP/secret.txt" \
  -o "$TMP/cover.txt" \
  --seed "در یک گفت‌وگوی کوتاه درباره‌ی فناوری و اخبار روز صحبت می‌کنیم." \
  --quality top-k 60 \
  --quality temperature 0.8 \
  --chunk-bytes 256 --crc on --ecc rs --nsym 10

echo "[COVER]"; head -n 3 "$TMP/cover.txt" || true

# اگر فعلاً متن→spans آماده نیست، این بخش را روی spans.json اجرا کن
# neuralstego cover-reveal -p "Pa$$w0rd" -i "$TMP/cover.txt" -o "$TMP/recovered.txt" --seed "... "
