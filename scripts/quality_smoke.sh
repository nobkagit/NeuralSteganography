#!/usr/bin/env bash
set -euo pipefail

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "گزارش سری: جلسه اصلی فردا ساعت ۱۵ برگزار می‌شود." > "${TMP}/secret.txt"

neuralstego cover-generate \
  -p "Pa$$w0rd" \
  -i "${TMP}/secret.txt" \
  -o "${TMP}/cover.txt" \
  --seed "در یک گفت‌وگوی کوتاه درباره‌ی فناوری و اخبار روز صحبت می‌کنیم." \
  --quality top-k 60 --quality temperature 0.8 \
  --quality-gate on \
  --max-ppl 100 \
  --max-ngram-repeat 0.25 \
  --min-ttr 0.30 \
  --regen-attempts 2

neuralstego quality-audit -i "${TMP}/cover.txt"

word_count=$(wc -w < "${TMP}/cover.txt")
echo "[OK] Quality smoke completed: ${word_count} words"
