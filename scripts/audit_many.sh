#!/usr/bin/env bash
set -euo pipefail

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

SEEDS=(
  "در یک گفت‌وگوی کوتاه درباره‌ی فناوری صحبت می‌کنیم."
  "امروز درباره‌ی کتاب‌ها و هنر صحبت می‌کنیم."
  "وضعیت بازار و اقتصاد را مرور می‌کنیم."
  "از خاطرات سفر و طبیعت حرف می‌زنیم."
  "کمی درباره‌ی ورزش و سلامت صحبت می‌کنیم."
)

SECRETS=(
  "جلسه‌ی تیمی فردا ساعت ۱۰ برگزار می‌شود."
  "پروتوتایپ نسخه‌ی جدید آماده است؛ لطفاً بازخورد بدهید."
  "رمز عبور ایمیل تا پایان هفته تغییر خواهد کرد."
)

printf "secret\tseed\tcover\tquality\n"

for secret_idx in "${!SECRETS[@]}"; do
  secret_label=$(printf "S%02d" "$((secret_idx + 1))")
  secret_path="${TMP}/secret_${secret_idx}.txt"
  printf '%s' "${SECRETS[$secret_idx]}" > "${secret_path}"

  for seed_idx in "${!SEEDS[@]}"; do
    seed_label=$(printf "seed%02d" "$((seed_idx + 1))")
    seed_text="${SEEDS[$seed_idx]}"
    cover_path="${TMP}/cover_${secret_idx}_${seed_idx}.txt"

    cover_status="pass"
    if ! neuralstego cover-generate \
      -i "${secret_path}" \
      -o "${cover_path}" \
      --seed "${seed_text}" \
      --quality top-k 70 --quality temperature 0.8 \
      --quality-gate on \
      --max-ppl 110 \
      --max-ngram-repeat 0.30 \
      --min-ttr 0.28 \
      --regen-attempts 2 >/dev/null; then
      cover_status="fail"
    fi

    if [ ! -f "${cover_path}" ]; then
      touch "${cover_path}"
    fi

    audit_status="pass"
    if ! neuralstego quality-audit -i "${cover_path}" --max-ppl 110 --max-ngram-repeat 0.30 --min-ttr 0.28 >/dev/null; then
      audit_status="fail"
    fi

    printf "%s\t%s\t%s\t%s\n" "${secret_label}" "${seed_label}" "${cover_status}" "${audit_status}"
  done
done
