#!/usr/bin/env bash
set -euo pipefail

if ! command -v neuralstego >/dev/null 2>&1; then
  echo "neuralstego CLI not found; activate your venv first." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SECRET_FILE="$TMP_DIR/secret.txt"
COVER_FILE="$TMP_DIR/cover.txt"
RECOVERED_FILE="$TMP_DIR/recovered.txt"

read -rsp "Secret message (hidden input): " SECRET
echo
read -rp "Seed text (optional, press enter to skip): " SEED_TEXT
read -rp "Enable quality gate? [y/N]: " ENABLE_GATE

printf "%s" "$SECRET" > "$SECRET_FILE"
unset SECRET  # keep the plaintext out of the shell environment

SEED_ARGS=()
if [[ -n "$SEED_TEXT" ]]; then
  SEED_ARGS=(--seed "$SEED_TEXT")
fi

QUALITY_GATE="off"
if [[ "$ENABLE_GATE" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  QUALITY_GATE="on"
fi

GEN_ARGS=(
  neuralstego cover-generate
  -i "$SECRET_FILE"
  -o "$COVER_FILE"
  --chunk-bytes 256
  --crc on
  --ecc rs
  --nsym 10
  --quality-gate "$QUALITY_GATE"
)
GEN_ARGS+=("${SEED_ARGS[@]}")
"${GEN_ARGS[@]}"

REVEAL_ARGS=(
  neuralstego cover-reveal
  -i "$COVER_FILE"
  -o "$RECOVERED_FILE"
  --crc on
  --ecc rs
  --nsym 10
)
REVEAL_ARGS+=("${SEED_ARGS[@]}")
"${REVEAL_ARGS[@]}"

if cmp -s "$SECRET_FILE" "$RECOVERED_FILE"; then
  echo "[OK] Secret round-trip succeeded."
else
  echo "[WARN] Round-trip mismatch; inspect $RECOVERED_FILE" >&2
  exit 1
fi

echo "Cover saved at $COVER_FILE"
echo "Recovered secret stored at $RECOVERED_FILE"
