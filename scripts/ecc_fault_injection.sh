#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

MESSAGE="$TMPDIR/message.bin"
python - <<'PY' > "$MESSAGE"
import os
import secrets

data = secrets.token_bytes(8192)
os.write(1, data)
PY

CONFIGS=(
  "chunk=128 crc=on ecc=none nsym=0"
  "chunk=128 crc=on ecc=rs nsym=10"
  "chunk=512 crc=off ecc=none nsym=0"
  "chunk=512 crc=on ecc=rs nsym=10"
)

for cfg in "${CONFIGS[@]}"; do
  eval "$cfg"
  echo "=== Configuration: chunk=$chunk crc=$crc ecc=$ecc nsym=$nsym ==="
  OUT="$TMPDIR/out_${chunk}_${crc}_${ecc}.json"
  if ! python -m neuralstego.cli encode \
    --lm mock \
    --chunk-bytes "$chunk" \
    --crc "$crc" \
    --ecc "$ecc" \
    --nsym "$nsym" \
    --seed-text "متن پایه." \
    --input "$MESSAGE" \
    --output "$OUT"; then
    echo "Encode failed (likely missing dependencies)" >&2
    continue
  fi

  SCENARIO_A="$TMPDIR/scenario_a_${chunk}_${crc}_${ecc}.json"
  python - "$OUT" "$SCENARIO_A" <<'PY'
import json
import sys

src, dst = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as fh:
    data = json.load(fh)

alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
if data["spans"]:
    span_bytes = bytes(int(v) % 256 for v in data["spans"][0])
    packet = json.loads(span_bytes.decode("utf-8"))
    payload = bytearray(packet["payload"].encode("ascii"))
    if payload:
        current = payload[0]
        idx = alphabet.find(bytes([current]))
        if idx == -1:
            payload[0] = alphabet[0]
        else:
            payload[0] = alphabet[(idx + 1) % len(alphabet)]
        packet["payload"] = payload.decode("ascii")
        mutated = json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")
        data["spans"][0] = [int(b) for b in mutated]

with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh)
PY

  echo "Scenario A (bit flip in first span):"
  if python -m neuralstego.cli decode \
    --lm mock \
    --chunk-bytes "$chunk" \
    --crc "$crc" \
    --ecc "$ecc" \
    --nsym "$nsym" \
    --seed-text "متن پایه." \
    --input "$SCENARIO_A" \
    --output "$TMPDIR/decoded_a"; then
    echo "  -> decode succeeded"
  else
    echo "  -> decode failed"
  fi

  SCENARIO_B="$TMPDIR/scenario_b_${chunk}_${crc}_${ecc}.json"
  python - "$OUT" "$SCENARIO_B" <<'PY'
import json
import sys

src, dst = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as fh:
    data = json.load(fh)

if data["spans"]:
    mid = len(data["spans"]) // 2
    data["spans"].pop(mid)

with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh)
PY

  echo "Scenario B (remove middle span):"
  if python -m neuralstego.cli decode \
    --lm mock \
    --chunk-bytes "$chunk" \
    --crc "$crc" \
    --ecc "$ecc" \
    --nsym "$nsym" \
    --seed-text "متن پایه." \
    --input "$SCENARIO_B" \
    --output "$TMPDIR/decoded_b"; then
    echo "  -> decode succeeded"
  else
    echo "  -> decode failed"
  fi
  echo

done
