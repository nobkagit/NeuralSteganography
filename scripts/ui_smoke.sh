#!/usr/bin/env bash
set -euo pipefail
export NEURALSTEGO_DEVICE=${NEURALSTEGO_DEVICE:-cpu}
python -m neuralstego.app.gradio_app --port 7860 --share false &
PID=$!
sleep 5
echo "[OK] Gradio app started on http://127.0.0.1:7860"
kill $PID || true
