#!/usr/bin/env bash
set -euo pipefail
export NEURALSTEGO_DEVICE=${NEURALSTEGO_DEVICE:-auto}
# پورت دلخواه: 7860
python -m neuralstego.app.gradio_app --port "${PORT:-7860}" --share "${SHARE:-false}"
