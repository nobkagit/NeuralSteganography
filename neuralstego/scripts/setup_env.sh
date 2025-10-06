#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
python -m venv "${PROJECT_ROOT}/.venv"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "${PROJECT_ROOT}[dev]"
