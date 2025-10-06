#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}/src"

python -m neuralstego --help
