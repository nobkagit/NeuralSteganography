#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

input_file="${tmp_dir}/secret.txt"
envelope_file="${tmp_dir}/secret.enc"
output_file="${tmp_dir}/secret.out"

cat >"${input_file}" <<'EOF'
سلام دنیا!
Hidden message for NeuralStego.
EOF

neuralstego encrypt -p "Pa$$w0rd" -i "${input_file}" -o "${envelope_file}"
neuralstego decrypt -p "Pa$$w0rd" -i "${envelope_file}" -o "${output_file}"

if diff -q "${input_file}" "${output_file}" >/dev/null; then
  echo "OK: crypto round-trip succeeded"
else
  echo "ERROR: encrypted round-trip mismatch" >&2
  exit 1
fi
