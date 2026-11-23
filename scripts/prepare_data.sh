#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

for users in 500 30000; do
  instance_name="users_${users}_contents_100_seed_0"
  uv run python scripts/generate_data.py "${instance_name}" \
    --users "${users}" \
    --contents 100 \
    --seed 0
done
