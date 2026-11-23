#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

batch_id="$(date +%Y%m%d_%H%M%S)"
instance_name="${MODEL_COMPARISON_INSTANCE_NAME:-users_10000_contents_100_seed_0}"
experiment_name="${MODEL_COMPARISON_EXPERIMENT_NAME:-model_comparison}"
time_limit="${MODEL_COMPARISON_TIME_LIMIT:-60}"
relative_gap_tol="${MODEL_COMPARISON_RELATIVE_GAP_TOL:-0.001}"

echo "Running model: pulp (HiGHS)"
uv run python scripts/run_experiment.py \
  --set "instance_name=${instance_name}" \
  --set model_name=pulp \
  --set pulp.solver=highs \
  --set "time_limit=${time_limit}" \
  --set "relative_gap_tol=${relative_gap_tol}" \
  --set "experiment_name=${experiment_name}" \
  --set "run_name=${batch_id}_pulp_highs"

echo "Running model: lagrangian"
uv run python scripts/run_experiment.py \
  --set "instance_name=${instance_name}" \
  --set model_name=lagrangian \
  --set "time_limit=${time_limit}" \
  --set "relative_gap_tol=${relative_gap_tol}" \
  --set "experiment_name=${experiment_name}" \
  --set "run_name=${batch_id}_lagrangian"

echo "Running model: descartes_mip"
uv run python scripts/run_experiment.py \
  --set "instance_name=${instance_name}" \
  --set model_name=descartes_mip \
  --set "time_limit=${time_limit}" \
  --set "relative_gap_tol=${relative_gap_tol}" \
  --set "experiment_name=${experiment_name}" \
  --set "run_name=${batch_id}_descartes_mip"

echo "Running model: greedy"
uv run python scripts/run_experiment.py \
  --set "instance_name=${instance_name}" \
  --set model_name=greedy \
  --set "experiment_name=${experiment_name}" \
  --set "run_name=${batch_id}_greedy"
