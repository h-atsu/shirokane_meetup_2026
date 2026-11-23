#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

batch_id="$(date +%Y%m%d_%H%M%S)"
instance_name="users_500_contents_100_seed_0"
experiment_name="small_experiment"
time_limit=60
relative_gap_tol=0.0001
lagrangian_max_iterations=1000000000

echo "Running model: greedy"
uv run python scripts/run_experiment.py \
  --set "instance_name=${instance_name}" \
  --set model_name=greedy \
  --set penalty=10 \
  --set "experiment_name=${experiment_name}" \
  --set "run_name=${batch_id}_greedy"

for rho in 0 10 100 1000; do
  echo "Running model: pulp (HiGHS, rho=${rho})"
  uv run python scripts/run_experiment.py \
    --set "instance_name=${instance_name}" \
    --set model_name=pulp \
    --set pulp.solver=highs \
    --set "penalty=${rho}" \
    --set "experiment_name=${experiment_name}" \
    --set "run_name=${batch_id}_pulp_highs_rho_${rho}"

  echo "Running model: lagrangian (rho=${rho})"
  uv run python scripts/run_experiment.py \
    --set "instance_name=${instance_name}" \
    --set model_name=lagrangian \
    --set "penalty=${rho}" \
    --set "time_limit=${time_limit}" \
    --set "relative_gap_tol=${relative_gap_tol}" \
    --set "lagrangian.max_iterations=${lagrangian_max_iterations}" \
    --set "experiment_name=${experiment_name}" \
    --set "run_name=${batch_id}_lagrangian_rho_${rho}"
done
