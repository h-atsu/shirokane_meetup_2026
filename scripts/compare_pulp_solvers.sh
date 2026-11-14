#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

batch_id="$(date +%Y%m%d_%H%M%S)"
experiment_name="pulp_solver_comparison"
solvers=(cbc highs scip)

for solver in "${solvers[@]}"; do
  echo "Running solver: ${solver}"
  uv run python scripts/run_experiment.py \
    --set instance_name=users_10000_contents_100_seed_0 \
    --set model_name=pulp \
    --set "pulp.solver=${solver}" \
    --set "experiment_name=${experiment_name}" \
    --set "run_name=${batch_id}_${solver}"
done
