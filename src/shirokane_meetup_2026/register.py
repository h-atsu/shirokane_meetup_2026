"""最適化実行の設定、評価指標、成果物をMLflowへ登録する。"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow

from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.metric import Metrics
from shirokane_meetup_2026.domain.solve_stats import SolveStats


def _flatten_params(
    values: dict[str, Any],
    parent_key: str = "",
) -> dict[str, Any]:
    """ネストした設定をドット区切りのMLflowパラメータへ変換する。"""
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        parameter_name = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, parameter_name))
        elif isinstance(value, list):
            flattened[parameter_name] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            flattened[parameter_name] = "null"
        elif isinstance(value, str | int | float | bool):
            flattened[parameter_name] = value
        else:
            flattened[parameter_name] = str(value)
    return flattened


class MLflowRegister:
    def __init__(self, tracking_uri: str | None = None) -> None:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

    def register(
        self,
        config: Config,
        solve_stats: SolveStats,
        metrics: Metrics | None,
        artifact_dir: str | Path,
    ) -> str:
        """最適化実行のパラメータ、指標、成果物をMLflowへ登録する。"""
        mlflow.set_experiment(experiment_name=config.experiment_name)
        with mlflow.start_run(run_name=config.run_name) as active_run:
            mlflow.log_params(_flatten_params(asdict(config)))
            mlflow.set_tags(
                {
                    "optimization_status": solve_stats.status,
                    "has_solution": str(solve_stats.has_solution).lower(),
                    "solver": solve_stats.solver,
                }
            )

            run_metrics = {
                "build_time": solve_stats.build_time,
                "solve_time": solve_stats.solve_time,
            }
            if solve_stats.relative_gap is not None:
                run_metrics["relative_gap"] = solve_stats.relative_gap
            if metrics is not None:
                run_metrics.update(
                    {name: float(value) for name, value in asdict(metrics).items()}
                )
            mlflow.log_metrics(run_metrics)
            mlflow.log_artifacts(local_dir=str(Path(artifact_dir)))
            return active_run.info.run_id
