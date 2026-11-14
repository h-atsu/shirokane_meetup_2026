import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.data_models import InputData, SolutionData
from shirokane_meetup_2026.domain.metric import Metrics
from shirokane_meetup_2026.domain.solve_stats import SolveStats
from shirokane_meetup_2026.loader import InstanceLoader


class ResultWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: Config) -> None:
        OmegaConf.save(
            config=OmegaConf.structured(config),
            f=self.output_dir / "config.yaml",
        )

    def write_solve_stats(self, solve_stats: SolveStats) -> None:
        self._write_json("solve_stats.json", asdict(solve_stats))

    def write_solution(
        self,
        loader: InstanceLoader,
        input_data: InputData,
        solution: SolutionData,
        metrics: Metrics,
    ) -> None:
        self._write_json("metrics.json", asdict(metrics))
        self._write_result_parquet(loader, input_data, solution)

    def _write_result_parquet(
        self,
        loader: InstanceLoader,
        input_data: InputData,
        solution: SolutionData,
    ) -> None:
        assignment = solution.assignment.astype(bool, copy=False)
        user_ids = loader.users.select("user_id").collect().get_column("user_id")
        content_ids = (
            loader.contents.select("content_id").collect().get_column("content_id")
        )
        content_id_values = content_ids.to_numpy()

        selected_content_ids = [content_id_values[row].tolist() for row in assignment]
        used_capacity = assignment @ input_data.weight
        user_score = (input_data.score * assignment).sum(
            axis=1,
            dtype=np.float64,
        )

        pl.DataFrame(
            {
                "user_id": user_ids,
                "content_ids": pl.Series(
                    selected_content_ids,
                    dtype=pl.List(content_ids.dtype),
                ),
                "num_deliveries": assignment.sum(axis=1),
                "used_capacity": used_capacity,
                "total_score": user_score,
            }
        ).write_parquet(
            self.output_dir / "df_result.parquet",
            compression="zstd",
        )

        delivery_count = assignment.sum(axis=0)
        shortage = np.maximum(input_data.minimum - delivery_count, 0)
        pl.DataFrame(
            {
                "content_id": content_ids,
                "delivery_count": delivery_count,
                "minimum_delivery_count": input_data.minimum,
                "shortage": shortage,
            }
        ).write_parquet(
            self.output_dir / "df_content_result.parquet",
            compression="zstd",
        )

    def _write_json(self, filename: str, value: Any) -> None:
        (self.output_dir / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
