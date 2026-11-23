from time import perf_counter

import numpy as np

from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.data_models import InputData, SolutionData
from shirokane_meetup_2026.domain.solve_stats import SolveStats
from shirokane_meetup_2026.models.model_base import ModelBase


class GreedyModel(ModelBase):
    """ユーザーごとにスコア密度順で選択する貪欲法。"""

    def __init__(self, input_data: InputData, config: Config) -> None:
        super().__init__(input_data, config)
        self._order: np.ndarray | None = None
        self._solution: SolutionData | None = None
        self.solve_stats: SolveStats | None = None

    def build_model(self) -> None:
        if not np.all(np.isfinite(self.input_data.score)):
            raise ValueError("score must contain only finite values")
        if not np.all(np.isfinite(self.input_data.weight)) or np.any(
            self.input_data.weight <= 0
        ):
            raise ValueError("weight must contain positive finite values")
        if not np.all(np.isfinite(self.input_data.capacity)) or np.any(
            self.input_data.capacity < 0
        ):
            raise ValueError("capacity must contain non-negative finite values")

        score_density = self.input_data.score / self.input_data.weight[None, :]
        self._order = np.argsort(-score_density, axis=1, kind="stable")
        self._solution = None
        self.solve_stats = None

    def solve(self) -> SolveStats:
        if self._order is None:
            raise RuntimeError("build_model() must be called before solve()")

        started_at = perf_counter()
        users, contents = self.input_data.score.shape
        assignment = np.zeros((users, contents), dtype=np.bool_)
        remaining_capacity = self.input_data.capacity.astype(np.float64, copy=True)
        rows = np.arange(users)

        for rank in range(contents):
            content = self._order[:, rank]
            score = self.input_data.score[rows, content]
            weight = self.input_data.weight[content]
            selected = (score > 0) & (weight <= remaining_capacity)
            assignment[rows[selected], content[selected]] = True
            remaining_capacity[selected] -= weight[selected]

        delivery_count = assignment.sum(axis=0)
        shortage = np.maximum(self.input_data.minimum - delivery_count, 0)
        objective_value = float(
            self.input_data.score[assignment].sum(dtype=np.float64)
            - self.config.penalty * shortage.sum()
        )
        self._solution = SolutionData(assignment=assignment)
        self.solve_stats = SolveStats(
            solver="Greedy-ScorePerWeight",
            status="Heuristic",
            has_solution=True,
            build_time=0.0,
            solve_time=perf_counter() - started_at,
            objective_value=objective_value,
            best_bound=None,
            relative_gap=None,
        )
        return self.solve_stats

    def get_solution(self) -> SolutionData:
        if self._solution is None:
            raise RuntimeError("solve() must be called before get_solution()")
        return self._solution
