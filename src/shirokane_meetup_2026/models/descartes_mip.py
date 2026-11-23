import logging

import numpy as np
from descartes_mip import (
    IterationRecord,
    LagrangianResult,
    Model,
    Variable,
    VariableType,
    quicksum,
)

from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.data_models import InputData, SolutionData
from shirokane_meetup_2026.domain.solve_stats import SolveStats
from shirokane_meetup_2026.models.model_base import ModelBase

logger = logging.getLogger(__name__)


class DescartesMipModel(ModelBase):
    """descartes-mipによる自動ラグランジュ分解モデル。"""

    def __init__(self, input_data: InputData, config: Config) -> None:
        super().__init__(input_data, config)
        self.problem = Model("content_delivery")
        self.assignment_variables: list[list[Variable]] = []
        self.result: LagrangianResult | None = None
        self.solve_stats: SolveStats | None = None
        self._gap_reached = False

    def build_model(self) -> None:
        self._validate_config()
        users, contents = self.input_data.score.shape
        self.assignment_variables = [
            [
                self.problem.add_var(
                    f"x_{user}_{content}",
                    vartype=VariableType.BINARY,
                )
                for content in range(contents)
            ]
            for user in range(users)
        ]

        for user in range(users):
            self.problem.add_constraint(
                quicksum(
                    float(self.input_data.weight[content])
                    * self.assignment_variables[user][content]
                    for content in range(contents)
                )
                <= float(self.input_data.capacity[user]),
                name=f"capacity_{user}",
            )

        minimum_constraints = [
            self.problem.add_constraint(
                quicksum(
                    self.assignment_variables[user][content] for user in range(users)
                )
                >= float(self.input_data.minimum[content]),
                name=f"minimum_delivery_{content}",
            )
            for content in range(contents)
        ]
        self.problem.relax(minimum_constraints, penalty=self.config.penalty)
        self.problem.maximize(
            quicksum(
                float(self.input_data.score[user, content])
                * self.assignment_variables[user][content]
                for user in range(users)
                for content in range(contents)
            )
        )
        self.result = None
        self.solve_stats = None

    def solve(self) -> SolveStats:
        if not self.assignment_variables:
            raise RuntimeError("build_model() must be called before solve()")

        settings = self.config.descartes_mip
        self._gap_reached = False
        parallel = settings.parallel
        self.result = self.problem.solve(
            highs_options={"threads": 1 if parallel else self.config.threads},
            max_iterations=settings.max_iterations,
            time_limit=self.config.time_limit,
            parallel=parallel,
            workers=self.config.threads if parallel else None,
            polyak_factor=settings.polyak_factor,
            callback=self._iteration_callback,
        )
        has_solution = self.result.soft_feasible
        self.solve_stats = SolveStats(
            solver="descartes-mip/HiGHS",
            status=self._status(self.result),
            has_solution=has_solution,
            build_time=0.0,
            solve_time=self.result.elapsed_time,
            objective_value=(self.result.best_primal_bound if has_solution else None),
            best_bound=(
                self.result.best_dual_bound
                if self.result.has_certified_bounds
                else None
            ),
            relative_gap=(
                self.result.relative_gap if self.result.has_certified_bounds else None
            ),
        )
        return self.solve_stats

    def get_solution(self) -> SolutionData:
        if self.result is None or self.solve_stats is None:
            raise RuntimeError("solve() must be called before get_solution()")
        if not self.solve_stats.has_solution:
            raise RuntimeError("descartes-mip did not return a feasible solution")

        users, contents = self.input_data.score.shape
        assignment = np.fromiter(
            (
                self.result.variable_values[variable.name] > 0.5
                for row in self.assignment_variables
                for variable in row
            ),
            dtype=np.bool_,
            count=users * contents,
        ).reshape(users, contents)
        return SolutionData(assignment=assignment)

    def _iteration_callback(self, record: IterationRecord) -> bool:
        relative_gap = record.absolute_gap / max(
            1.0,
            abs(record.best_primal_bound),
            abs(record.best_dual_bound),
        )
        logger.info(
            "iteration=%d elapsed=%.3f "
            "current_lower=%.6f best_lower=%.6f "
            "current_upper=%.6f best_upper=%.6f "
            "relative_gap=%.6f total_shortage=%.0f step_size=%.6f",
            record.iteration + 1,
            record.elapsed_time,
            record.primal_bound,
            record.best_primal_bound,
            record.dual_bound,
            record.best_dual_bound,
            relative_gap,
            sum(record.violations.values()),
            record.step_size,
        )
        self._gap_reached = relative_gap <= self.config.relative_gap_tol
        return self._gap_reached

    def _validate_config(self) -> None:
        settings = self.config.descartes_mip
        if self.config.penalty <= 0:
            raise ValueError("descartes-mip requires a positive penalty")
        if self.config.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.config.relative_gap_tol < 0:
            raise ValueError("relative_gap_tol must be non-negative")
        if self.config.threads < 1:
            raise ValueError("threads must be positive")
        if settings.max_iterations < 1:
            raise ValueError("descartes_mip.max_iterations must be positive")
        if not 0 < settings.polyak_factor <= 2:
            raise ValueError("descartes_mip.polyak_factor must be in (0, 2]")

    def _status(self, result: LagrangianResult) -> str:
        if self._gap_reached:
            return "GapLimit"
        return {
            "max_iterations": "IterationLimit",
            "time_limit": "TimeLimit",
            "callback": "Callback",
        }[result.termination_reason]
