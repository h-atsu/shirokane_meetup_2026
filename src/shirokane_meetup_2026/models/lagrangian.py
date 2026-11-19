import logging
from time import perf_counter

import numpy as np

from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.data_models import InputData, SolutionData
from shirokane_meetup_2026.domain.solve_stats import SolveStats
from shirokane_meetup_2026.models.model_base import ModelBase

logger = logging.getLogger(__name__)


class LagrangianModel(ModelBase):
    """最低配信回数制約を緩和し、ユーザー単位に分解して求解する。"""

    _MAX_BATCH_WORK_BYTES = 128 * 1024 * 1024
    _STALL_PATIENCE = 10

    def __init__(self, input_data: InputData, config: Config) -> None:
        super().__init__(input_data, config)
        self._weight: np.ndarray | None = None
        self._capacity: np.ndarray | None = None
        self._batch_size = 0
        self._solution: SolutionData | None = None
        self.solve_stats: SolveStats | None = None

    def build_model(self) -> None:
        """動的計画法で利用する整数配列とバッチサイズを準備する。"""
        users, contents = self.input_data.score.shape
        if users == 0 or contents == 0:
            raise ValueError("score must contain at least one user and one content")
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
        if not np.all(np.isfinite(self.input_data.minimum)) or np.any(
            self.input_data.minimum < 0
        ):
            raise ValueError("minimum must contain non-negative finite values")
        if not np.all(self.input_data.weight == np.floor(self.input_data.weight)):
            raise ValueError("LagrangianModel requires integer weights")
        if not np.all(self.input_data.capacity == np.floor(self.input_data.capacity)):
            raise ValueError("LagrangianModel requires integer capacities")
        if self.config.penalty < 0:
            raise ValueError("penalty must be non-negative")
        if self.config.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.config.relative_gap_tol < 0:
            raise ValueError("relative_gap_tol must be non-negative")
        if self.config.lagrangian.max_iterations < 1:
            raise ValueError("lagrangian.max_iterations must be positive")
        if self.config.lagrangian.initial_step_size <= 0:
            raise ValueError("lagrangian.initial_step_size must be positive")

        self._weight = self.input_data.weight.astype(np.int64, copy=False)
        self._capacity = self.input_data.capacity.astype(np.int64, copy=False)
        capacity_states = int(self._capacity.max()) + 1
        bytes_per_user = max(1, contents * capacity_states)
        self._batch_size = max(
            1,
            min(users, self._MAX_BATCH_WORK_BYTES // bytes_per_user),
        )
        self._solution = None
        self.solve_stats = None

    def solve(self) -> SolveStats:
        """射影劣勾配法でラグランジュ双対問題を反復求解する。"""
        if self._weight is None or self._capacity is None:
            raise RuntimeError("build_model() must be called before solve()")

        started_at = perf_counter()
        users, contents = self.input_data.score.shape
        multiplier = np.zeros(contents, dtype=np.float64)
        best_assignment = np.zeros((users, contents), dtype=np.bool_)
        best_lower = self._primal_objective(best_assignment)
        best_upper = float("inf")
        best_gap = float("inf")
        step_coefficient = float(self.config.lagrangian.initial_step_size)
        stalled_iterations = 0
        status = "IterationLimit"

        for iteration in range(self.config.lagrangian.max_iterations):
            if iteration > 0 and perf_counter() - started_at >= self.config.time_limit:
                status = "TimeLimit"
                break

            assignment, subproblem_value = self._solve_subproblems(multiplier)
            delivery_count = assignment.sum(axis=0, dtype=np.float64)
            dual_value = float(subproblem_value - multiplier @ self.input_data.minimum)
            previous_upper = best_upper
            best_upper = min(best_upper, dual_value)

            primal_value = self._primal_objective(assignment, delivery_count)
            if primal_value > best_lower:
                best_lower = primal_value
                best_assignment = assignment.copy()

            best_gap = self._relative_gap(best_lower, best_upper)
            logger.info(
                "iteration=%d elapsed=%.3f "
                "current_lower=%.6f best_lower=%.6f "
                "current_upper=%.6f best_upper=%.6f "
                "relative_gap=%.6f total_shortage=%.0f",
                iteration + 1,
                perf_counter() - started_at,
                primal_value,
                best_lower,
                dual_value,
                best_upper,
                best_gap,
                np.maximum(self.input_data.minimum - delivery_count, 0).sum(),
            )
            if best_gap <= self.config.relative_gap_tol:
                status = "GapLimit"
                break

            tolerance = 1e-9 * max(1.0, abs(best_upper))
            if not np.isfinite(previous_upper) or (
                dual_value < previous_upper - tolerance
            ):
                stalled_iterations = 0
            else:
                stalled_iterations += 1
                if stalled_iterations >= self._STALL_PATIENCE:
                    step_coefficient *= 0.9
                    stalled_iterations = 0

            gradient = delivery_count - self.input_data.minimum
            projected_gradient = gradient.copy()
            projected_gradient[(multiplier <= 0) & (gradient > 0)] = 0
            projected_gradient[(multiplier >= self.config.penalty) & (gradient < 0)] = 0
            norm_squared = float(projected_gradient @ projected_gradient)
            if norm_squared <= 1e-20:
                status = "Converged"
                break

            step_size = (
                step_coefficient * max(dual_value - best_lower, 0.0) / norm_squared
            )
            if step_size <= 0:
                status = "Converged"
                break
            multiplier = np.clip(
                multiplier - step_size * gradient,
                0.0,
                self.config.penalty,
            )
        else:
            status = "IterationLimit"

        solve_time = perf_counter() - started_at
        best_upper = max(best_upper, best_lower)
        best_gap = self._relative_gap(best_lower, best_upper)
        self._solution = SolutionData(assignment=best_assignment)
        self.solve_stats = SolveStats(
            solver="Lagrangian-NumPy",
            status=status,
            has_solution=True,
            build_time=0.0,
            solve_time=solve_time,
            objective_value=best_lower,
            best_bound=best_upper,
            relative_gap=best_gap,
        )
        return self.solve_stats

    def get_solution(self) -> SolutionData:
        if self._solution is None:
            raise RuntimeError("solve() must be called before get_solution()")
        return self._solution

    def _solve_subproblems(
        self,
        multiplier: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """全ユーザーの0-1ナップサック部分問題をバッチ求解する。"""
        if self._weight is None or self._capacity is None:
            raise RuntimeError("build_model() must be called before solve()")

        users, contents = self.input_data.score.shape
        max_capacity = int(self._capacity.max())
        capacity_axis = np.arange(max_capacity + 1, dtype=np.int64)
        assignment = np.zeros((users, contents), dtype=np.bool_)
        total_value = 0.0

        for start in range(0, users, self._batch_size):
            stop = min(start + self._batch_size, users)
            batch_users = stop - start
            rows = np.arange(batch_users)
            score = self.input_data.score[start:stop]
            capacity = self._capacity[start:stop]
            value = np.zeros((batch_users, max_capacity + 1), dtype=np.float64)
            take = np.zeros(
                (contents, batch_users, max_capacity + 1),
                dtype=np.bool_,
            )

            for content in range(contents):
                source_capacity = capacity_axis - self._weight[content]
                feasible = source_capacity >= 0
                source_capacity = np.maximum(source_capacity, 0)
                candidate = (
                    value[:, source_capacity]
                    + score[:, content, None]
                    + multiplier[content]
                )
                candidate[:, ~feasible] = -np.inf
                improve = candidate > value + 1e-12
                take[content] = improve
                value = np.where(improve, candidate, value)

            remaining_capacity = capacity.copy()
            for content in range(contents - 1, -1, -1):
                selected = take[content, rows, remaining_capacity]
                assignment[start:stop, content] = selected
                remaining_capacity -= selected * self._weight[content]
            total_value += float(value[rows, capacity].sum(dtype=np.float64))

        return assignment, total_value

    def _primal_objective(
        self,
        assignment: np.ndarray,
        delivery_count: np.ndarray | None = None,
    ) -> float:
        if delivery_count is None:
            delivery_count = assignment.sum(axis=0, dtype=np.float64)
        score = float(self.input_data.score[assignment].sum(dtype=np.float64))
        shortage = np.maximum(self.input_data.minimum - delivery_count, 0)
        return score - self.config.penalty * float(shortage.sum())

    @staticmethod
    def _relative_gap(lower_bound: float, upper_bound: float) -> float:
        if not np.isfinite(upper_bound):
            return float("inf")
        return max(0.0, upper_bound - lower_bound) / max(1.0, abs(lower_bound))
