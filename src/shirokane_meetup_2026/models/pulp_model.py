from itertools import chain

import numpy as np
import pulp

from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.data_models import InputData, SolutionData
from shirokane_meetup_2026.domain.solve_stats import SolveStats
from shirokane_meetup_2026.models.model_base import ModelBase


class PulpModel(ModelBase):
    def __init__(self, input_data: InputData, config: Config) -> None:
        super().__init__(input_data, config)
        self.problem = pulp.LpProblem("content_delivery", pulp.LpMaximize)
        self.assignment_variables: list[list[pulp.LpVariable]] = []
        self.shortage_variables: list[pulp.LpVariable] = []
        self.solve_stats: SolveStats | None = None

    def build_model(self) -> None:
        self._add_variable()
        self._add_constraints()
        self._add_objective()

    def _add_variable(self) -> None:
        users, contents = self.input_data.score.shape
        self.assignment_variables = self.problem.add_variable_matrix(
            "x",
            (range(users), range(contents)),
            cat="Binary",
        )
        self.shortage_variables = self.problem.add_variable_matrix(
            "s",
            range(contents),
            lowBound=0,
        )

    def _add_constraints(self) -> None:
        users, contents = self.input_data.score.shape

        for user in range(users):
            used_capacity = pulp.lpSum(
                self.assignment_variables[user][content]
                * float(self.input_data.weight[content])
                for content in range(contents)
            )
            self.problem.add(
                used_capacity <= float(self.input_data.capacity[user]),
                name=f"capacity_{user}",
            )

        for content in range(contents):
            delivery_count = pulp.lpSum(
                self.assignment_variables[user][content] for user in range(users)
            )
            self.problem.add(
                delivery_count + self.shortage_variables[content]
                >= float(self.input_data.minimum[content]),
                name=f"minimum_delivery_{content}",
            )

    def _add_objective(self) -> None:
        users, contents = self.input_data.score.shape
        score_terms = (
            self.assignment_variables[user][content]
            * float(self.input_data.score[user, content])
            for user in range(users)
            for content in range(contents)
        )
        penalty_terms = (
            -self.config.penalty * variable for variable in self.shortage_variables
        )
        self.problem.setObjective(pulp.lpSum(chain(score_terms, penalty_terms)))

    def solve(self) -> SolveStats:
        stats = self.problem.solve(self._create_solver())
        self.solve_stats = SolveStats(
            solver=stats.solver,
            status=stats.status_str,
            has_solution=stats.has_solution,
            build_time=0.0,
            solve_time=stats.time,
            objective_value=stats.objective,
            best_bound=stats.best_bound,
            relative_gap=stats.gap_rel,
        )
        return self.solve_stats

    def _create_solver(self) -> pulp.LpSolver:
        match self.config.pulp.solver.lower():
            case "highs":
                solver = pulp.HiGHS(
                    msg=True,
                    timeLimit=self.config.time_limit,
                    gapRel=self.config.relative_gap,
                    threads=self.config.threads,
                )
            case "cbc" | "coin":
                solver = pulp.COIN_CMD(
                    msg=True,
                    timeLimit=self.config.time_limit,
                    gapRel=self.config.relative_gap,
                    threads=self.config.threads,
                )
            case "scip":
                solver = pulp.SCIP_PY(
                    msg=True,
                    timeLimit=self.config.time_limit,
                    gapRel=self.config.relative_gap,
                    threads=self.config.threads,
                )
            case solver_name:
                raise ValueError(f"Unsupported PuLP solver: {solver_name}")

        if not solver.available():
            raise RuntimeError(
                f"PuLP solver is not available: {self.config.pulp.solver}"
            )
        return solver

    def get_solution(self) -> SolutionData:
        if self.solve_stats is None:
            raise RuntimeError("solve() must be called before get_solution()")
        if not self.solve_stats.has_solution:
            raise RuntimeError("The solver did not return a feasible solution")

        users, contents = self.input_data.score.shape
        assignment = np.fromiter(
            (
                (self.assignment_variables[user][content].value() or 0.0) > 0.5
                for user in range(users)
                for content in range(contents)
            ),
            dtype=np.bool_,
            count=users * contents,
        ).reshape(users, contents)
        return SolutionData(assignment=assignment)
