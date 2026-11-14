from dataclasses import dataclass


@dataclass(slots=True)
class SolveStats:
    solver: str
    status: str
    has_solution: bool
    build_time: float
    solve_time: float
    objective_value: float | None
    best_bound: float | None
    relative_gap: float | None
