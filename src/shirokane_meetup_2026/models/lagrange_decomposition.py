from shirokane_meetup_2026.domain.data_models import SolutionData
from shirokane_meetup_2026.domain.solve_stats import SolveStats
from shirokane_meetup_2026.models.model_base import ModelBase


class LagrangianModel(ModelBase):
    """Lagrangian decomposition implementation placeholder."""

    def build_model(self) -> None:
        raise NotImplementedError(
            "Lagrangian model construction is not implemented yet"
        )

    def solve(self) -> SolveStats:
        raise NotImplementedError("Lagrangian solve is not implemented yet")

    def get_solution(self) -> SolutionData:
        raise NotImplementedError(
            "Lagrangian solution extraction is not implemented yet"
        )
