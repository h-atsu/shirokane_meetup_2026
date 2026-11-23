import numpy as np
import pytest

from shirokane_meetup_2026.domain.config import (
    Config,
    DescartesMipConfig,
    LagrangianConfig,
    ModelName,
    PulpConfig,
)
from shirokane_meetup_2026.domain.data_models import InputData
from shirokane_meetup_2026.evaluator import SolutionEvaluator
from shirokane_meetup_2026.models.greedy import GreedyModel


def _config() -> Config:
    return Config(
        instance_name="test",
        experiment_name="test",
        run_name="test",
        model_name=ModelName.GREEDY,
        penalty=10.0,
        time_limit=1.0,
        relative_gap_tol=0.001,
        threads=1,
        pulp=PulpConfig(),
        lagrangian=LagrangianConfig(),
        descartes_mip=DescartesMipConfig(),
    )


def test_selects_contents_by_score_per_weight() -> None:
    input_data = InputData(
        score=np.array([[10.0, 9.0, 6.0]]),
        weight=np.array([6, 5, 4]),
        capacity=np.array([10]),
        minimum=np.array([0, 0, 0]),
    )
    config = _config()
    model = GreedyModel(input_data, config)

    model.build_model()
    stats = model.solve()
    solution = model.get_solution()
    metrics = SolutionEvaluator(input_data, config, solution).evaluate()

    np.testing.assert_array_equal(solution.assignment, [[False, True, True]])
    assert stats.status == "Heuristic"
    assert stats.objective_value == pytest.approx(15.0)
    assert stats.best_bound is None
    assert stats.relative_gap is None
    assert metrics.num_capacity_violations == 0


def test_skips_non_positive_scores() -> None:
    input_data = InputData(
        score=np.array([[1.0, 0.0, -1.0]]),
        weight=np.array([1, 1, 1]),
        capacity=np.array([3]),
        minimum=np.array([0, 0, 0]),
    )
    model = GreedyModel(input_data, _config())

    model.build_model()
    model.solve()

    np.testing.assert_array_equal(
        model.get_solution().assignment,
        [[True, False, False]],
    )
