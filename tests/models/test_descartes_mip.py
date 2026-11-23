import logging

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
from shirokane_meetup_2026.models.descartes_mip import DescartesMipModel


def _config() -> Config:
    return Config(
        instance_name="test",
        experiment_name="test",
        run_name="test",
        model_name=ModelName.DESCARTES_MIP,
        penalty=10.0,
        time_limit=2.0,
        relative_gap_tol=0.001,
        threads=1,
        pulp=PulpConfig(),
        lagrangian=LagrangianConfig(),
        descartes_mip=DescartesMipConfig(
            max_iterations=100,
            polyak_factor=1.0,
            parallel=False,
        ),
    )


def test_solves_decomposed_problem_and_reports_bounds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    input_data = InputData(
        score=np.array(
            [
                [10.0, 1.0],
                [9.0, 2.0],
                [1.0, 10.0],
            ]
        ),
        weight=np.array([1, 1]),
        capacity=np.array([1, 1, 1]),
        minimum=np.array([2, 1]),
    )
    config = _config()
    model = DescartesMipModel(input_data, config)

    caplog.set_level(logging.INFO)
    model.build_model()
    stats = model.solve()
    solution = model.get_solution()
    metrics = SolutionEvaluator(input_data, config, solution).evaluate()

    assert stats.status == "GapLimit"
    assert stats.has_solution
    assert stats.objective_value == pytest.approx(29.0)
    assert stats.best_bound == pytest.approx(29.0)
    assert stats.relative_gap == pytest.approx(0.0)
    assert metrics.objective_value == pytest.approx(stats.objective_value)
    assert metrics.total_shortage == 0
    assert metrics.num_capacity_violations == 0
    assert "iteration=1" in caplog.text


def test_requires_positive_relaxation_penalty() -> None:
    input_data = InputData(
        score=np.array([[1.0]]),
        weight=np.array([1]),
        capacity=np.array([1]),
        minimum=np.array([0]),
    )
    config = _config()
    config.penalty = 0

    with pytest.raises(ValueError, match="positive penalty"):
        DescartesMipModel(input_data, config).build_model()
