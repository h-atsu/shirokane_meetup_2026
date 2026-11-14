from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, ValidationError

from shirokane_meetup_2026.domain.config import Config, ModelName

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "default.yaml"


def test_from_yaml_applies_typed_nested_overrides() -> None:
    config = Config.from_yaml(
        DEFAULT_CONFIG,
        [
            "model_name=lagrangian",
            "time_limit=120",
            "pulp.solver=cbc",
            "lagrangian.max_iterations=2000",
        ],
    )

    assert config.model_name is ModelName.LAGRANGIAN
    assert config.time_limit == 120.0
    assert config.pulp.solver == "cbc"
    assert config.lagrangian.max_iterations == 2_000


def test_from_yaml_rejects_unknown_override() -> None:
    with pytest.raises(ConfigKeyError):
        Config.from_yaml(DEFAULT_CONFIG, ["pulp.solvar=cbc"])


def test_from_yaml_rejects_invalid_override_type() -> None:
    with pytest.raises(ValidationError):
        Config.from_yaml(DEFAULT_CONFIG, ["threads=invalid"])
