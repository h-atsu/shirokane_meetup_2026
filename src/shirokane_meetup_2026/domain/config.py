from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from omegaconf import OmegaConf


class ModelName(StrEnum):
    PULP = "pulp"
    LAGRANGIAN = "lagrangian"


@dataclass(slots=True)
class PulpConfig:
    solver: str = "highs"


@dataclass(slots=True)
class LagrangianConfig:
    max_iterations: int = 1_000
    initial_step_size: float = 1.0


@dataclass(slots=True)
class Config:
    instance_name: str
    experiment_name: str
    run_name: str | None
    model_name: ModelName
    penalty: float
    time_limit: float
    relative_gap_tol: float
    threads: int
    pulp: PulpConfig = field(default_factory=PulpConfig)
    lagrangian: LagrangianConfig = field(default_factory=LagrangianConfig)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        overrides: Sequence[str] = (),
    ) -> Config:
        """YAML設定を読み込み、dotlist形式の上書きを適用する。"""
        schema = OmegaConf.structured(cls)
        file_config = OmegaConf.load(path)
        override_config = OmegaConf.from_dotlist(list(overrides))
        config = OmegaConf.merge(schema, file_config, override_config)
        value = OmegaConf.to_object(config)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value
