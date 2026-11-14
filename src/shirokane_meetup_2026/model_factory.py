from typing import assert_never

from shirokane_meetup_2026.domain.config import Config, ModelName
from shirokane_meetup_2026.domain.data_models import InputData
from shirokane_meetup_2026.models.lagrange_decomposition import LagrangianModel
from shirokane_meetup_2026.models.model_base import ModelBase
from shirokane_meetup_2026.models.pulp_model import PulpModel


def create_model(input_data: InputData, config: Config) -> ModelBase:
    match config.model_name:
        case ModelName.PULP:
            return PulpModel(input_data, config)
        case ModelName.LAGRANGIAN:
            return LagrangianModel(input_data, config)
        case unknown:
            assert_never(unknown)
