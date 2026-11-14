from abc import ABC, abstractmethod

from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.data_models import InputData, SolutionData
from shirokane_meetup_2026.domain.solve_stats import SolveStats


class ModelBase(ABC):
    def __init__(self, input_data: InputData, config: Config) -> None:
        self.input_data = input_data
        self.config = config

    @abstractmethod
    def build_model(self) -> None:
        pass

    @abstractmethod
    def solve(self) -> SolveStats:
        pass

    @abstractmethod
    def get_solution(self) -> SolutionData:
        pass
