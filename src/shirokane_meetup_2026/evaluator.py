import numpy as np

from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.data_models import InputData, SolutionData
from shirokane_meetup_2026.domain.metric import Metrics


class SolutionEvaluator:
    def __init__(
        self,
        input_data: InputData,
        config: Config,
        solution_data: SolutionData,
    ) -> None:
        self.input_data = input_data
        self.config = config
        self.solution_data = solution_data

    def evaluate(self) -> Metrics:
        """解の目的値、配信不足、偏り、容量制約違反を評価する。"""
        self.solution_data.validate_for(self.input_data)
        assignment = self.solution_data.assignment.astype(bool, copy=False)

        delivery_count = assignment.sum(axis=0)
        shortage = np.maximum(self.input_data.minimum - delivery_count, 0)
        used_capacity = assignment @ self.input_data.weight

        total_score = float(self.input_data.score[assignment].sum(dtype=np.float64))
        total_shortage = int(shortage.sum())
        total_penalty = self.config.penalty * total_shortage
        total_deliveries = int(delivery_count.sum())
        mean_delivery_count = float(delivery_count.mean())

        return Metrics(
            objective_value=total_score - total_penalty,
            total_score=total_score,
            total_penalty=total_penalty,
            total_shortage=total_shortage,
            num_contents_below_target=int(np.count_nonzero(shortage)),
            total_deliveries=total_deliveries,
            delivery_count_cv=(
                float(delivery_count.std() / mean_delivery_count)
                if mean_delivery_count > 0
                else 0.0
            ),
            num_capacity_violations=int(
                np.count_nonzero(used_capacity > self.input_data.capacity)
            ),
        )
