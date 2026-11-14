from pathlib import Path

import polars as pl

from shirokane_meetup_2026.domain.data_models import InputData


class InstanceLoader:
    def __init__(self, instance_dir: str | Path) -> None:
        self.instance_dir = Path(instance_dir)

    @property
    def users(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.instance_dir / "users.parquet")

    @property
    def contents(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.instance_dir / "contents.parquet")

    @property
    def user_contents(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.instance_dir / "user_contents.parquet")

    def load_input_data(self) -> InputData:
        capacity = (
            self.users.select("capacity").collect().get_column("capacity").to_numpy()
        )

        contents = self.contents.select("weight", "minimum_delivery_count").collect()
        weight = contents.get_column("weight").to_numpy()
        minimum = contents.get_column("minimum_delivery_count").to_numpy()

        score = (
            self.user_contents.select("score")
            .collect()
            .get_column("score")
            .to_numpy()
            .reshape(capacity.size, weight.size)
        )

        return InputData(
            score=score,
            weight=weight,
            capacity=capacity,
            minimum=minimum,
        )
