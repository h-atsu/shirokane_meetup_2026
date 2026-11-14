from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(slots=True)
class InputDataframes:
    users: pl.DataFrame
    contents: pl.DataFrame
    user_contents: pl.DataFrame


@dataclass(slots=True)
class InputData:
    score: np.ndarray  # [user, content]
    weight: np.ndarray  # [content]
    capacity: np.ndarray  # [user]
    minimum: np.ndarray  # [content]

    def __post_init__(self) -> None:
        if self.score.ndim != 2:
            raise ValueError(f"score must be 2-dimensional, got {self.score.shape}")

        users, contents = self.score.shape
        expected_shapes = {
            "weight": (contents,),
            "capacity": (users,),
            "minimum": (contents,),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = getattr(self, name).shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {actual_shape}"
                )


@dataclass(slots=True)
class SolutionData:
    assignment: np.ndarray  # [user, content]

    def __post_init__(self) -> None:
        if self.assignment.ndim != 2:
            raise ValueError(
                f"assignment must be 2-dimensional, got {self.assignment.shape}"
            )
        if np.any((self.assignment != 0) & (self.assignment != 1)):
            raise ValueError("assignment must contain only 0 or 1")

    def validate_for(self, data: InputData) -> None:
        if self.assignment.shape != data.score.shape:
            raise ValueError(
                f"assignment must have shape {data.score.shape}, "
                f"got {self.assignment.shape}"
            )
