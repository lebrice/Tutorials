from __future__ import annotations

from typing import Union
import numpy as np

Observation = Union[np.ndarray, list[float]]
Action = Union[int, float]


class Environment:
    def reset(self) -> Observation:
        ...

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        ...


env: Environment = ...
env.step("123")
bob = env.reset()
