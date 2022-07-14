from __future__ import annotations

from dataclasses import dataclass
from simple_parsing.helpers.hparams.hparam import uniform, log_uniform
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters


@dataclass
class ModelHparams(HyperParameters):

    # Optimizer learning rate
    lr: float = uniform(1e-6, 1e-3, default=1e-4)


hp = ModelHparams.sample()
