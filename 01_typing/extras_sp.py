from __future__ import annotations

from dataclasses import dataclass
from simple_parsing import ArgumentParser
from simple_parsing.helpers.hparams.hparam import uniform, log_uniform
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters


@dataclass
class ModelHparams(HyperParameters):
    """Model Hyper-Parameters."""

    # Optimizer learning rate
    lr: float = uniform(1e-6, 1e-3, default=1e-4)


hp = ModelHparams.sample()


def main():
    parser = ArgumentParser()
    parser.add_arguments(ModelHparams, "hparams")
    args = parser.parse_args()
    hp: ModelHparams = args.hparams
    print(hp)

if __name__ == "__main__":
    main()
    