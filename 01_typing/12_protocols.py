from typing import Protocol


class HasShape(Protocol):

    # Normally, this would be enough, but pytorch/numpy
    # shape: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        ...


# ----------------------

from functools import reduce
import operator


def total_dimensions(array: HasShape) -> int:
    return reduce(operator.mul, array.shape)


import torch
import numpy as np

list_dims = total_dimensions([1, 2, 3])  # BAD!
array_dims = total_dimensions(np.array([1, 2, 3]))
tensor_dims = total_dimensions(torch.tensor([1, 2, 3]))
