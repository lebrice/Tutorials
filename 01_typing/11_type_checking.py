import typing

if typing.TYPE_CHECKING:
    from torch import Tensor


def a_func(arg: Tensor) -> None:
    ...
