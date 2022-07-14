from __future__ import annotations
from typing import Any, Iterable, Protocol, TypeVar, Generic, Union
from typing_extensions import TypeVarTuple, Unpack

DType = TypeVar("DType")
Shape = TypeVarTuple("Shape")


class Array(Generic[DType, Unpack[Shape]]):
    def __abs__(self) -> Array[DType, Unpack[Shape]]:
        ...

    def __add__(
        self, other: Array[DType, Unpack[Shape]]
    ) -> Array[DType, Unpack[Shape]]:
        ...


from typing import Literal as L

x: Array[int, L[32], L[32], L[3]] = Array()
y: Array[int, L[10]] = Array()


B = TypeVar("B", bound=L[16, 32, 64, 128, 256, 512])


def batch(
    loader: Iterable[Array[DType, Unpack[Shape]]], batch_size: B
) -> Array[DType, B, Unpack[Shape]]:
    ...


x_batch = batch([x], 32)


H = TypeVar("H")
W = TypeVar("W")
C = TypeVar("C")


def to_tensor(
    image: Array[int, Unpack[Shape], H, W, C]
) -> Array[float, Unpack[Shape], C, H, W]:
    ...


x_tensor = to_tensor(x)

# ------------------------------------------------------------


# ------------------------------------------------------------

I = TypeVar("I")
O = TypeVar("O")


class Linear(Generic[I, O]):
    def __init__(self, in_features: I, out_features: O):
        ...

    def __call__(self, input: Array[float, B, I]) -> Array[float, B, O]:
        ...


C_i = TypeVar("C_i")
C_o = TypeVar("C_o")


class Conv2d(Generic[C_i, C_o]):
    def __init__(self, in_channels: C_i, out_channels: C_o):
        ...

    def __call__(self, input: Array[float, B, C_i, H, W]) -> Array[float, B, C_o, H, W]:
        ...


class Pool(Generic[O]):
    def __init__(self, out_dims: O):
        ...

    def __call__(self, input: Array[float, B, Unpack[Shape]]) -> Array[float, B, O]:
        ...


class Cifar10Network:
    def __init__(self):
        self.conv: Conv2d[L[3], L[128]] = Conv2d(3, 128)
        self.pool: Pool[L[128]] = Pool(128)
        self.fc: Linear[L[128], L[10]] = Linear(128, 10)

    def __call__(
        self, input: Array[int, B, L[32], L[32], L[3]]
    ) -> Array[float, B, L[10]]:
        x = to_tensor(input)
        h_x = self.conv(x)
        h_x = self.pool(h_x)
        logits = self.fc(h_x)
        return logits


network = Cifar10Network()

logits = network(x_batch)  # WORKS!

network(x_batch)  # WORKS!

network(x)  # BAD! (missing batch dimension)

wrong_size: Array[int, L[32], L[256], L[256], L[3]] = Array()
network(wrong_size)


weird_batch_size: Array[int, L[17], L[32], L[32], L[3]] = Array()
network(weird_batch_size)


weird_channels: Array[int, L[32], L[32], L[32], L[17]] = Array()
network(weird_channels)
