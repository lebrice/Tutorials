from typing import Any, Iterable, Sized, TypeVar, Generic, Sequence


T = TypeVar("T")


class Space(Generic[T]):
    def sample(self) -> T:
        ...

    def contains(self, v: Any) -> bool:
        ...


Box = Space[float]
Discrete = Space[int]
Tuples = Space[tuple]
Mappings = Space[dict]


# Generic function:


def first(l: Sequence[T]) -> T:
    return l[0]


a = first("hello")
b = first([1, 2, 3])

# Using generic classes is also good:


def total_length(items: Iterable[Sized]) -> int:
    return sum(len(element) for element in items)


l = total_length([[1, 2], [2, 3, 4], [5, 6]])
l = total_length([1, 2, 3])  # BAD!


def flatten(items: list[list[T]]) -> list[T]:
    return [item for sublist in items for item in sublist]


flat = flatten([[1, 2], [2, 3, 4], [5, 6]])
flat_str = flatten([["H"], ["e", "l", "l"], ["o"]])
