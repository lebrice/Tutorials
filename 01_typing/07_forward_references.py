from typing import TypeVar, Union

K = TypeVar("K")
V = TypeVar("V")

PossiblyNestedDict = dict[K, Union[V, "PossiblyNestedDict[K, V]"]]


shapes: PossiblyNestedDict[str, tuple[int, ...]] = {
    "x": (10, 3, 3),
    "y": {
        "label": (10, 1),
        "bounding_box": (
            10,
            2,
        ),
    },
}


def flatten(nested: PossiblyNestedDict[K, V]) -> dict[K, V]:
    """Flatten a dictionary of dictionaries. Ignores collisions for simplicity."""
    flattened: dict[K, V] = {}
    for k, v in nested.items():
        if isinstance(v, dict):
            flattened.update(flatten(v))
        else:
            flattened[k] = v
    return flattened


flattened_shapes = flatten(shapes)


from typing import Sequence

NestedSequence = Sequence[Union[V, "NestedSequence[V]"]]
