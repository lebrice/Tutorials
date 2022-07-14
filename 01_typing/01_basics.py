from typing import Any


def log(something: Any) -> None:
    """This function accepts any argument and returns nothing."""
    print("Logging:", something)


log(1.23)
log("john")


def greeting(name: str) -> str:
    return "Hello " + name


greeting("Bob")  # OK!
greeting(123)  # BAD!


# Suppose that we now want to make it work with both strings and integers:

from typing import Union


def new_greeting(name: Union[str, int]) -> str:
    return "Hello " + name  # we find a bug here!


new_greeting("Bob")
new_greeting(123)


def total_ndim(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
