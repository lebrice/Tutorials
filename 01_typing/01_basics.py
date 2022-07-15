from typing import Any

a: int = 132
a = "bob"  # BAD!


def foo(something):
    """This function accepts any argument and returns nothing."""
    print("Logging:", something)


foo(1.23)
foo("john")


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


numbers: list[int] = []
numbers.append("bob")  # BAD!
numbers.append(1.23)  # BAD! float can't be used like an int.
numbers.append(123)  # Fine!

values: list[float] = []
values.append("bob")  # BAD!
values.append(123)  # Fine! (int can be used like a float.)
values.append(1.23)
