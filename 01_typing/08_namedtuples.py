from typing import NamedTuple
import collections


Employee_ugly = collections.namedtuple("Employee", ["name", "id"])


class Employee(NamedTuple):
    name: str
    id: int = 3


employee = Employee("Guido")
assert employee.id == 3


def foo_with_many_outputs(x: int) -> tuple[int, int, int]:
    return x, x**2, x**3


class OutputTuple(NamedTuple):
    x: int
    x_squared: int

    x_cubed: int
    """ The third power of x."""


def foo(x: int) -> OutputTuple:
    return OutputTuple(x, x**2, x**3)


v = foo(123)
v.x_cubed
