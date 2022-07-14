from __future__ import annotations

# Type annotations are evaluated at runtime by default (until python 3.11)
# Adding the line above makes it so they are only evaluated as strings at runtime.


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.friends: list[Person] = []  # funky!
        self._grade: float | None = 0.0

    def greet(self, someone: Person) -> str:
        return f"{self.name} says: 'Hi {someone.name}!'"
