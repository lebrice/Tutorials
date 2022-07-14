from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Person:
    name: str
    age: int
    _grade: float = 0.0
    friends: list[Person] = field(default_factory=list)


bob = Person("Bob", 20)
alice = Person("Alice", 32)
print(bob, alice)
