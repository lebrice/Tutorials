class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self._grade: float = 0.0

    def __str__(self) -> str:
        return f"{self.name} is {self.age} years old"


bob = Person("Bob", 20)
print(bob.name)
