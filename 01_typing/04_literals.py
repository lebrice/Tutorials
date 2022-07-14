from pathlib import Path
from typing import Literal


Mode = Literal["r", "w", "a"]


def open(name: str, mode: Mode = "w"):
    ...


open("test.txt", mode="w")
open("test.txt", mode=123)
open("test.txt", mode="BOB")
