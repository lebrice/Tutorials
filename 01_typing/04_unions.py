from __future__ import annotations

from typing import Union

str_or_int = Union[str, int]
# (python 3.10+)
# str_or_int = str | int


def positive(v: int | float) -> bool:
    return v > 0
