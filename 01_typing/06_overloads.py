from typing import overload


@overload
def process(response: None) -> None:
    """Process a None response."""


@overload
def process(response: int) -> tuple[int, str]:
    """Process an int response."""


@overload
def process(response: bytes) -> str:
    """Process a bytes response."""


def process(response):
    ...


bob = process()
