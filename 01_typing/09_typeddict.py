from typing import TypedDict


class Point2D(TypedDict):
    x: int
    y: int
    label: str


a: Point2D = {"x": 1, "y": 2, "label": "good"}  # OK
b: Point2D = {"z": 3, "label": "bad"}  # Fails type check

# Actually just a dict.
assert Point2D(x=1, y=2, label="first") == dict(x=1, y=2, label="first")


class Point2Dor3D(Point2D, total=False):
    z: int


c: Point2Dor3D = {"x": 1, "y": 2, "label": "good"}

print(c["z"])
