from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..array_types import FloatArray


@dataclass(frozen=True, eq=False)
class Polygon:
    """
    Closed polygon in pixel coordinates.

    Attributes
    ----------
    points : FloatArray
        Vertices as ``(x, y)``, shape (P, 2); the last vertex connects back to the first.

    Raises
    ------
    ValueError
        If ``points`` is not of shape (P, 2) with at least three vertices, or a coordinate is not finite.
    """

    points: FloatArray

    MIN_POINT_COUNT: ClassVar[int] = 3

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"points must have shape (P, 2). got {self.points.shape}")
        if self.points.shape[0] < self.MIN_POINT_COUNT:
            raise ValueError(f"a polygon needs at least {self.MIN_POINT_COUNT} points. got {self.points.shape[0]}")
        if not bool(np.isfinite(self.points).all()):
            raise ValueError("points must be finite")

    @property
    def area(self) -> float:
        """
        Enclosed area by the shoelace formula.

        Returns
        -------
        float
            Non-negative area in square pixels.
        """
        x_values: FloatArray = self.points[:, 0]
        y_values: FloatArray = self.points[:, 1]
        signed_double_area: float = float(
            np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))
        )
        return abs(signed_double_area) / 2.0

    def to_flat_list(self) -> list[float]:
        """
        Vertices as one flat coordinate list.

        Returns
        -------
        list[float]
            ``[x1, y1, x2, y2, ...]``.
        """
        return [float(value) for value in self.points.reshape(-1)]

    def to_point_list(self) -> list[list[float]]:
        """
        Vertices as a list of coordinate pairs.

        Returns
        -------
        list[list[float]]
            ``[[x1, y1], [x2, y2], ...]``.
        """
        return [[float(x), float(y)] for x, y in self.points]
