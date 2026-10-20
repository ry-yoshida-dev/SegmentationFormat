from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from cv2.typing import MatLike

from ..array_types import BoolArray, FloatArray, IntArray, UInt8Array
from .polygon import Polygon


@dataclass(frozen=True)
class MaskPolygonizer:
    """
    Traces the outer boundaries of a binary mask as polygons.

    Boundaries follow pixel edges, pixel ``(x, y)`` spanning ``[x, x + 1] x [y, y + 1]``, so a polygon rasterized
    by pycocotools reproduces its region. Contours are traced by OpenCV on the mask upsampled twice, where every
    left or top boundary lies on an even index and every right or bottom boundary on an odd one, so
    ``ceil(index / 2)`` recovers the pixel corners. Regions are 8-connected; holes are not traced.

    Attributes
    ----------
    simplification_tolerance : float, optional
        Maximum distance in pixels between a boundary and its simplified polygon (Douglas-Peucker); ``0`` keeps
        the exact boundary. A region that would collapse below three vertices keeps its exact boundary.

    Raises
    ------
    ValueError
        If the tolerance is negative or not finite.
    """

    simplification_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.simplification_tolerance) and self.simplification_tolerance >= 0.0):
            raise ValueError(f"simplification_tolerance must be non-negative. got {self.simplification_tolerance}")

    def polygons(self, mask: BoolArray) -> tuple[Polygon, ...]:
        """
        Outer boundary of every connected region of a mask.

        Parameters
        ----------
        mask : BoolArray
            Binary mask, shape (H, W).

        Raises
        ------
        ValueError
            If the mask is not two-dimensional.

        Returns
        -------
        tuple[Polygon, ...]
            One polygon per region, largest area first; empty for an empty mask.
        """
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape (H, W). got {mask.shape}")
        upsampled: UInt8Array = mask.astype(np.uint8).repeat(2, axis=0).repeat(2, axis=1)
        contours: Sequence[MatLike]
        contours, _ = cv2.findContours(upsampled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[Polygon] = [self._trace(contour) for contour in contours]
        return tuple(sorted(polygons, key=lambda polygon: polygon.area, reverse=True))

    def _trace(self, contour: MatLike) -> Polygon:
        upsampled_points: IntArray = np.asarray(contour, dtype=np.int64).reshape(-1, 2)
        corners: IntArray = (upsampled_points + 1) // 2
        is_new_corner: BoolArray = np.any(corners != np.roll(corners, 1, axis=0), axis=1)
        exact: Polygon = Polygon(points=corners[is_new_corner].astype(np.float64))
        if self.simplification_tolerance == 0.0:
            return exact
        simplified_points: FloatArray = np.asarray(
            cv2.approxPolyDP(exact.points.astype(np.float32), self.simplification_tolerance, True), dtype=np.float64
        ).reshape(-1, 2)
        if simplified_points.shape[0] < Polygon.MIN_POINT_COUNT:
            return exact
        simplified: Polygon = Polygon(points=simplified_points)
        return simplified if simplified.area > 0.0 else exact
