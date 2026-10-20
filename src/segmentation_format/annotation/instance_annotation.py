import math
from dataclasses import dataclass

import numpy as np
from geometry import Box2D, Box2DFormat

from ..array_types import BoolArray, IntArray


@dataclass(frozen=True, eq=False)
class InstanceAnnotation:
    """
    One labeled object instance in an image.

    Attributes
    ----------
    mask : BoolArray
        Pixels of the instance, shape (H, W) of the whole image.
    class_name : str
        Category of the instance.
    confidence : float | None, optional
        Segmenter confidence in ``[0, 1]``; ``None`` for ground-truth labels.

    Raises
    ------
    TypeError
        If the mask is not boolean.
    ValueError
        If the mask is not two-dimensional or has no pixel, the class name is blank or contains a line break,
        or the confidence is outside ``[0, 1]``.
    """

    mask: BoolArray
    class_name: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise TypeError(f"mask must be boolean. got {self.mask.dtype}")
        if self.mask.ndim != 2:
            raise ValueError(f"mask must have shape (H, W). got {self.mask.shape}")
        if not bool(self.mask.any()):
            raise ValueError("mask must not be empty")
        if not self.class_name.strip():
            raise ValueError("class_name must not be blank")
        if "\n" in self.class_name or "\r" in self.class_name:
            raise ValueError(f"class_name must not contain line breaks. got {self.class_name!r}")
        if self.confidence is not None and not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]. got {self.confidence}")

    @property
    def area(self) -> int:
        """
        Number of pixels of the instance.

        Returns
        -------
        int
            Count of ``True`` pixels in ``mask``.
        """
        return int(np.count_nonzero(self.mask))

    @property
    def bounding_box(self) -> Box2D:
        """
        Tightest box around the mask.

        Returns
        -------
        Box2D
            XYXY box in pixel coordinates covering whole pixels, with exclusive right and bottom edges.
        """
        rows: IntArray = np.flatnonzero(self.mask.any(axis=1))
        columns: IntArray = np.flatnonzero(self.mask.any(axis=0))
        corners: list[float] = [
            float(columns[0]),
            float(rows[0]),
            float(columns[-1] + 1),
            float(rows[-1] + 1),
        ]
        return Box2D.register(value=np.array(corners, dtype=np.float64), box2d_format=Box2DFormat.XYXY)
