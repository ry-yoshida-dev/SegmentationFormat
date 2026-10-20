import numpy as np

from segmentation_format.array_types import BoolArray


def rectangle_mask(width: int, height: int, x_min: int, y_min: int, x_max: int, y_max: int) -> BoolArray:
    """
    Build a mask with one filled rectangle.

    Parameters
    ----------
    width : int
        Mask width.
    height : int
        Mask height.
    x_min : int
        First column of the rectangle.
    y_min : int
        First row of the rectangle.
    x_max : int
        Column after the rectangle.
    y_max : int
        Row after the rectangle.

    Returns
    -------
    BoolArray
        Mask of shape (height, width).
    """
    mask: BoolArray = np.zeros((height, width), dtype=np.bool_)
    mask[y_min:y_max, x_min:x_max] = True
    return mask
