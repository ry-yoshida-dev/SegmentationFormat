from dataclasses import dataclass


@dataclass(frozen=True)
class ImageSize:
    """
    Pixel size of an annotated image.

    Attributes
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Raises
    ------
    ValueError
        If width or height is not positive.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be positive. got ({self.width}, {self.height})")

    @property
    def mask_shape(self) -> tuple[int, int]:
        """
        Shape of a mask covering the image.

        Returns
        -------
        tuple[int, int]
            ``(height, width)``.
        """
        return (self.height, self.width)
