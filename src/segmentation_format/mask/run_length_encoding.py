from dataclasses import dataclass

import numpy as np

from ..array_types import BoolArray, IntArray


@dataclass(frozen=True)
class RunLengthEncoding:
    """
    Uncompressed COCO run-length encoding of a binary mask.

    Pixels are read in column-major order; ``counts`` alternates runs of ``False`` and ``True`` pixels,
    starting with ``False`` (a leading ``0`` when the first pixel is set).

    Attributes
    ----------
    counts : tuple[int, ...]
        Run lengths.
    height : int
        Mask height in pixels.
    width : int
        Mask width in pixels.

    Raises
    ------
    ValueError
        If the size is not positive, a run is negative, or the runs do not cover the mask exactly.
    """

    counts: tuple[int, ...]
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"height and width must be positive. got ({self.height}, {self.width})")
        if any(count < 0 for count in self.counts):
            raise ValueError("counts must not be negative")
        if sum(self.counts) != self.height * self.width:
            raise ValueError(f"counts must sum to {self.height * self.width}. got {sum(self.counts)}")

    @classmethod
    def from_mask(cls, mask: BoolArray) -> "RunLengthEncoding":
        """
        Encode a binary mask.

        Parameters
        ----------
        mask : BoolArray
            Binary mask, shape (H, W).

        Raises
        ------
        ValueError
            If the mask is not two-dimensional or has no pixel.

        Returns
        -------
        RunLengthEncoding
            Encoding of ``mask``.
        """
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError(f"mask must have a non-empty shape (H, W). got {mask.shape}")
        pixels: BoolArray = mask.ravel(order="F")
        change_positions: IntArray = np.flatnonzero(pixels[1:] != pixels[:-1]).astype(np.int64) + 1
        run_boundaries: IntArray = np.concatenate(([0], change_positions, [pixels.size])).astype(np.int64)
        run_lengths: list[int] = [int(length) for length in np.diff(run_boundaries)]
        counts: list[int] = [0, *run_lengths] if bool(pixels[0]) else run_lengths
        height, width = mask.shape
        return cls(counts=tuple(counts), height=int(height), width=int(width))

    def to_mask(self) -> BoolArray:
        """
        Decode into a binary mask.

        Returns
        -------
        BoolArray
            Mask of shape (height, width).
        """
        run_values: BoolArray = np.arange(len(self.counts)) % 2 == 1
        pixels: BoolArray = np.repeat(run_values, self.counts)
        return pixels.reshape((self.height, self.width), order="F")
