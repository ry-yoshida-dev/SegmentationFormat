from dataclasses import dataclass
from pathlib import Path

from .image_size import ImageSize
from .instance_annotation import InstanceAnnotation


@dataclass(frozen=True, eq=False)
class ImageAnnotation:
    """
    Instances labeled in one image.

    Attributes
    ----------
    image_path : Path
        Location of the image file.
    image_size : ImageSize
        Size of the image the masks cover.
    instances : tuple[InstanceAnnotation, ...], optional
        Labeled instances; empty for a background image.

    Raises
    ------
    ValueError
        If a mask shape differs from the image size.
    """

    image_path: Path
    image_size: ImageSize
    instances: tuple[InstanceAnnotation, ...] = ()

    def __post_init__(self) -> None:
        for instance in self.instances:
            if instance.mask.shape != self.image_size.mask_shape:
                raise ValueError(
                    f"mask of {self.image_path} has shape {instance.mask.shape}, "
                    + f"expected {self.image_size.mask_shape}"
                )

    @property
    def class_names(self) -> tuple[str, ...]:
        """
        Classes used in this image.

        Returns
        -------
        tuple[str, ...]
            Distinct class names in order of first appearance.
        """
        return tuple(dict.fromkeys(instance.class_name for instance in self.instances))
