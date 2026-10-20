from pathlib import Path

import cv2
import numpy as np

from ...annotation import AnnotationDataset, ImageAnnotation
from ...array_types import UInt8Array, UInt16Array
from ..class_list_file import ClassListFile
from ..per_image import PerImageAnnotationWriter


class SemanticMaskWriter(PerImageAnnotationWriter):
    """
    Writes semantic label maps as single-channel PNG files, one per image.

    Pixel value ``0`` is background and value ``i + 1`` is class ``i`` of the dataset; where instances overlap,
    later instances cover earlier ones. Maps are 8-bit, or 16-bit when more than 255 classes are listed.
    ``classes.txt`` lists the background name followed by the dataset classes, so line ``v`` names pixel value
    ``v``. Instances and confidences are not kept.

    Parameters
    ----------
    background_class_name : str, optional
        Name written for pixel value ``0``.

    Raises
    ------
    ValueError
        If the background name is blank or contains a line break.
    """

    def __init__(self, background_class_name: str = "_background_") -> None:
        if not background_class_name.strip():
            raise ValueError("background_class_name must not be blank")
        if "\n" in background_class_name or "\r" in background_class_name:
            raise ValueError(f"background_class_name must not contain line breaks. got {background_class_name!r}")
        self.background_class_name: str = background_class_name
        self.class_list_file: ClassListFile = ClassListFile()

    @property
    def file_suffix(self) -> str:
        """
        Extension of the label files.

        Returns
        -------
        str
            ``".png"``.
        """
        return ".png"

    def build_label_map(self, image: ImageAnnotation, dataset: AnnotationDataset) -> UInt8Array | UInt16Array:
        """
        Render the label map of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to render.
        dataset : AnnotationDataset
            Dataset providing class ids.

        Returns
        -------
        UInt8Array | UInt16Array
            Class id plus one per pixel, shape (H, W).
        """
        label_map: UInt8Array | UInt16Array = (
            np.zeros(image.image_size.mask_shape, dtype=np.uint8)
            if len(dataset.class_names) <= np.iinfo(np.uint8).max
            else np.zeros(image.image_size.mask_shape, dtype=np.uint16)
        )
        for instance in image.instances:
            label_map[instance.mask] = dataset.class_index(instance.class_name) + 1
        return label_map

    def write_label(self, image: ImageAnnotation, dataset: AnnotationDataset, label_path: Path) -> None:
        """
        Write the PNG label map of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to write.
        dataset : AnnotationDataset
            Dataset providing class ids.
        label_path : Path
            Destination PNG file.

        Raises
        ------
        RuntimeError
            If OpenCV fails to encode the PNG.
        """
        is_encoded, encoded_png = cv2.imencode(".png", self.build_label_map(image, dataset))
        if not is_encoded:
            raise RuntimeError(f"failed to encode the label map of {image.image_path}")
        label_path.write_bytes(encoded_png.tobytes())

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write one label map per image and the class list.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to export.
        output_directory : Path
            Directory receiving the files.

        Raises
        ------
        ValueError
            If the background name is a dataset class, more than 65535 classes are listed, or two images share
            a stem, ignoring case.

        Returns
        -------
        tuple[Path, ...]
            Label maps in image order followed by ``classes.txt``.
        """
        if self.background_class_name in dataset.class_names:
            raise ValueError(f"background_class_name {self.background_class_name!r} is a dataset class")
        if len(dataset.class_names) > np.iinfo(np.uint16).max:
            raise ValueError(
                f"at most {np.iinfo(np.uint16).max} classes fit a label map. got {len(dataset.class_names)}"
            )
        label_paths: tuple[Path, ...] = super().write(dataset, output_directory)
        class_names: list[str] = [self.background_class_name, *dataset.class_names]
        return (*label_paths, self.class_list_file.write(class_names, output_directory))
