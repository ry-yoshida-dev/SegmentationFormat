from abc import abstractmethod
from collections import Counter
from pathlib import Path

from ..annotation import AnnotationDataset, ImageAnnotation
from .base import AnnotationWriter


class PerImageAnnotationWriter(AnnotationWriter):
    """
    Writer producing one label file per image, named after the image stem.
    """

    @property
    @abstractmethod
    def file_suffix(self) -> str:
        """
        Extension of the label files.

        Returns
        -------
        str
            Suffix including the leading dot.
        """

    @abstractmethod
    def write_label(self, image: ImageAnnotation, dataset: AnnotationDataset, label_path: Path) -> None:
        """
        Write the label file of one image, replacing an existing file.

        Parameters
        ----------
        image : ImageAnnotation
            Image to write.
        dataset : AnnotationDataset
            Dataset providing the class list.
        label_path : Path
            Destination file inside an existing directory.
        """

    def label_path(self, image: ImageAnnotation, output_directory: Path) -> Path:
        """
        Location of the label file of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Annotated image.
        output_directory : Path
            Directory receiving the label files.

        Returns
        -------
        Path
            ``output_directory / (image stem + file_suffix)``.
        """
        return output_directory / f"{image.image_path.stem}{self.file_suffix}"

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write one label file per image.

        Parameters
        ----------
        dataset : AnnotationDataset
            Images, instances and class list to export.
        output_directory : Path
            Directory receiving the label files.

        Raises
        ------
        ValueError
            If two images share a stem, ignoring case, so their label files would collide.

        Returns
        -------
        tuple[Path, ...]
            Label files in image order.
        """
        label_paths: tuple[Path, ...] = tuple(self.label_path(image, output_directory) for image in dataset.images)
        name_counts: Counter[str] = Counter(label_path.name.casefold() for label_path in label_paths)
        colliding_names: list[str] = [name for name, count in name_counts.items() if count > 1]
        if colliding_names:
            raise ValueError(f"images share a file stem, label files would collide: {colliding_names}")
        output_directory.mkdir(parents=True, exist_ok=True)
        for image, label_path in zip(dataset.images, label_paths, strict=True):
            self.write_label(image, dataset, label_path)
        return label_paths
