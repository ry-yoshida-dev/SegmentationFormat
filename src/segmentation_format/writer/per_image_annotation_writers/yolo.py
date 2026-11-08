from pathlib import Path

from ...annotation import AnnotationDataset, ImageAnnotation, InstanceAnnotation
from ...mask import MaskPolygonizer, Polygon
from ..class_list_file import ClassListFile
from ..per_image import PerImageAnnotationWriter


class YoloWriter(PerImageAnnotationWriter):
    """
    Writes Ultralytics YOLO segmentation text labels.

    Each line is ``class_id x1 y1 x2 y2 ...`` with polygon vertices normalized to ``[0, 1]``; images without
    instances get an empty file. YOLO allows one polygon per instance, so only the largest region of a mask is
    written. Class names are written to ``classes.txt``, one per line in id order.

    Parameters
    ----------
    is_confidence_included : bool, optional
        Whether to append the confidence as the last column, as Ultralytics ``save_conf`` does.
    decimal_places : int, optional
        Digits after the decimal point of normalized values.
    polygonizer : MaskPolygonizer, optional
        Boundary tracer of the masks.

    Raises
    ------
    ValueError
        If ``decimal_places`` is negative.
    """

    def __init__(
        self,
        is_confidence_included: bool = False,
        decimal_places: int = 6,
        polygonizer: MaskPolygonizer = MaskPolygonizer(),
    ) -> None:
        if decimal_places < 0:
            raise ValueError(f"decimal_places must not be negative. got {decimal_places}")
        self.is_confidence_included: bool = is_confidence_included
        self.decimal_places: int = decimal_places
        self.polygonizer: MaskPolygonizer = polygonizer
        self.class_list_file: ClassListFile = ClassListFile()

    @property
    def file_suffix(self) -> str:
        """
        Extension of the label files.

        Returns
        -------
        str
            ``".txt"``.
        """
        return ".txt"

    def build_lines(self, image: ImageAnnotation, dataset: AnnotationDataset) -> list[str]:
        """
        Render the label lines of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to render.
        dataset : AnnotationDataset
            Dataset providing class ids.

        Raises
        ------
        ValueError
            If confidences are included but an instance has none.

        Returns
        -------
        list[str]
            One line per instance, without line breaks.
        """
        return [self._format_line(instance, image, dataset) for instance in image.instances]

    def write_label(self, image: ImageAnnotation, dataset: AnnotationDataset, label_path: Path) -> None:
        """
        Write the label file of one image as UTF-8.

        Parameters
        ----------
        image : ImageAnnotation
            Image to write.
        dataset : AnnotationDataset
            Dataset providing class ids.
        label_path : Path
            Destination text file.

        Raises
        ------
        ValueError
            If confidences are included but an instance has none.
        """
        lines: list[str] = self.build_lines(image, dataset)
        label_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write one label file per image and the class list.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to export.
        output_directory : Path
            Directory receiving the files.

        Raises
        ------
        ValueError
            If two images share a stem or an image stem matches the class file name, ignoring case.

        Returns
        -------
        tuple[Path, ...]
            Label files in image order followed by ``classes.txt``.
        """
        class_file_name: str = self.class_list_file.file_name
        if any(
            self.label_path(image, output_directory).name.casefold() == class_file_name.casefold()
            for image in dataset.images
        ):
            raise ValueError(f"an image stem collides with {class_file_name}")
        label_paths: tuple[Path, ...] = super().write(dataset, output_directory)
        return (*label_paths, self.class_list_file.write(dataset.class_names, output_directory))

    def _format_line(self, instance: InstanceAnnotation, image: ImageAnnotation, dataset: AnnotationDataset) -> str:
        largest_polygon: Polygon = self.polygonizer.polygons(instance.mask)[0]
        width: int = image.image_size.width
        height: int = image.image_size.height
        values: list[float] = [
            coordinate for x, y in largest_polygon.to_point_list() for coordinate in (x / width, y / height)
        ]
        if self.is_confidence_included:
            if instance.confidence is None:
                raise ValueError(f"instance of class {instance.class_name!r} in {image.image_path} has no confidence")
            values.append(instance.confidence)
        formatted_values: list[str] = [f"{value:.{self.decimal_places}f}" for value in values]
        return " ".join([str(dataset.class_index(instance.class_name)), *formatted_values])
