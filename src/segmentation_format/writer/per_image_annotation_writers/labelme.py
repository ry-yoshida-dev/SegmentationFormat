from pathlib import Path

from file_handler import FileHandler

from ...annotation import AnnotationDataset, ImageAnnotation, InstanceAnnotation
from ...mask import MaskPolygonizer, Polygon
from ..json_value import JsonValue
from ..per_image import PerImageAnnotationWriter


class LabelMeWriter(PerImageAnnotationWriter):
    """
    Writes LabelMe JSON with polygon shapes, one file per image.

    Every region of a mask becomes one polygon shape; the shapes of an instance with several regions share a
    ``group_id`` (its position in the image), otherwise ``group_id`` is ``None``. ``imagePath`` is relative to
    the output directory so LabelMe can open the image, and ``imageData`` is left empty.

    Parameters
    ----------
    polygonizer : MaskPolygonizer, optional
        Boundary tracer of the masks.
    """

    labelme_version: str = "5.5.0"

    def __init__(self, polygonizer: MaskPolygonizer = MaskPolygonizer()) -> None:
        self.polygonizer: MaskPolygonizer = polygonizer

    @property
    def file_suffix(self) -> str:
        """
        Extension of the label files.

        Returns
        -------
        str
            ``".json"``.
        """
        return ".json"

    def build_document(self, image: ImageAnnotation, output_directory: Path) -> dict[str, JsonValue]:
        """
        Build the LabelMe document of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to render.
        output_directory : Path
            Directory the JSON file is written to, used to relativize ``imagePath``.

        Returns
        -------
        dict[str, JsonValue]
            LabelMe document.
        """
        shapes: list[JsonValue] = [
            shape
            for instance_index, instance in enumerate(image.instances)
            for shape in self._build_shapes(instance, instance_index)
        ]
        image_path: Path = image.image_path.resolve().relative_to(output_directory.resolve(), walk_up=True)
        document: dict[str, JsonValue] = {
            "version": self.labelme_version,
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.as_posix(),
            "imageData": None,
            "imageHeight": image.image_size.height,
            "imageWidth": image.image_size.width,
        }
        return document

    def write_label(self, image: ImageAnnotation, dataset: AnnotationDataset, label_path: Path) -> None:
        """
        Write the JSON file of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to write.
        dataset : AnnotationDataset
            Dataset the image belongs to.
        label_path : Path
            Destination JSON file.
        """
        FileHandler.save(self.build_document(image, label_path.parent), str(label_path))

    def _build_shapes(self, instance: InstanceAnnotation, instance_index: int) -> list[JsonValue]:
        polygons: tuple[Polygon, ...] = self.polygonizer.polygons(instance.mask)
        group_id: int | None = instance_index if len(polygons) > 1 else None
        shapes: list[JsonValue] = [
            {
                "label": instance.class_name,
                "points": [[x, y] for x, y in polygon.to_point_list()],
                "group_id": group_id,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            }
            for polygon in polygons
        ]
        return shapes
