from collections import Counter
from pathlib import Path

from file_handler import FileHandler
from geometry import Box2DFormat

from ...annotation import AnnotationDataset, InstanceAnnotation
from ...mask import MaskPolygonizer, RunLengthEncoding
from ..base import AnnotationWriter
from ..json_value import JsonValue
from .coco_mask_encoding import CocoMaskEncoding


class CocoWriter(AnnotationWriter):
    """
    Writes MS COCO instance segmentation JSON.

    Image, annotation and category ids start at 1 and ``file_name`` is the image file name without directories.
    ``bbox`` is the ``[x_min, y_min, width, height]`` pixel bounds of the mask and ``area`` its pixel count.
    ``segmentation`` holds either one flat polygon per region or an uncompressed RLE, which pycocotools reads
    without ``iscrowd``; ``iscrowd`` is always ``0``.

    Parameters
    ----------
    file_name : str, optional
        Name of the JSON file.
    mask_encoding : CocoMaskEncoding, optional
        Encoding of ``segmentation``; run-length encoding is lossless, polygons drop holes.
    polygonizer : MaskPolygonizer, optional
        Boundary tracer used by the polygon encoding.
    is_confidence_included : bool, optional
        Whether to add ``score`` to annotations that carry a confidence.
    """

    def __init__(
        self,
        file_name: str = "annotations.json",
        mask_encoding: CocoMaskEncoding = CocoMaskEncoding.POLYGON,
        polygonizer: MaskPolygonizer = MaskPolygonizer(),
        is_confidence_included: bool = True,
    ) -> None:
        self.file_name: str = file_name
        self.mask_encoding: CocoMaskEncoding = mask_encoding
        self.polygonizer: MaskPolygonizer = polygonizer
        self.is_confidence_included: bool = is_confidence_included

    def build_document(self, dataset: AnnotationDataset) -> dict[str, JsonValue]:
        """
        Build the COCO document.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to convert.

        Raises
        ------
        ValueError
            If two images share a file name, ignoring case.

        Returns
        -------
        dict[str, JsonValue]
            Document with ``images``, ``annotations`` and ``categories``.
        """
        name_counts: Counter[str] = Counter(image.image_path.name.casefold() for image in dataset.images)
        duplicate_names: list[str] = [name for name, count in name_counts.items() if count > 1]
        if duplicate_names:
            raise ValueError(f"images share a file name, COCO file_name would be ambiguous: {duplicate_names}")
        images: list[JsonValue] = []
        annotations: list[JsonValue] = []
        for image_id, image in enumerate(dataset.images, start=1):
            images.append(
                {
                    "id": image_id,
                    "file_name": image.image_path.name,
                    "width": image.image_size.width,
                    "height": image.image_size.height,
                }
            )
            for instance in image.instances:
                annotations.append(self._build_annotation(instance, len(annotations) + 1, image_id, dataset))
        categories: list[JsonValue] = [
            {"id": category_id, "name": class_name, "supercategory": ""}
            for category_id, class_name in enumerate(dataset.class_names, start=1)
        ]
        return {"images": images, "annotations": annotations, "categories": categories}

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write the COCO JSON file.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to export.
        output_directory : Path
            Directory receiving the file.

        Raises
        ------
        ValueError
            If two images share a file name, ignoring case.

        Returns
        -------
        tuple[Path, ...]
            The JSON file.
        """
        output_path: Path = output_directory / self.file_name
        FileHandler.save(self.build_document(dataset), str(output_path))
        return (output_path,)

    def _build_annotation(
        self, instance: InstanceAnnotation, annotation_id: int, image_id: int, dataset: AnnotationDataset
    ) -> dict[str, JsonValue]:
        bbox: list[JsonValue] = [float(value) for value in instance.bounding_box.to_format(Box2DFormat.XYWH)]
        annotation: dict[str, JsonValue] = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": dataset.class_index(instance.class_name) + 1,
            "bbox": bbox,
            "area": float(instance.area),
            "iscrowd": 0,
            "segmentation": self._build_segmentation(instance),
        }
        if self.is_confidence_included and instance.confidence is not None:
            annotation["score"] = instance.confidence
        return annotation

    def _build_segmentation(self, instance: InstanceAnnotation) -> JsonValue:
        match self.mask_encoding:
            case CocoMaskEncoding.POLYGON:
                polygons: list[JsonValue] = [
                    [coordinate for coordinate in polygon.to_flat_list()]
                    for polygon in self.polygonizer.polygons(instance.mask)
                ]
                return polygons
            case CocoMaskEncoding.RUN_LENGTH:
                encoding: RunLengthEncoding = RunLengthEncoding.from_mask(instance.mask)
                return {"size": [encoding.height, encoding.width], "counts": [count for count in encoding.counts]}
