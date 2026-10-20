import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from cv2.typing import MatLike
from masks import rectangle_mask

from segmentation_format import (
    AnnotationDataset,
    AnnotationFormat,
    CocoMaskEncoding,
    CocoWriter,
    ImageAnnotation,
    InstanceAnnotation,
    LabelMeWriter,
    RunLengthEncoding,
    SemanticMaskWriter,
    YoloWriter,
)


def test_coco_document_uses_mask_bounds_and_polygons(dataset: AnnotationDataset, tmp_path: Path) -> None:
    written: tuple[Path, ...] = CocoWriter().write(dataset, tmp_path / "coco")
    assert written == (tmp_path / "coco" / "annotations.json",)
    document = json.loads(written[0].read_text(encoding="utf-8"))
    assert [image["file_name"] for image in document["images"]] == ["street.jpg", "empty.png"]
    assert [category["id"] for category in document["categories"]] == [1, 2, 3]
    person, car = document["annotations"]
    assert {key: value for key, value in person.items() if key != "segmentation"} == {
        "id": 1,
        "image_id": 1,
        "category_id": 2,
        "bbox": [10.0, 20.0, 40.0, 40.0],
        "area": 1600.0,
        "iscrowd": 0,
        "score": 0.9,
    }
    assert len(person["segmentation"]) == 1
    assert len(person["segmentation"][0]) == 8
    assert (car["id"], car["category_id"], car["bbox"]) == (2, 3, [100.0, 0.0, 100.0, 100.0])
    assert len(car["segmentation"]) == 2


def test_coco_run_length_encoding_round_trips(dataset: AnnotationDataset) -> None:
    document = CocoWriter(mask_encoding=CocoMaskEncoding.RUN_LENGTH).build_document(dataset)
    annotations = document["annotations"]
    assert isinstance(annotations, list)
    car = annotations[1]
    assert isinstance(car, dict)
    segmentation = car["segmentation"]
    assert isinstance(segmentation, dict)
    assert segmentation["size"] == [100, 200]
    counts = segmentation["counts"]
    assert isinstance(counts, list)
    decoded = RunLengthEncoding(counts=tuple(int(str(count)) for count in counts), height=100, width=200).to_mask()
    assert np.array_equal(decoded, dataset.images[0].instances[1].mask)


def test_coco_rejects_duplicate_file_names(dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = dataset.images[0]
    namesake: ImageAnnotation = replace(street, image_path=tmp_path / "other" / "Street.JPG")
    ambiguous: AnnotationDataset = AnnotationDataset(images=(street, namesake), class_names=dataset.class_names)
    with pytest.raises(ValueError, match="file name"):
        CocoWriter().write(ambiguous, tmp_path / "coco")
    assert not (tmp_path / "coco").exists()


def test_coco_can_omit_scores(dataset: AnnotationDataset) -> None:
    document = CocoWriter(is_confidence_included=False).build_document(dataset)
    assert isinstance(document["annotations"], list)
    assert all(isinstance(annotation, dict) and "score" not in annotation for annotation in document["annotations"])


def test_yolo_writes_normalized_polygons_and_classes(dataset: AnnotationDataset, tmp_path: Path) -> None:
    output_directory: Path = tmp_path / "yolo"
    written: tuple[Path, ...] = YoloWriter(decimal_places=4).write(dataset, output_directory)
    assert [path.name for path in written] == ["street.txt", "empty.txt", "classes.txt"]
    person_line, car_line = (output_directory / "street.txt").read_text().splitlines()
    person_values: list[str] = person_line.split()
    assert person_values[0] == "1"
    assert sorted(zip(person_values[1::2], person_values[2::2], strict=True)) == [
        ("0.0500", "0.2000"),
        ("0.0500", "0.6000"),
        ("0.2500", "0.2000"),
        ("0.2500", "0.6000"),
    ]
    car_values: list[str] = car_line.split()
    assert car_values[0] == "2"
    assert min(float(value) for value in car_values[1::2]) == pytest.approx(150 / 200)
    assert (output_directory / "empty.txt").read_text() == ""
    assert (output_directory / "classes.txt").read_text() == "dog\nperson\ncar\n"


def test_yolo_can_append_confidence(dataset: AnnotationDataset) -> None:
    lines: list[str] = YoloWriter(is_confidence_included=True, decimal_places=2).build_lines(dataset.images[0], dataset)
    assert lines[0].split()[-1] == "0.90"
    assert len(lines[0].split()) == 1 + 8 + 1


def test_yolo_confidence_column_requires_confidence(dataset: AnnotationDataset) -> None:
    street: ImageAnnotation = dataset.images[0]
    unscored: ImageAnnotation = replace(street, instances=(replace(street.instances[0], confidence=None),))
    with pytest.raises(ValueError, match="no confidence"):
        YoloWriter(is_confidence_included=True).build_lines(unscored, dataset)


def test_yolo_rejects_stem_matching_class_file_ignoring_case(dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = dataset.images[0]
    shadowing: ImageAnnotation = replace(street, image_path=street.image_path.with_name("Classes.jpg"))
    shadowing_dataset: AnnotationDataset = AnnotationDataset(images=(shadowing,), class_names=dataset.class_names)
    with pytest.raises(ValueError, match="collides"):
        YoloWriter().write(shadowing_dataset, tmp_path / "yolo")


def test_per_image_writer_rejects_colliding_stems(dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = dataset.images[0]
    twin: ImageAnnotation = replace(street, image_path=street.image_path.with_name("STREET.png"))
    colliding: AnnotationDataset = AnnotationDataset(images=(street, twin), class_names=dataset.class_names)
    with pytest.raises(ValueError, match="collide"):
        LabelMeWriter().write(colliding, tmp_path / "labelme")
    assert not (tmp_path / "labelme").exists()


def test_labelme_writes_grouped_polygons(dataset: AnnotationDataset, tmp_path: Path) -> None:
    LabelMeWriter().write(dataset, tmp_path / "labelme")
    document = json.loads((tmp_path / "labelme" / "street.json").read_text(encoding="utf-8"))
    assert document["imagePath"] == "../images/street.jpg"
    assert (document["imageWidth"], document["imageHeight"]) == (200, 100)
    shapes = document["shapes"]
    assert [shape["label"] for shape in shapes] == ["person", "car", "car"]
    assert all(shape["shape_type"] == "polygon" for shape in shapes)
    assert [shape["group_id"] for shape in shapes] == [None, 1, 1]
    assert sorted(shapes[0]["points"]) == [[10.0, 20.0], [10.0, 60.0], [50.0, 20.0], [50.0, 60.0]]


def test_semantic_mask_writes_shifted_class_ids(dataset: AnnotationDataset, tmp_path: Path) -> None:
    output_directory: Path = tmp_path / "semantic"
    written: tuple[Path, ...] = SemanticMaskWriter().write(dataset, output_directory)
    assert [path.name for path in written] == ["street.png", "empty.png", "classes.txt"]
    label_map: MatLike | None = cv2.imread(str(output_directory / "street.png"), cv2.IMREAD_UNCHANGED)
    assert label_map is not None
    assert label_map.dtype == np.uint8
    assert label_map.shape == (100, 200)
    assert (int(label_map[30, 20]), int(label_map[5, 110]), int(label_map[0, 0])) == (2, 3, 0)
    assert (output_directory / "classes.txt").read_text() == "_background_\ndog\nperson\ncar\n"


def test_semantic_mask_paints_later_instances_on_top(dataset: AnnotationDataset) -> None:
    street: ImageAnnotation = dataset.images[0]
    covering: InstanceAnnotation = InstanceAnnotation(mask=rectangle_mask(200, 100, 0, 0, 30, 30), class_name="dog")
    layered: ImageAnnotation = replace(street, instances=(*street.instances, covering))
    label_map = SemanticMaskWriter().build_label_map(layered, dataset)
    assert (int(label_map[25, 15]), int(label_map[40, 40])) == (1, 2)


def test_semantic_mask_uses_16_bits_beyond_255_classes(dataset: AnnotationDataset) -> None:
    many_classes: AnnotationDataset = AnnotationDataset.from_images(
        dataset.images, class_names=[f"class_{index}" for index in range(300)]
    )
    label_map = SemanticMaskWriter().build_label_map(many_classes.images[0], many_classes)
    assert label_map.dtype == np.uint16
    assert int(label_map[5, 110]) == 302


def test_semantic_mask_rejects_background_named_like_a_class(dataset: AnnotationDataset, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="background_class_name"):
        SemanticMaskWriter(background_class_name="dog").write(dataset, tmp_path / "semantic")


@pytest.mark.parametrize("annotation_format", list(AnnotationFormat))
def test_every_format_writes_files(
    annotation_format: AnnotationFormat, dataset: AnnotationDataset, tmp_path: Path
) -> None:
    written: tuple[Path, ...] = annotation_format.create_writer().write(dataset, tmp_path / annotation_format.value)
    assert written
    assert all(path.is_file() for path in written)
    assert annotation_format.display_name
