from pathlib import Path

import numpy as np
import pytest
from masks import rectangle_mask

from segmentation_format import AnnotationDataset, ImageAnnotation, ImageSize, InstanceAnnotation


def test_non_boolean_mask_is_rejected() -> None:
    with pytest.raises(TypeError, match="boolean"):
        InstanceAnnotation(mask=np.ones((2, 2), dtype=np.uint8), class_name="cat")  # pyright: ignore[reportArgumentType]


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        InstanceAnnotation(mask=np.zeros((4, 4), dtype=np.bool_), class_name="cat")


def test_confidence_outside_unit_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        InstanceAnnotation(mask=rectangle_mask(4, 4, 0, 0, 2, 2), class_name="cat", confidence=1.5)


def test_class_name_with_line_break_is_rejected() -> None:
    with pytest.raises(ValueError, match="line breaks"):
        InstanceAnnotation(mask=rectangle_mask(4, 4, 0, 0, 2, 2), class_name="cat\ndog")


def test_instance_derives_area_and_exclusive_bounding_box() -> None:
    instance: InstanceAnnotation = InstanceAnnotation(mask=rectangle_mask(20, 10, 2, 3, 7, 9), class_name="cat")
    assert instance.area == 30
    assert instance.bounding_box.value.tolist() == [2.0, 3.0, 7.0, 9.0]


def test_mask_of_other_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected"):
        ImageAnnotation(
            image_path=Path("a.jpg"),
            image_size=ImageSize(10, 10),
            instances=(InstanceAnnotation(mask=rectangle_mask(12, 10, 0, 0, 5, 5), class_name="cat"),),
        )


def test_dataset_orders_given_classes_first(dataset: AnnotationDataset) -> None:
    assert dataset.class_names == ("dog", "person", "car")
    assert dataset.class_index("car") == 2


def test_dataset_rejects_unlisted_class(dataset: AnnotationDataset) -> None:
    with pytest.raises(ValueError, match="missing from class_names"):
        AnnotationDataset(images=dataset.images, class_names=("person",))


def test_dataset_rejects_duplicate_image_paths(dataset: AnnotationDataset) -> None:
    with pytest.raises(ValueError, match="image paths"):
        AnnotationDataset(images=(dataset.images[0], dataset.images[0]), class_names=dataset.class_names)


def test_dataset_rejects_unknown_class_lookup(dataset: AnnotationDataset) -> None:
    with pytest.raises(KeyError, match="unknown class"):
        dataset.class_index("bird")
