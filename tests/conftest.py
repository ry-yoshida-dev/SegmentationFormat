from pathlib import Path

import pytest
from masks import rectangle_mask

from segmentation_format import AnnotationDataset, ImageAnnotation, ImageSize, InstanceAnnotation
from segmentation_format.array_types import BoolArray


@pytest.fixture
def dataset(tmp_path: Path) -> AnnotationDataset:
    car_mask: BoolArray = rectangle_mask(200, 100, 100, 0, 120, 10) | rectangle_mask(200, 100, 150, 50, 200, 100)
    street: ImageAnnotation = ImageAnnotation(
        image_path=tmp_path / "images" / "street.jpg",
        image_size=ImageSize(width=200, height=100),
        instances=(
            InstanceAnnotation(mask=rectangle_mask(200, 100, 10, 20, 50, 60), class_name="person", confidence=0.9),
            InstanceAnnotation(mask=car_mask, class_name="car", confidence=0.5),
        ),
    )
    empty: ImageAnnotation = ImageAnnotation(
        image_path=tmp_path / "images" / "empty.png", image_size=ImageSize(width=64, height=32)
    )
    return AnnotationDataset.from_images([street, empty], class_names=["dog"])
