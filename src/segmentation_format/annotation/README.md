# annotation

## Overview

Immutable, validated annotation model shared by every writer. Each instance is a boolean mask of the whole image;
its bounding box and area are derived from the mask. Invalid data (non-boolean or empty masks, masks of another
size than their image, unknown classes) is rejected at construction.

## Components

| Component | Description |
| --------- | ----------- |
| [image_size.py](./image_size.py) | `ImageSize`: image width and height, and the mask shape they imply. |
| [instance_annotation.py](./instance_annotation.py) | `InstanceAnnotation`: mask, class name, optional confidence, derived area and box. |
| [image_annotation.py](./image_annotation.py) | `ImageAnnotation`: image path, size and its instances. |
| [annotation_dataset.py](./annotation_dataset.py) | `AnnotationDataset`: images and the ordered class list defining class ids. |

## Examples

```python
instance = InstanceAnnotation(mask=mask, class_name="car")
instance.area
instance.bounding_box
dataset = AnnotationDataset.from_images(images, class_names=["person", "car"])
dataset.class_index("car")
```
