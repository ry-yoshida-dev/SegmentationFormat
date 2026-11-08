# per_image_annotation_writers

## Overview

`PerImageAnnotationWriter` implementations writing one label file per image. Polygon formats take a
`MaskPolygonizer` to control simplification.

## Components

| Component | Description |
| --------- | ----------- |
| [yolo.py](./yolo.py) | `YoloWriter`: normalized YOLO segmentation polygons and `classes.txt`. |
| [labelme.py](./labelme.py) | `LabelMeWriter`: LabelMe JSON with polygon shapes. |
| [semantic_mask.py](./semantic_mask.py) | `SemanticMaskWriter`: PNG label maps and `classes.txt` starting with the background. |

## Examples

```python
SemanticMaskWriter(background_class_name="background").write(dataset, Path("export/masks"))
```
