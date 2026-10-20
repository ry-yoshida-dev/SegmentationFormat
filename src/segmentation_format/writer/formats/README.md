# formats

## Overview

Concrete writers, one per segmentation format. Single-file formats extend `AnnotationWriter`; formats with one label
file per image extend `PerImageAnnotationWriter`. Polygon formats take a `MaskPolygonizer` to control simplification.
JSON files are saved with [FileHandler](https://github.com/ry-yoshida-dev/FileHandler), text files as UTF-8 with
`\n` line endings, and PNG label maps with OpenCV. COCO refuses images sharing a file name in different directories.

## Components

| Component | Description |
| --------- | ----------- |
| [coco.py](./coco.py) | `CocoWriter`: MS COCO instance segmentation JSON with optional `score`. |
| [coco_mask_encoding.py](./coco_mask_encoding.py) | `CocoMaskEncoding`: polygon or run-length `segmentation`. |
| [yolo.py](./yolo.py) | `YoloWriter`: normalized YOLO segmentation polygons and `classes.txt`. |
| [labelme.py](./labelme.py) | `LabelMeWriter`: LabelMe JSON with polygon shapes. |
| [semantic_mask.py](./semantic_mask.py) | `SemanticMaskWriter`: PNG label maps and `classes.txt` starting with the background. |

## Examples

```python
CocoWriter(mask_encoding=CocoMaskEncoding.RUN_LENGTH).write(dataset, Path("export/coco"))
SemanticMaskWriter(background_class_name="background").write(dataset, Path("export/masks"))
```
