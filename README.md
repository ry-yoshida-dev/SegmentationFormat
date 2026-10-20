# SegmentationFormat

## Overview

Export instance segmentation masks to common annotation formats. Instances are boolean NumPy masks covering the whole
image; the library does not depend on any segmenter: build an `AnnotationDataset` from your own masks and pick a
writer. Boxes and areas are derived from the masks, and polygons follow pixel edges so that pycocotools rasterizes
them back to the original region. It is the segmentation counterpart of
[ObjectDetectionFormat](https://github.com/ry-yoshida-dev/ObjectDetectionFormat).

| Format | Output | Mask encoding |
| ------ | ------ | ------------- |
| MS COCO | `annotations.json` | polygons or uncompressed RLE, mask `bbox` and pixel `area`, optional `score` |
| YOLO segmentation | `<stem>.txt` per image + `classes.txt` | `class_id x1 y1 x2 y2 ...` normalized, largest region only, optional confidence column |
| LabelMe | `<stem>.json` per image | one `polygon` shape per region, regions of an instance share a `group_id` |
| Semantic label map | `<stem>.png` per image + `classes.txt` | 8/16-bit PNG, `0` background, class id plus one elsewhere |

Polygons drop holes; use `CocoMaskEncoding.RUN_LENGTH` for lossless COCO export.
See [src/segmentation_format](src/segmentation_format/README.md) for the package layout.

## Installation

```bash
pip install git+https://github.com/ry-yoshida-dev/SegmentationFormat.git
pip install -e ".[dev]"
```

## Examples

```python
from pathlib import Path

import numpy as np
from segmentation_format import (
    AnnotationDataset,
    AnnotationFormat,
    CocoMaskEncoding,
    CocoWriter,
    ImageAnnotation,
    ImageSize,
    InstanceAnnotation,
    MaskPolygonizer,
    YoloWriter,
)

mask = np.zeros((1080, 1920), dtype=np.bool_)
mask[20:400, 10:200] = True
image = ImageAnnotation(
    image_path=Path("images/street.jpg"),
    image_size=ImageSize(width=1920, height=1080),
    instances=(InstanceAnnotation(mask=mask, class_name="person", confidence=0.87),),
)
dataset = AnnotationDataset.from_images([image], class_names=["person", "car"])

AnnotationFormat.COCO.create_writer().write(dataset, Path("export/coco"))
CocoWriter(mask_encoding=CocoMaskEncoding.RUN_LENGTH).write(dataset, Path("export/coco_rle"))
YoloWriter(polygonizer=MaskPolygonizer(simplification_tolerance=1.0)).write(dataset, Path("export/labels"))
```

Converting an [OpenVocabularySegmentation](https://github.com/ry-yoshida-dev/OpenVocabularySegmentation) result:

```python
image = ImageAnnotation(
    image_path=path,
    image_size=ImageSize(width=result.image_size.width, height=result.image_size.height),
    instances=tuple(
        InstanceAnnotation(
            mask=segmentation.mask,
            class_name=segmentation.class_name,
            confidence=segmentation.confidence,
        )
        for segmentation in result
    ),
)
```
