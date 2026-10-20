# segmentation_format

## Overview

Format-independent instance mask model, mask encoders, and writers serializing the model into segmentation formats.

## Components

| Component | Description |
| --------- | ----------- |
| [annotation/](./annotation/README.md) | Validated instance masks, images and the dataset with its class list. |
| [mask/](./mask/README.md) | Mask encodings: pixel-edge polygons and COCO run-length encoding. |
| [writer/](./writer/README.md) | Writers for COCO, YOLO segmentation, LabelMe and semantic label maps, and the `AnnotationFormat` enum. |
| [array_types.py](./array_types.py) | NumPy array aliases shared by the package. |
