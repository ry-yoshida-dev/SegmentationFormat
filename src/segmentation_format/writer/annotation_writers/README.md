# annotation_writers

## Overview

`AnnotationWriter` implementations writing the whole dataset to one file.

## Components

| Component | Description |
| --------- | ----------- |
| [coco.py](./coco.py) | `CocoWriter`: MS COCO instance segmentation JSON with optional `score`. |

## Examples

```python
CocoWriter(mask_encoding=CocoMaskEncoding.RUN_LENGTH).write(dataset, Path("export/coco"))
```
