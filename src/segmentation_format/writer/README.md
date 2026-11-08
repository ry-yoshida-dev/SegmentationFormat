# writer

## Overview

Writers turning an `AnnotationDataset` into files. Every writer implements
`AnnotationWriter.write(dataset, output_directory)`, creates the directory if needed and returns the written paths.
Per-image formats name each label file after the image stem and refuse to write when two stems collide.

## Components

| Component | Description |
| --------- | ----------- |
| [annotation_writer.py](./annotation_writer.py) | `AnnotationWriter`: abstract writer. |
| [annotation_writers/](./annotation_writers/README.md) | Single-file writer for COCO. |
| [per_image_annotation_writer.py](./per_image_annotation_writer.py) | `PerImageAnnotationWriter`: abstract writer of one label file per image. |
| [per_image_annotation_writers/](./per_image_annotation_writers/README.md) | Per-image writers for YOLO segmentation, LabelMe and semantic label maps. |
| [coco_mask_encoding.py](./coco_mask_encoding.py) | `CocoMaskEncoding`: polygon or run-length `segmentation` of `CocoWriter`. |
| [class_list_file.py](./class_list_file.py) | `ClassListFile`: `classes.txt` with one class name per line. |
| [annotation_format.py](./annotation_format.py) | `AnnotationFormat`: supported formats, display names and default writers. |
| [json_value.py](./json_value.py) | `JsonValue`: recursive type of JSON documents. |

## Examples

```python
for annotation_format in AnnotationFormat:
    annotation_format.create_writer().write(dataset, Path("export") / annotation_format.value)
```
