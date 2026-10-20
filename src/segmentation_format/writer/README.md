# writer

## Overview

Writers turning an `AnnotationDataset` into files. Every writer implements
`AnnotationWriter.write(dataset, output_directory)`, creates the directory if needed and returns the written paths.
Per-image formats name each label file after the image stem and refuse to write when two stems collide.

## Components

| Component | Description |
| --------- | ----------- |
| [base.py](./base.py) | `AnnotationWriter`: abstract writer. |
| [per_image.py](./per_image.py) | `PerImageAnnotationWriter`: base for formats with one label file per image, written by `write_label`. |
| [class_list_file.py](./class_list_file.py) | `ClassListFile`: `classes.txt` with one class name per line. |
| [formats/](./formats/README.md) | Concrete writers for COCO, YOLO segmentation, LabelMe and semantic label maps. |
| [annotation_format.py](./annotation_format.py) | `AnnotationFormat`: supported formats, display names and default writers. |
| [json_value.py](./json_value.py) | `JsonValue`: recursive type of JSON documents. |

## Examples

```python
for annotation_format in AnnotationFormat:
    annotation_format.create_writer().write(dataset, Path("export") / annotation_format.value)
```
