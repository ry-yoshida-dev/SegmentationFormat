# tests

## Overview

Unit tests of the annotation model, the mask encodings and every writer.

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | Shared dataset fixture with a labeled image (one single-region and one two-region instance) and an empty image. |
| [masks.py](./masks.py) | Rectangle mask construction helper shared by tests. |
| [test_annotation.py](./test_annotation.py) | Validation of instances, images and datasets. |
| [test_mask.py](./test_mask.py) | Polygon tracing, simplification and run-length encoding. |
| [test_writers.py](./test_writers.py) | Output of each format writer. |

## Examples

```bash
pytest
```
