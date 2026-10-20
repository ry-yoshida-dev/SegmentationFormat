# mask

## Overview

Encodings of binary masks used by the writers. Polygons follow pixel edges (pixel `(x, y)` spans
`[x, x + 1] x [y, y + 1]`), which pycocotools rasterizes back to the original region with holes filled. Contours are
traced by OpenCV on the mask upsampled twice, so tracing stays fast on noisy masks.

## Components

| Component | Description |
| --------- | ----------- |
| [polygon.py](./polygon.py) | `Polygon`: validated closed polygon with its area and flat or paired coordinates. |
| [mask_polygonizer.py](./mask_polygonizer.py) | `MaskPolygonizer`: outer boundary of every 8-connected region, optionally simplified. |
| [run_length_encoding.py](./run_length_encoding.py) | `RunLengthEncoding`: lossless uncompressed COCO RLE, encode and decode. |

## Examples

```python
polygons = MaskPolygonizer(simplification_tolerance=1.0).polygons(mask)
largest = polygons[0].to_flat_list()
encoding = RunLengthEncoding.from_mask(mask)
assert (encoding.to_mask() == mask).all()
```
