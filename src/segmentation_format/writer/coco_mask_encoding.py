from enum import Enum


class CocoMaskEncoding(Enum):
    """
    Encodings of the COCO ``segmentation`` field.
    """

    POLYGON = "polygon"
    RUN_LENGTH = "run_length"
