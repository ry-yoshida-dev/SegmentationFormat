from enum import Enum

from .base import AnnotationWriter
from .formats import CocoWriter, LabelMeWriter, SemanticMaskWriter, YoloWriter


class AnnotationFormat(Enum):
    """
    Supported export formats.
    """

    COCO = "coco"
    YOLO = "yolo"
    LABELME = "labelme"
    SEMANTIC_MASK = "semantic_mask"

    @property
    def display_name(self) -> str:
        """
        Human-readable format name.

        Returns
        -------
        str
            Name suitable for menus.
        """
        match self:
            case AnnotationFormat.COCO:
                return "MS COCO (JSON)"
            case AnnotationFormat.YOLO:
                return "YOLO Segmentation (TXT)"
            case AnnotationFormat.LABELME:
                return "LabelMe (JSON)"
            case AnnotationFormat.SEMANTIC_MASK:
                return "Semantic Label Map (PNG)"

    def create_writer(self) -> AnnotationWriter:
        """
        Build a writer of this format with default options.

        Returns
        -------
        AnnotationWriter
            Writer producing this format.
        """
        match self:
            case AnnotationFormat.COCO:
                return CocoWriter()
            case AnnotationFormat.YOLO:
                return YoloWriter()
            case AnnotationFormat.LABELME:
                return LabelMeWriter()
            case AnnotationFormat.SEMANTIC_MASK:
                return SemanticMaskWriter()
