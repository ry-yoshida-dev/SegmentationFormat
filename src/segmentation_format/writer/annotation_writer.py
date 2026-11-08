from abc import ABC, abstractmethod
from pathlib import Path

from ..annotation import AnnotationDataset


class AnnotationWriter(ABC):
    """
    Serializes an annotation dataset into one segmentation format.
    """

    @abstractmethod
    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write the dataset below ``output_directory``, creating the directory if needed.

        Parameters
        ----------
        dataset : AnnotationDataset
            Images, instances and class list to export.
        output_directory : Path
            Directory receiving the files.

        Returns
        -------
        tuple[Path, ...]
            Written files.
        """
