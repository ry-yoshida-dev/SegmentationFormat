from collections.abc import Sequence
from pathlib import Path


class ClassListFile:
    """
    Text file listing class names, one per line in id order.

    Parameters
    ----------
    file_name : str, optional
        Name of the file.
    """

    def __init__(self, file_name: str = "classes.txt") -> None:
        self.file_name: str = file_name

    def path(self, output_directory: Path) -> Path:
        """
        Location of the file.

        Parameters
        ----------
        output_directory : Path
            Directory receiving the file.

        Returns
        -------
        Path
            ``output_directory / file_name``.
        """
        return output_directory / self.file_name

    def write(self, class_names: Sequence[str], output_directory: Path) -> Path:
        """
        Write the class names as UTF-8, replacing an existing file.

        Parameters
        ----------
        class_names : Sequence[str]
            Names in id order.
        output_directory : Path
            Existing directory receiving the file.

        Returns
        -------
        Path
            The written file.
        """
        output_path: Path = self.path(output_directory)
        output_path.write_text("".join(f"{class_name}\n" for class_name in class_names), encoding="utf-8", newline="\n")
        return output_path
