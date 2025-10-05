import abc
from pathlib import Path


class Blender(abc.ABC):
    @abc.abstractmethod
    def blend(
        self, image_paths: list[Path], destination_path: Path, temp_dir: str
    ) -> None:
        """Blend images and save the result to the destination path.

        Args:
            image_paths: List of paths to aligned images to blend
            destination_path: The destination path where the result will be saved
            temp_dir: Optional temporary directory for intermediate files
        """
        raise NotImplementedError
