import abc
from pathlib import Path


class Aligner(abc.ABC):
    @abc.abstractmethod
    def align(
        self, reference_path: Path, image_paths: list[Path], temp_dir: str
    ) -> list[Path]:
        """Align images to a reference image.

        Args:
            reference_path: The path to the reference image to align all other images to
            image_paths: List of image paths to align to the reference
            temp_dir: Temporary directory to save aligned images

        Returns:
            List of paths to aligned images (including the reference image)
        """
        raise NotImplementedError
