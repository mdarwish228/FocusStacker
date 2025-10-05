"""Image utility functions for common OpenCV operations."""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from ...common.exceptions import (
    FocusStackerDirectoryException,
    FocusStackerFileException,
    FocusStackerImageProcessingException,
    FocusStackerMemoryException,
)
from .resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utility class for common image operations using OpenCV."""

    @staticmethod
    def load_image(path: Union[str, Path]) -> np.ndarray:
        """Load image from path and convert to RGB format.

        Args:
            path: Path to the image file

        Returns:
            Image array in RGB format

        Raises:
            FocusStackerFileException: If image cannot be loaded
            FocusStackerImageProcessingException: If image processing fails
            FocusStackerMemoryException: If insufficient memory
        """
        path = Path(path)

        # Check memory before loading large image
        try:
            # Get file size to estimate memory usage
            file_size_mb = path.stat().st_size / (1024 * 1024)
            estimated_memory_gb = (
                file_size_mb * 4 / 1024
            )  # Rough estimate: 4x file size in memory

            ResourceMonitor.check_memory_availability(estimated_memory_gb)
        except Exception as e:
            logger.warning(f"Could not estimate memory for {path}: {e}")

        try:
            img = cv2.imread(str(path))
            if img is None:
                raise FocusStackerFileException(f"Could not load image from {path}")

            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except MemoryError as e:
            raise FocusStackerMemoryException(
                f"Insufficient memory to load image {path}: {e}"
            ) from e
        except Exception as e:
            if isinstance(e, (FocusStackerFileException, FocusStackerMemoryException)):
                raise
            raise FocusStackerImageProcessingException(
                f"Image processing failed while loading {path}: {e}"
            ) from e

    @staticmethod
    def save_image(
        image: np.ndarray, path: Union[str, Path], quality: int = 100
    ) -> None:
        """Save image to path, converting from RGB to BGR format.

        Args:
            image: Image array in RGB format
            path: Destination path for the image
            quality: JPEG quality (1-100)

        Raises:
            FocusStackerFileException: If image cannot be saved
            FocusStackerImageProcessingException: If image processing fails
            FocusStackerDirectoryException: If insufficient disk space
        """
        path = Path(path)

        # Check disk space before saving
        try:
            # Estimate file size (rough approximation)
            height, width, channels = image.shape
            estimated_size_mb = (
                (height * width * channels) / (1024 * 1024) * 0.1
            )  # Rough compression estimate
            estimated_size_gb = estimated_size_mb / 1024

            ResourceMonitor.check_disk_space(path.parent, estimated_size_gb)
        except Exception as e:
            logger.warning(f"Could not estimate disk space for {path}: {e}")

        # Create parent directories if they don't exist
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise FocusStackerFileException(
                f"Permission denied creating directory for {path}: {e}"
            ) from e
        except Exception as e:
            raise FocusStackerFileException(
                f"Failed to create directory for {path}: {e}"
            ) from e

        # Convert from RGB back to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ensure image is in correct format for JPEG (uint8, 0-255 range)
        if img_bgr.dtype != np.uint8:
            # Clip values to valid range and convert to uint8
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

        # Save the image
        try:
            success = cv2.imwrite(
                str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if not success:
                raise FocusStackerFileException(f"Failed to save image to {path}")

        except MemoryError as e:
            raise FocusStackerMemoryException(
                f"Insufficient memory to save image {path}: {e}"
            ) from e
        except OSError as e:
            if "No space left" in str(e):
                raise FocusStackerDirectoryException(
                    f"Insufficient disk space to save image {path}: {e}"
                ) from e
            raise FocusStackerFileException(
                f"Failed to save image to {path}: {e}"
            ) from e
        except Exception as e:
            if isinstance(e, (FocusStackerFileException, FocusStackerMemoryException)):
                raise
            raise FocusStackerImageProcessingException(
                f"Image processing failed while saving {path}: {e}"
            ) from e

    @staticmethod
    def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale.

        Args:
            image: Image array in RGB format

        Returns:
            Grayscale image array
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    @staticmethod
    def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to BGR format.

        Args:
            image: Image array in RGB format

        Returns:
            Image array in BGR format
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB format.

        Args:
            image: Image array in BGR format

        Returns:
            Image array in RGB format
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def resize_image(
        image: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        """Resize image to specified dimensions.

        Args:
            image: Input image array
            size: Target size as (width, height)
            interpolation: Interpolation method

        Returns:
            Resized image array
        """
        return cv2.resize(image, size, interpolation=interpolation)

    @staticmethod
    def gaussian_blur(
        image: np.ndarray, kernel_size: tuple[int, int], sigma: float
    ) -> np.ndarray:
        """Apply Gaussian blur to image.

        Args:
            image: Input image array
            kernel_size: Blur kernel size as (width, height)
            sigma: Gaussian standard deviation

        Returns:
            Blurred image array
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)

    @staticmethod
    def laplacian_filter(image: np.ndarray, ddepth: int = cv2.CV_64F) -> np.ndarray:
        """Apply Laplacian filter for edge detection.

        Args:
            image: Input image array
            ddepth: Output image depth

        Returns:
            Laplacian filtered image array
        """
        return cv2.Laplacian(image, ddepth)

    @staticmethod
    def sobel_filter(
        image: np.ndarray, dx: int, dy: int, ddepth: int = cv2.CV_64F, ksize: int = 3
    ) -> np.ndarray:
        """Apply Sobel filter for gradient calculation.

        Args:
            image: Input image array
            dx: Order of derivative in x direction
            dy: Order of derivative in y direction
            ddepth: Output image depth
            ksize: Kernel size

        Returns:
            Sobel filtered image array
        """
        return cv2.Sobel(image, ddepth, dx, dy, ksize=ksize)

    @staticmethod
    def filter_2d(
        image: np.ndarray, kernel: np.ndarray, ddepth: int = -1
    ) -> np.ndarray:
        """Apply 2D filter to image.

        Args:
            image: Input image array
            kernel: Filter kernel
            ddepth: Output image depth

        Returns:
            Filtered image array
        """
        return cv2.filter2D(image, ddepth, kernel)

    @staticmethod
    def pyramid_down(image: np.ndarray) -> np.ndarray:
        """Downsample image using Gaussian pyramid.

        Args:
            image: Input image array

        Returns:
            Downsampled image array
        """
        return cv2.pyrDown(image)

    @staticmethod
    def pyramid_up(image: np.ndarray) -> np.ndarray:
        """Upsample image using Gaussian pyramid.

        Args:
            image: Input image array

        Returns:
            Upsampled image array
        """
        return cv2.pyrUp(image)

    @staticmethod
    def warp_perspective(
        image: np.ndarray, matrix: np.ndarray, size: tuple[int, int]
    ) -> np.ndarray:
        """Apply perspective transformation to image.

        Args:
            image: Input image array
            matrix: Transformation matrix
            size: Output image size as (width, height)

        Returns:
            Transformed image array
        """
        return cv2.warpPerspective(image, matrix, size)

    @staticmethod
    def create_sift_detector(
        nfeatures: int = 8000,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.02,
        edge_threshold: float = 15.0,
        sigma: float = 1.2,
    ) -> cv2.SIFT:
        """Create SIFT detector with specified parameters.

        Args:
            nfeatures: Maximum number of features
            n_octave_layers: Number of layers in each octave
            contrast_threshold: Contrast threshold
            edge_threshold: Edge threshold
            sigma: Sigma value

        Returns:
            SIFT detector object
        """
        return cv2.SIFT_create(  # type: ignore[attr-defined]
            nfeatures=nfeatures,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

    @staticmethod
    def create_flann_matcher(
        algorithm: int = 1, trees: int = 8, checks: int = 128
    ) -> cv2.FlannBasedMatcher:
        """Create FLANN-based matcher.

        Args:
            algorithm: Index algorithm
            trees: Number of trees
            checks: Number of checks

        Returns:
            FLANN matcher object
        """
        index_params = {
            "algorithm": algorithm,
            "trees": trees,
        }
        search_params = {"checks": checks}
        return cv2.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def find_homography(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        method: int = cv2.RANSAC,
        ransac_reproj_threshold: float = 10.0,
        max_iters: int = 10000,
        confidence: float = 0.999,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find homography matrix between point sets.

        Args:
            src_points: Source points
            dst_points: Destination points
            method: Method for homography estimation
            ransac_reproj_threshold: RANSAC reprojection threshold
            max_iters: Maximum number of iterations
            confidence: Confidence level

        Returns:
            Tuple of (homography_matrix, mask)
        """
        return cv2.findHomography(
            src_points,
            dst_points,
            method=method,
            ransacReprojThreshold=ransac_reproj_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
