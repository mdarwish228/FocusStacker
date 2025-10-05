import logging
from pathlib import Path

import cv2
import numpy as np

from ....common.enums import BlenderChoice
from ....common.exceptions import (
    FocusStackerDirectoryException,
    FocusStackerFileException,
    FocusStackerMemoryException,
)
from ...config.blender import LaplacianPyramidMaxSharpnessConfig
from ...models.decorators import Blender
from ...util.image import ImageUtils
from ...util.resource_monitor import ResourceMonitor
from .base import Blender as BlenderBase

logger = logging.getLogger(__name__)


@Blender(BlenderChoice.LAPLACIAN_PYRAMID_MAX_SHARPNESS)
class LaplacianPyramidMaxSharpnessBlender(BlenderBase):
    def __init__(self, *, config: LaplacianPyramidMaxSharpnessConfig) -> None:
        self.config = config

    def blend(
        self, image_paths: list[Path], destination_path: Path, temp_dir: str
    ) -> None:
        """Stack images using Laplacian pyramid merging and save to destination."""
        logger.info(
            f"Starting Laplacian pyramid stacking with {len(image_paths)} images"
        )

        # Load images from paths
        image_arrays = []
        for path in image_paths:
            img = ImageUtils.load_image(path)
            if img is not None:
                image_arrays.append(img)

        # Perform focus stacking with memory optimization
        stacked_image = self._focus_stack_pyramid_optimized(image_paths, temp_dir)

        # Save stacked result to destination
        ImageUtils.save_image(stacked_image, destination_path, quality=100)
        logger.info("Laplacian pyramid stacking completed successfully")

    def _calculate_enhanced_sharpness(self, img: np.ndarray) -> np.ndarray:
        """Calculate enhanced sharpness map using multiple focus detection methods."""
        # Constants for Sobel kernel size
        SOBEL_KERNEL_SIZE = 3  # Standard Sobel kernel size

        # Convert to grayscale for sharpness analysis
        gray = ImageUtils.rgb_to_grayscale(img)

        # Method 1: Laplacian variance (good for general sharpness)
        laplacian = ImageUtils.laplacian_filter(gray, cv2.CV_64F)
        laplacian_sharpness = np.abs(laplacian)

        # Method 2: Sobel gradient magnitude (good for edge detection)
        sobel_x = ImageUtils.sobel_filter(gray, 1, 0, cv2.CV_64F, SOBEL_KERNEL_SIZE)
        sobel_y = ImageUtils.sobel_filter(gray, 0, 1, cv2.CV_64F, SOBEL_KERNEL_SIZE)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Method 3: Local variance (good for texture and out-of-focus detection)
        kernel_size = self.config.local_variance_kernel_size
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (
            kernel_size * kernel_size
        )
        local_mean = ImageUtils.filter_2d(gray.astype(np.float32), kernel)
        local_variance = ImageUtils.filter_2d(
            (gray.astype(np.float32) - local_mean) ** 2, kernel
        )

        # Method 4: Tenengrad (gradient-based focus measure)
        tenengrad_map = gradient_magnitude**2

        # Method 5: Variance of gradient (detects smooth vs textured areas)
        gradient_variance = ImageUtils.filter_2d(
            gradient_magnitude.astype(np.float32), kernel
        )
        gradient_variance = np.abs(gradient_magnitude - gradient_variance)

        # Combine methods with weights from config
        combined_sharpness = (
            self.config.laplacian_weight * laplacian_sharpness
            + self.config.gradient_weight * gradient_magnitude
            + self.config.local_variance_weight * local_variance
            + self.config.tenengrad_weight * tenengrad_map
            + self.config.gradient_variance_weight * gradient_variance
        )

        # Normalize to 0-1 range
        combined_sharpness = (combined_sharpness - np.min(combined_sharpness)) / (
            np.max(combined_sharpness)
            - np.min(combined_sharpness)
            + self.config.epsilon
        )

        # Apply Gaussian smoothing for spatial consistency
        sigma = self.config.spatial_smoothing_sigma
        blur_size = self.config.spatial_smoothing_blur_size
        combined_sharpness = ImageUtils.gaussian_blur(
            combined_sharpness, (blur_size, blur_size), sigma
        )

        return combined_sharpness

    def _build_gaussian_pyramid(self, img: np.ndarray) -> list[np.ndarray]:
        """Build Gaussian pyramid with customizable downsampling factor."""
        pyramid = []
        current = img.astype(np.float32)
        pyramid.append(current)

        for _level in range(self.config.levels - 1):
            # Custom downsampling factor
            new_width = int(current.shape[1] * self.config.downsample_factor)
            new_height = int(current.shape[0] * self.config.downsample_factor)

            # Ensure minimum size to avoid too small images
            min_size = self.config.minimum_pyramid_size
            if new_width < min_size or new_height < min_size:
                break

            current = ImageUtils.resize_image(
                current,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,  # Best for downsampling
            )
            pyramid.append(current)

        return pyramid

    def _build_laplacian_pyramid(
        self, gaussian_pyramid: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Build Laplacian pyramid from Gaussian pyramid."""
        laplacian_pyramid = []

        # All levels except the last (smallest)
        for i in range(len(gaussian_pyramid) - 1):
            # Upsample the next level
            upsampled = ImageUtils.pyramid_up(gaussian_pyramid[i + 1])

            # Ensure same size (handle odd dimensions)
            if upsampled.shape != gaussian_pyramid[i].shape:
                upsampled = ImageUtils.resize_image(
                    upsampled,
                    (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]),
                    interpolation=cv2.INTER_CUBIC,  # Better interpolation
                )

            # Laplacian level = current - upsampled
            laplacian = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(laplacian)

        # Add the smallest Gaussian level
        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    def _reconstruct_from_laplacian_pyramid(
        self, laplacian_pyramid: list[np.ndarray]
    ) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        result = laplacian_pyramid[-1].copy()

        # Reconstruct from smallest to largest
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            # Upsample current result
            upsampled = ImageUtils.pyramid_up(result)

            # Ensure same size
            if upsampled.shape != laplacian_pyramid[i].shape:
                upsampled = ImageUtils.resize_image(
                    upsampled,
                    (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]),
                )

            # Add Laplacian level
            result = upsampled + laplacian_pyramid[i]

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _focus_stack_pyramid(self, images: list[np.ndarray]) -> np.ndarray:
        """Enhanced multi-scale focus stacking using advanced Laplacian pyramid merging."""
        # Build pyramids for all images
        gaussian_pyramids = []
        laplacian_pyramids = []

        for _, img in enumerate(images):
            gaussian_pyramid = self._build_gaussian_pyramid(img)
            laplacian_pyramid = self._build_laplacian_pyramid(gaussian_pyramid)
            gaussian_pyramids.append(gaussian_pyramid)
            laplacian_pyramids.append(laplacian_pyramid)

            # Free intermediate objects immediately
            del gaussian_pyramid

        # Calculate enhanced sharpness at each pyramid level using original images
        sharpness_maps = []
        for _i, img in enumerate(images):
            sharpness_map = self._calculate_enhanced_sharpness(img)
            sharpness_maps.append(sharpness_map)

        # Build sharpness pyramids
        sharpness_pyramids = []
        for _i, sharpness_map in enumerate(sharpness_maps):
            sharpness_pyramid = self._build_gaussian_pyramid(sharpness_map)
            sharpness_pyramids.append(sharpness_pyramid)

            # Free intermediate sharpness map
            del sharpness_map

        # Merge pyramids with enhanced sharpness detection and spatial consistency
        merged_pyramid = []
        num_levels = len(laplacian_pyramids[0])

        for level_idx in range(num_levels):
            logger.info(f"Processing pyramid level {level_idx + 1}/{num_levels}")
            level_shape = laplacian_pyramids[0][level_idx].shape

            # Initialize merged level
            merged_level = np.zeros(level_shape, dtype=np.float32)

            # Apply spatial consistency by smoothing sharpness maps at each level
            smoothed_sharpness = []
            blur_size = self.config.spatial_consistency_blur_size
            sigma = self.config.spatial_consistency_sigma
            for i in range(len(images)):
                # Apply Gaussian blur for spatial consistency
                smoothed = ImageUtils.gaussian_blur(
                    sharpness_pyramids[i][level_idx], (blur_size, blur_size), sigma
                )
                smoothed_sharpness.append(smoothed)

            # For each pixel, select from the image with highest sharpness (vectorized approach)

            # Vectorized approach: Find best image index for each pixel
            # Stack all sharpness maps into a 3D array (height, width, num_images)
            sharpness_stack = np.stack(smoothed_sharpness, axis=2)

            # Find the image index with maximum sharpness for each pixel
            best_image_indices = np.argmax(sharpness_stack, axis=2)

            # Create a mask for each image and select pixels vectorized
            merged_level = np.zeros(level_shape, dtype=np.float32)
            for img_idx in range(len(images)):
                # Create boolean mask for pixels where this image is best
                mask = best_image_indices == img_idx

                # Select pixels from this image where mask is True
                merged_level[mask] = laplacian_pyramids[img_idx][level_idx][mask]

            merged_pyramid.append(merged_level)

        # Reconstruct final image from merged pyramid
        result = self._reconstruct_from_laplacian_pyramid(merged_pyramid)

        # Free all large objects
        del (
            gaussian_pyramids,
            laplacian_pyramids,
            sharpness_maps,
            sharpness_pyramids,
            merged_pyramid,
        )
        ResourceMonitor.force_garbage_collection()

        return result

    def _focus_stack_pyramid_optimized(
        self, image_paths: list[Path], temp_dir: str
    ) -> np.ndarray:
        """Memory-optimized focus stacking - process one image at a time."""
        # Process each image individually and save to disk
        pyramid_paths = []
        sharpness_paths = []

        for i, path in enumerate(image_paths):
            # Load image
            img = ImageUtils.load_image(path)

            # Build pyramid
            gaussian_pyramid = self._build_gaussian_pyramid(img)
            laplacian_pyramid = self._build_laplacian_pyramid(gaussian_pyramid)

            # Calculate sharpness
            sharpness_map = self._calculate_enhanced_sharpness(img)
            sharpness_pyramid = self._build_gaussian_pyramid(sharpness_map)

            # Save to disk (uncompressed for speed)
            try:
                pyramid_path = Path(temp_dir) / f"pyramid_{i}.npz"
                sharpness_path = Path(temp_dir) / f"sharpness_{i}.npz"
            except Exception as e:
                raise FocusStackerDirectoryException(
                    f"Failed to create temporary file paths: {e}"
                ) from e

            # Check disk space before saving
            try:
                # Estimate file sizes for pyramid data
                pyramid_size_gb = sum(arr.nbytes for arr in laplacian_pyramid) / (
                    1024**3
                )
                sharpness_size_gb = sum(arr.nbytes for arr in sharpness_pyramid) / (
                    1024**3
                )
                total_size_gb = pyramid_size_gb + sharpness_size_gb

                ResourceMonitor.check_disk_space(temp_dir, total_size_gb)
            except Exception as e:
                logger.warning(f"Could not estimate disk space for pyramid data: {e}")

            try:
                np.savez(pyramid_path, *laplacian_pyramid)
                np.savez(sharpness_path, *sharpness_pyramid)
            except MemoryError as e:
                raise FocusStackerMemoryException(
                    f"Insufficient memory to save pyramid data for image {i}: {e}"
                ) from e
            except OSError as e:
                if "No space left" in str(e):
                    raise FocusStackerDirectoryException(
                        f"Insufficient disk space to save pyramid data for image {i}: {e}"
                    ) from e
                raise FocusStackerFileException(
                    f"Failed to save pyramid data to temporary files: {e}"
                ) from e
            except Exception as e:
                if isinstance(
                    e, (FocusStackerMemoryException, FocusStackerFileException)
                ):
                    raise
                raise FocusStackerFileException(
                    f"Failed to save pyramid data to temporary files: {e}"
                ) from e

            pyramid_paths.append(pyramid_path)
            sharpness_paths.append(sharpness_path)

            # Free memory immediately
            del (
                img,
                gaussian_pyramid,
                laplacian_pyramid,
                sharpness_map,
                sharpness_pyramid,
            )

            # Force garbage collection after each image processing
            ResourceMonitor.force_garbage_collection()

        # Load from disk for merging
        merged_pyramid = self._merge_pyramids_from_disk(pyramid_paths, sharpness_paths)

        # Reconstruct final image from merged pyramid
        result = self._reconstruct_from_laplacian_pyramid(merged_pyramid)

        # Free memory
        del merged_pyramid
        ResourceMonitor.force_garbage_collection()

        return result

    def _merge_pyramids_from_disk(
        self, pyramid_paths: list[Path], sharpness_paths: list[Path]
    ) -> list[np.ndarray]:
        """Merge pyramids from disk files."""
        # Load first pyramid to get dimensions
        first_pyramid_data = np.load(pyramid_paths[0])
        first_pyramid = [
            first_pyramid_data[f"arr_{i}"] for i in range(len(first_pyramid_data.files))
        ]
        num_levels = len(first_pyramid)
        merged_pyramid = []

        for level_idx in range(num_levels):
            level_shape = first_pyramid[level_idx].shape

            # Load all pyramids and sharpness maps for this level
            level_pyramids = []
            level_sharpness = []

            for pyramid_path, sharpness_path in zip(pyramid_paths, sharpness_paths):
                # Load pyramid level
                pyramid_data = np.load(pyramid_path)
                pyramid = [
                    pyramid_data[f"arr_{i}"] for i in range(len(pyramid_data.files))
                ]
                level_pyramids.append(pyramid[level_idx])

                # Load sharpness level
                sharpness_data = np.load(sharpness_path)
                sharpness_pyramid = [
                    sharpness_data[f"arr_{i}"] for i in range(len(sharpness_data.files))
                ]
                level_sharpness.append(sharpness_pyramid[level_idx])

                # Free memory
                del pyramid_data, pyramid, sharpness_data, sharpness_pyramid

            # Apply spatial consistency by smoothing sharpness maps
            smoothed_sharpness = []
            blur_size = self.config.spatial_consistency_blur_size
            sigma = self.config.spatial_consistency_sigma

            for sharpness in level_sharpness:
                smoothed = ImageUtils.gaussian_blur(
                    sharpness, (blur_size, blur_size), sigma
                )
                smoothed_sharpness.append(smoothed)

            # Vectorized pixel selection
            sharpness_stack = np.stack(smoothed_sharpness, axis=2)
            best_image_indices = np.argmax(sharpness_stack, axis=2)

            # Create merged level
            merged_level = np.zeros(level_shape, dtype=np.float32)
            for img_idx in range(len(level_pyramids)):
                mask = best_image_indices == img_idx
                merged_level[mask] = level_pyramids[img_idx][mask]

            merged_pyramid.append(merged_level)

            # Free memory
            del (
                level_pyramids,
                level_sharpness,
                smoothed_sharpness,
                sharpness_stack,
                best_image_indices,
                merged_level,
            )

        # Free memory
        del first_pyramid_data, first_pyramid
        ResourceMonitor.force_garbage_collection()

        return merged_pyramid
