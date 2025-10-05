import logging
from pathlib import Path

import cv2
import numpy as np

from ....common.enums import BlenderChoice
from ...config.blender import LaplacianPyramidBalancedConfig
from ...models.decorators import Blender
from ...util.image import ImageUtils
from ...util.resource_monitor import ResourceMonitor
from .base import Blender as BlenderBase

logger = logging.getLogger(__name__)


@Blender(BlenderChoice.LAPLACIAN_PYRAMID_BALANCED)
class LaplacianPyramidBalancedBlender(BlenderBase):
    def __init__(self, *, config: LaplacianPyramidBalancedConfig) -> None:
        self.config = config

    def _calculate_enhanced_sharpness(self, img: np.ndarray) -> np.ndarray:
        """Calculate enhanced sharpness map using multiple focus detection methods."""
        # Convert to grayscale for sharpness analysis
        gray = ImageUtils.rgb_to_grayscale(img)

        # Method 1: Laplacian variance (good for general sharpness)
        laplacian = ImageUtils.laplacian_filter(gray, cv2.CV_64F)
        laplacian_sharpness = np.abs(laplacian)

        # Method 2: Sobel gradient magnitude (good for edge detection)
        SOBEL_KERNEL_SIZE = 3
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
        if self.config.spatial_smoothing_blur_size > 0:
            sigma = self.config.spatial_smoothing_sigma
            blur_size = self.config.spatial_smoothing_blur_size
            # Ensure float32 for OpenCV operations
            combined_sharpness = ImageUtils.gaussian_blur(
                combined_sharpness.astype(np.float32), (blur_size, blur_size), sigma
            )

        return combined_sharpness

    def _build_gaussian_pyramid(self, img: np.ndarray) -> list[np.ndarray]:
        """Build Gaussian pyramid using optimized cv2.pyrDown."""
        pyramid = []
        current = img.astype(np.float32)
        pyramid.append(current)

        for _level in range(self.config.levels - 1):
            # Use cv2.pyrDown for optimized Gaussian + 2x downsample
            current = ImageUtils.pyramid_down(current)

            # Ensure minimum size to avoid too small images
            min_size = self.config.minimum_pyramid_size
            if current.shape[0] < min_size or current.shape[1] < min_size:
                break

            pyramid.append(current)

        return pyramid

    def _build_laplacian_pyramid(
        self, gaussian_pyramid: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Build Laplacian pyramid from Gaussian pyramid using float16."""
        laplacian_pyramid = []

        # All levels except the last (smallest)
        for i in range(len(gaussian_pyramid) - 1):
            # Upsample the next level using cv2.pyrUp
            upsampled = ImageUtils.pyramid_up(gaussian_pyramid[i + 1])

            # Ensure same size (handle odd dimensions)
            if upsampled.shape != gaussian_pyramid[i].shape:
                upsampled = ImageUtils.resize_image(
                    upsampled,
                    (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            # Laplacian level = current - upsampled (use float16 for memory)
            laplacian = (gaussian_pyramid[i] - upsampled).astype(np.float16)
            laplacian_pyramid.append(laplacian)

        # Add the smallest Gaussian level (use float16)
        laplacian_pyramid.append(gaussian_pyramid[-1].astype(np.float16))

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

        return result

    def _merge_pyramids_soft_blend(
        self,
        level_pyramids: list[np.ndarray],
        sharpness_stack: np.ndarray,
    ) -> np.ndarray:
        """Merge Laplacian pyramids with soft blending (weighted by sharpness).

        This avoids hard seams/bars at pyramid edges.
        """
        # get shape
        h, w = level_pyramids[0].shape[:2]

        # --- soft weights ---
        # normalize sharpness to [0..1] weights across images
        weights = sharpness_stack.astype(np.float32)
        weights_sum = np.sum(weights, axis=2, keepdims=True) + self.config.epsilon
        weights /= weights_sum  # each pixel: sum of weights = 1

        # --- blend pyramids ---
        merged_level = np.zeros((h, w, level_pyramids[0].shape[2]), dtype=np.float32)
        for i, level_img in enumerate(level_pyramids):
            # expand weights for color channels if needed
            w_map = weights[..., i]
            if level_img.ndim == 3 and w_map.ndim == 2:
                w_map = w_map[..., None]
            merged_level += level_img.astype(np.float32) * w_map

        return merged_level

    def _focus_stack_pyramid_optimized(
        self, image_paths: list[Path], temp_dir: str
    ) -> np.ndarray:
        """Memory-optimized focus stacking - process level-by-level."""
        # Determine number of levels by building pyramid for first image
        first_img = ImageUtils.load_image(image_paths[0])
        first_gaussian = self._build_gaussian_pyramid(first_img)
        num_levels = len(first_gaussian)

        # Free first image data
        del first_img, first_gaussian
        ResourceMonitor.force_garbage_collection()

        # Process level-by-level to minimize memory usage
        merged_pyramid = []

        for level_idx in range(num_levels):
            logger.info(f"Processing pyramid level {level_idx + 1}/{num_levels}")

            # Collect this level from all images
            level_pyramids = []
            level_sharpness = []

            for path in image_paths:
                # Load image
                img = ImageUtils.load_image(path)

                # Build Gaussian pyramid
                gaussian_pyramid = self._build_gaussian_pyramid(img)

                # Get this level's Laplacian
                if level_idx < len(gaussian_pyramid) - 1:
                    # Regular Laplacian level
                    upsampled = ImageUtils.pyramid_up(gaussian_pyramid[level_idx + 1])
                    if upsampled.shape != gaussian_pyramid[level_idx].shape:
                        upsampled = ImageUtils.resize_image(
                            upsampled,
                            (
                                gaussian_pyramid[level_idx].shape[1],
                                gaussian_pyramid[level_idx].shape[0],
                            ),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    laplacian_level = (gaussian_pyramid[level_idx] - upsampled).astype(
                        np.float16
                    )
                else:
                    # Smallest level (last Gaussian)
                    laplacian_level = gaussian_pyramid[level_idx].astype(np.float16)

                level_pyramids.append(laplacian_level)

                # Calculate sharpness for this level only (grayscale)
                sharpness_map = self._calculate_enhanced_sharpness(img)
                sharpness_gaussian = self._build_gaussian_pyramid(sharpness_map)
                sharpness_level = sharpness_gaussian[level_idx].astype(np.float16)
                level_sharpness.append(sharpness_level)

                # Free memory immediately
                del img, sharpness_map, sharpness_gaussian
                if level_idx < len(gaussian_pyramid) - 1:
                    del upsampled
                del gaussian_pyramid, laplacian_level, sharpness_level

                # Force garbage collection after each image processing
                ResourceMonitor.force_garbage_collection()

            # Apply spatial consistency by smoothing sharpness maps
            smoothed_sharpness = []
            blur_size = self.config.spatial_consistency_blur_size
            sigma = self.config.spatial_consistency_sigma

            for sharpness in level_sharpness:
                if blur_size > 0:
                    # Ensure float32 for OpenCV operations
                    smoothed = ImageUtils.gaussian_blur(
                        sharpness.astype(np.float32), (blur_size, blur_size), sigma
                    )
                else:
                    smoothed = sharpness
                smoothed_sharpness.append(smoothed)

            # Stack all sharpness maps into a 3D array (height, width, num_images)
            sharpness_stack = np.stack(smoothed_sharpness, axis=2)

            # Merge using soft blending
            merged_level = self._merge_pyramids_soft_blend(
                level_pyramids, sharpness_stack
            )
            merged_pyramid.append(merged_level)

            # Free memory
            del (
                level_pyramids,
                level_sharpness,
                smoothed_sharpness,
                sharpness_stack,
                merged_level,
            )

        # Reconstruct final image from merged pyramid
        result = self._reconstruct_from_laplacian_pyramid(merged_pyramid)

        # Free memory
        del merged_pyramid
        ResourceMonitor.force_garbage_collection()

        return result

    def blend(
        self, image_paths: list[Path], destination_path: Path, temp_dir: str
    ) -> None:
        """Stack images using soft-blended Laplacian pyramid merging and save to destination."""
        logger.info(
            f"Starting balanced Laplacian pyramid stacking with {len(image_paths)} images"
        )

        # Perform focus stacking with memory optimization
        stacked_image = self._focus_stack_pyramid_optimized(image_paths, temp_dir)

        # Save stacked result to destination
        ImageUtils.save_image(stacked_image, destination_path, quality=100)
        logger.info(f"Saved stacked image to {destination_path}")
