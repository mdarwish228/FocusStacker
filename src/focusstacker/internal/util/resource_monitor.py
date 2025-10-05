"""Resource monitoring utilities for memory and disk space management."""

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import psutil

from ...common.exceptions import (
    FocusStackerDirectoryException,
    FocusStackerMemoryException,
)

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources and provide early warnings for potential issues."""

    # Default thresholds (can be overridden)
    MEMORY_WARNING_THRESHOLD = 0.85  # 85% memory usage
    MEMORY_CRITICAL_THRESHOLD = 0.95  # 95% memory usage
    DISK_WARNING_THRESHOLD = 0.90  # 90% disk usage
    DISK_CRITICAL_THRESHOLD = 0.95  # 95% disk usage
    MIN_FREE_DISK_GB = 2.0  # Minimum 2GB free disk space

    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage as a percentage (0.0 to 1.0)."""
        return psutil.virtual_memory().percent / 100.0

    @staticmethod
    def get_memory_info() -> dict:
        """Get detailed memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
            "free_gb": memory.free / (1024**3),
        }

    @staticmethod
    def get_disk_usage(path: Union[str, Path]) -> dict:
        """Get disk usage information for a specific path."""
        path = Path(path)
        usage = shutil.disk_usage(path)
        return {
            "total_gb": usage.total / (1024**3),
            "used_gb": usage.used / (1024**3),
            "free_gb": usage.free / (1024**3),
            "percent": (usage.used / usage.total) * 100,
        }

    @classmethod
    def check_memory_availability(
        cls,
        estimated_usage_gb: Optional[float] = None,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
    ) -> None:
        """Check if sufficient memory is available.

        Args:
            estimated_usage_gb: Estimated additional memory needed in GB
            warning_threshold: Memory usage threshold for warnings (0.0-1.0)
            critical_threshold: Memory usage threshold for critical errors (0.0-1.0)

        Raises:
            FocusStackerMemoryException: If memory is critically low
        """
        warning_threshold = warning_threshold or cls.MEMORY_WARNING_THRESHOLD
        critical_threshold = critical_threshold or cls.MEMORY_CRITICAL_THRESHOLD

        memory_info = cls.get_memory_info()
        current_usage = memory_info["percent"] / 100.0

        # Check if adding estimated usage would exceed critical threshold
        if estimated_usage_gb:
            estimated_usage_percent = estimated_usage_gb / memory_info["total_gb"]
            projected_usage = current_usage + estimated_usage_percent

            if projected_usage > critical_threshold:
                raise FocusStackerMemoryException(
                    f"Projected memory usage ({projected_usage:.1%}) would exceed "
                    f"critical threshold ({critical_threshold:.1%}). "
                    f"Available: {memory_info['available_gb']:.1f}GB, "
                    f"Estimated needed: {estimated_usage_gb:.1f}GB"
                )

        # Check current usage
        if current_usage > critical_threshold:
            raise FocusStackerMemoryException(
                f"Memory usage ({current_usage:.1%}) exceeds critical threshold "
                f"({critical_threshold:.1%}). Available: {memory_info['available_gb']:.1f}GB"
            )
        elif current_usage > warning_threshold:
            logger.warning(
                f"Memory usage ({current_usage:.1%}) is high. Available: {memory_info['available_gb']:.1f}GB"
            )

    @classmethod
    def check_disk_space(
        cls,
        path: Union[str, Path],
        estimated_usage_gb: Optional[float] = None,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        min_free_gb: Optional[float] = None,
    ) -> None:
        """Check if sufficient disk space is available.

        Args:
            path: Path to check disk space for
            estimated_usage_gb: Estimated additional disk space needed in GB
            warning_threshold: Disk usage threshold for warnings (0.0-1.0)
            critical_threshold: Disk usage threshold for critical errors (0.0-1.0)
            min_free_gb: Minimum free space required in GB

        Raises:
            FocusStackerDirectoryException: If disk space is critically low
        """
        warning_threshold = warning_threshold or cls.DISK_WARNING_THRESHOLD
        critical_threshold = critical_threshold or cls.DISK_CRITICAL_THRESHOLD
        min_free_gb = min_free_gb or cls.MIN_FREE_DISK_GB

        disk_info = cls.get_disk_usage(path)
        current_usage = disk_info["percent"] / 100.0
        free_gb = disk_info["free_gb"]

        # Check minimum free space
        if free_gb < min_free_gb:
            raise FocusStackerDirectoryException(
                f"Insufficient free disk space: {free_gb:.1f}GB available, "
                f"minimum required: {min_free_gb:.1f}GB"
            )

        # Check if adding estimated usage would exceed critical threshold
        if estimated_usage_gb:
            if free_gb < estimated_usage_gb:
                raise FocusStackerDirectoryException(
                    f"Insufficient free disk space: {free_gb:.1f}GB available, "
                    f"estimated needed: {estimated_usage_gb:.1f}GB"
                )

            projected_usage = (disk_info["used_gb"] + estimated_usage_gb) / disk_info[
                "total_gb"
            ]
            if projected_usage > critical_threshold:
                raise FocusStackerDirectoryException(
                    f"Projected disk usage ({projected_usage:.1%}) would exceed "
                    f"critical threshold ({critical_threshold:.1%}). "
                    f"Free space: {free_gb:.1f}GB, estimated needed: {estimated_usage_gb:.1f}GB"
                )

        # Check current usage
        if current_usage > critical_threshold:
            raise FocusStackerDirectoryException(
                f"Disk usage ({current_usage:.1%}) exceeds critical threshold "
                f"({critical_threshold:.1%}). Free space: {free_gb:.1f}GB"
            )
        elif current_usage > warning_threshold:
            logger.warning(
                f"Disk usage ({current_usage:.1%}) is high. Free space: {free_gb:.1f}GB"
            )

    @classmethod
    def estimate_image_memory_usage(
        cls,
        width: int,
        height: int,
        channels: int = 3,
        dtype_bytes: int = 1,
        pyramid_levels: int = 5,
    ) -> float:
        """Estimate memory usage for image processing operations.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            channels: Number of color channels
            dtype_bytes: Bytes per pixel (1 for uint8, 4 for float32, 8 for float64)
            pyramid_levels: Number of pyramid levels

        Returns:
            Estimated memory usage in GB
        """
        # Base image size
        base_size = width * height * channels * dtype_bytes

        # Pyramid levels (each level is 1/4 the size of the previous)
        pyramid_size = 0
        current_size = base_size
        for _ in range(pyramid_levels):
            pyramid_size += current_size
            current_size = current_size // 4

        # Additional overhead for processing (sharpness maps, temporary arrays, etc.)
        overhead_multiplier = 3.0

        total_bytes = (base_size + pyramid_size) * overhead_multiplier
        return total_bytes / (1024**3)  # Convert to GB

    @classmethod
    def estimate_disk_usage(
        cls,
        num_images: int,
        width: int,
        height: int,
        channels: int = 3,
        dtype_bytes: int = 1,
        pyramid_levels: int = 5,
        compression_ratio: float = 0.3,
    ) -> float:
        """Estimate disk usage for temporary files.

        Args:
            num_images: Number of images to process
            width: Image width in pixels
            height: Image height in pixels
            channels: Number of color channels
            dtype_bytes: Bytes per pixel
            pyramid_levels: Number of pyramid levels
            compression_ratio: Compression ratio for saved files (0.1-1.0)

        Returns:
            Estimated disk usage in GB
        """
        # Estimate memory per image
        memory_per_image = cls.estimate_image_memory_usage(
            width, height, channels, dtype_bytes, pyramid_levels
        )

        # Apply compression ratio for disk storage
        disk_per_image = memory_per_image * compression_ratio

        # Total for all images plus some overhead
        total_gb = (disk_per_image * num_images) * 1.2  # 20% overhead

        return total_gb

    @classmethod
    def force_garbage_collection(cls) -> dict:
        """Force garbage collection and return memory info.

        Returns:
            Memory information after garbage collection
        """
        gc.collect()
        memory_info = cls.get_memory_info()

        return memory_info

    @classmethod
    def log_resource_status(cls, context: str = "") -> None:
        """Log current resource status.

        Args:
            context: Context description for the log message
        """
        memory_info = cls.get_memory_info()
        logger.info(
            f"{context} - Memory: {memory_info['used_gb']:.1f}GB/{memory_info['total_gb']:.1f}GB "
            f"({memory_info['percent']:.1f}%), Available: {memory_info['available_gb']:.1f}GB"
        )

    @classmethod
    def validate_temp_directory(cls, temp_dir: Union[str, Path]) -> None:
        """Validate that a temporary directory has sufficient space and permissions.

        Args:
            temp_dir: Path to temporary directory

        Raises:
            FocusStackerDirectoryException: If directory is invalid
        """
        temp_path = Path(temp_dir)

        # Check if directory exists and is writable
        if not temp_path.exists():
            raise FocusStackerDirectoryException(
                f"Temporary directory does not exist: {temp_path}"
            )

        if not os.access(temp_path, os.W_OK):
            raise FocusStackerDirectoryException(
                f"Temporary directory is not writable: {temp_path}"
            )

        # Check disk space
        cls.check_disk_space(temp_path)


def check_resources_before_processing(
    image_paths: list[Path],
    temp_dir: Union[str, Path],
    estimated_memory_gb: Optional[float] = None,
    estimated_disk_gb: Optional[float] = None,
) -> None:
    """Comprehensive resource check before starting image processing.

    Args:
        image_paths: List of image paths to process
        temp_dir: Temporary directory path
        estimated_memory_gb: Estimated memory usage in GB
        estimated_disk_gb: Estimated disk usage in GB

    Raises:
        FocusStackerMemoryException: If memory is insufficient
        FocusStackerDirectoryException: If disk space is insufficient
    """
    logger.info("Performing resource availability checks...")

    # Validate temporary directory
    ResourceMonitor.validate_temp_directory(temp_dir)

    # Check memory availability
    if estimated_memory_gb:
        ResourceMonitor.check_memory_availability(estimated_memory_gb)
    else:
        # Auto-estimate based on image sizes
        total_memory_gb = 0
        for image_path in image_paths:
            try:
                # Get image dimensions (this is a lightweight operation)
                import cv2

                img = cv2.imread(str(image_path))
                if img is not None:
                    height, width, channels = img.shape
                    memory_gb = ResourceMonitor.estimate_image_memory_usage(
                        width, height, channels
                    )
                    total_memory_gb += memory_gb
                del img  # Free immediately
            except Exception as e:
                logger.warning(f"Could not estimate memory for {image_path}: {e}")

        if total_memory_gb > 0:
            ResourceMonitor.check_memory_availability(total_memory_gb)

    # Check disk space
    if estimated_disk_gb:
        ResourceMonitor.check_disk_space(temp_dir, estimated_disk_gb)
    else:
        # Auto-estimate based on image sizes
        total_disk_gb = 0
        for image_path in image_paths:
            try:
                import cv2

                img = cv2.imread(str(image_path))
                if img is not None:
                    height, width, channels = img.shape
                    disk_gb = ResourceMonitor.estimate_disk_usage(
                        1, width, height, channels
                    )
                    total_disk_gb += disk_gb
                del img  # Free immediately
            except Exception as e:
                logger.warning(f"Could not estimate disk usage for {image_path}: {e}")

        if total_disk_gb > 0:
            ResourceMonitor.check_disk_space(temp_dir, total_disk_gb)

    # Log current resource status
    ResourceMonitor.log_resource_status("Before processing")

    logger.info("Resource checks completed successfully")
