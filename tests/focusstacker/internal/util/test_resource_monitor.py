"""Unit tests for ResourceMonitor class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from focusstacker.common.exceptions import (
    FocusStackerDirectoryException,
    FocusStackerMemoryException,
)
from focusstacker.internal.util.resource_monitor import (
    ResourceMonitor,
    check_resources_before_processing,
)


class TestResourceMonitorMemory:
    """Test memory monitoring functionality."""

    @patch("focusstacker.internal.util.resource_monitor.psutil.virtual_memory")
    def test_get_memory_usage(self, mock_virtual_memory: Mock) -> None:
        """Test getting memory usage percentage."""
        # Setup mock
        mock_memory = Mock()
        mock_memory.percent = 75.5
        mock_virtual_memory.return_value = mock_memory

        # Execute
        usage = ResourceMonitor.get_memory_usage()

        # Verify
        assert usage == 0.755
        mock_virtual_memory.assert_called_once()

    @patch("focusstacker.internal.util.resource_monitor.psutil.virtual_memory")
    def test_get_memory_info(self, mock_virtual_memory: Mock) -> None:
        """Test getting detailed memory information."""
        # Setup mock
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_memory.used = 8 * 1024**3  # 8GB
        mock_memory.percent = 50.0
        mock_memory.free = 4 * 1024**3  # 4GB
        mock_virtual_memory.return_value = mock_memory

        # Execute
        info = ResourceMonitor.get_memory_info()

        # Verify
        assert info["total_gb"] == 16.0
        assert info["available_gb"] == 8.0
        assert info["used_gb"] == 8.0
        assert info["percent"] == 50.0
        assert info["free_gb"] == 4.0

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.get_memory_info"
    )
    def test_memory_availability_success(self, mock_get_memory_info: Mock) -> None:
        """Test successful memory availability check."""
        # Setup mock - low memory usage
        mock_get_memory_info.return_value = {
            "total_gb": 16.0,
            "available_gb": 12.0,
            "used_gb": 4.0,
            "percent": 25.0,
            "free_gb": 8.0,
        }

        # Execute - should not raise exception
        ResourceMonitor.check_memory_availability(estimated_usage_gb=2.0)

        # Verify
        mock_get_memory_info.assert_called_once()

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.get_memory_info"
    )
    def test_memory_warning_threshold_exceeded(
        self, mock_get_memory_info: Mock
    ) -> None:
        """Test memory warning threshold exceeded."""
        # Setup mock - high memory usage (90%)
        mock_get_memory_info.return_value = {
            "total_gb": 16.0,
            "available_gb": 1.6,
            "used_gb": 14.4,
            "percent": 90.0,
            "free_gb": 1.6,
        }

        # Execute - should not raise exception but log warning
        with patch("focusstacker.internal.util.resource_monitor.logger") as mock_logger:
            ResourceMonitor.check_memory_availability()  # No estimated usage to avoid projection check

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Memory usage (90.0%) is high" in warning_call

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.get_memory_info"
    )
    def test_memory_critical_threshold_exceeded(
        self, mock_get_memory_info: Mock
    ) -> None:
        """Test memory critical threshold exceeded."""
        # Setup mock - critical memory usage (96%)
        mock_get_memory_info.return_value = {
            "total_gb": 16.0,
            "available_gb": 0.64,
            "used_gb": 15.36,
            "percent": 96.0,
            "free_gb": 0.64,
        }

        # Execute and verify exception
        with pytest.raises(FocusStackerMemoryException) as exc_info:
            ResourceMonitor.check_memory_availability()  # No estimated usage to avoid projection check

        assert "Memory usage (96.0%) exceeds critical threshold" in str(exc_info.value)

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.get_memory_info"
    )
    def test_estimated_memory_projection_exceeds_critical(
        self, mock_get_memory_info: Mock
    ) -> None:
        """Test that estimated memory usage projection exceeds critical threshold."""
        # Setup mock - moderate memory usage (70%)
        mock_get_memory_info.return_value = {
            "total_gb": 16.0,
            "available_gb": 4.8,
            "used_gb": 11.2,
            "percent": 70.0,
            "free_gb": 4.8,
        }

        # Execute with large estimated usage that would push over critical threshold
        with pytest.raises(FocusStackerMemoryException) as exc_info:
            ResourceMonitor.check_memory_availability(estimated_usage_gb=8.0)

        assert "Projected memory usage" in str(exc_info.value)
        assert "would exceed critical threshold" in str(exc_info.value)

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.get_memory_info"
    )
    def test_custom_memory_thresholds(self, mock_get_memory_info: Mock) -> None:
        """Test custom memory thresholds."""
        # Setup mock - 80% memory usage
        mock_get_memory_info.return_value = {
            "total_gb": 16.0,
            "available_gb": 3.2,
            "used_gb": 12.8,
            "percent": 80.0,
            "free_gb": 3.2,
        }

        # Execute with custom critical threshold (75%)
        with pytest.raises(FocusStackerMemoryException) as exc_info:
            ResourceMonitor.check_memory_availability(
                critical_threshold=0.75,  # No estimated usage to avoid projection check
            )

        assert "Memory usage (80.0%) exceeds critical threshold (75.0%)" in str(
            exc_info.value
        )


class TestResourceMonitorDisk:
    """Test disk space monitoring functionality."""

    @patch("focusstacker.internal.util.resource_monitor.shutil.disk_usage")
    def test_get_disk_usage(self, mock_disk_usage: Mock, temp_dir: Path) -> None:
        """Test getting disk usage information."""
        # Setup mock
        mock_usage = Mock()
        mock_usage.total = 1000 * 1024**3  # 1000GB
        mock_usage.used = 600 * 1024**3  # 600GB
        mock_usage.free = 400 * 1024**3  # 400GB
        mock_disk_usage.return_value = mock_usage

        # Execute
        usage = ResourceMonitor.get_disk_usage(temp_dir)

        # Verify
        assert usage["total_gb"] == 1000.0
        assert usage["used_gb"] == 600.0
        assert usage["free_gb"] == 400.0
        assert usage["percent"] == 60.0

    @patch("focusstacker.internal.util.resource_monitor.ResourceMonitor.get_disk_usage")
    def test_disk_warning_threshold_exceeded(
        self, mock_get_disk_usage: Mock, temp_dir: Path
    ) -> None:
        """Test disk warning threshold exceeded."""
        # Setup mock - high disk usage (92%)
        mock_get_disk_usage.return_value = {
            "total_gb": 1000.0,
            "used_gb": 920.0,
            "free_gb": 80.0,
            "percent": 92.0,
        }

        # Execute - should not raise exception but log warning
        with patch("focusstacker.internal.util.resource_monitor.logger") as mock_logger:
            ResourceMonitor.check_disk_space(temp_dir, estimated_usage_gb=10.0)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Disk usage (92.0%) is high" in warning_call

    @patch("focusstacker.internal.util.resource_monitor.ResourceMonitor.get_disk_usage")
    def test_disk_critical_threshold_exceeded(
        self, mock_get_disk_usage: Mock, temp_dir: Path
    ) -> None:
        """Test disk critical threshold exceeded."""
        # Setup mock - critical disk usage (96%)
        mock_get_disk_usage.return_value = {
            "total_gb": 1000.0,
            "used_gb": 960.0,
            "free_gb": 40.0,
            "percent": 96.0,
        }

        # Execute and verify exception
        with pytest.raises(FocusStackerDirectoryException) as exc_info:
            ResourceMonitor.check_disk_space(
                temp_dir
            )  # No estimated usage to avoid projection check

        assert "Disk usage (96.0%) exceeds critical threshold" in str(exc_info.value)

    @patch("focusstacker.internal.util.resource_monitor.ResourceMonitor.get_disk_usage")
    def test_minimum_free_space_insufficient(
        self, mock_get_disk_usage: Mock, temp_dir: Path
    ) -> None:
        """Test insufficient minimum free space."""
        # Setup mock - low free space (1GB)
        mock_get_disk_usage.return_value = {
            "total_gb": 1000.0,
            "used_gb": 999.0,
            "free_gb": 1.0,
            "percent": 99.9,
        }

        # Execute and verify exception
        with pytest.raises(FocusStackerDirectoryException) as exc_info:
            ResourceMonitor.check_disk_space(temp_dir)

        assert (
            "Insufficient free disk space: 1.0GB available, minimum required: 2.0GB"
            in str(exc_info.value)
        )

    @patch("focusstacker.internal.util.resource_monitor.ResourceMonitor.get_disk_usage")
    def test_estimated_disk_projection_exceeds_critical(
        self, mock_get_disk_usage: Mock, temp_dir: Path
    ) -> None:
        """Test that estimated disk usage projection exceeds critical threshold."""
        # Setup mock - moderate disk usage (80%)
        mock_get_disk_usage.return_value = {
            "total_gb": 1000.0,
            "used_gb": 800.0,
            "free_gb": 200.0,
            "percent": 80.0,
        }

        # Execute with large estimated usage that would push over critical threshold
        with pytest.raises(FocusStackerDirectoryException) as exc_info:
            ResourceMonitor.check_disk_space(temp_dir, estimated_usage_gb=200.0)

        assert "Projected disk usage" in str(exc_info.value)
        assert "would exceed critical threshold" in str(exc_info.value)

    @patch("focusstacker.internal.util.resource_monitor.ResourceMonitor.get_disk_usage")
    def test_insufficient_free_space_for_estimated_usage(
        self, mock_get_disk_usage: Mock, temp_dir: Path
    ) -> None:
        """Test insufficient free space for estimated usage."""
        # Setup mock - moderate free space (50GB)
        mock_get_disk_usage.return_value = {
            "total_gb": 1000.0,
            "used_gb": 950.0,
            "free_gb": 50.0,
            "percent": 95.0,
        }

        # Execute with estimated usage larger than free space
        with pytest.raises(FocusStackerDirectoryException) as exc_info:
            ResourceMonitor.check_disk_space(temp_dir, estimated_usage_gb=100.0)

        assert (
            "Insufficient free disk space: 50.0GB available, estimated needed: 100.0GB"
            in str(exc_info.value)
        )


class TestResourceMonitorEstimation:
    """Test resource estimation algorithms."""

    def test_estimate_image_memory_usage(self) -> None:
        """Test image memory usage estimation."""
        # Test with standard image dimensions
        memory_gb = ResourceMonitor.estimate_image_memory_usage(
            width=1920, height=1080, channels=3, dtype_bytes=1, pyramid_levels=5
        )

        # Should return a reasonable estimate
        assert memory_gb > 0
        assert memory_gb < 1.0  # Should be less than 1GB for 1080p image

    def test_estimate_image_memory_usage_large_image(self) -> None:
        """Test memory estimation for large image."""
        # Test with 4K image
        memory_gb = ResourceMonitor.estimate_image_memory_usage(
            width=3840, height=2160, channels=3, dtype_bytes=1, pyramid_levels=5
        )

        # Should return a larger estimate
        assert memory_gb > 0
        assert memory_gb < 5.0  # Should be less than 5GB for 4K image

    def test_estimate_image_memory_usage_float32(self) -> None:
        """Test memory estimation with float32 data type."""
        # Test with float32 (4 bytes per pixel)
        memory_gb = ResourceMonitor.estimate_image_memory_usage(
            width=1920, height=1080, channels=3, dtype_bytes=4, pyramid_levels=5
        )

        # Should be larger than uint8 version
        memory_gb_uint8 = ResourceMonitor.estimate_image_memory_usage(
            width=1920, height=1080, channels=3, dtype_bytes=1, pyramid_levels=5
        )

        assert memory_gb > memory_gb_uint8

    def test_estimate_disk_usage(self) -> None:
        """Test disk usage estimation."""
        # Test with multiple images
        disk_gb = ResourceMonitor.estimate_disk_usage(
            num_images=5,
            width=1920,
            height=1080,
            channels=3,
            dtype_bytes=1,
            pyramid_levels=5,
            compression_ratio=0.3,
        )

        # Should return a reasonable estimate
        assert disk_gb > 0
        assert disk_gb < 2.0  # Should be less than 2GB for 5 images

    def test_estimate_disk_usage_compression_ratio(self) -> None:
        """Test disk usage estimation with different compression ratios."""
        # Test with high compression
        disk_gb_high_compression = ResourceMonitor.estimate_disk_usage(
            num_images=3,
            width=1920,
            height=1080,
            channels=3,
            dtype_bytes=1,
            pyramid_levels=5,
            compression_ratio=0.1,
        )

        # Test with low compression
        disk_gb_low_compression = ResourceMonitor.estimate_disk_usage(
            num_images=3,
            width=1920,
            height=1080,
            channels=3,
            dtype_bytes=1,
            pyramid_levels=5,
            compression_ratio=0.5,
        )

        # High compression should use less disk space
        assert disk_gb_high_compression < disk_gb_low_compression


class TestResourceMonitorValidation:
    """Test directory validation functionality."""

    def test_validate_temp_directory_success(self) -> None:
        """Test successful temporary directory validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise exception
            ResourceMonitor.validate_temp_directory(temp_dir)

    def test_validate_temp_directory_not_exists(self) -> None:
        """Test validation of non-existent directory."""
        non_existent_dir = Path("/tmp/non_existent_directory_12345")

        # Execute and verify exception
        with pytest.raises(FocusStackerDirectoryException) as exc_info:
            ResourceMonitor.validate_temp_directory(non_existent_dir)

        assert "Temporary directory does not exist" in str(exc_info.value)

    @patch("focusstacker.internal.util.resource_monitor.os.access")
    def test_validate_temp_directory_not_writable(self, mock_access: Mock) -> None:
        """Test validation of non-writable directory."""
        # Setup mock to simulate non-writable directory
        mock_access.return_value = False

        # Execute and verify exception
        with pytest.raises(FocusStackerDirectoryException) as exc_info:
            ResourceMonitor.validate_temp_directory("/tmp")

        assert "Temporary directory is not writable" in str(exc_info.value)

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.check_disk_space"
    )
    def test_validate_temp_directory_calls_disk_check(
        self, mock_check_disk_space: Mock
    ) -> None:
        """Test that validation calls disk space check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Execute
            ResourceMonitor.validate_temp_directory(temp_dir)

            # Verify disk space check was called
            mock_check_disk_space.assert_called_once()


class TestResourceMonitorUtilities:
    """Test utility functions."""

    @patch("focusstacker.internal.util.resource_monitor.gc.collect")
    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.get_memory_info"
    )
    def test_force_garbage_collection(
        self, mock_get_memory_info: Mock, mock_gc_collect: Mock
    ) -> None:
        """Test forced garbage collection."""
        # Setup mock
        mock_get_memory_info.return_value = {
            "total_gb": 16.0,
            "available_gb": 8.0,
            "used_gb": 8.0,
            "percent": 50.0,
            "free_gb": 4.0,
        }

        # Execute
        result = ResourceMonitor.force_garbage_collection()

        # Verify
        mock_gc_collect.assert_called_once()
        mock_get_memory_info.assert_called_once()
        assert result == mock_get_memory_info.return_value

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.get_memory_info"
    )
    def test_log_resource_status(self, mock_get_memory_info: Mock) -> None:
        """Test resource status logging."""
        # Setup mock
        mock_get_memory_info.return_value = {
            "total_gb": 16.0,
            "available_gb": 8.0,
            "used_gb": 8.0,
            "percent": 50.0,
            "free_gb": 4.0,
        }

        # Execute
        with patch("focusstacker.internal.util.resource_monitor.logger") as mock_logger:
            ResourceMonitor.log_resource_status("Test context")

        # Verify
        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "Test context" in log_call
        assert "Memory: 8.0GB/16.0GB (50.0%)" in log_call
        assert "Available: 8.0GB" in log_call


class TestCheckResourcesBeforeProcessing:
    """Test comprehensive resource checking function."""

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.validate_temp_directory"
    )
    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.check_memory_availability"
    )
    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.check_disk_space"
    )
    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.log_resource_status"
    )
    def test_check_resources_with_estimates(
        self,
        mock_log_status: Mock,
        mock_check_disk: Mock,
        mock_check_memory: Mock,
        mock_validate_temp: Mock,
        temp_dir: Path,
        aligned_image_paths: list[Path],
    ) -> None:
        """Test resource checking with provided estimates."""
        # Execute
        check_resources_before_processing(
            image_paths=aligned_image_paths,
            temp_dir=temp_dir,
            estimated_memory_gb=2.0,
            estimated_disk_gb=1.0,
        )

        # Verify all checks were called
        mock_validate_temp.assert_called_once_with(temp_dir)
        mock_check_memory.assert_called_once_with(2.0)
        mock_check_disk.assert_called_once_with(temp_dir, 1.0)
        mock_log_status.assert_called_once_with("Before processing")

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.validate_temp_directory"
    )
    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.check_memory_availability"
    )
    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.check_disk_space"
    )
    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.log_resource_status"
    )
    @patch("cv2.imread")
    def test_check_resources_auto_estimate(
        self,
        mock_imread: Mock,
        mock_log_status: Mock,
        mock_check_disk: Mock,
        mock_check_memory: Mock,
        mock_validate_temp: Mock,
        temp_dir: Path,
        aligned_image_paths: list[Path],
    ) -> None:
        """Test resource checking with auto-estimation."""
        # Setup mock image
        mock_image = Mock()
        mock_image.shape = (1080, 1920, 3)  # height, width, channels
        mock_imread.return_value = mock_image

        # Execute
        check_resources_before_processing(
            image_paths=aligned_image_paths,
            temp_dir=temp_dir,
        )

        # Verify all checks were called
        mock_validate_temp.assert_called_once_with(temp_dir)
        mock_check_memory.assert_called_once()
        mock_check_disk.assert_called_once()
        mock_log_status.assert_called_once_with("Before processing")

        # Verify images were read for estimation (called twice: once for memory, once for disk)
        assert mock_imread.call_count == len(aligned_image_paths) * 2

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.validate_temp_directory"
    )
    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.log_resource_status"
    )
    @patch("cv2.imread")
    def test_check_resources_image_read_failure(
        self,
        mock_imread: Mock,
        mock_log_status: Mock,
        mock_validate_temp: Mock,
        temp_dir: Path,
        aligned_image_paths: list[Path],
    ) -> None:
        """Test resource checking when image reading fails."""
        # Setup mock to simulate image read failure by raising an exception
        mock_imread.side_effect = Exception("Image read failed")

        # Execute - should not raise exception, just log warning
        with patch("focusstacker.internal.util.resource_monitor.logger") as mock_logger:
            check_resources_before_processing(
                image_paths=aligned_image_paths,
                temp_dir=temp_dir,
            )

        # Verify warning was logged (called twice: once for memory, once for disk)
        assert mock_logger.warning.call_count == len(aligned_image_paths) * 2
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        memory_warnings = [
            call for call in warning_calls if "Could not estimate memory for" in call
        ]
        disk_warnings = [
            call
            for call in warning_calls
            if "Could not estimate disk usage for" in call
        ]
        assert len(memory_warnings) == len(aligned_image_paths)
        assert len(disk_warnings) == len(aligned_image_paths)

    @patch(
        "focusstacker.internal.util.resource_monitor.ResourceMonitor.validate_temp_directory"
    )
    def test_check_resources_validation_failure(
        self,
        mock_validate_temp: Mock,
        temp_dir: Path,
        aligned_image_paths: list[Path],
    ) -> None:
        """Test resource checking when temp directory validation fails."""
        # Setup mock to simulate validation failure
        mock_validate_temp.side_effect = FocusStackerDirectoryException(
            "Invalid directory"
        )

        # Execute and verify exception
        with pytest.raises(FocusStackerDirectoryException) as exc_info:
            check_resources_before_processing(
                image_paths=aligned_image_paths,
                temp_dir=temp_dir,
            )

        assert "Invalid directory" in str(exc_info.value)
