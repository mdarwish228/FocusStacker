"""Unit tests for registry module."""

from typing import cast
from unittest.mock import Mock, patch

from focusstacker.common.enums import AlignerChoice, BlenderChoice
from focusstacker.internal.config.base import AlignerConfig, BlenderConfig
from focusstacker.internal.models.align.base import Aligner as AlignerBase
from focusstacker.internal.models.registry import (
    register_aligner,
    register_aligner_config,
    register_blender,
    register_blender_config,
)
from focusstacker.internal.models.stack.base import Blender as BlenderBase


class TestRegistryFunctions:
    """Test registry registration functions."""

    @patch("focusstacker.internal.models.registry._aligner_map")
    def test_register_aligner(self, mock_aligner_map: Mock) -> None:
        """Test aligner registration."""
        # Create mock aligner class
        mock_aligner_class = Mock()
        mock_aligner_class.__name__ = "MockAligner"

        # Register the aligner
        result = register_aligner(
            AlignerChoice.SIFT, cast("type[AlignerBase]", mock_aligner_class)
        )

        # Verify registration
        assert result is mock_aligner_class

        # Verify it's in the registry
        mock_aligner_map.__setitem__.assert_called_once_with(
            AlignerChoice.SIFT, mock_aligner_class
        )

    @patch("focusstacker.internal.models.registry._aligner_config_map")
    def test_register_aligner_config(self, mock_aligner_config_map: Mock) -> None:
        """Test aligner config registration."""
        # Create mock config class
        mock_config_class = Mock()
        mock_config_class.__name__ = "MockAlignerConfig"

        # Register the config
        result = register_aligner_config(
            AlignerChoice.SIFT, cast("type[AlignerConfig]", mock_config_class)
        )

        # Verify registration
        assert result is mock_config_class

        # Verify it's in the registry
        mock_aligner_config_map.__setitem__.assert_called_once_with(
            AlignerChoice.SIFT, mock_config_class
        )

    @patch("focusstacker.internal.models.registry._blenders_map")
    def test_register_blender(self, mock_blenders_map: Mock) -> None:
        """Test blender registration."""
        # Create mock blender class
        mock_blender_class = Mock()
        mock_blender_class.__name__ = "MockBlender"

        # Register the blender
        result = register_blender(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            cast("type[BlenderBase]", mock_blender_class),
        )

        # Verify registration
        assert result is mock_blender_class

        # Verify it's in the registry
        mock_blenders_map.__setitem__.assert_called_once_with(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED, mock_blender_class
        )

    @patch("focusstacker.internal.models.registry._blender_config_map")
    def test_register_blender_config(self, mock_blender_config_map: Mock) -> None:
        """Test blender config registration."""
        # Create mock config class
        mock_config_class = Mock()
        mock_config_class.__name__ = "MockBlenderConfig"

        # Register the config
        result = register_blender_config(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            cast("type[BlenderConfig]", mock_config_class),
        )

        # Verify registration
        assert result is mock_config_class

        # Verify it's in the registry
        mock_blender_config_map.__setitem__.assert_called_once_with(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED, mock_config_class
        )

    @patch("focusstacker.internal.models.registry._aligner_map")
    def test_register_multiple_aligners(self, mock_aligner_map: Mock) -> None:
        """Test registering multiple aligners."""
        # Create mock aligner classes
        mock_aligner1 = Mock()
        mock_aligner2 = Mock()

        # Register both aligners
        register_aligner(AlignerChoice.SIFT, cast("type[AlignerBase]", mock_aligner1))
        # Note: Only SIFT is available in the enum, so we'll test overwriting
        register_aligner(AlignerChoice.SIFT, cast("type[AlignerBase]", mock_aligner2))

        # Verify both registrations were called
        assert mock_aligner_map.__setitem__.call_count == 2
        mock_aligner_map.__setitem__.assert_any_call(AlignerChoice.SIFT, mock_aligner1)
        mock_aligner_map.__setitem__.assert_any_call(AlignerChoice.SIFT, mock_aligner2)

    @patch("focusstacker.internal.models.registry._blenders_map")
    def test_register_multiple_blenders(self, mock_blenders_map: Mock) -> None:
        """Test registering multiple blenders."""
        # Create mock blender classes
        mock_blender1 = Mock()
        mock_blender2 = Mock()

        # Register both blenders
        register_blender(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            cast("type[BlenderBase]", mock_blender1),
        )
        register_blender(
            BlenderChoice.LAPLACIAN_PYRAMID_MAX_SHARPNESS,
            cast("type[BlenderBase]", mock_blender2),
        )

        # Verify both registrations were called
        assert mock_blenders_map.__setitem__.call_count == 2
        mock_blenders_map.__setitem__.assert_any_call(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED, mock_blender1
        )
        mock_blenders_map.__setitem__.assert_any_call(
            BlenderChoice.LAPLACIAN_PYRAMID_MAX_SHARPNESS, mock_blender2
        )

    @patch("focusstacker.internal.models.registry._blender_config_map")
    @patch("focusstacker.internal.models.registry._blenders_map")
    @patch("focusstacker.internal.models.registry._aligner_config_map")
    @patch("focusstacker.internal.models.registry._aligner_map")
    def test_registry_isolation(
        self,
        mock_aligner_map: Mock,
        mock_aligner_config_map: Mock,
        mock_blenders_map: Mock,
        mock_blender_config_map: Mock,
    ) -> None:
        """Test that different registry types don't interfere with each other."""
        # Create mock classes
        mock_aligner = Mock()
        mock_aligner_config = Mock()
        mock_blender = Mock()
        mock_blender_config = Mock()

        # Register all types
        register_aligner(AlignerChoice.SIFT, cast("type[AlignerBase]", mock_aligner))
        register_aligner_config(
            AlignerChoice.SIFT, cast("type[AlignerConfig]", mock_aligner_config)
        )
        register_blender(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            cast("type[BlenderBase]", mock_blender),
        )
        register_blender_config(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            cast("type[BlenderConfig]", mock_blender_config),
        )

        # Verify all registrations were called
        mock_aligner_map.__setitem__.assert_called_once_with(
            AlignerChoice.SIFT, mock_aligner
        )
        mock_aligner_config_map.__setitem__.assert_called_once_with(
            AlignerChoice.SIFT, mock_aligner_config
        )
        mock_blenders_map.__setitem__.assert_called_once_with(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED, mock_blender
        )
        mock_blender_config_map.__setitem__.assert_called_once_with(
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED, mock_blender_config
        )

        # Verify they're different objects
        assert mock_aligner is not mock_aligner_config
        assert mock_blender is not mock_blender_config
        assert mock_aligner is not mock_blender
        assert mock_aligner_config is not mock_blender_config

    @patch("focusstacker.internal.models.registry._aligner_config_map")
    @patch("focusstacker.internal.models.registry._aligner_map")
    def test_registry_preserves_class_reference(
        self,
        mock_aligner_map: Mock,
        mock_aligner_config_map: Mock,
    ) -> None:
        """Test that registry stores exact class references."""

        # Create a real class (not a mock)
        class TestAligner:
            pass

        class TestConfig(AlignerConfig):
            pass

        # Register real classes
        register_aligner(AlignerChoice.SIFT, cast("type[AlignerBase]", TestAligner))
        register_aligner_config(
            AlignerChoice.SIFT, cast("type[AlignerConfig]", TestConfig)
        )

        # Verify exact class references are stored
        mock_aligner_map.__setitem__.assert_called_once_with(
            AlignerChoice.SIFT, TestAligner
        )
        mock_aligner_config_map.__setitem__.assert_called_once_with(
            AlignerChoice.SIFT, TestConfig
        )
