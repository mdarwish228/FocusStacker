"""Unit tests for factory module."""

import pytest

from focusstacker.common.enums import AlignerChoice, BlenderChoice
from focusstacker.internal.config.base import FocusStackingConfig
from focusstacker.internal.models.factory import FocusStackingConfigFactory


class TestFocusStackingConfigFactory:
    """Test FocusStackingConfigFactory class."""

    def test_create_returns_focus_stacking_config(self) -> None:
        """Test that factory returns a FocusStackingConfig instance."""
        # Execute
        result = FocusStackingConfigFactory.create(
            aligner=AlignerChoice.SIFT,
            blender=BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
        )

        # Verify result type
        assert isinstance(result, FocusStackingConfig)
        assert hasattr(result, "aligner")
        assert hasattr(result, "blender")

    def test_create_with_custom_levels(self) -> None:
        """Test factory creation with custom levels."""
        # Execute with custom levels
        result = FocusStackingConfigFactory.create(
            aligner=AlignerChoice.SIFT,
            blender=BlenderChoice.LAPLACIAN_PYRAMID_MAX_SHARPNESS,
            levels=8,
        )

        # Verify result type
        assert isinstance(result, FocusStackingConfig)
        assert hasattr(result, "aligner")
        assert hasattr(result, "blender")

    def test_create_with_different_aligner_blender_combinations(self) -> None:
        """Test factory creation with different aligner/blender combinations."""
        # Test all combinations
        combinations = [
            (AlignerChoice.SIFT, BlenderChoice.LAPLACIAN_PYRAMID_BALANCED),
            (AlignerChoice.SIFT, BlenderChoice.LAPLACIAN_PYRAMID_MAX_SHARPNESS),
        ]

        for aligner_choice, blender_choice in combinations:
            # Execute
            result = FocusStackingConfigFactory.create(
                aligner=aligner_choice,
                blender=blender_choice,
                levels=3,
            )

            # Verify result
            assert isinstance(result, FocusStackingConfig)
            assert hasattr(result, "aligner")
            assert hasattr(result, "blender")

    def test_create_class_method(self) -> None:
        """Test that create is a class method."""
        # Verify it can be called on the class
        assert hasattr(FocusStackingConfigFactory, "create")
        assert callable(FocusStackingConfigFactory.create)

        # Verify it can be called on an instance
        factory = FocusStackingConfigFactory()
        assert hasattr(factory, "create")
        assert callable(factory.create)

    def test_create_with_minimum_levels(self) -> None:
        """Test factory creation with minimum levels."""
        # Execute with minimum levels
        result = FocusStackingConfigFactory.create(
            aligner=AlignerChoice.SIFT,
            blender=BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            levels=1,
        )

        # Verify result type
        assert isinstance(result, FocusStackingConfig)
        assert hasattr(result, "aligner")
        assert hasattr(result, "blender")

    def test_create_with_maximum_levels(self) -> None:
        """Test factory creation with maximum levels."""
        # Execute with maximum levels
        result = FocusStackingConfigFactory.create(
            aligner=AlignerChoice.SIFT,
            blender=BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            levels=10,
        )

        # Verify result type
        assert isinstance(result, FocusStackingConfig)
        assert hasattr(result, "aligner")
        assert hasattr(result, "blender")

    def test_create_with_default_parameters(self) -> None:
        """Test factory creation with default parameters."""
        # Execute with only required parameters
        result = FocusStackingConfigFactory.create(
            aligner=AlignerChoice.SIFT,
            blender=BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
        )

        # Verify result type
        assert isinstance(result, FocusStackingConfig)
        assert hasattr(result, "aligner")
        assert hasattr(result, "blender")

    @pytest.mark.parametrize("levels", [1, 3, 5, 8, 10])
    def test_create_with_various_levels(self, levels: int) -> None:
        """Test factory creation with various level values."""
        # Execute with parameterized levels
        result = FocusStackingConfigFactory.create(
            aligner=AlignerChoice.SIFT,
            blender=BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            levels=levels,
        )

        # Verify result type
        assert isinstance(result, FocusStackingConfig)
        assert hasattr(result, "aligner")
        assert hasattr(result, "blender")
