"""Unit tests for stack.py module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from focusstacker.common.enums import AlignerChoice, BlenderChoice
from focusstacker.common.exceptions import (
    FocusStackerAlignmentException,
    FocusStackerDirectoryException,
    FocusStackerStackingException,
    FocusStackerValidationException,
)
from focusstacker.stack import stack_images


class TestStackImagesValidation:
    """Test input validation in stack_images function."""

    def test_valid_inputs_success(
        self,
        aligned_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that valid inputs pass validation and processing completes successfully."""
        # Execute the function with real data
        stack_images(
            image_paths=[str(p) for p in aligned_image_paths],
            destination_image_path=str(output_path),
            aligner=default_aligner,
            blender=default_blender,
            levels=default_levels,
        )

        # Verify that the output file was created
        assert output_path.exists(), f"Output file {output_path} was not created"
        assert output_path.is_file(), f"Output file {output_path} is not a file"
        assert output_path.stat().st_size > 0, f"Output file {output_path} is empty"

        # Verify output file is a valid image (basic check)
        assert output_path.suffix.lower() in [".jpg", ".jpeg"], (
            f"Output file {output_path} has invalid extension"
        )

    def test_misaligned_images_raises_alignment_error(
        self,
        misaligned_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that misaligned images raise FocusStackerAlignmentException."""
        # Execute the function with misaligned images
        with pytest.raises(FocusStackerAlignmentException):
            stack_images(
                image_paths=[str(p) for p in misaligned_image_paths],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=default_levels,
            )

        # Verify that no output file was created due to alignment failure
        assert not output_path.exists(), (
            f"Output file {output_path} should not exist after alignment failure"
        )

    def test_invalid_image_paths_raises_validation_error(
        self,
        invalid_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that invalid image paths raise validation error."""
        with pytest.raises(FocusStackerValidationException):
            stack_images(
                image_paths=[str(p) for p in invalid_image_paths],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=default_levels,
            )

    def test_directory_path_raises_validation_error(
        self,
        directory_path: Path,
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that directory path raises validation error."""
        with pytest.raises(FocusStackerValidationException):
            stack_images(
                image_paths=[str(directory_path)],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=default_levels,
            )

    def test_too_few_images_raises_validation_error(
        self,
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that too few images raise validation error."""
        single_image = Path("tests/fixtures/images/aligned1.jpg")
        with pytest.raises(FocusStackerValidationException):
            stack_images(
                image_paths=[str(single_image)],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=default_levels,
            )

    def test_invalid_levels_raises_validation_error(
        self,
        aligned_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
    ) -> None:
        """Test that invalid levels raise validation error."""
        # Test levels too low
        with pytest.raises(FocusStackerValidationException):
            stack_images(
                image_paths=[str(p) for p in aligned_image_paths],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=2,
            )

        # Test levels too high
        with pytest.raises(FocusStackerValidationException):
            stack_images(
                image_paths=[str(p) for p in aligned_image_paths],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=9,
            )

    def test_existing_output_file_raises_validation_error(
        self,
        aligned_image_paths: list[Path],
        temp_dir: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that existing output file raises validation error."""
        existing_output = temp_dir / "existing_output.jpg"
        existing_output.touch()  # Create the file

        with pytest.raises(FocusStackerValidationException):
            stack_images(
                image_paths=[str(p) for p in aligned_image_paths],
                destination_image_path=str(existing_output),
                aligner=default_aligner,
                blender=default_blender,
                levels=default_levels,
            )


class TestStackImagesAlignment:
    """Test alignment functionality."""

    @patch("focusstacker.stack.FocusStackingConfigFactory")
    def test_alignment_success(
        self,
        mock_factory: Mock,
        aligned_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test successful alignment with mock."""
        # Setup mock
        mock_aligner = Mock()
        mock_blender = Mock()
        mock_aligner.align.return_value = aligned_image_paths
        mock_blender.blend.return_value = None

        mock_config = Mock()
        mock_config.aligner = mock_aligner
        mock_config.blender = mock_blender
        mock_factory.create.return_value = mock_config

        # Execute
        stack_images(
            image_paths=[str(p) for p in aligned_image_paths],
            destination_image_path=str(output_path),
            aligner=default_aligner,
            blender=default_blender,
            levels=default_levels,
        )

        # Verify
        mock_aligner.align.assert_called_once()
        mock_blender.blend.assert_called_once()

    @patch("focusstacker.stack.FocusStackingConfigFactory")
    def test_alignment_failure_raises_exception(
        self,
        mock_factory: Mock,
        misaligned_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that alignment failure raises FocusStackerAlignmentException."""
        # Setup mock to raise alignment error
        mock_aligner = Mock()
        mock_blender = Mock()
        mock_aligner.align.side_effect = Exception("Alignment failed")

        mock_config = Mock()
        mock_config.aligner = mock_aligner
        mock_config.blender = mock_blender
        mock_factory.create.return_value = mock_config

        # Execute and verify
        with pytest.raises(FocusStackerAlignmentException, match="Alignment failed"):
            stack_images(
                image_paths=[str(p) for p in misaligned_image_paths],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=default_levels,
            )


class TestStackImagesBlending:
    """Test blending functionality."""

    @patch("focusstacker.stack.FocusStackingConfigFactory")
    def test_blending_success(
        self,
        mock_factory: Mock,
        aligned_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test successful blending with mock."""
        # Setup mock
        mock_aligner = Mock()
        mock_blender = Mock()
        mock_aligner.align.return_value = aligned_image_paths
        mock_blender.blend.return_value = None

        mock_config = Mock()
        mock_config.aligner = mock_aligner
        mock_config.blender = mock_blender
        mock_factory.create.return_value = mock_config

        # Execute
        stack_images(
            image_paths=[str(p) for p in aligned_image_paths],
            destination_image_path=str(output_path),
            aligner=default_aligner,
            blender=default_blender,
            levels=default_levels,
        )

        # Verify blending was called
        mock_blender.blend.assert_called_once()

    @patch("focusstacker.stack.FocusStackingConfigFactory")
    def test_blending_failure_raises_exception(
        self,
        mock_factory: Mock,
        aligned_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that blending failure raises FocusStackerStackingException."""
        # Setup mock to raise blending error
        mock_aligner = Mock()
        mock_blender = Mock()
        mock_aligner.align.return_value = aligned_image_paths
        mock_blender.blend.side_effect = Exception("Blending failed")

        mock_config = Mock()
        mock_config.aligner = mock_aligner
        mock_config.blender = mock_blender
        mock_factory.create.return_value = mock_config

        # Execute and verify
        with pytest.raises(FocusStackerStackingException, match="Blending failed"):
            stack_images(
                image_paths=[str(p) for p in aligned_image_paths],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=default_levels,
            )


class TestStackImagesResourceManagement:
    """Test resource management and cleanup."""

    @patch("focusstacker.stack.tempfile.TemporaryDirectory")
    def test_temp_directory_creation_failure_raises_exception(
        self,
        mock_temp_dir: Mock,
        aligned_image_paths: list[Path],
        output_path: Path,
        default_aligner: AlignerChoice,
        default_blender: BlenderChoice,
        default_levels: int,
    ) -> None:
        """Test that temp directory creation failure raises FocusStackerDirectoryException."""
        # Setup mock to raise OSError
        mock_temp_dir.side_effect = OSError("Cannot create temp directory")

        # Execute and verify
        with pytest.raises(
            FocusStackerDirectoryException, match="Failed to create temporary directory"
        ):
            stack_images(
                image_paths=[str(p) for p in aligned_image_paths],
                destination_image_path=str(output_path),
                aligner=default_aligner,
                blender=default_blender,
                levels=default_levels,
            )


class TestStackImagesParameterVariations:
    """Test different parameter combinations."""

    @pytest.mark.parametrize("aligner", [AlignerChoice.SIFT])
    @pytest.mark.parametrize(
        "blender",
        [
            BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
            BlenderChoice.LAPLACIAN_PYRAMID_MAX_SHARPNESS,
        ],
    )
    @pytest.mark.parametrize("levels", [3, 5, 8])
    @patch("focusstacker.stack.FocusStackingConfigFactory")
    def test_different_parameter_combinations(
        self,
        mock_factory: Mock,
        aligned_image_paths: list[Path],
        output_path: Path,
        aligner: AlignerChoice,
        blender: BlenderChoice,
        levels: int,
    ) -> None:
        """Test different combinations of aligner, blender, and levels."""
        # Setup mock
        mock_aligner = Mock()
        mock_blender = Mock()
        mock_aligner.align.return_value = aligned_image_paths
        mock_blender.blend.return_value = None

        mock_config = Mock()
        mock_config.aligner = mock_aligner
        mock_config.blender = mock_blender
        mock_factory.create.return_value = mock_config

        # Execute
        stack_images(
            image_paths=[str(p) for p in aligned_image_paths],
            destination_image_path=str(output_path),
            aligner=aligner,
            blender=blender,
            levels=levels,
        )

        # Verify factory was called with correct parameters
        mock_factory.create.assert_called_once_with(
            aligner=aligner,
            blender=blender,
            levels=levels,
        )
