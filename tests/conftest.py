"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from focusstacker.common.enums import AlignerChoice, BlenderChoice


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="test_focus_stack_") as tmp:
        yield Path(tmp)


@pytest.fixture
def aligned_image_paths() -> list[Path]:
    """Images that should align well (same scene, different focus)."""
    return [
        Path("tests/images/aligned1.jpg"),
        Path("tests/images/aligned2.jpg"),
        Path("tests/images/aligned3.jpg"),
    ]


@pytest.fixture
def misaligned_image_paths() -> list[Path]:
    """Images that should fail to align (different scenes)."""
    return [
        Path("tests/images/misaligned1.jpg"),
        Path("tests/images/misaligned2.jpg"),
        Path("tests/images/misaligned3.jpg"),
    ]


@pytest.fixture
def invalid_image_paths() -> list[Path]:
    """Invalid image paths for error testing."""
    return [
        Path("tests/images/nonexistent1.jpg"),
        Path("tests/images/nonexistent2.jpg"),
    ]


@pytest.fixture
def directory_path() -> Path:
    """Directory path (not a file) for error testing."""
    return Path("tests/images/")


@pytest.fixture
def default_aligner() -> AlignerChoice:
    """Default aligner choice for testing."""
    return AlignerChoice.SIFT


@pytest.fixture
def default_blender() -> BlenderChoice:
    """Default blender choice for testing."""
    return BlenderChoice.LAPLACIAN_PYRAMID_BALANCED


@pytest.fixture
def default_levels() -> int:
    """Default pyramid levels for testing."""
    return 5


@pytest.fixture
def output_path(temp_dir: Path) -> Path:
    """Output path for test results."""
    return temp_dir / "test_output.jpg"
