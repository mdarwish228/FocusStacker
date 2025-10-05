"""Base configuration classes for aligners and blenders."""

from dataclasses import dataclass

from ..models.align.base import Aligner
from ..models.stack.base import Blender


@dataclass
class FocusStackingConfig:
    """Complete focus stacking configuration."""

    aligner: Aligner
    blender: Blender


@dataclass
class AlignerConfig:
    """Base config class for aligners."""

    pass


@dataclass
class BlenderConfig:
    """Base config class for blenders."""

    # Shared Laplacian pyramid parameters
    laplacian_weight: float = 0.45
    gradient_weight: float = 0.2
    local_variance_weight: float = 0.18
    tenengrad_weight: float = 0.12
    gradient_variance_weight: float = 0.05
    spatial_smoothing_sigma: float = 0.5
    levels: int = 5
    spatial_consistency_sigma: float = 1.0
    local_variance_kernel_size: int = 3
    spatial_consistency_blur_size: int = 5
    spatial_smoothing_blur_size: int = 3
    minimum_pyramid_size: int = 32
    epsilon: float = 1e-8
