from dataclasses import dataclass

from ...common.enums import BlenderChoice
from ..models.decorators import BlenderConfigDecorator
from .base import BlenderConfig


@BlenderConfigDecorator(BlenderChoice.LAPLACIAN_PYRAMID_MAX_SHARPNESS)
@dataclass
class LaplacianPyramidMaxSharpnessConfig(BlenderConfig):
    """Laplacian pyramid max sharpness blending parameters."""

    name: str = "laplacian_pyramid_max_sharpness"
    downsample_factor: float = 0.5


@BlenderConfigDecorator(BlenderChoice.LAPLACIAN_PYRAMID_BALANCED)
@dataclass
class LaplacianPyramidBalancedConfig(BlenderConfig):
    """Config for soft-blended Laplacian pyramid blending."""

    name: str = "laplacian_pyramid_balanced"
