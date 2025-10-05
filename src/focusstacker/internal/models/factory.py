# type: ignore[unknown-argument]

from ...common.enums import AlignerChoice, BlenderChoice
from ..config.base import FocusStackingConfig

# Import modules to trigger decorator registration
from .align import sift  # noqa: F401
from .registry import (
    _aligner_config_map,
    _aligner_map,
    _blender_config_map,
    _blenders_map,
)
from .stack import (
    laplacian_pyramid_balanced,  # noqa: F401
    laplacian_pyramid_max_sharpness,  # noqa: F401
)


class FocusStackingConfigFactory:
    """Factory that dispatches to appropriate config class based on enum."""

    @classmethod
    def create(
        cls, aligner: AlignerChoice, blender: BlenderChoice, levels: int = 5
    ) -> FocusStackingConfig:
        blender_instance = _blenders_map[blender](
            config=_blender_config_map[blender](levels=levels)
        )
        aligner_instance = _aligner_map[aligner](config=_aligner_config_map[aligner]())

        return FocusStackingConfig(
            aligner=aligner_instance,
            blender=blender_instance,
        )
