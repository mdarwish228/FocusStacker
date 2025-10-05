"""Registry for auto-discovered aligners and blenders."""

from ...common.enums import AlignerChoice, BlenderChoice
from ..config.base import AlignerConfig, BlenderConfig
from .align.base import Aligner as AlignerBase
from .stack.base import Blender as BlenderBase

# Registry dictionaries for auto-discovered components
_aligner_map: dict[AlignerChoice, type[AlignerBase]] = {}
_blenders_map: dict[BlenderChoice, type[BlenderBase]] = {}
_aligner_config_map: dict[AlignerChoice, type[AlignerConfig]] = {}
_blender_config_map: dict[BlenderChoice, type[BlenderConfig]] = {}


def register_aligner(
    choice: AlignerChoice, cls: type[AlignerBase]
) -> type[AlignerBase]:
    """Register an aligner class with its enum choice."""
    _aligner_map[choice] = cls
    return cls


def register_aligner_config(
    choice: AlignerChoice, cls: type[AlignerConfig]
) -> type[AlignerConfig]:
    """Register an aligner config class with its enum choice."""
    _aligner_config_map[choice] = cls
    return cls


def register_blender(
    choice: BlenderChoice, cls: type[BlenderBase]
) -> type[BlenderBase]:
    """Register a blender class with its enum choice."""
    _blenders_map[choice] = cls
    return cls


def register_blender_config(
    choice: BlenderChoice, cls: type[BlenderConfig]
) -> type[BlenderConfig]:
    """Register a blender config class with its enum choice."""
    _blender_config_map[choice] = cls
    return cls
