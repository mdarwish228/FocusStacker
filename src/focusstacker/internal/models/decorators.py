from typing import Callable, TypeVar

from ...common.enums import AlignerChoice, BlenderChoice
from ..config.base import AlignerConfig, BlenderConfig
from .align.base import Aligner as AlignerBase
from .registry import (
    register_aligner,
    register_aligner_config,
    register_blender,
    register_blender_config,
)
from .stack.base import Blender as BlenderBase

T = TypeVar("T")


def Aligner(choice: AlignerChoice) -> Callable[[type[AlignerBase]], type[AlignerBase]]:
    """Decorator to register an aligner class with its enum choice."""

    def decorator(cls: type[AlignerBase]) -> type[AlignerBase]:
        return register_aligner(choice, cls)

    return decorator


def AlignerConfigDecorator(
    choice: AlignerChoice,
) -> Callable[[type[AlignerConfig]], type[AlignerConfig]]:
    """Decorator to register an aligner config class with its enum choice."""

    def decorator(cls: type[AlignerConfig]) -> type[AlignerConfig]:
        return register_aligner_config(choice, cls)

    return decorator


def Blender(choice: BlenderChoice) -> Callable[[type[BlenderBase]], type[BlenderBase]]:
    """Decorator to register a blender class with its enum choice."""

    def decorator(cls: type[BlenderBase]) -> type[BlenderBase]:
        return register_blender(choice, cls)

    return decorator


def BlenderConfigDecorator(
    choice: BlenderChoice,
) -> Callable[[type[BlenderConfig]], type[BlenderConfig]]:
    """Decorator to register a blender config class with its enum choice."""

    def decorator(cls: type[BlenderConfig]) -> type[BlenderConfig]:
        return register_blender_config(choice, cls)

    return decorator
