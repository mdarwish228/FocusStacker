import logging
import tempfile
from pathlib import Path
from typing import Union

from .common.enums import AlignerChoice, BlenderChoice
from .common.exceptions import (
    FocusStackerAlignmentException,
    FocusStackerDirectoryException,
    FocusStackerStackingException,
)
from .internal.models.factory import FocusStackingConfigFactory
from .internal.models.validation import FocusStackerInputValidation
from .internal.util.resource_monitor import (
    ResourceMonitor,
    check_resources_before_processing,
)

logger = logging.getLogger(__name__)


def stack_images(
    image_paths: list[Union[str, Path]],
    destination_image_path: Union[str, Path],
    *,
    aligner: AlignerChoice = AlignerChoice.SIFT,
    blender: BlenderChoice = BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
    levels: int = 5,
) -> None:
    logger.info(f"Starting focus stacking with {len(image_paths)} images")
    logger.info(f"Destination: {destination_image_path}")
    logger.info(f"Aligner: {aligner}")
    logger.info(f"Blender: {blender}")
    logger.info(f"Levels: {levels}")

    # Validate all inputs using Pydantic (handles str -> Path conversion)
    # The validation model now handles its own exception conversion
    validated_input = FocusStackerInputValidation(
        image_paths=image_paths,
        destination_image_path=destination_image_path,
        aligner=aligner,
        blender=blender,
        levels=levels,
    )

    # Convert paths to Path objects
    image_paths = [Path(p) for p in validated_input.image_paths]
    destination_path = Path(validated_input.destination_image_path)

    logger.info(f"Processing {len(image_paths)} source images")

    config = FocusStackingConfigFactory.create(
        aligner=validated_input.aligner,
        blender=validated_input.blender,
        levels=validated_input.levels,
    )

    # Select reference image (middle image for better alignment)
    middle_idx = len(image_paths) // 2
    reference_path = image_paths[middle_idx]

    # Separate reference from other images
    other_paths = [path for i, path in enumerate(image_paths) if i != middle_idx]

    try:
        with tempfile.TemporaryDirectory(prefix="focus_stack_") as temp_dir:
            # Check resource availability before processing
            check_resources_before_processing(image_paths, temp_dir)

            # Apply alignment using all configured aligners
            try:
                aligned_paths = config.aligner.align(
                    reference_path, other_paths, temp_dir
                )
            except Exception as e:
                raise FocusStackerAlignmentException(f"Alignment failed: {e}") from e

            # Apply blending using the configured blender
            try:
                config.blender.blend(aligned_paths, destination_path, temp_dir)
            except Exception as e:
                raise FocusStackerStackingException(f"Stacking failed: {e}") from e

            # Log final resource status
            ResourceMonitor.log_resource_status("After processing")

            # Force garbage collection after processing
            ResourceMonitor.force_garbage_collection()

    except OSError as e:
        raise FocusStackerDirectoryException(
            f"Failed to create temporary directory: {e}"
        ) from e
