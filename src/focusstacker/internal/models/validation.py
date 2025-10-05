from typing import Annotated, ClassVar, Union

from pydantic import BaseModel, Field, FilePath, ValidationError, field_validator
from pydantic.types import NewPath

from ...common.enums import AlignerChoice, BlenderChoice
from ...common.exceptions import FocusStackerValidationException


class FocusStackerInputValidation(BaseModel):
    """Input validation model for stack_images function."""

    VALID_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg"}

    def __init__(self, **data):
        """Initialize with custom validation error handling."""
        try:
            super().__init__(**data)
        except (ValidationError, ValueError) as e:
            # Convert Pydantic validation errors to custom exceptions
            if isinstance(e, ValidationError) and hasattr(e, "errors"):
                errors = e.errors()
            else:
                errors = [{"msg": str(e), "type": "value_error", "loc": ["unknown"]}]
            raise FocusStackerValidationException(
                self._format_validation_errors(errors)
            ) from e

    image_paths: Annotated[
        list[FilePath],
        Field(
            min_length=2,
            max_length=50,
            description="List of source image paths (2-50 images required for focus stacking)",
            examples=[["image1.jpg", "image2.jpg", "image3.jpg"]],
        ),
    ]
    destination_image_path: Annotated[
        NewPath,
        Field(
            description="Output path for the focus-stacked image",
            examples=["images.jpg", "output.jpg"],
        ),
    ]

    aligner: Annotated[
        AlignerChoice,
        Field(
            description="Aligner to use for focus stacking. SIFT is supported.",
        ),
    ]
    blender: Annotated[
        BlenderChoice,
        Field(
            description="Blender to use for focus stacking. Laplacian pyramid is supported.",
        ),
    ]
    levels: Annotated[
        int,
        Field(
            ge=3,
            le=8,
            description="Number of pyramid levels for Laplacian pyramid stacking (3-8).",
        ),
    ]

    @field_validator("image_paths")
    @classmethod
    def validate_image_extensions(cls, v):
        """Validate that all image paths have .jpg or .jpeg extensions."""
        for path in v:
            cls._validate_image_extension(path, "image")
        return v

    @field_validator("destination_image_path")
    @classmethod
    def validate_destination_extension(cls, v):
        """Validate that destination path has .jpg or .jpeg extension."""
        cls._validate_image_extension(v, "destination")
        return v

    @classmethod
    def _validate_image_extension(
        cls, path: Union[NewPath, FilePath], field_name: str
    ) -> None:
        """Shared validation logic for image file extensions."""
        if path.suffix.lower() not in cls.VALID_EXTENSIONS:
            raise ValueError(
                f"Unsupported {field_name} format '{path.suffix.lower()}' for '{path}'. "
                f"Supported formats: {', '.join(cls.VALID_EXTENSIONS)}"
            )

    @staticmethod
    def _format_validation_errors(errors: list) -> str:
        """Format Pydantic validation errors into user-friendly messages."""
        formatted_errors = []

        for error in errors:
            field = error.get("loc", ["unknown"])[-1]
            error_type = error.get("type", "unknown")
            message = error.get("msg", "Validation error")
            input_value = error.get("input", "unknown")

            # Customize messages based on error type - only for types we actually use
            if error_type == "path_not_file":
                formatted_errors.append(f"Path '{input_value}' is not a file.")
            else:
                # Fallback for unknown error types
                formatted_errors.append(f"{field}: {message}")

        return "; ".join(formatted_errors)
