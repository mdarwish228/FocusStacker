# FocusStacker

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**FocusStacker** is a high-performance Python library for advanced focus stacking using Laplacian pyramid merging. It combines multiple images taken at different focus distances to create a single image with extended depth of field.

## Features

- **Advanced Laplacian Pyramid Merging**: Uses sophisticated algorithms for optimal focus stacking
- **SIFT-based Alignment**: Automatic image alignment using Scale-Invariant Feature Transform
- **Memory Efficient**: Optimized for processing large images with minimal memory usage
- **Configurable Parameters**: Fine-tune alignment and blending parameters
- **Resource Monitoring**: Built-in memory and disk space monitoring
- **Comprehensive Error Handling**: Detailed exceptions for debugging
- **Type Safety**: Full type annotations for better development experience

## Installation

### Using pip

```bash
pip install focusstacker
```

### Using uv

```bash
uv add focusstacker
```

## Quick Start

### Basic Usage

```python
from focusstacker import stack_images

# Define your input images
image_paths = [
    "images/focus_1.jpg",
    "images/focus_2.jpg", 
    "images/focus_3.jpg"
]

# Stack the images
stack_images(
    image_paths=image_paths,
    destination_image_path="output/stacked_image.jpg"
)
```

### Advanced Usage

```python
from focusstacker import stack_images
from focusstacker.common.enums import AlignerChoice, BlenderChoice

stack_images(
    image_paths=[
        "images/focus_1.jpg",
        "images/focus_2.jpg",
        "images/focus_3.jpg",
        "images/focus_4.jpg"
    ],
    destination_image_path="output/high_quality_stack.jpg",
    aligner=AlignerChoice.SIFT,
    blender=BlenderChoice.LAPLACIAN_PYRAMID_BALANCED,
    levels=6  # More pyramid levels for higher quality
)
```

## API Reference

### `stack_images()`

The main function for focus stacking.

#### Parameters

- **`image_paths`** (`list[str | Path]`): List of input image paths (2-50 images)
- **`destination_image_path`** (`str | Path`): Output path for the stacked image
- **`aligner`** (`AlignerChoice`, optional): Alignment algorithm (default: `SIFT`)
- **`blender`** (`BlenderChoice`, optional): Blending algorithm (default: `LAPLACIAN_PYRAMID_BALANCED`)
- **`levels`** (`int`, optional): Number of pyramid levels 3-8 (default: 5)

#### Supported Formats

- **Input**: JPEG (.jpg, .jpeg)
- **Output**: JPEG (.jpg, .jpeg)

## Configuration Options

### Alignment Algorithms

#### SIFT (Scale-Invariant Feature Transform)
- **Best for**: General photography, landscapes, macro
- **Features**: Robust to scale, rotation, and illumination changes
- **Performance**: High accuracy, moderate speed

### Blending Algorithms

#### Laplacian Pyramid Balanced
- **Best for**: Most photography scenarios
- **Features**: Soft blending, natural transitions
- **Performance**: Balanced quality and speed

#### Laplacian Pyramid Max Sharpness
- **Best for**: Technical photography, maximum detail
- **Features**: Aggressive sharpness selection
- **Performance**: Highest detail, slower processing

### Pyramid Levels

- **3-4 levels**: Fast processing, good for previews
- **5-6 levels**: Balanced quality and speed (recommended)
- **7-8 levels**: Highest quality, slower processing

## Error Handling

FocusStacker provides comprehensive error handling with specific exception types:

```python
from focusstacker.common.exceptions import (
    FocusStackerValidationException,
    FocusStackerAlignmentException,
    FocusStackerStackingException,
    FocusStackerFileException,
    FocusStackerMemoryException
)

try:
    stack_images(
        image_paths=["img1.jpg", "img2.jpg"],
        destination_image_path="result.jpg"
    )
except FocusStackerValidationException as e:
    print(f"Validation error: {e}")
except FocusStackerAlignmentException as e:
    print(f"Alignment failed: {e}")
except FocusStackerStackingException as e:
    print(f"Stacking failed: {e}")
except FocusStackerMemoryException as e:
    print(f"Insufficient memory: {e}")
except FocusStackerFileException as e:
    print(f"File operation failed: {e}")
```

### Exception Types

- **`FocusStackerValidationException`**: Input validation errors
- **`FocusStackerAlignmentException`**: Image alignment failures
- **`FocusStackerStackingException`**: Blending/stacking failures
- **`FocusStackerFileException`**: File I/O errors
- **`FocusStackerMemoryException`**: Memory/resource issues
- **`FocusStackerImageProcessingException`**: Image processing errors
- **`FocusStackerDirectoryException`**: Directory operation errors

## Best Practices

### Image Preparation

1. **Consistent Lighting**: Use consistent lighting across all images
2. **Stable Camera**: Use a tripod or stable surface
3. **Overlapping Focus**: Ensure focus ranges overlap between images
4. **Image Quality**: Use high-quality, sharp images
5. **File Format**: Use JPEG format for best compatibility

### Performance Optimization

1. **Image Size**: Resize large images before processing if memory is limited
2. **Pyramid Levels**: Use fewer levels for faster processing
3. **Batch Processing**: Process multiple stacks sequentially
4. **Memory Monitoring**: Monitor system resources during processing

## System Requirements

### Minimum Requirements
- Python 3.9+
- 4GB RAM
- 1GB free disk space

### Recommended Requirements
- 8GB+ RAM
- 5GB+ free disk space
- SSD storage for better performance

### Dependencies
- `numpy>=1.24.4`
- `opencv-contrib-python>=4.12.0.88`
- `psutil>=7.1.0`
- `pydantic>=2.11.9`

## Troubleshooting

### Common Issues

#### Memory Errors
```
FocusStackerMemoryException: Insufficient memory
```
**Solution**: Reduce image size or use fewer pyramid levels

#### Alignment Failures
```
FocusStackerAlignmentException: Alignment failed
```
**Solution**: Ensure images have sufficient overlap and features

#### File Access Errors
```
FocusStackerFileException: Could not load image
```
**Solution**: Check file paths and permissions

### Performance Tips

1. **Use SSD storage** for temporary files
2. **Close other applications** during processing
3. **Monitor system resources** with task manager
4. **Use appropriate pyramid levels** for your needs

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/focus-stacking.git
cd focus-stacking

# Install development dependencies
uv install --dev

# Run tests
uv run pytest

# Run type checking
uv run ty check

# Run linting
uv run ruff check
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for computer vision algorithms
- NumPy for numerical computations
- The focus stacking community for inspiration and feedback

## Changelog

### v0.1.0
- Initial release
- SIFT-based alignment
- Laplacian pyramid blending
- Memory-efficient processing
- Comprehensive error handling
