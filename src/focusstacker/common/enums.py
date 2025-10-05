from enum import Enum


class BlenderChoice(Enum):
    LAPLACIAN_PYRAMID_MAX_SHARPNESS = "laplacian_pyramid_max_sharpness"
    LAPLACIAN_PYRAMID_BALANCED = "laplacian_pyramid_balanced"


class AlignerChoice(Enum):
    SIFT = "sift"
