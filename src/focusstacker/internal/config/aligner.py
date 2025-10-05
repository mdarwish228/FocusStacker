from dataclasses import dataclass

from ...common.enums import AlignerChoice
from ..models.decorators import AlignerConfigDecorator
from .base import AlignerConfig


@AlignerConfigDecorator(AlignerChoice.SIFT)
@dataclass
class SIFTConfig(AlignerConfig):
    """SIFT-based alignment pipeline parameters."""

    nfeatures: int = 8000
    n_octave_layers: int = 3
    contrast_threshold: float = 0.02
    edge_threshold: float = 15.0
    sigma: float = 1.2

    flann_index_algorithm: int = 1
    flann_trees: int = 8
    flann_checks: int = 128
    flann_k_matches: int = 2
    lowe_ratio_threshold: float = 0.72
    min_matches_required: int = 300

    ransac_reproj_threshold: float = 10.0
    ransac_max_iters: int = 10000
    ransac_confidence: float = 0.999
