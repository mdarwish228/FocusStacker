import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from ....common.enums import AlignerChoice
from ....common.exceptions import (
    FocusStackerAlignmentException,
    FocusStackerDirectoryException,
    FocusStackerFileException,
    FocusStackerMemoryException,
)
from ...config.aligner import SIFTConfig
from ...models.decorators import Aligner
from ...util.image import ImageUtils
from ...util.resource_monitor import ResourceMonitor
from .base import Aligner as AlignerBase

logger = logging.getLogger(__name__)


@Aligner(AlignerChoice.SIFT)
class SiftAligner(AlignerBase):
    def __init__(self, *, config: SIFTConfig) -> None:
        self.config = config

    def align(
        self, reference_path: Path, image_paths: list[Path], temp_dir: str
    ) -> list[Path]:
        """Align images to a reference image using SIFT feature matching."""
        logger.info(f"SIFT alignment starting with {len(image_paths)} images to align")

        if len(image_paths) == 0:
            logger.warning("No images to align, returning reference image only")
            return [reference_path]

        # Copy reference image to temp directory
        ref_aligned_path = Path(temp_dir) / f"aligned_{reference_path.name}"

        # Check disk space before copying
        try:
            file_size_gb = reference_path.stat().st_size / (1024**3)
            ResourceMonitor.check_disk_space(temp_dir, file_size_gb)
        except Exception as e:
            logger.warning(
                f"Could not estimate disk space for copying {reference_path}: {e}"
            )

        try:
            shutil.copy2(reference_path, ref_aligned_path)
        except OSError as e:
            if "No space left" in str(e):
                raise FocusStackerDirectoryException(
                    f"Insufficient disk space to copy reference image: {e}"
                ) from e
            raise FocusStackerFileException(
                f"Failed to copy reference image: {e}"
            ) from e
        except Exception as e:
            raise FocusStackerFileException(
                f"Failed to copy reference image: {e}"
            ) from e

        aligned_paths = [ref_aligned_path]

        for _i, image_path in enumerate(image_paths):
            # Align each image to the reference
            aligned_path = self._align_sift_refined(
                reference_path, image_path, temp_dir
            )
            aligned_paths.append(aligned_path)

        # Final aggressive garbage collection
        return aligned_paths

    def _align_sift_refined(
        self, reference_path: Path, target_path: Path, temp_dir: str
    ) -> Path:
        """SIFT alignment with multi-resolution matching for memory efficiency."""
        # Load both images from files
        ref_img = ImageUtils.load_image(reference_path)
        target_img = ImageUtils.load_image(target_path)

        if ref_img is None or target_img is None:
            return target_path

        # Multi-resolution approach: detect features on downsampled images
        downsample_factor = self._calculate_downsample_factor(ref_img, target_img)

        # Downsample images for feature detection
        ref_small = self._downsample_image(ref_img, downsample_factor)
        target_small = self._downsample_image(target_img, downsample_factor)

        # Convert to grayscale
        gray_ref_small = ImageUtils.rgb_to_grayscale(ref_small)
        gray_target_small = ImageUtils.rgb_to_grayscale(target_small)

        # Enhanced SIFT parameters from config
        sift = ImageUtils.create_sift_detector(
            nfeatures=self.config.nfeatures,
            n_octave_layers=self.config.n_octave_layers,
            contrast_threshold=self.config.contrast_threshold,
            edge_threshold=self.config.edge_threshold,
            sigma=self.config.sigma,
        )

        logger.info(
            f"Detecting SIFT features for {target_path.name} (downsampled {downsample_factor:.2f}x)..."
        )
        kp_ref, des_ref = sift.detectAndCompute(gray_ref_small, None)
        kp_target, des_target = sift.detectAndCompute(gray_target_small, None)

        logger.info(
            f"Found {len(kp_ref) if kp_ref else 0} features in reference, {len(kp_target) if kp_target else 0} in target"
        )

        if des_ref is None or des_target is None:
            msg = f"Insufficient features detected for {target_path}"
            logger.error(msg)
            raise FocusStackerAlignmentException(msg)

        # Enhanced FLANN matcher with config parameters
        flann = ImageUtils.create_flann_matcher(
            algorithm=self.config.flann_index_algorithm,
            trees=self.config.flann_trees,
            checks=self.config.flann_checks,
        )

        matches = flann.knnMatch(des_ref, des_target, k=self.config.flann_k_matches)

        # Stricter Lowe's ratio test for better quality matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if (
                    m.distance < self.config.lowe_ratio_threshold * n.distance
                ):  # Lowe's ratio test
                    good_matches.append(m)

        if (
            len(good_matches) < self.config.min_matches_required
        ):  # Check minimum matches threshold
            msg = f"Insufficient matches ({len(good_matches)} < {self.config.min_matches_required}) for {target_path}"
            logger.error(msg)
            raise FocusStackerAlignmentException(msg)

        # Extract matched keypoints for homography estimation (on downsampled images)
        src_pts_small = np.float32(
            [kp_ref[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts_small = np.float32(
            [kp_target[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        # Enhanced homography estimation with config parameters (on downsampled images)
        H_small, mask = ImageUtils.find_homography(
            dst_pts_small,
            src_pts_small,
            method=cv2.RANSAC,
            ransac_reproj_threshold=self.config.ransac_reproj_threshold,
            max_iters=self.config.ransac_max_iters,
            confidence=self.config.ransac_confidence,
        )

        if H_small is None:
            msg = f"Failed to compute homography for {target_path}"
            logger.error(msg)
            raise FocusStackerAlignmentException(msg)

        # Scale homography matrix to full resolution
        H_full = self._scale_homography(H_small, downsample_factor)

        # Calculate RANSAC confidence (inlier ratio)
        if mask is not None:
            inliers = np.sum(mask)
            total_points = len(good_matches)
            confidence_ratio = inliers / total_points if total_points > 0 else 0
            logger.info(
                f"RANSAC confidence for {target_path.name}: {inliers}/{total_points} inliers ({confidence_ratio:.3f})"
            )
        else:
            logger.info(f"RANSAC confidence for {target_path.name}: mask not available")

        # Apply transformation to full resolution images
        h, w = ref_img.shape[:2]
        aligned_img = ImageUtils.warp_perspective(target_img, H_full, (w, h))

        # Save aligned image to temporary directory
        try:
            aligned_path = Path(temp_dir) / f"aligned_{target_path.name}"
        except Exception as e:
            raise FocusStackerDirectoryException(
                f"Failed to create temporary file path: {e}"
            ) from e

        # Save aligned image using ImageUtils
        try:
            ImageUtils.save_image(aligned_img, aligned_path, quality=100)
        except MemoryError as e:
            raise FocusStackerMemoryException(
                f"Insufficient memory to save aligned image: {e}"
            ) from e

        # Free memory immediately
        del ref_img, target_img, aligned_img
        del ref_small, target_small, gray_ref_small, gray_target_small, sift, flann
        del kp_ref, des_ref, kp_target, des_target, matches, good_matches
        del src_pts_small, dst_pts_small, H_small, H_full

        # Force garbage collection for large image objects
        ResourceMonitor.force_garbage_collection()

        return aligned_path

    def _calculate_downsample_factor(
        self, ref_img: np.ndarray, target_img: np.ndarray
    ) -> float:
        """Calculate optimal downsampling factor based on image resolution."""
        # Calculate megapixels for both images
        ref_mp = (ref_img.shape[0] * ref_img.shape[1]) / 1_000_000
        target_mp = (target_img.shape[0] * target_img.shape[1]) / 1_000_000

        # Use the larger image for downsampling decision
        max_mp = max(ref_mp, target_mp)

        if max_mp > 20:
            # High resolution (> 20 MP): downsample to 25% (16x memory reduction)
            factor = 0.25
        elif max_mp >= 10:
            # Medium resolution (10-20 MP): downsample to 50% (4x memory reduction)
            factor = 0.5
        else:
            # Low resolution (< 10 MP): use full resolution
            factor = 1.0

        return factor

    def _downsample_image(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Downsample image by the given factor using high-quality interpolation."""
        if factor >= 1.0:
            return img

        h, w = img.shape[:2]
        new_h = int(h * factor)
        new_w = int(w * factor)

        # Use INTER_AREA for downsampling (best quality for shrinking)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _scale_homography(self, H: np.ndarray, downsample_factor: float) -> np.ndarray:
        """Scale homography matrix from downsampled coordinates to full resolution."""
        if downsample_factor >= 1.0:
            return H

        # Create scaling matrix
        scale = 1.0 / downsample_factor
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

        # Scale homography: H_full = S * H_small * S^-1
        S_inv = np.array(
            [[downsample_factor, 0, 0], [0, downsample_factor, 0], [0, 0, 1]]
        )

        return S @ H @ S_inv
