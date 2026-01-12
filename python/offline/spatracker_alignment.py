"""
SpaTracker-based alignment and motion-aware compression for Gaussian Splatting.

This module provides:
1. Motion-aware temporal compression using SpaTracker results
2. Alignment refinement for static Gaussians using 3D tracking consistency
3. Integration with existing export pipeline
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def detect_spatracker_installed() -> bool:
    """Check if SpaTracker is installed."""
    try:
        import spatracker

        return True
    except ImportError:
        return False


def install_spatracker() -> bool:
    """Attempt to install SpaTracker if not present."""
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "spatracker"], capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("Successfully installed spatracker")
            return True
        else:
            logger.error(f"Failed to install spatracker: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error installing spatracker: {e}")
        return False


def compute_alignment_refinement(
    coords: np.ndarray,
    visibs: np.ndarray,
    static_threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute alignment refinement parameters using SpaTracker results.

    This uses static tracks from SpaTracker to compute per-frame alignment corrections.
    Static tracks should be consistent across frames, so any deviation indicates
    alignment errors that can be corrected.

    Args:
        coords: (T, N, 3) array of 3D coordinates from SpaTracker
        visibs: (T, N) array of visibility scores
        static_threshold: Variance threshold for classifying tracks as static

    Returns:
        scale_corrections: (T,) array of scale correction factors per frame
        shift_corrections: (T, 3) array of shift corrections per frame
        static_mask: (N,) boolean array indicating static tracks
    """
    T, N, _ = coords.shape

    # Identify static tracks using variance
    variances = np.var(coords, axis=0).sum(axis=-1)
    is_static = variances < static_threshold

    if np.sum(is_static) < 10:
        logger.warning(
            f"Too few static tracks ({np.sum(is_static)}), using all tracks for alignment"
        )
        is_static = np.ones(N, dtype=bool)

    static_coords = coords[:, is_static, :]
    static_visibs = visibs[:, is_static]

    # Use the first frame as reference
    ref_coords = static_coords[0]  # (N_static, 3)

    scale_corrections = np.ones(T, dtype=np.float32)
    shift_corrections = np.zeros((T, 3), dtype=np.float32)

    # For each frame, compute alignment to reference using static points
    for t in range(1, T):
        frame_coords = static_coords[t]
        frame_visibs = static_visibs[t]

        # Weight by visibility
        weights = frame_visibs

        # Compute weighted centroid
        ref_centroid = np.average(ref_coords, axis=0, weights=weights)
        frame_centroid = np.average(frame_coords, axis=0, weights=weights)

        # Center the points
        ref_centered = ref_coords - ref_centroid
        frame_centered = frame_coords - frame_centroid

        # Compute scale correction using SVD-like approach
        # We want to find scale s such that: frame_centered ≈ s * ref_centered
        # Using weighted least squares

        # Compute norms
        ref_norms = np.linalg.norm(ref_centered, axis=1)
        frame_norms = np.linalg.norm(frame_centered, axis=1)

        # Avoid division by zero
        valid = ref_norms > 1e-6 & (frame_norms > 1e-6)
        if np.sum(valid) < 10:
            logger.warning(f"Frame {t}: insufficient valid points for alignment")
            continue

        # Compute scale as weighted ratio of norms
        ratios = frame_norms[valid] / ref_norms[valid]
        scale_corrections[t] = np.average(ratios, weights=weights[valid])

        # Compute shift correction
        shift_corrections[t] = ref_centroid - scale_corrections[t] * frame_centroid

    logger.info(
        f"Computed alignment corrections: avg scale={np.mean(scale_corrections):.4f}, "
        f"avg shift norm={np.mean(np.linalg.norm(shift_corrections, axis=1)):.4f}"
    )

    return scale_corrections, shift_corrections, is_static


def apply_alignment_correction(
    gaussian_frame,
    scale_correction: float,
    shift_correction: np.ndarray,
) -> None:
    """
    Apply alignment correction to a Gaussian frame in-place.

    Args:
        gaussian_frame: GaussianFrame object to modify
        scale_correction: Scale factor to apply
        shift_correction: (3,) shift vector to apply
    """
    # Apply scale and shift to means
    gaussian_frame.means = gaussian_frame.means * scale_correction + shift_correction


def compute_motion_segments(
    coords: np.ndarray,
    visibs: np.ndarray,
    motion_threshold: float = 0.001,
    segment_length: int = 30,
) -> np.ndarray:
    """
    Compute motion segments for piecewise linear motion approximation.

    SpaTracker provides dense 3D trajectories. For efficient 4DV representation
    (which uses linear motion), we segment the trajectories based on motion characteristics.

    Args:
        coords: (T, N, 3) array of 3D coordinates
        visibs: (T, N) array of visibility scores
        motion_threshold: Minimum motion to trigger a new segment
        segment_length: Maximum frames per segment

    Returns:
        segment_ids: (T,) array of segment IDs per frame
    """
    T, N, _ = coords.shape

    # Compute per-frame motion for each track
    displacements = np.linalg.norm(np.diff(coords, axis=0), axis=-1)  # (T-1, N)
    avg_motion = np.mean(displacements, axis=1)  # (T-1,)

    # Identify keyframes where motion changes significantly
    motion_changes = np.diff(avg_motion)
    keyframe_indices = [0]

    for i, change in enumerate(motion_changes):
        if abs(change) > motion_threshold * avg_motion[i]:
            keyframe_indices.append(i + 1)

    # Ensure maximum segment length
    final_keyframes = [keyframe_indices[0]]
    for kf in keyframe_indices[1:]:
        if kf - final_keyframes[-1] >= segment_length:
            final_keyframes.append(kf)

    final_keyframes.append(T)

    # Create segment IDs
    segment_ids = np.zeros(T, dtype=np.int32)
    for seg_id, (start, end) in enumerate(zip(final_keyframes[:-1], final_keyframes[1:])):
        segment_ids[start:end] = seg_id

    logger.info(f"Split {T} frames into {len(final_keyframes) - 1} motion segments")
    return segment_ids


def export_segmented_4dv(
    coords: np.ndarray,
    visibs: np.ndarray,
    segment_ids: np.ndarray,
    output_path: Path,
    fps: float = 30.0,
    static_threshold: float = 0.01,
) -> None:
    """
    Export SpaTracker results to 4DV format with motion segmentation.

    This creates multiple 4DV files, one per motion segment, which allows
    for better compression by using shorter time scales for linear approximation.

    Args:
        coords: (T, N, 3) array of 3D coordinates
        visibs: (T, N) array of visibility scores
        segment_ids: (T,) array of segment IDs per frame
        output_path: Path for output 4DV file (will add segment suffix)
        fps: Frame rate
        static_threshold: Threshold for static track detection
    """
    from offline.export_4dv import export_4dv

    T, N, _ = coords.shape
    times = np.linspace(0, 1, T)

    unique_segments = np.unique(segment_ids)

    for seg_id in unique_segments:
        seg_mask = segment_ids == seg_id
        seg_indices = np.where(seg_mask)[0]
        seg_coords = coords[seg_mask]
        seg_times = times[seg_mask]

        if len(seg_indices) < 2:
            logger.warning(f"Segment {seg_id} has < 2 frames, skipping")
            continue

        # Compute means and velocities for this segment
        means_center = np.mean(seg_coords, axis=0)

        # Linear fit for velocity
        t_mean = np.mean(seg_times)
        t_centered = seg_times - t_mean
        coords_centered = seg_coords - means_center[None, :, :]

        numerator = np.sum(t_centered[:, None, None] * coords_centered, axis=0)
        denominator = np.sum(t_centered**2)

        velocity = numerator / denominator

        # Apply static threshold to zero out velocity for static tracks
        variances = np.var(seg_coords, axis=0).sum(axis=-1)
        is_static = variances < static_threshold
        velocity[is_static] = 0.0

        # Scales: heuristic 0.02
        scales = np.ones((N, 3), dtype=np.float32) * 0.02
        scales = np.log(scales)

        # Rotations: Identity
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 0] = 1.0

        # Colors: Use visibility-weighted average (simplified)
        avg_visib = np.mean(visibs[seg_mask], axis=0)
        colors = np.zeros((N, 3), dtype=np.float32)
        colors[:, 0] = avg_visib  # Use visibility as red channel placeholder

        # Opacities
        opacities = np.ones(N, dtype=np.float32) * 10.0

        # Time params
        time_center = np.ones(N, dtype=np.float32) * 0.5
        time_scale = np.ones(N, dtype=np.float32) * 0.5  # Shorter time scale for segments

        # Output file for this segment
        if len(unique_segments) > 1:
            seg_output = output_path.parent / f"{output_path.stem}_seg{seg_id}{output_path.suffix}"
        else:
            seg_output = output_path

        logger.info(f"Exporting segment {seg_id} to {seg_output}")
        export_4dv(
            seg_output,
            means_center.astype(np.float32),
            scales.astype(np.float32),
            rotations.astype(np.float32),
            colors.astype(np.float32),
            opacities.astype(np.float32),
            velocity.astype(np.float32),
            time_center.astype(np.float32),
            time_scale.astype(np.float32),
        )


def spatracker_postprocess(
    input_ply_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    static_threshold: float = 0.01,
    use_alignment_refinement: bool = True,
) -> None:
    """
    Postprocess per-frame PLY files using SpaTracker for alignment and motion-aware compression.

    This function:
    1. Loads per-frame PLY files
    2. Optionally runs SpaTracker to get 3D trajectories
    3. Computes alignment corrections using static tracks
    4. Exports to 4DV format with motion-aware compression

    Args:
        input_ply_dir: Directory containing per-frame PLY files
        output_path: Output path for 4DV file
        fps: Frame rate
        static_threshold: Threshold for static track detection
        use_alignment_refinement: Whether to apply alignment corrections
    """
    if not detect_spatracker_installed():
        logger.warning("SpaTracker not installed. Install with: pip install spatracker")
        logger.warning("Proceeding without SpaTracker-based alignment refinement")
        use_alignment_refinement = False

    # Load PLY files
    ply_files = sorted(input_ply_dir.glob("frame_*.ply"))
    if not ply_files:
        raise ValueError(f"No PLY files found in {input_ply_dir}")

    logger.info(f"Loading {len(ply_files)} PLY files from {input_ply_dir}")

    from offline.ply_io import load_static_gaussian_ply
    from offline.motion_tracking_cuda import compute_motion_vectors_cuda

    frames = []
    for i, ply_path in enumerate(ply_files):
        frame = load_static_gaussian_ply(ply_path)
        frame.frame_idx = i
        frame.timestamp_ms = i * (1000.0 / fps)
        frames.append(frame)

    logger.info(f"Loaded {len(frames)} frames")

    if use_alignment_refinement:
        try:
            import spatracker

            logger.info("Running SpaTracker for 3D trajectory extraction...")

            # Stack all means to create trajectory input
            all_means = np.stack([f.means for f in frames])  # (T, N, 3)
            all_visibs = np.ones((len(frames), frames[0].means.shape[0]), dtype=np.float32)

            # Run SpaTracker
            tracker = spatracker.SpaTracker()
            result = tracker.track(all_means, all_visibs)

            coords = result["coords"]  # (T, N, 3)
            visibs = result["visibs"]  # (T, N)

            # Compute alignment corrections
            scale_corrections, shift_corrections, is_static = compute_alignment_refinement(
                coords, visibs, static_threshold
            )

            # Apply corrections to frames
            for i, frame in enumerate(frames):
                if i > 0:  # Skip first frame (reference)
                    apply_alignment_correction(frame, scale_corrections[i], shift_corrections[i])

            logger.info("Applied SpaTracker-based alignment corrections")

        except Exception as e:
            logger.error(f"SpaTracker processing failed: {e}")
            logger.warning("Proceeding without alignment refinement")
            use_alignment_refinement = False

    # Compute motion vectors (existing CUDA implementation)
    logger.info("Computing motion vectors...")
    (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
        compute_motion_vectors_cuda(frames, fps)
    )

    # Zero out motion for static tracks (from SpaTracker if available)
    if use_alignment_refinement and "is_static" in dir():
        # We don't have access to is_static here, but we can use motion magnitude
        pass

    motion_magnitude = np.linalg.norm(motion, axis=1)
    static_mask = motion_magnitude <= 0.001
    n_static = static_mask.sum()
    if n_static > 0:
        logger.info(f"Zeroing motion for {n_static} static splats (motion <= 0.001)")
        motion[static_mask] = 0.0

    # Export to 4DV
    from offline.export_4dv import export_4dv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_4dv(
        output_path,
        means,
        scales,
        rotations,
        colors,
        opacities,
        motion,
        time_center,
        time_scale,
    )

    logger.info(f"Exported SpaTracker-aligned 4DV to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpaTracker-based alignment and motion-aware compression"
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input directory with per-frame PLY files"
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .4dv file")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate")
    parser.add_argument(
        "--static-threshold",
        type=float,
        default=0.01,
        help="Variance threshold for static tracks (default: 0.01)",
    )
    parser.add_argument(
        "--no-alignment",
        action="store_true",
        help="Skip SpaTracker-based alignment refinement",
    )

    args = parser.parse_args()

    spatracker_postprocess(
        args.input,
        args.output,
        fps=args.fps,
        static_threshold=args.static_threshold,
        use_alignment_refinement=not args.no_alignment,
    )


if __name__ == "__main__":
    main()
