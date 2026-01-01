"""Video and frame processing utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .types import GaussianFrame

logger = logging.getLogger(__name__)


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    frame_skip: int = 1,
    max_frames: int | None = None,
) -> tuple[list[Path], list[float]]:
    """Extract frames from video.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_skip: Process every Nth frame
        max_frames: Maximum frames to extract (None for all)

    Returns:
        Tuple of (frame_paths, timestamps_ms)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_paths = []
    timestamps_ms = []
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            if max_frames is not None and saved >= max_frames:
                break

            frame_path = output_dir / f"frame_{saved:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            timestamps_ms.append(idx * 1000.0 / fps)
            saved += 1

            if saved % 50 == 0:
                logger.info(f"Extracted {saved} frames")

        idx += 1

    cap.release()
    logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
    return frame_paths, timestamps_ms


def prune_gaussian_frame(frame: GaussianFrame, opacity_threshold: float = 0.05) -> GaussianFrame:
    """Prune Gaussians with low opacity.

    Args:
        frame: GaussianFrame to prune
        opacity_threshold: Minimum sigmoid(opacity) to keep (probability threshold)

    Returns:
        New GaussianFrame with pruned data
    """
    # Opacities are logits (inverse sigmoid)
    # sigmoid(x) = 1 / (1 + exp(-x))
    # we want sigmoid(op) > threshold
    probs = 1.0 / (1.0 + np.exp(-frame.opacities))
    mask = probs > opacity_threshold

    n_before = len(frame.means)
    n_after = np.sum(mask)

    if n_after < n_before:
        logger.info(
            f"Pruned frame {frame.frame_idx}: {n_before} -> {n_after} splats (threshold {opacity_threshold})"
        )

    return GaussianFrame(
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        means=frame.means[mask],
        scales=frame.scales[mask],
        rotations=frame.rotations[mask],
        colors=frame.colors[mask],
        opacities=frame.opacities[mask],
    )
