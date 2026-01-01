"""Export functions for Gaussian Splatting."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from .ply_io import load_static_gaussian_ply, write_freetimegs_ply, write_static_gaussian_ply
from .processors.da3 import DA3GaussianProcessor
from .processors.sharp import SharpGaussianProcessor
from .types import GaussianFrame
from .video_utils import extract_video_frames, prune_gaussian_frame

logger = logging.getLogger(__name__)


def _export_sog(
    output_path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    motion: np.ndarray,
    time_center: np.ndarray,
    time_scale: np.ndarray,
) -> None:
    """Export to SOG format."""
    from sogs.compression import run_compression
    import torch

    # Prepare splats dict for SOGS
    splats = {
        "means": torch.from_numpy(means).float().cuda(),
        "scales": torch.from_numpy(scales).float().cuda(),
        "quats": torch.from_numpy(rotations).float().cuda(),
        "opacities": torch.from_numpy(opacities).float().cuda(),
        "sh0": torch.from_numpy(colors).float().cuda(),
        "motion": torch.from_numpy(motion).float().cuda(),
        "t": torch.from_numpy(time_center).float().cuda(),
        "t_scale": torch.from_numpy(time_scale).float().cuda(),
    }

    logger.info(f"Compressing to SOG format: {output_path}")

    if output_path.suffix == ".sog":
        comp_dir = output_path.parent / (output_path.stem + "_sog_temp")
        comp_dir.mkdir(parents=True, exist_ok=True)
        run_compression(str(comp_dir), splats, verbose=True)

        # Zip it up
        shutil.make_archive(str(output_path.with_suffix("")), "zip", comp_dir)
        shutil.move(str(output_path.with_suffix(".zip")), str(output_path))
        shutil.rmtree(comp_dir)
        logger.info(f"Saved SOG file to {output_path}")
    else:
        # Assume directory
        output_path.mkdir(parents=True, exist_ok=True)
        run_compression(str(output_path), splats, verbose=True)


def _export_4dv(
    output_path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    motion: np.ndarray,
    time_center: np.ndarray,
    time_scale: np.ndarray,
) -> None:
    """Export to 4DV format."""
    from .export_4dv import export_4dv

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
        sh_rest=None,  # Currently no SH rest in this flow
    )


def export_video_to_gaussian_plys(
    video_path: Path,
    output_path: Path,
    mode: str = "frames",
    format: str = "ply",
    model_id: str = "depth-anything/DA3-GIANT",
    frame_skip: int = 5,
    chunk_size: int = 10,
    max_frames: int | None = None,
    device: str = "cuda",
    process_res: int = 518,
    opacity_threshold: float = 0.0,
    flip_y: bool = False,
) -> None:
    """Convert video to Gaussian PLY files.

    Args:
        video_path: Input video file
        output_path: Output path (directory for 'frames' mode, file for 'freetimegs')
        mode: 'frames' for per-frame PLYs, 'freetimegs' for single temporal PLY
        format: Output format ('ply', 'sog', '4dv')
        model_id: DA3 model ID (must support infer_gs=True)
        frame_skip: Process every Nth frame
        chunk_size: Number of frames to process together
        max_frames: Maximum frames to process (None for all)
        device: PyTorch device
        process_res: Processing resolution for DA3
        opacity_threshold: Prune Gaussians with opacity below this threshold
        flip_y: If True, negate Y coordinates to flip the coordinate system
    """
    temp_dir = (
        output_path.parent / "temp_frames" if mode == "freetimegs" else output_path / "temp_frames"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, timestamps_ms = extract_video_frames(
        video_path, temp_dir, frame_skip=frame_skip, max_frames=max_frames
    )

    if not frame_paths:
        raise ValueError("No frames extracted from video")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if "sharp" in model_id.lower():
        # Parse SHARP model configuration
        model_path = None
        vit_preset = "dinov2l16_384"  # Default

        if ":" in model_id:
            # Format: sharp:dinov3l16_384 or sharp:/path/to/model.pt
            parts = model_id.split(":", 1)
            config_part = parts[1]

            if Path(config_part).exists() or "\\" in config_part or "/" in config_part:
                model_path = config_part
            else:
                vit_preset = config_part

        processor = SharpGaussianProcessor(
            model_path=model_path, device=device, vit_preset=vit_preset
        )
    else:
        processor = DA3GaussianProcessor(
            model_id=model_id,
            device=device,
            process_res=process_res,
        )

    all_frames: list[GaussianFrame] = []

    for chunk_start in range(0, len(frame_paths), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(frame_paths))
        chunk_paths = frame_paths[chunk_start:chunk_end]
        chunk_timestamps = timestamps_ms[chunk_start:chunk_end]

        logger.info(
            f"Processing chunk {chunk_start // chunk_size + 1}: frames {chunk_start}-{chunk_end - 1}"
        )

        try:
            # For frames mode, process individually to get per-frame PLYs
            # For freetimegs mode, process merged for unified scene
            per_frame = mode == "frames"
            chunk_frames = processor.process_frames(
                chunk_paths, chunk_timestamps, per_frame=per_frame
            )

            for i, frame in enumerate(chunk_frames):
                if per_frame:
                    frame.frame_idx = chunk_start + i

                # Apply pruning if requested
                if opacity_threshold > 0:
                    frame = prune_gaussian_frame(frame, opacity_threshold)
                    # Update the list with pruned frame if needed
                    chunk_frames[i] = frame

            all_frames.extend(chunk_frames)

            if mode == "frames":
                output_path.mkdir(parents=True, exist_ok=True)
                for frame in chunk_frames:
                    ply_path = output_path / f"frame_{frame.frame_idx:06d}.ply"
                    write_static_gaussian_ply(
                        ply_path,
                        frame.means,
                        frame.scales,
                        frame.rotations,
                        frame.colors,
                        frame.opacities,
                        flip_y=flip_y,
                    )
        except Exception as e:
            logger.error(f"Failed to process chunk: {e}")
            raise

    if mode == "freetimegs":
        from .motion_tracking_cpu import compute_motion_vectors

        logger.info("Computing motion vectors (CPU-accelerated with FAISS)...")
        (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
            compute_motion_vectors(all_frames, fps)
        )

        # Zero out motion for static splats (motion magnitude <= 0.001)
        motion_magnitude = np.linalg.norm(motion, axis=1)
        static_mask = motion_magnitude <= 0.001
        n_static = static_mask.sum()
        if n_static > 0:
            logger.info(f"Zeroing motion for {n_static} static splats (motion <= 0.001)")
            motion[static_mask] = 0.0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "sog":
            _export_sog(
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
        elif format == "4dv":
            _export_4dv(
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
        else:
            write_freetimegs_ply(
                output_path,
                means,
                scales,
                rotations,
                colors,
                opacities,
                motion,
                time_center,
                time_scale,
                flip_y=flip_y,
            )

        logger.info(f"Export complete: {output_path}")


def postprocess_plys_to_freetimegs(
    input_dir: Path,
    output_path: Path,
    format: str = "ply",
    fps: float = 30.0,
    max_match_distance: float = 0.05,
    ply_pattern: str = "frame_*.ply",
    flip_y: bool = False,
) -> None:
    """Postprocess existing per-frame PLY files to a single FreeTimeGS PLY.

    Args:
        input_dir: Directory containing per-frame PLY files
        output_path: Output FreeTimeGS PLY file path
        fps: Assumed frame rate if not derivable from filenames
        max_match_distance: Maximum distance for matching Gaussians across frames
        ply_pattern: Glob pattern for PLY files
        flip_y: If True, negate Y coordinates to flip the coordinate system
    """
    ply_files = sorted(input_dir.glob(ply_pattern))

    if not ply_files:
        raise ValueError(f"No PLY files found in {input_dir} matching '{ply_pattern}'")

    logger.info(f"Found {len(ply_files)} PLY files to postprocess")

    frames: list[GaussianFrame] = []
    for i, ply_path in enumerate(ply_files):
        logger.info(f"Loading {i + 1}/{len(ply_files)}: {ply_path.name}")
        frame = load_static_gaussian_ply(ply_path)
        frame.frame_idx = i
        frame.timestamp_ms = i * (1000.0 / fps)
        frames.append(frame)

    logger.info(
        f"Loaded {len(frames)} frames, total {sum(len(f.means) for f in frames)} Gaussian observations"
    )

    from .motion_tracking_cpu import compute_motion_vectors

    logger.info("Computing motion vectors (CPU-accelerated with FAISS)...")
    (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
        compute_motion_vectors(frames, fps, max_match_distance=max_match_distance)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "sog":
        _export_sog(
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
    elif format == "4dv":
        _export_4dv(
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
    else:
        write_freetimegs_ply(
            output_path,
            means,
            scales,
            rotations,
            colors,
            opacities,
            motion,
            time_center,
            time_scale,
            flip_y=flip_y,
        )

    logger.info(f"Wrote FreeTimeGS Gaussians to {output_path}")


def export_images_to_gaussian_plys(
    input_dir: Path,
    output_path: Path,
    mode: str = "frames",
    format: str = "ply",
    fps: float = 30.0,
    model_id: str = "depth-anything/DA3-GIANT",
    image_pattern: str = "*.jpg",
    max_frames: int | None = None,
    device: str = "cuda",
    process_res: int = 518,
    opacity_threshold: float = 0.0,
    masks_dir: Path | None = None,
    mask_first_frame: bool = True,
    remove_black_splats: bool = True,
    save_frequency: int = 5,
    flip_y: bool = False,
) -> None:
    """Process images with DA3 and export to Gaussian PLY files.

    Args:
        input_dir: Directory containing input images
        output_path: Output path (directory for 'frames', file for 'freetimegs')
        mode: 'frames' for per-frame PLYs, 'freetimegs' for single temporal PLY
        format: Output format ('ply', 'sog', '4dv')
        fps: Assumed frame rate for temporal normalization
        model_id: DA3 model ID
        image_pattern: Glob pattern for images
        max_frames: Maximum frames to process
        device: PyTorch device
        process_res: Processing resolution for DA3
        opacity_threshold: Prune Gaussians with opacity below this threshold
        flip_y: If True, negate Y coordinates to flip the coordinate system
    """
    image_paths = sorted(input_dir.glob(image_pattern))
    if max_frames:
        image_paths = image_paths[:max_frames]

    if not image_paths:
        raise ValueError(f"No images found in {input_dir} matching '{image_pattern}'")

    logger.info(f"Found {len(image_paths)} images to process")

    timestamps_ms = [i * (1000.0 / fps) for i in range(len(image_paths))]

    if "sharp" in model_id.lower():
        # Heuristic: if model_id contains "sharp", use Sharp processor
        model_path = (
            model_id if Path(model_id).exists() or "\\" in model_id or "/" in model_id else None
        )
        if model_id.lower() == "sharp":
            model_path = None

        processor = SharpGaussianProcessor(model_path=model_path, device=device)
    else:
        processor = DA3GaussianProcessor(
            model_id=model_id,
            device=device,
            process_res=process_res,
        )

    # Process frames individually first
    if isinstance(processor, SharpGaussianProcessor):
        frames = processor.process_frames(
            image_paths,
            timestamps_ms,
            per_frame=True,
            masks_dir=masks_dir,
            mask_first_frame=mask_first_frame,
            remove_black_splats=remove_black_splats,
        )
    else:
        frames = processor.process_frames(image_paths, timestamps_ms, per_frame=True)

    if not frames:
        raise RuntimeError("No frames processed successfully")

    # Apply pruning if requested
    if opacity_threshold > 0:
        frames = [prune_gaussian_frame(f, opacity_threshold) for f in frames]

    logger.info(
        f"Processed {len(frames)} frames, total {sum(len(f.means) for f in frames)} Gaussians"
    )

    if mode == "frames":
        output_path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            ply_path = output_path / f"frame_{frame.frame_idx:06d}.ply"
            write_static_gaussian_ply(
                ply_path,
                frame.means,
                frame.scales,
                frame.rotations,
                frame.colors,
                frame.opacities,
                flip_y=flip_y,
            )
            if (i + 1) % save_frequency == 0:
                logger.info(f"Saved {i + 1}/{len(frames)} PLY files")
        logger.info(f"Exported {len(frames)} PLY files to {output_path}")
        return

    # FreeTimeGS mode
    from .motion_tracking_cpu import compute_motion_vectors

    logger.info("Computing motion vectors (CPU-accelerated with FAISS)...")
    (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
        compute_motion_vectors(frames, fps)
    )

    # Zero out motion for static splats (motion magnitude <= 0.001)
    motion_magnitude = np.linalg.norm(motion, axis=1)
    static_mask = motion_magnitude <= 0.001
    n_static = static_mask.sum()
    if n_static > 0:
        logger.info(f"Zeroing motion for {n_static} static splats (motion <= 0.001)")
        motion[static_mask] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "sog":
        _export_sog(
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
    elif format == "4dv":
        _export_4dv(
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
    else:
        write_freetimegs_ply(
            output_path,
            means,
            scales,
            rotations,
            colors,
            opacities,
            motion,
            time_center,
            time_scale,
            flip_y=flip_y,
        )

    logger.info(f"Wrote FreeTimeGS Gaussians to {output_path}")
