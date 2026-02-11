"""Export functions for Gaussian Splatting."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .ply_io import (
    load_static_gaussian_ply,
    write_delta_compressed_freetimegs_ply,
    write_freetimegs_ply,
    write_static_gaussian_ply,
)
from .processors.da3 import DA3GaussianProcessor
from .processors.fastgs import FastGSProcessor
from .processors.matrix3d import Matrix3DGaussianProcessor
from .processors.sharp import SharpGaussianProcessor
from .types import GaussianFrame
from .video_utils import extract_video_frames, prune_gaussian_frame

logger = logging.getLogger(__name__)


def extract_audio_ffmpeg(video_path: Path, output_path: Path) -> bool:
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found, skipping audio extraction")
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path.expanduser()),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ab",
            "192k",
            "-y",
            str(output_path.expanduser()),
        ]

        logger.info(f"Extracting audio to {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Audio extracted successfully: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found in PATH")
        return False


def check_existing_plys(output_path: Path) -> set[int]:
    existing_frames = set()
    for ply_file in output_path.glob("frame_*.ply"):
        try:
            frame_num = int(ply_file.stem.split("_")[1])
            existing_frames.add(frame_num)
        except (ValueError, IndexError):
            continue
    return existing_frames


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
    print(f"DEBUG: _export_sog called with output_path={output_path}")
    logger.info(f"SOG export: means.shape={means.shape}")
    logger.info(
        f"motion: {motion.shape if motion is not None else None}, time_center: {time_center.shape if time_center is not None else None}, time_scale: {time_scale.shape if time_scale is not None else None}"
    )
    try:
        from sogs.compression import run_compression
        import torch

        logger.info(f"Preparing SOG compression for {output_path}")

        # Prepare splats dict for SOGS (keep on CPU for FAISS)
        splats = {
            "means": torch.from_numpy(means).float(),
            "scales": torch.from_numpy(scales).float(),
            "quats": torch.from_numpy(rotations).float(),
            "opacities": torch.from_numpy(opacities).float(),
            "sh0": torch.from_numpy(colors).float(),
        }

        # Add FreeTimeGS fields
        if motion is not None and len(motion) > 0:
            splats["motion"] = torch.from_numpy(motion).float()
        if time_center is not None and len(time_center) > 0:
            splats["t"] = torch.from_numpy(time_center).float()
        if time_scale is not None and len(time_scale) > 0:
            splats["t_scale"] = torch.from_numpy(time_scale).float()

        logger.info(f"Compressing to SOG format: {output_path}")

        if output_path.suffix == ".sog":
            run_compression(str(output_path), splats, verbose=True)
        else:
            # Assume directory
            output_path.mkdir(parents=True, exist_ok=True)
            run_compression(str(output_path), splats, verbose=True)
    except Exception as e:
        import traceback

        logger.error(f"SOG compression failed: {e}")
        logger.error(traceback.format_exc())
        logger.info("Falling back to PLY export")
        from .ply_io import write_freetimegs_ply

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
        )


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
    masks_dir: Path | None = None,
    mask_first_frame: bool = False,
    remove_black_splats: bool = True,
    extract_audio: bool = False,
    resume_processing: bool = True,
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
        masks_dir: Directory containing masks for background removal
        mask_first_frame: Whether to apply mask to the first frame
        remove_black_splats: Whether to remove black/background splats
        extract_audio: Whether to also extract audio to MP3
        resume_processing: Whether to skip already processed frames
    """
    if extract_audio:
        audio_path = output_path / "audio.mp3"
        if not audio_path.exists():
            extract_audio_ffmpeg(video_path, audio_path)

    existing_frames = set()
    if resume_processing:
        existing_frames = check_existing_plys(output_path)
        if existing_frames:
            logger.info(
                f"Found {len(existing_frames)} existing PLY files, will skip processed frames"
            )

    temp_dir = (
        output_path.parent / "temp_frames" if mode == "freetimegs" else output_path / "temp_frames"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, timestamps_ms = extract_video_frames(
        video_path, temp_dir, frame_skip=frame_skip, max_frames=max_frames
    )

    if not frame_paths:
        raise ValueError("No frames extracted from video")

    frames_to_process = []
    timestamps_to_process = []

    for i, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps_ms)):
        if i not in existing_frames:
            frames_to_process.append(frame_path)
            timestamps_to_process.append(timestamp)

    if not frames_to_process:
        logger.info("All frames already processed!")
        return

    logger.info(f"Processing {len(frames_to_process)} frames ({len(existing_frames)} already done)")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if "matrix3d" in model_id.lower():
        processor = Matrix3DGaussianProcessor(device=device)
    elif "sharp" in model_id.lower():
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
    elif "sam3dbody" in model_id.lower():
        # SAM 3D Body for human scenes
        # Format: sam3dbody:facebook/sam-3d-body-vith or sam3dbody:/path/to/model.ckpt
        human_model_id = "facebook/sam-3d-body-vith"  # Default
        if ":" in model_id:
            parts = model_id.split(":", 1)
            human_model_id = parts[1]

        from .processors.sam3dbody import Sam3DBodyProcessor

        processor = Sam3DBodyProcessor(
            hf_repo_id=human_model_id,
            device=device,
            points_per_person=10000,
            bbox_threshold=0.8,
        )
    elif "hybrid" in model_id.lower():
        # Hybrid: SAM 3D Body (humans) + depth model (background)
        # Format: hybrid:facebook/sam-3d-body-vith+depth-anything/DA3-GIANT
        # or hybrid:/path/to/human.ckpt:/path/to/depth.pt
        human_model_id = "facebook/sam-3d-body-vith"
        depth_model_id = "depth-anything/DA3-GIANT"
        depth_type = "da3"  # Default depth processor type

        if ":" in model_id:
            parts = model_id.split(":", 1)
            config = parts[1]

            # Parse human:depth configuration
            if "+" in config:
                # Format: human_model+depth_model
                model_parts = config.split("+", 1)
                human_model_id = model_parts[0] if model_parts[0] else human_model_id
                depth_config = model_parts[1] if len(model_parts) > 1 else depth_model_id

                # Detect depth type from depth_config
                if "moge" in depth_config.lower():
                    depth_type = "moge"
                    depth_model_id = depth_config
                elif "sharp" in depth_config.lower():
                    depth_type = "sharp"
                    depth_model_id = depth_config
                else:
                    depth_type = "da3"
                    depth_model_id = depth_config

        from .processors.hybrid import HybridProcessor

        processor = HybridProcessor(
            human_model_id=human_model_id,
            depth_model_id=depth_model_id,
            depth_processor_type=depth_type,
            device=device,
            points_per_person=10000,
            bbox_threshold=0.8,
            depth_process_res=process_res,
            remove_overlap=True,
        )
    elif "motioncrafter" in model_id.lower():
        from .processors.motioncrafter import MotionCrafterProcessor

        config_path = None
        model_path = None

        if ":" in model_id:
            parts = model_id.split(":", 1)
            p = parts[1]
            if p.endswith(".yaml"):
                config_path = p
            else:
                model_path = p

        processor = MotionCrafterProcessor(
            model_path=model_path,
            config_path=config_path,
            device=device,
            process_res=process_res,
        )
    else:
        processor = DA3GaussianProcessor(
            model_id=model_id,
            device=device,
            process_res=process_res,
        )

    all_frames: list[GaussianFrame] = []

    for chunk_start in range(0, len(frames_to_process), chunk_size):
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

            if isinstance(processor, DA3GaussianProcessor):
                chunk_frames = processor.process_frames(
                    chunk_paths,
                    chunk_timestamps,
                    per_frame=per_frame,
                    masks_dir=masks_dir,
                    mask_first_frame=mask_first_frame,
                    remove_black_splats=remove_black_splats,
                )
            elif isinstance(processor, SharpGaussianProcessor):
                chunk_frames = processor.process_frames(
                    chunk_paths,
                    chunk_timestamps,
                    per_frame=per_frame,
                    masks_dir=masks_dir,
                    mask_first_frame=mask_first_frame,
                )
            else:
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

    if mode == "frames":
        logger.info(f"Exported frames to {output_path}")
        return

    if mode != "freetimegs":
        raise ValueError(f"Invalid mode: {mode}")

    from .motion_tracking_cuda import compute_motion_vectors_cuda

    # Extract flow priors if available (from MotionCrafter)
    all_flows = None
    if all_frames and hasattr(all_frames[0], "flow") and all_frames[0].flow is not None:
        logger.info("Extracting flow priors for motion tracking...")
        all_flows = [f.flow for f in all_frames]
        # If any flow is None, disable flow priors to avoid crashes
        if any(f is None for f in all_flows):
            logger.warning("Some frames missing flow data, disabling flow priors")
            all_flows = None

    logger.info("Computing motion vectors (GPU-accelerated with cuTile)...")
    (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
        compute_motion_vectors_cuda(all_frames, fps, all_flows=all_flows)
    )

    # Zero out motion for static splats (motion magnitude <= 0.001)
    motion_magnitude = np.linalg.norm(motion, axis=1)
    static_mask = motion_magnitude <= 0.001
    n_static = static_mask.sum()
    if n_static > 0:
        logger.info(f"Zeroing motion for {n_static} static splats (motion <= 0.001)")
        motion[static_mask] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"DEBUG: format = '{format}'")
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
    fps: float = 30.0,
    max_match_distance: float = 0.05,
    ply_pattern: str = "frame_*.ply",
    flip_y: bool = False,
    format: str = "ply",
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
    print(f"DEBUG: postprocess_plys_to_freetimegs called with format='{format}'")
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

    from .motion_tracking_cuda import compute_motion_vectors_cuda

    logger.info("Computing motion vectors (GPU-accelerated with cuTile)...")
    (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
        compute_motion_vectors_cuda(frames, fps, max_match_distance=max_match_distance)
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
    debug_output_dir: Path | None = None,
    refine_boundaries: bool = False,
    boundary_min_angle: float = 3.0,
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

    if "matrix3d" in model_id.lower():
        processor = Matrix3DGaussianProcessor(device=device)
    elif "moge" in model_id.lower():
        from .processors.moge import MoGeGaussianProcessor

        processor = MoGeGaussianProcessor(
            model_id=model_id,
            device=device,
            debug_output_dir=debug_output_dir,
            refine_boundaries=refine_boundaries,
            boundary_min_angle=boundary_min_angle,
        )
    elif "motioncrafter" in model_id.lower():
        from .processors.motioncrafter import MotionCrafterProcessor

        config_path = None
        model_path = None

        if ":" in model_id:
            parts = model_id.split(":", 1)
            p = parts[1]
            if p.endswith(".yaml"):
                config_path = p
            else:
                model_path = p

        processor = MotionCrafterProcessor(
            model_path=model_path,
            config_path=config_path,
            device=device,
            process_res=process_res,
        )
    elif "sharp" in model_id.lower():
        model_path = (
            model_id if Path(model_id).exists() or "\\" in model_id or "/" in model_id else None
        )
        if model_id.lower() == "sharp":
            model_path = None

        processor = SharpGaussianProcessor(model_path=model_path, device=device)
    elif "fastgs" in model_id.lower():
        from .processors.fastgs import FastGSProcessor

        fastgs_path = None
        iterations = 30_000

        if ":" in model_id:
            parts = model_id.split(":", 1)
            config = parts[1]

            if Path(config).exists() or "\\" in config or "/" in config:
                fastgs_path = config
            else:
                try:
                    iterations = int(config)
                except ValueError:
                    pass

        processor = FastGSProcessor(
            device=device,
            iterations=iterations,
            fastgs_path=fastgs_path,
            white_background=False,
            eval=False,
            sh_degree=3,
        )
    elif "trellis.2" in model_id.lower():
        from .processors.trellis2 import Trellis2Processor

        processor = Trellis2Processor(model_id=model_id, device=device)

        # TRELLIS.2 produces GLB meshes or VXZ o-voxel files, not Gaussians
        # Determine output format based on --format arg or file extension
        output_format = format.lower() if format else "glb"

        # Handle frames mode (export each image separately)
        if mode == "frames":
            output_path.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(image_paths):
                if output_format == "vxz":
                    out_file = output_path / f"{p.stem}.vxz"
                    processor.export_vxz(p, out_file)
                else:
                    # Default to GLB
                    out_file = output_path / f"{p.stem}.glb"
                    processor.export_glb(p, out_file)
            return

        # Handle single output mode
        if output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            if output_format == "vxz":
                out_file = output_path / f"{image_paths[0].stem}.vxz"
            else:
                out_file = output_path / f"{image_paths[0].stem}.glb"
        else:
            out_file = output_path

        if output_format == "vxz":
            processor.export_vxz(image_paths[0], out_file)
        else:
            processor.export_glb(image_paths[0], out_file)
        return

    elif "trellis" in model_id.lower():
        from .processors.trellis import TrellisProcessor

        processor = TrellisProcessor(model_id=model_id, device=device)
    else:
        processor = DA3GaussianProcessor(
            model_id=model_id,
            device=device,
            process_res=process_res,
        )

    # Process frames individually first
    if isinstance(processor, DA3GaussianProcessor):
        frames = processor.process_frames(
            image_paths,
            timestamps_ms,
            per_frame=True,
            masks_dir=masks_dir,
            mask_first_frame=mask_first_frame,
            remove_black_splats=remove_black_splats,
        )
    elif isinstance(processor, SharpGaussianProcessor):
        frames = processor.process_frames(
            image_paths,
            timestamps_ms,
            per_frame=True,
            masks_dir=masks_dir,
            mask_first_frame=mask_first_frame,
        )
    else:
        # Generic processor (Trellis)
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

    # FreeTimeGS mode - support delta compression as alternative
    mode_lower = mode.lower()

    # Extract flow priors if available (from MotionCrafter)
    all_flows = None
    if all_frames and hasattr(all_frames[0], "flow") and all_frames[0].flow is not None:
        logger.info("Extracting flow priors for motion tracking...")
        all_flows = [f.flow for f in all_frames]
        # Ensure flows are not None (some frames might be skipped or fail)
        # If any flow is None, we should probably disable flow priors to avoid crashes
        if any(f is None for f in all_flows):
            logger.warning("Some frames missing flow data, disabling flow priors")
            all_flows = None

    if "delta" in mode_lower:
        from .motion_tracking_cuda import compute_motion_vectors_delta_compression_cuda

        use_int8 = "int8" in mode_lower
        compression_ratio = 51.0  # Default target ratio

        logger.info(
            f"Computing delta-compressed motion vectors (GPU-accelerated with cuTile, ratio={compression_ratio}x)..."
        )
        (
            means,
            scales,
            rotations,
            colors,
            opacities,
            deltas,
            time_center,
            time_scale,
            compression_scale,
        ) = compute_motion_vectors_delta_compression_cuda(
            all_frames,
            fps,
            compression_ratio_target=compression_ratio,
            use_int8=use_int8,
            all_flows=all_flows,
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
                deltas.astype(np.float32),  # Convert back for SOG compatibility
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
                deltas.astype(np.float32),  # Convert back for 4DV compatibility
                time_center,
                time_scale,
            )
        else:
            write_delta_compressed_freetimegs_ply(
                output_path,
                means,
                scales,
                rotations,
                colors,
                opacities,
                deltas,
                time_center,
                time_scale,
                compression_scale=compression_scale,
                use_int8=use_int8,
                flip_y=flip_y,
            )

        logger.info(f"Export complete: {output_path} (delta-compressed)")
        return
    else:
        from .motion_tracking_cuda import compute_motion_vectors_cuda

        logger.info("Computing motion vectors (GPU-accelerated with cuTile)...")
        (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
            compute_motion_vectors_cuda(frames, fps)
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
