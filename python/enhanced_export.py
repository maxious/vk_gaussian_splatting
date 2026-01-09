#!/usr/bin/env python3
"""Enhanced video to Gaussian PLY exporter with audio extraction and improved chunking."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from offline.exporters import export_video_to_gaussian_plys
from offline.video_utils import extract_video_frames

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
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ab",
            "192k",
            "-y",
            str(output_path),
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

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ab",
            "192k",
            "-y",
            str(output_path),
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


async def extract_video_frames_async(
    video_path: Path, temp_dir: Path, frame_skip: int = 1, max_frames: int | None = None
) -> tuple[list[Path], list[int]]:
    logger.info(f"Extracting frames asynchronously to {temp_dir}")

    loop = asyncio.get_event_loop()
    frame_paths, timestamps_ms = await loop.run_in_executor(
        None, extract_video_frames, video_path, temp_dir, frame_skip, max_frames
    )

    logger.info(f"Extracted {len(frame_paths)} frames")
    return frame_paths, [int(t) for t in timestamps_ms]


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced video to Gaussian PLY exporter with audio extraction"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--mode",
        choices=["frames", "freetimegs"],
        default="frames",
        help="Export mode: 'frames' for per-frame PLYs, 'freetimegs' for temporal PLY",
    )
    parser.add_argument(
        "--format",
        choices=["ply", "sog", "4dv"],
        default="ply",
        help="Output format: 'ply' (standard), 'sog' (compressed static/dynamic), '4dv' (compressed dynamic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sharp",
        help="Model ID: 'sharp', 'depth-anything/DA3-GIANT', etc.",
    )
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--chunk-size", type=int, default=10, help="Frames per processing chunk")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--process-res", type=int, default=518, help="Processing resolution")
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.05,
        help="Prune Gaussians with opacity below this threshold",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Negate Y coordinates to flip coordinate system (useful for SHARP models)",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Directory containing mask images for background removal",
    )
    parser.add_argument(
        "--no-mask-first-frame",
        action="store_true",
        help="Apply mask to first frame (default: skip first frame)",
    )
    parser.add_argument(
        "--no-remove-black-splats",
        action="store_true",
        help="Keep black splats instead of removing them (default: remove)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device")
    parser.add_argument("--extract-audio", action="store_true", help="Also extract audio to MP3")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.verbose:
        for lib in ["matplotlib", "PIL", "urllib3", "timm", "huggingface_hub", "torch"]:
            logging.getLogger(lib).setLevel(logging.INFO)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input video not found: {args.input}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    existing_frames = check_existing_plys(args.output)
    if existing_frames:
        logger.info(f"Found {len(existing_frames)} existing PLY files, will skip processed frames")

    if args.extract_audio:
        audio_path = args.output / "audio.mp3"
        if not audio_path.exists():
            extract_audio_ffmpeg(args.input, audio_path)
        else:
            logger.info(f"Audio already exists: {audio_path}")

    temp_dir = args.output / "temp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_paths, timestamps_ms = asyncio.run(
            extract_video_frames_async(args.input, temp_dir, args.frame_skip, args.max_frames)
        )
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Check for existing PLY files
    existing_frames = check_existing_plys(args.output)
    if existing_frames:
        logger.info(f"Found {len(existing_frames)} existing PLY files, will skip processed frames")

    # Extract audio if requested
    if args.extract_audio:
        audio_path = args.output / "audio.mp3"
        if not audio_path.exists():
            extract_audio_ffmpeg(args.input, audio_path)
        else:
            logger.info(f"Audio already exists: {audio_path}")

    # Enhanced frame extraction with async support
    temp_dir = args.output / "temp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Use async frame extraction
    try:
        frame_paths, timestamps_ms = asyncio.run(
            extract_video_frames_async(args.input, temp_dir, args.frame_skip, args.max_frames)
        )
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        sys.exit(1)

    if not frame_paths:
        logger.error("No frames extracted from video")
        sys.exit(1)

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

    try:
        export_video_to_gaussian_plys(
            args.input,
            args.output,
            mode=args.mode,
            format=args.format,
            model_id=args.model,
            frame_skip=args.frame_skip,
            chunk_size=args.chunk_size,
            max_frames=args.max_frames,
            device=args.device,
            process_res=args.process_res,
            opacity_threshold=args.opacity_threshold,
            flip_y=args.flip_y,
            masks_dir=args.masks_dir,
            mask_first_frame=not args.no_mask_first_frame,
            remove_black_splats=not args.no_remove_black_splats,
        )

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        logger.info(f"Processing complete! Output in {args.output}")

        final_ply_count = len(list(args.output.glob("frame_*.ply")))
        logger.info(f"Final PLY count: {final_ply_count}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
