"""Command-line interface for Gaussian Splatting export."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .exporters import (
    export_images_to_gaussian_plys,
    export_video_to_gaussian_plys,
    postprocess_plys_to_freetimegs,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export video to Gaussian Splatting PLY files using DA3"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Depth extraction subcommand
    depth_parser = subparsers.add_parser(
        "depth", help="Extract depth maps from video using DA3 (multi-XPU optimized)"
    )
    depth_parser.add_argument("--input", "-i", type=Path, required=True, help="Input video file")
    depth_parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    depth_parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3METRIC-LARGE",
        help="DA3 model ID (default: depth-anything/DA3METRIC-LARGE)",
    )
    depth_parser.add_argument(
        "--device-spec",
        type=str,
        default="auto",
        help="Device specification: 'auto', 'xpu', 'xpu:0', 'xpu:0,1', 'cuda', 'cuda:0'",
    )
    depth_parser.add_argument(
        "--process-res",
        type=int,
        default=518,
        help="Processing resolution for DA3 (default: 518)",
    )
    depth_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device for inference (default: 4)",
    )
    depth_parser.add_argument(
        "--format",
        type=str,
        default="vdz",
        choices=["vdz", "video"],
        help="Output format: 'vdz' for compressed depth sequence, 'video' for H.265 lossless",
    )
    depth_parser.add_argument("-v", "--verbose", action="store_true")

    # Depth extraction subcommand
    extract_parser = subparsers.add_parser(
        "extract-depth", help="Extract depth frames from VDZ file as images"
    )
    extract_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input VDZ depth sequence file"
    )
    extract_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for depth images"
    )
    extract_parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "exr"],
        help="Output image format (default: png)",
    )
    extract_parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        choices=["viridis", "magma", "plasma", "inferno", "gray", "raw"],
        help="Colormap for depth visualization (default: viridis). Use 'raw' for metric depth.",
    )
    extract_parser.add_argument(
        "--frame-range",
        type=str,
        default=None,
        help="Frame range to extract (e.g., '0-100' or '50,60,70')",
    )
    extract_parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Resize output images to max width (maintains aspect ratio)",
    )
    extract_parser.add_argument("-v", "--verbose", action="store_true")

    export_parser = subparsers.add_parser("export", help="Export video to Gaussian PLYs")
    export_parser.add_argument("--input", "-i", type=Path, required=True, help="Input video file")
    export_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path (directory for frames mode, file for freetimegs)",
    )
    export_parser.add_argument(
        "--mode",
        choices=["frames", "freetimegs"],
        default="frames",
        help="Export mode: 'frames' for per-frame PLYs, 'freetimegs' for temporal PLY",
    )
    export_parser.add_argument(
        "--format",
        choices=["ply", "sog", "4dv"],
        default="ply",
        help="Output format: 'ply' (standard), 'sog' (compressed static/dynamic), '4dv' (compressed dynamic)",
    )
    export_parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3-GIANT",
        help="DA3 model ID (must support infer_gs)",
    )
    export_parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame")
    export_parser.add_argument(
        "--chunk-size", type=int, default=10, help="Frames per processing chunk"
    )
    export_parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process"
    )
    export_parser.add_argument(
        "--process-res", type=int, default=518, help="Processing resolution for DA3"
    )
    export_parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.0,
        help="Prune Gaussians with opacity below this threshold (e.g. 0.05)",
    )
    export_parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Negate Y coordinates to flip the coordinate system (useful for SHARP models)",
    )
    export_parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Directory containing mask images for background removal",
    )
    export_parser.add_argument(
        "--no-mask-first-frame",
        action="store_true",
        help="Apply mask to the first frame (default: skip first frame)",
    )
    export_parser.add_argument(
        "--no-remove-black-splats",
        action="store_true",
        help="Keep black splats instead of removing them (default: remove)",
    )
    export_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'xpu', 'cpu', 'mps' (auto-detects best available)",
    )
    export_parser.add_argument(
        "--extract-audio", action="store_true", help="Also extract audio to MP3"
    )
    export_parser.add_argument(
        "--resume-processing",
        action="store_true",
        default=True,
        help="Skip already processed frames",
    )
    export_parser.add_argument("-v", "--verbose", action="store_true")

    postprocess_parser = subparsers.add_parser(
        "postprocess", help="Postprocess per-frame PLYs to FreeTimeGS PLY with motion vectors"
    )
    postprocess_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory containing per-frame PLY files",
    )
    postprocess_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output FreeTimeGS PLY file"
    )
    postprocess_parser.add_argument(
        "--format",
        choices=["ply", "sog", "4dv"],
        default="ply",
        help="Output format: 'ply', 'sog', '4dv'",
    )
    postprocess_parser.add_argument(
        "--fps", type=float, default=30.0, help="Frame rate for temporal normalization"
    )
    postprocess_parser.add_argument(
        "--max-match-distance",
        type=float,
        default=0.05,
        help="Maximum distance for matching Gaussians across frames",
    )
    postprocess_parser.add_argument(
        "--pattern", type=str, default="frame_*.ply", help="Glob pattern for PLY files"
    )
    postprocess_parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Negate Y coordinates to flip the coordinate system (useful for SHARP models). Only use if input PLYs weren't already flipped.",
    )
    postprocess_parser.add_argument("-v", "--verbose", action="store_true")

    images_parser = subparsers.add_parser(
        "images", help="Process images with DA3 and export to Gaussian PLY files"
    )
    images_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input directory containing images"
    )
    images_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path (directory for frames, file for freetimegs)",
    )
    images_parser.add_argument(
        "--mode",
        choices=["frames", "freetimegs"],
        default="frames",
        help="Export mode: 'frames' for per-frame PLYs, 'freetimegs' for temporal PLY",
    )
    images_parser.add_argument(
        "--format",
        choices=["ply", "sog", "4dv", "glb", "vxz"],
        default="ply",
        help="Output format: 'ply' (standard), 'sog' (compressed static/dynamic), '4dv' (compressed dynamic), 'glb' (TRELLIS.2 mesh), 'vxz' (TRELLIS.2 o-voxel)",
    )
    images_parser.add_argument(
        "--fps", type=float, default=30.0, help="Frame rate for temporal normalization"
    )
    images_parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3-GIANT",
        help="Model ID. Options: 'depth-anything/DA3-GIANT', 'sharp', 'microsoft/TRELLIS-image-large', 'microsoft/TRELLIS.2-4B'",
    )
    images_parser.add_argument(
        "--pattern", type=str, default="*.jpg", help="Glob pattern for image files"
    )
    images_parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process"
    )
    images_parser.add_argument(
        "--process-res", type=int, default=518, help="Processing resolution for DA3"
    )
    images_parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Directory containing mask images for background removal",
    )
    images_parser.add_argument(
        "--no-mask-first-frame",
        action="store_true",
        help="Apply mask to the first frame (default: skip first frame)",
    )
    images_parser.add_argument(
        "--no-remove-black-splats",
        action="store_true",
        help="Keep black splats instead of removing them (default: remove)",
    )
    images_parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Negate Y coordinates to flip the coordinate system (useful for SHARP models). If using postprocess afterward, don't flip there too.",
    )
    images_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'xpu', 'cpu', 'mps' (auto-detects best available)",
    )
    images_parser.add_argument("-v", "--verbose", action="store_true")

    legacy_parser = subparsers.add_parser("legacy", help="Legacy CLI (deprecated)")
    legacy_parser.add_argument("--input", "-i", type=Path, required=True)
    legacy_parser.add_argument("--output", "-o", type=Path, required=True)
    legacy_parser.add_argument("--mode", choices=["frames", "freetimegs"], default="frames")
    legacy_parser.add_argument("--model", type=str, default="depth-anything/DA3-GIANT")
    legacy_parser.add_argument("--frame-skip", type=int, default=5)
    legacy_parser.add_argument("--chunk-size", type=int, default=10)
    legacy_parser.add_argument("--max-frames", type=int, default=None)
    legacy_parser.add_argument("--process-res", type=int, default=518)
    legacy_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'xpu', 'cpu', 'mps' (auto-detects best available)",
    )
    legacy_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        if hasattr(args, "input"):
            args.command = "legacy"
        else:
            parser.print_help()
            return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.verbose:
        for lib in ["matplotlib", "PIL", "urllib3", "timm", "huggingface_hub", "torch"]:
            logging.getLogger(lib).setLevel(logging.INFO)

    if args.command == "depth":
        from .depth import run_depth

        run_depth(args)

    elif args.command == "extract-depth":
        from .extract_depth import run_extract_depth

        run_extract_depth(args)

    elif args.command == "export" or args.command == "legacy":
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
            opacity_threshold=args.opacity_threshold if hasattr(args, "opacity_threshold") else 0.0,
            flip_y=getattr(args, "flip_y", False),
            masks_dir=getattr(args, "masks_dir", None),
            mask_first_frame=not getattr(args, "no_mask_first_frame", False),
            remove_black_splats=not getattr(args, "no_remove_black_splats", False),
            extract_audio=getattr(args, "extract_audio", False),
            resume_processing=getattr(args, "resume_processing", True),
        )

    elif args.command == "postprocess":
        postprocess_plys_to_freetimegs(
            args.input,
            args.output,
            fps=args.fps,
            max_match_distance=args.max_match_distance,
            ply_pattern=args.pattern,
            flip_y=getattr(args, "flip_y", False),
            format=args.format,
        )

    elif args.command == "images":
        export_images_to_gaussian_plys(
            args.input,
            args.output,
            mode=args.mode,
            fps=args.fps,
            model_id=args.model,
            image_pattern=args.pattern,
            max_frames=args.max_frames,
            device=args.device,
            process_res=args.process_res,
            opacity_threshold=args.opacity_threshold if hasattr(args, "opacity_threshold") else 0.0,
            masks_dir=args.masks_dir if hasattr(args, "masks_dir") else None,
            mask_first_frame=not getattr(args, "no_mask_first_frame", False),
            remove_black_splats=not getattr(args, "no_remove_black_splats", False),
            flip_y=getattr(args, "flip_y", False),
        )


if __name__ == "__main__":
    main()
