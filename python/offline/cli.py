"""Command-line interface for Gaussian Splatting export."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .exporters import (
    export_images_to_gaussian_plys,
    export_video_to_gaussian_plys,
    postprocess_plys_to_freetimegs,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export video to Gaussian Splatting PLY files using InfiniDepth/MoGe/SHARP/UniSHARP/TripoSplat/SAM 3D Body/Hybrid"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Depth extraction subcommand
    depth_parser = subparsers.add_parser(
        "depth", help="Extract depth maps from video using InfiniDepth (multi-XPU optimized)"
    )
    depth_parser.add_argument("--input", "-i", type=Path, required=True, help="Input video file")
    depth_parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    depth_parser.add_argument(
        "--model",
        type=str,
        default="InfiniDepth",
        help="Model ID (default: InfiniDepth)",
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
        default=768,
        help="Processing resolution (default: 768)",
    )
    depth_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device for inference (default: 4)",
    )
    depth_parser.add_argument("-v", "--verbose", action="store_true")

    masks_parser = subparsers.add_parser(
        "masks", help="Generate masks for images using BEN2 or BiRefNet"
    )
    masks_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input image directory"
    )
    masks_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output mask directory"
    )
    masks_parser.add_argument("--device", type=str, default="cuda")
    masks_parser.add_argument(
        "--pattern", type=str, default="*.jpg", help="Glob pattern for input images"
    )
    masks_parser.add_argument(
        "--model",
        type=str,
        default="ben2",
        choices=["ben2", "birefnet", "birefnet-lite"],
        help="Model to use for mask generation",
    )
    masks_parser.add_argument(
        "--refine",
        action="store_true",
        help="Enable foreground refinement (BEN2 only)",
    )
    masks_parser.add_argument("-v", "--verbose", action="store_true")

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
        choices=["frames", "freetimegs", "freetimegs-delta", "freetimegs-delta-int8"],
        default="frames",
        help="Export mode: 'frames' for per-frame PLYs, 'freetimegs' for temporal PLY, "
        "'freetimegs-delta' for compressed temporal (Int16), 'freetimegs-delta-int8' for high compression (Int8)",
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
        default="InfiniDepth",
        help="Model: InfiniDepth/MoGe/SHARP/UniSHARP/TripoSplat/MotionCrafter for depth, 'sam3dbody' for humans, 'hybrid' for SAM 3D Body + depth "
        "(default: InfiniDepth). "
        "SAM 3D Body: sam3dbody:facebook/sam-3d-body-vith. "
        "Hybrid: hybrid:human+depth (e.g., hybrid:facebook/sam-3d-body-vith+InfiniDepth). "
        "TripoSplat: triposplat[:N[:steps=S][:cfg=C]] (e.g., triposplat:131072:steps=10). "
        "MotionCrafter: motioncrafter:path/to/config.yaml or motioncrafter:path/to/checkpoint.ckpt",
    )
    export_parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame")
    export_parser.add_argument(
        "--chunk-size", type=int, default=10, help="Frames per processing chunk"
    )
    export_parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process"
    )
    export_parser.add_argument(
        "--process-res",
        type=int,
        default=1024,
        help="Processing resolution (higher = more Gaussians, but slower)",
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
        "--enable-skyseg",
        action="store_true",
        help="Enable sky segmentation to remove sky regions (InfiniDepth only)",
    )
    export_parser.add_argument(
        "--color-correct",
        action="store_true",
        help="Apply trajectory-consensus per-frame affine color correction to reduce temporal flicker (mined from FreeTimeGS++)",
    )
    export_parser.add_argument(
        "--temporal-gating",
        action="store_true",
        help="Keep persistent/static Gaussians visible across the whole clip instead of fading at the ends (FreeTimeGS++ gated marginalization approximation)",
    )
    export_parser.add_argument(
        "--flow-velocity-weight",
        type=float,
        default=0.0,
        help="Blend weight [0,1] for the 3D scene-flow velocity prior when available (mined from FreeTimeGS++ velocity distillation; 0 disables)",
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
    postprocess_parser.add_argument(
        "--color-correct",
        action="store_true",
        help="Apply trajectory-consensus per-frame affine color correction to reduce temporal flicker (mined from FreeTimeGS++)",
    )
    postprocess_parser.add_argument(
        "--temporal-gating",
        action="store_true",
        help="Keep persistent/static Gaussians visible across the whole clip instead of fading at the ends (FreeTimeGS++ gated marginalization approximation)",
    )
    postprocess_parser.add_argument("-v", "--verbose", action="store_true")

    zipsplat_parser = subparsers.add_parser(
        "zipsplat",
        help="Convert a 4DAnyone multi-view result into per-timestamp ZipSplat PLYs",
    )
    zipsplat_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="4DAnyone fdanyone/<clip> result directory",
    )
    zipsplat_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for per-timestamp PLY files",
    )
    zipsplat_parser.add_argument(
        "--weights", default="zipsplat", help="ZipSplat weights name or path"
    )
    zipsplat_parser.add_argument("--frame-skip", type=int, default=1)
    zipsplat_parser.add_argument("--max-frames", type=int, default=None)
    zipsplat_parser.add_argument(
        "--compression-ratio",
        type=float,
        default=None,
        help="Optional ZipSplat query-sampling ratio in (0, 1]",
    )
    zipsplat_parser.add_argument("-v", "--verbose", action="store_true")

    omnimatte_parser = subparsers.add_parser(
        "omnimatte", help="OmnimatteZero: Background Generation & Object Extraction"
    )
    omnimatte_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input video file"
    )
    omnimatte_parser.add_argument(
        "--mask", "-m", type=Path, required=True, help="Input mask video (total mask)"
    )
    omnimatte_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory"
    )
    omnimatte_parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    omnimatte_parser.add_argument(
        "--extract-foreground",
        action="store_true",
        help="Extract foreground layer after background generation",
    )
    omnimatte_parser.add_argument("--width", type=int, default=768)
    omnimatte_parser.add_argument("--height", type=int, default=512)
    omnimatte_parser.add_argument("--device", type=str, default="cuda")
    omnimatte_parser.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help="Path to GGUF quantized model file (from calcuis/ltxv-gguf). "
        "If provided, uses GGUF instead of Diffusers repo (lower VRAM).",
    )

    images_parser = subparsers.add_parser(
        "images", help="Process images with InfiniDepth and export to Gaussian PLY files"
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
        choices=["frames", "freetimegs", "freetimegs-delta", "freetimegs-delta-int8"],
        default="frames",
        help="Export mode: 'frames' for per-frame PLYs, 'freetimegs' for temporal PLY, "
        "'freetimegs-delta' for compressed temporal (Int16), 'freetimegs-delta-int8' for high compression (Int8)",
    )
    images_parser.add_argument(
        "--format",
        choices=["ply", "sog", "4dv"],
        default="ply",
        help="Output format: 'ply' (standard), 'sog' (compressed static/dynamic), '4dv' (compressed dynamic)",
    )
    images_parser.add_argument(
        "--fps", type=float, default=30.0, help="Frame rate for temporal normalization"
    )
    images_parser.add_argument(
        "--model",
        type=str,
        default="InfiniDepth",
        help="Model ID. Options: 'InfiniDepth', 'sharp', 'unisharp', 'triposplat[:N[:steps=S][:cfg=C]]', 'motioncrafter', "
        "'tttlrm' (full model) or 'tttlrm-ar' (autoregressive, lower memory), "
        "optionally 'tttlrm:/path/to/checkpoint.pt' (multi-view LRM, input must be JSON manifests)",
    )
    images_parser.add_argument(
        "--pattern", type=str, default="*.jpg", help="Glob pattern for image files"
    )
    images_parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process"
    )
    images_parser.add_argument(
        "--process-res",
        type=int,
        default=1024,
        help="Processing resolution (higher = more Gaussians, but slower)",
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
        "--enable-skyseg",
        action="store_true",
        help="Enable sky segmentation to remove sky regions (InfiniDepth only)",
    )
    images_parser.add_argument(
        "--color-correct",
        action="store_true",
        help="Apply trajectory-consensus per-frame affine color correction to reduce temporal flicker (mined from FreeTimeGS++)",
    )
    images_parser.add_argument(
        "--temporal-gating",
        action="store_true",
        help="Keep persistent/static Gaussians visible across the whole clip instead of fading at the ends (FreeTimeGS++ gated marginalization approximation)",
    )
    images_parser.add_argument(
        "--flow-velocity-weight",
        type=float,
        default=0.0,
        help="Blend weight [0,1] for the 3D scene-flow velocity prior when available (mined from FreeTimeGS++ velocity distillation; 0 disables)",
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
    images_parser.add_argument(
        "--debug-output-dir",
        type=Path,
        default=None,
        help="Directory for debug outputs (PLY point cloud, GLB mesh). Only for MoGe model.",
    )
    images_parser.add_argument(
        "--refine-boundaries",
        action="store_true",
        help="Apply boundary depth refinement to reduce halo artifacts (MoGe only)",
    )
    images_parser.add_argument(
        "--boundary-min-angle",
        type=float,
        default=3.0,
        help="Minimum angle (degrees) for boundary detection (default: 3.0)",
    )
    images_parser.add_argument("-v", "--verbose", action="store_true")

    legacy_parser = subparsers.add_parser("legacy", help="Legacy CLI (deprecated)")
    legacy_parser.add_argument("--input", "-i", type=Path, required=True)
    legacy_parser.add_argument("--output", "-o", type=Path, required=True)
    legacy_parser.add_argument("--mode", choices=["frames", "freetimegs"], default="frames")
    legacy_parser.add_argument("--model", type=str, default="InfiniDepth")
    legacy_parser.add_argument("--frame-skip", type=int, default=5)
    legacy_parser.add_argument("--chunk-size", type=int, default=10)
    legacy_parser.add_argument("--max-frames", type=int, default=None)
    legacy_parser.add_argument("--process-res", type=int, default=768)
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

    elif args.command == "masks":
        from .masking import generate_masks

        generate_masks(
            args.input,
            args.output,
            device=args.device,
            pattern=args.pattern,
            refine=args.refine,
            model=args.model,
        )

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
            enable_skyseg=getattr(args, "enable_skyseg", False),
            color_correction=getattr(args, "color_correct", False),
            temporal_gating=getattr(args, "temporal_gating", False),
            flow_prior_weight=getattr(args, "flow_velocity_weight", 0.0),
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
            color_correction=getattr(args, "color_correct", False),
            temporal_gating=getattr(args, "temporal_gating", False),
        )

    elif args.command == "zipsplat":
        from .processors.zipsplat import export_4danyone_with_zipsplat

        export_4danyone_with_zipsplat(
            args.input,
            args.output,
            weights=args.weights,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
            compression_ratio=args.compression_ratio,
        )

    elif args.command == "omnimatte":
        from .processors.omnimatte import OmnimatteProcessor

        processor = OmnimatteProcessor(
            device=args.device, gguf_path=str(args.gguf) if args.gguf else None
        )
        output_dir = args.output
        output_dir.mkdir(parents=True, exist_ok=True)

        bg_output = output_dir / "background.mp4"
        processor.remove_object(
            args.input,
            args.mask,
            bg_output,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
        )

        if args.extract_foreground:
            fg_output = output_dir / "foreground.mp4"
            processor.extract_foreground(
                args.input,
                bg_output,
                fg_output,
                height=args.height,
                width=args.width,
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
            debug_output_dir=getattr(args, "debug_output_dir", None),
            refine_boundaries=getattr(args, "refine_boundaries", False),
            boundary_min_angle=getattr(args, "boundary_min_angle", 3.0),
            enable_skyseg=getattr(args, "enable_skyseg", False),
            color_correction=getattr(args, "color_correct", False),
            temporal_gating=getattr(args, "temporal_gating", False),
            flow_prior_weight=getattr(args, "flow_velocity_weight", 0.0),
        )


if __name__ == "__main__":
    main()
