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

    trellis2_xpu_parser = subparsers.add_parser(
        "trellis2-xpu", help="Stage 1: Run TRELLIS.2 on XPU and save mesh data to disk"
    )
    trellis2_xpu_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input image or directory of images"
    )
    trellis2_xpu_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for .pt mesh files"
    )
    trellis2_xpu_parser.add_argument(
        "--model",
        type=str,
        default="microsoft/TRELLIS.2-4B",
        help="TRELLIS.2 model ID (e.g., 'microsoft/TRELLIS.2-4B')",
    )
    trellis2_xpu_parser.add_argument(
        "--xpu-device", type=str, default="xpu:0", help="XPU device (e.g., 'xpu:0', 'xpu:1')"
    )
    trellis2_xpu_parser.add_argument(
        "--pattern", type=str, default="*.{jpg,jpeg,png,webp}", help="Glob pattern for image files"
    )
    trellis2_xpu_parser.add_argument(
        "--pipeline-type",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        default="1024_cascade",
        help="TRELLIS.2 pipeline type",
    )
    trellis2_xpu_parser.add_argument(
        "--max-num-tokens", type=int, default=49152, help="Maximum tokens for cascade pipeline"
    )
    trellis2_xpu_parser.add_argument(
        "--num-samples", type=int, default=1, help="Number of samples per image"
    )
    trellis2_xpu_parser.add_argument(
        "--multi-xpu",
        action="store_true",
        help="Enable Multi-XPU inference using MultiXPUSpconv (requires >1 XPU)",
    )
    trellis2_xpu_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    trellis2_xpu_parser.add_argument("-v", "--verbose", action="store_true")

    trellis2_cuda_parser = subparsers.add_parser(
        "trellis2-cuda", help="Stage 2: Load saved mesh data and export to GLB/VXZ using CUDA"
    )
    trellis2_cuda_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory or file containing .pt mesh files",
    )
    trellis2_cuda_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for GLB/VXZ files"
    )
    trellis2_cuda_parser.add_argument(
        "--format",
        choices=["glb", "vxz"],
        default="glb",
        help="Output format: 'glb' (mesh with texture), 'vxz' (o-voxel format)",
    )
    trellis2_cuda_parser.add_argument(
        "--cuda-device", type=str, default="cuda:0", help="CUDA device (e.g., 'cuda:0')"
    )
    trellis2_cuda_parser.add_argument(
        "--model",
        type=str,
        default="microsoft/TRELLIS.2-4B",
        help="TRELLIS.2 model ID (for decoding latent data)",
    )
    trellis2_cuda_parser.add_argument(
        "--pattern", type=str, default="*.pt", help="Glob pattern for .pt mesh files"
    )
    trellis2_cuda_parser.add_argument(
        "--decimation-target",
        type=int,
        default=1000000,
        help="Target face count for mesh simplification",
    )
    trellis2_cuda_parser.add_argument(
        "--texture-size", type=int, default=4096, help="Texture resolution for GLB export"
    )
    trellis2_cuda_parser.add_argument(
        "--chunk-size", type=int, default=256, help="Chunk size for VXZ export"
    )
    trellis2_cuda_parser.add_argument(
        "--compression",
        choices=["lzma", "zlib", "none"],
        default="lzma",
        help="Compression for VXZ export",
    )
    trellis2_cuda_parser.add_argument(
        "--compression-level", type=int, default=9, help="Compression level for VXZ export (0-9)"
    )
    trellis2_cuda_parser.add_argument("-v", "--verbose", action="store_true")

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
        # Silence noisy libraries even in verbose mode
        for lib in ["matplotlib", "PIL", "urllib3", "timm", "huggingface_hub", "torch"]:
            logging.getLogger(lib).setLevel(logging.INFO)

    if args.command == "export" or args.command == "legacy":
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
    elif args.command == "trellis2-xpu":
        from .processors.trellis2_xpu import Trellis2XPUProcessor
        from .vkgs_trellis2.modules.sparse.conv import config as conv_config

        if args.multi_xpu:
            logger.info("Enabling Multi-XPU inference")
            conv_config.FLEX_GEMM_USE_MULTI_XPU = True

        processor = Trellis2XPUProcessor(
            model_id=args.model,
            xpu_device=args.xpu_device,
        )

        if args.input.is_file():
            mesh_path = args.output / f"{args.input.stem}.pt"
            processor.infer_and_save(
                args.input,
                mesh_path,
                pipeline_type=args.pipeline_type,
                num_samples=args.num_samples,
                seed=args.seed,
                max_num_tokens=args.max_num_tokens,
            )
        else:
            image_paths = sorted(args.input.glob(args.pattern))
            args.output.mkdir(parents=True, exist_ok=True)

            for i, image_path in enumerate(image_paths, 1):
                logger.info(f"Processing {i}/{len(image_paths)}: {image_path}")
                mesh_path = args.output / f"{image_path.stem}.pt"
                processor.infer_and_save(
                    image_path,
                    mesh_path,
                    pipeline_type=args.pipeline_type,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    max_num_tokens=args.max_num_tokens,
                )
    elif args.command == "trellis2-cuda":
        from .processors.trellis2_cuda import Trellis2CUDAProcessor

        processor = Trellis2CUDAProcessor(
            cuda_device=args.cuda_device,
            model_id=args.model,
        )

        if args.input.is_file():
            output_path = (
                args.output
                if args.output.suffix == f".{args.format}"
                else args.output / args.input.with_suffix(f".{args.format}").name
            )
            if args.format == "glb":
                processor.load_and_export_glb(
                    args.input,
                    output_path,
                    decimation_target=args.decimation_target,
                    texture_size=args.texture_size,
                )
            else:
                processor.load_and_export_vxz(
                    args.input,
                    output_path,
                    chunk_size=args.chunk_size,
                    compression=args.compression,
                    compression_level=args.compression_level,
                )
        else:
            mesh_paths = sorted(args.input.glob(args.pattern))
            args.output.mkdir(parents=True, exist_ok=True)

            for i, mesh_path in enumerate(mesh_paths, 1):
                logger.info(f"Exporting {i}/{len(mesh_paths)}: {mesh_path}")
                output_path = args.output / f"{mesh_path.stem}.{args.format}"

                if args.format == "glb":
                    processor.load_and_export_glb(
                        mesh_path,
                        output_path,
                        decimation_target=args.decimation_target,
                        texture_size=args.texture_size,
                    )
                else:
                    processor.load_and_export_vxz(
                        mesh_path,
                        output_path,
                        chunk_size=args.chunk_size,
                        compression=args.compression,
                        compression_level=args.compression_level,
                    )


if __name__ == "__main__":
    main()
