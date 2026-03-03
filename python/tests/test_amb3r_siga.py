"""Run AMB3R + COLMAP export on SIGA2025VVC dataset.

Usage:
    cd python
    uv run --extra cuda --extra amb3r python tests/test_amb3r_siga.py \
        --root /media/maxious/Data/SIGA2025VVC-Dataset/compression/test/006_1_seq1/ \
        --output /tmp/amb3r_siga_output \
        --num-cameras 8 \
        --frame-idx 0 \
        --process-res 896 490
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_siga_frames(
    dataset_root: Path,
    frame_idx: int = 0,
    num_cameras: int | None = None,
    camera_ids: list[str] | None = None,
    use_train_split: bool = False,
    no_masks: bool = False,
) -> tuple[list[Path], list[Path] | None, list[str]]:
    """Load frame paths from SIGA2025VVC dataset.

    Returns (image_paths, mask_paths_or_None, camera_ids_used).
    """
    from offline.datasets.siga2025vvc_to_lrm_json import _parse_opencv_yaml

    prefix = "train_" if use_train_split else ""
    extri_file = dataset_root / f"{prefix}extri.yml"
    logger.info(f"Loading camera list from {extri_file.name}")
    extri = _parse_opencv_yaml(extri_file)
    cam_ids = extri.get("names", [])

    if camera_ids is not None:
        cam_ids = [c for c in cam_ids if c in camera_ids]
    if num_cameras is not None and num_cameras < len(cam_ids):
        # Pick evenly spaced cameras for better coverage
        indices = np.linspace(0, len(cam_ids) - 1, num_cameras, dtype=int)
        cam_ids = [cam_ids[i] for i in indices]

    image_paths = []
    mask_paths = []
    used_ids = []
    has_masks = (dataset_root / "masks").is_dir() and not no_masks

    for cam_id in cam_ids:
        img_path = dataset_root / "images" / cam_id / f"{frame_idx:06d}.jpg"
        if img_path.exists():
            image_paths.append(img_path)
            used_ids.append(cam_id)
            if has_masks:
                mask_path = dataset_root / "masks" / cam_id / f"{frame_idx:06d}.png"
                mask_paths.append(mask_path if mask_path.exists() else None)
        else:
            logger.warning(f"Image not found: {img_path}")

    # Only return masks if all frames have them
    if has_masks and all(m is not None for m in mask_paths):
        return image_paths, mask_paths, used_ids
    else:
        if has_masks and mask_paths:
            missing = sum(1 for m in mask_paths if m is None)
            logger.warning(f"{missing}/{len(mask_paths)} masks missing, disabling masking")
        return image_paths, None, used_ids


def main():
    parser = argparse.ArgumentParser(description="Run AMB3R on SIGA2025VVC dataset")
    parser.add_argument(
        "--root", type=Path, required=True,
        help="SIGA2025VVC dataset root (e.g. .../006_1_seq1/)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/amb3r_siga_output"),
        help="Output directory",
    )
    parser.add_argument(
        "--frame-idx", type=int, default=0,
        help="Temporal frame index to process",
    )
    parser.add_argument(
        "--num-cameras", type=int, default=8,
        help="Number of cameras to use (evenly spaced)",
    )
    parser.add_argument(
        "--cameras", type=str, nargs="*", default=None,
        help="Specific camera IDs (overrides --num-cameras)",
    )
    parser.add_argument(
        "--use-train-split", action="store_true",
        help="Use train_extri.yml/train_intri.yml camera list instead of extri.yml",
    )
    parser.add_argument(
        "--no-masks", action="store_true",
        help="Disable foreground masking",
    )
    parser.add_argument(
        "--process-res", type=int, nargs=2, default=[896, 490],
        metavar=("W", "H"),
        help="Processing resolution (W H), must be divisible by 14",
    )
    parser.add_argument(
        "--window-size", type=int, default=8,
        help="AMB3R window size (frames per batch)",
    )
    parser.add_argument(
        "--max-points-per-frame", type=int, default=10000,
        help="Max 3D points per frame for COLMAP export",
    )
    parser.add_argument(
        "--conf-thresh", type=float, default=30.0,
        help="Confidence threshold percentile",
    )
    parser.add_argument(
        "--ckpt-path", type=str, default=None,
        help="Path to AMB3R checkpoint (skip HuggingFace download)",
    )
    parser.add_argument(
        "--skip-fastgs", action="store_true",
        help="Skip FastGS training, export point cloud PLY directly",
    )
    args = parser.parse_args()

    # Validate resolution
    w, h = args.process_res
    assert w % 14 == 0, f"Width {w} must be divisible by 14"
    assert h % 14 == 0, f"Height {h} must be divisible by 14"

    # Load frames
    logger.info(f"Loading SIGA dataset from {args.root}")
    image_paths, mask_paths, cam_ids = load_siga_frames(
        args.root,
        frame_idx=args.frame_idx,
        num_cameras=args.num_cameras,
        camera_ids=args.cameras,
        use_train_split=args.use_train_split,
        no_masks=args.no_masks,
    )
    logger.info(f"Using {len(image_paths)} cameras: {cam_ids}")
    logger.info(f"Processing resolution: {w}x{h}")
    if mask_paths:
        logger.info(f"Using {len(mask_paths)} foreground masks")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run AMB3R inference
    from offline.processors.amb3r import AMB3RFastGSProcessor

    processor = AMB3RFastGSProcessor(
        device="cuda",
        process_res=(w, h),
        window_size=args.window_size,
        conf_thresh_percentile=args.conf_thresh,
        max_points_per_frame=args.max_points_per_frame,
        ckpt_path=args.ckpt_path,
    )

    logger.info("Running AMB3R inference...")
    t0 = time.time()
    amb3r_output = processor._run_amb3r_inference(image_paths)
    t1 = time.time()
    logger.info(f"AMB3R inference took {t1 - t0:.1f}s")

    # Log statistics
    wp = amb3r_output["world_points"]
    conf = amb3r_output["confidence"]
    logger.info(f"World points shape: {wp.shape}, range: [{wp.min():.3f}, {wp.max():.3f}]")
    logger.info(f"Confidence shape: {conf.shape}, range: [{conf.min():.3f}, {conf.max():.3f}], mean: {conf.mean():.3f}")

    # Export COLMAP
    colmap_dir = args.output / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting COLMAP reconstruction...")
    from offline.colmap_export import export_amb3r_to_colmap

    export_amb3r_to_colmap(
        world_points=amb3r_output["world_points"],
        confidence=amb3r_output["confidence"],
        extrinsics_w2c=amb3r_output["extrinsics_w2c"],
        intrinsics=amb3r_output["intrinsics"],
        image_paths=[str(p) for p in image_paths],
        export_dir=str(colmap_dir),
        conf_thresh_percentile=args.conf_thresh,
        max_points_per_frame=args.max_points_per_frame,
        mask_paths=[str(p) for p in mask_paths] if mask_paths else None,
    )

    # Create PLY from point cloud
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(colmap_dir))
    logger.info(
        f"COLMAP: {len(reconstruction.images)} images, "
        f"{len(reconstruction.points3D)} 3D points, "
        f"{len(reconstruction.cameras)} cameras"
    )

    pts = np.array([p.xyz for p in reconstruction.points3D.values()], dtype=np.float32)
    colors_uint8 = np.array([p.color for p in reconstruction.points3D.values()], dtype=np.uint8)
    num_points = len(pts)

    logger.info(f"Creating Gaussian PLY from {num_points} points")

    # Create simple Gaussians from the point cloud
    means = pts
    scales = np.full((num_points, 3), -5.0, dtype=np.float32)
    rotations = np.zeros((num_points, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    colors_float = colors_uint8.astype(np.float32) / 255.0
    sh_dc = (colors_float - 0.5) / 0.28209479177387814
    opacities = np.full((num_points, 1), 5.0, dtype=np.float32)

    from offline.ply_io import write_static_gaussian_ply

    ply_path = args.output / "amb3r_siga.ply"
    write_static_gaussian_ply(ply_path, means, scales, rotations, sh_dc, opacities)

    logger.info(f"PLY written: {ply_path} ({ply_path.stat().st_size / 1024:.1f} KB, {num_points} splats)")
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
