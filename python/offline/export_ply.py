"""Export preprocessed depth sequence to PLY point cloud.

Fuses per-frame depth maps into a globally consistent point cloud,
optionally using camera poses for proper 3D reconstruction.

Usage:
    python -m offline.export_ply --input ./preprocessed/ --output scene.ply
    
    # Or with installed package:
    vkgs-export-ply --input ./preprocessed/ --output scene.ply
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import zlib
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .formats import (
    VDZ_HEADER_STRUCT,
    VDZ_HEADER_SIZE,
    read_camera_poses,
    write_ply_header,
    write_ply_points,
    CameraPose,
)

logger = logging.getLogger(__name__)


def read_vdz_frames(vdz_path: Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """Read VDZ frames from sequence file.
    
    Yields:
        (frame_idx, timestamp_ms, depth_map)
    """
    with open(vdz_path, "rb") as f:
        frame_idx = 0
        
        while True:
            header_bytes = f.read(VDZ_HEADER_SIZE)
            if len(header_bytes) < VDZ_HEADER_SIZE:
                break
            
            (magic, version, dtype, timestamp_ms, width, height,
             scale, bias, z_max) = VDZ_HEADER_STRUCT.unpack(header_bytes)
            
            # Calculate payload size
            raw_size = width * height * 2  # uint16
            
            if magic == b"VDZ2":
                # Compressed - read until we have enough decompressed data
                # This is tricky because zlib doesn't have a fixed size
                # Read in chunks and decompress
                compressed_data = b""
                decompressor = zlib.decompressobj()
                decompressed = b""
                
                while len(decompressed) < raw_size:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    try:
                        decompressed += decompressor.decompress(chunk)
                    except zlib.error:
                        # Might have read too much, seek back
                        break
                
                # Handle any leftover
                remaining = decompressor.unused_data
                if remaining:
                    f.seek(-len(remaining), 1)
                
                raw_bytes = decompressed[:raw_size]
            else:
                # Uncompressed
                raw_bytes = f.read(raw_size)
            
            if len(raw_bytes) < raw_size:
                break
            
            # Decode depth
            encoded = np.frombuffer(raw_bytes, dtype="<u2").reshape(height, width)
            depth = encoded.astype(np.float32) * scale + bias
            
            yield frame_idx, timestamp_ms, depth
            frame_idx += 1


def depth_to_points_with_pose(
    depth: np.ndarray,
    color: np.ndarray,
    pose: CameraPose,
    conf_threshold: float = 0.0,
    sample_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth map to world-space points using camera pose.
    
    Returns:
        (points, colors) - Nx3 arrays
    """
    H, W = depth.shape
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Get intrinsics
    K = pose.intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Unproject to camera space
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors_flat = color.reshape(-1, 3)
    
    # Filter invalid depths
    valid_mask = (depth.flatten() > 0.01) & (depth.flatten() < 100.0)
    
    # Random sampling
    if sample_ratio < 1.0:
        n_valid = np.sum(valid_mask)
        n_samples = int(n_valid * sample_ratio)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > n_samples:
            selected = np.random.choice(valid_indices, n_samples, replace=False)
            valid_mask = np.zeros_like(valid_mask)
            valid_mask[selected] = True
    
    points_cam = points_cam[valid_mask]
    colors_flat = colors_flat[valid_mask]
    
    # Transform to world space using c2w
    c2w = pose.extrinsics
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    
    points_world = points_cam @ R.T + t
    
    return points_world, colors_flat


def depth_to_points_simple(
    depth: np.ndarray,
    color: np.ndarray,
    fov_deg: float = 60.0,
    sample_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth map to points using assumed FOV (no pose).
    
    Points are in camera space (useful for single-frame visualization).
    """
    H, W = depth.shape
    
    # Estimate intrinsics from FOV
    fov_rad = np.radians(fov_deg)
    fx = fy = W / (2 * np.tan(fov_rad / 2))
    cx, cy = W / 2, H / 2
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3)
    
    # Filter and sample
    valid_mask = (depth.flatten() > 0.01) & (depth.flatten() < 100.0)
    
    if sample_ratio < 1.0:
        n_valid = np.sum(valid_mask)
        n_samples = int(n_valid * sample_ratio)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > n_samples:
            selected = np.random.choice(valid_indices, n_samples, replace=False)
            valid_mask = np.zeros_like(valid_mask)
            valid_mask[selected] = True
    
    return points[valid_mask], colors[valid_mask]


def export_ply(
    preprocessed_dir: Path,
    output_path: Path,
    video_path: Optional[Path] = None,
    sample_ratio: float = 0.01,
    frame_skip: int = 5,
    use_poses: bool = True,
) -> None:
    """Export preprocessed depth sequence to PLY.
    
    Args:
        preprocessed_dir: Directory containing depth_sequence.vdz and camera_poses.txt
        output_path: Output PLY file path
        video_path: Original video for RGB colors (optional, uses metadata if not provided)
        sample_ratio: Fraction of points to keep per frame
        frame_skip: Only process every Nth frame
        use_poses: Use camera poses for world-space reconstruction
    """
    vdz_path = preprocessed_dir / "depth_sequence.vdz"
    meta_path = preprocessed_dir / "metadata.json"
    
    if not vdz_path.exists():
        raise FileNotFoundError(f"VDZ sequence not found: {vdz_path}")
    
    # Load metadata
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if video_path is None:
            video_path = Path(meta.get("video_path", ""))
    
    # Load camera poses if available
    poses: list[CameraPose] = []
    if use_poses:
        poses_path = preprocessed_dir / "camera_poses.txt"
        if poses_path.exists():
            poses = read_camera_poses(preprocessed_dir)
            logger.info(f"Loaded {len(poses)} camera poses")
        else:
            logger.warning("No camera poses found, using simple projection")
            use_poses = False
    
    # Open video for colors
    cap = None
    if video_path and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        logger.info(f"Using video for colors: {video_path}")
    
    # Collect points
    all_points = []
    all_colors = []
    
    logger.info("Processing depth frames...")
    
    for frame_idx, timestamp_ms, depth in read_vdz_frames(vdz_path):
        if frame_idx % frame_skip != 0:
            continue
        
        # Get color frame
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to match depth if needed
                if color.shape[:2] != depth.shape:
                    color = cv2.resize(color, (depth.shape[1], depth.shape[0]))
            else:
                color = np.full((*depth.shape, 3), 128, dtype=np.uint8)
        else:
            # Gray fallback
            color = np.full((*depth.shape, 3), 128, dtype=np.uint8)
        
        # Convert to points
        if use_poses and frame_idx < len(poses):
            points, colors = depth_to_points_with_pose(
                depth, color, poses[frame_idx], sample_ratio=sample_ratio
            )
        else:
            points, colors = depth_to_points_simple(
                depth, color, sample_ratio=sample_ratio
            )
        
        all_points.append(points)
        all_colors.append(colors)
        
        if frame_idx % 50 == 0:
            logger.info(f"Processed frame {frame_idx}")
    
    if cap is not None:
        cap.release()
    
    # Combine and write
    if not all_points:
        logger.error("No points extracted!")
        return
    
    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
    
    logger.info(f"Writing {len(points)} points to {output_path}")
    
    with open(output_path, "wb") as f:
        write_ply_header(f, len(points))
        write_ply_points(f, points, colors)
    
    logger.info("Export complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Export preprocessed depth sequence to PLY point cloud"
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                       help="Preprocessed directory containing depth_sequence.vdz")
    parser.add_argument("--output", "-o", type=Path, required=True,
                       help="Output PLY file")
    parser.add_argument("--video", type=Path, default=None,
                       help="Original video for RGB colors")
    parser.add_argument("--sample-ratio", type=float, default=0.01,
                       help="Fraction of points per frame (default: 0.01)")
    parser.add_argument("--frame-skip", type=int, default=5,
                       help="Process every Nth frame (default: 5)")
    parser.add_argument("--no-poses", action="store_true",
                       help="Don't use camera poses")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    export_ply(
        args.input,
        args.output,
        video_path=args.video,
        sample_ratio=args.sample_ratio,
        frame_skip=args.frame_skip,
        use_poses=not args.no_poses,
    )


if __name__ == "__main__":
    main()
