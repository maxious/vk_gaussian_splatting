"""Test PLY export from preprocessed depth sequence."""

import sys
from pathlib import Path

import cv2
import numpy as np

from offline.formats import (
    read_camera_poses,
    write_ply_header,
    write_ply_points,
)
from offline.export_ply import depth_to_points_with_pose, depth_to_points_simple

# Paths
output_dir = Path(r"D:\maxio\Documents\GitHub\vk_gaussian_splatting\_downloaded_resources\test_preprocess")
video_path = Path(r"D:\maxio\Documents\GitHub\vk_gaussian_splatting\_downloaded_resources\BigBuckBunny_320x180.mp4")

# Load camera poses
print("=== Loading camera poses ===")
poses = read_camera_poses(output_dir)
print(f"Loaded {len(poses)} poses")
for i, pose in enumerate(poses[:3]):
    print(f"  Pose {i}: position = {pose.extrinsics[:3, 3]}")

# Read VDZ frames manually (our test file)
print("\n=== Reading VDZ frames ===")
from offline.formats import VDZ_HEADER_STRUCT, VDZ_HEADER_SIZE
import zlib

vdz_path = output_dir / "test_depth.vdz"
frames = []

with open(vdz_path, "rb") as f:
    frame_idx = 0
    while True:
        header = f.read(VDZ_HEADER_SIZE)
        if len(header) < VDZ_HEADER_SIZE:
            break
        
        magic, version, dtype, ts, w, h, scale, bias, z_max = VDZ_HEADER_STRUCT.unpack(header)
        raw_size = w * h * 2
        
        if magic == b"VDZ2":
            # Read compressed data - we need to handle this carefully
            # Read a large chunk and decompress
            pos = f.tell()
            compressed = f.read(raw_size * 2)  # Read more than needed
            try:
                raw = zlib.decompress(compressed)[:raw_size]
                # Seek back to correct position
                obj = zlib.decompressobj()
                obj.decompress(compressed)
                unused = len(obj.unused_data)
                f.seek(pos + len(compressed) - unused)
            except:
                break
        else:
            raw = f.read(raw_size)
        
        if len(raw) < raw_size:
            break
        
        encoded = np.frombuffer(raw, dtype="<u2").reshape(h, w)
        depth = encoded.astype(np.float32) * scale + bias
        frames.append((frame_idx, ts, depth))
        frame_idx += 1

print(f"Read {len(frames)} depth frames")

# Open video for colors
cap = cv2.VideoCapture(str(video_path))

# Convert to point cloud
print("\n=== Converting to point cloud ===")
all_points = []
all_colors = []

sample_ratio = 0.1  # 10% of points
for frame_idx, ts, depth in frames:
    # Get video frame
    video_frame_idx = frame_idx * 24  # We extracted every second
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
    ret, frame = cap.read()
    if ret:
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to match depth
        color = cv2.resize(color, (depth.shape[1], depth.shape[0]))
    else:
        color = np.full((*depth.shape, 3), 128, dtype=np.uint8)
    
    # Use camera pose if available
    if frame_idx < len(poses):
        points, colors = depth_to_points_with_pose(
            depth, color, poses[frame_idx], sample_ratio=sample_ratio
        )
    else:
        points, colors = depth_to_points_simple(depth, color, sample_ratio=sample_ratio)
    
    all_points.append(points)
    all_colors.append(colors)
    print(f"  Frame {frame_idx}: {len(points)} points")

cap.release()

# Combine and write PLY
points = np.concatenate(all_points, axis=0).astype(np.float32)
colors = np.concatenate(all_colors, axis=0).astype(np.uint8)

print(f"\n=== Writing PLY ===")
print(f"Total points: {len(points)}")

ply_path = output_dir / "test_pointcloud.ply"
with open(ply_path, "wb") as f:
    write_ply_header(f, len(points))
    write_ply_points(f, points, colors)

print(f"Saved to: {ply_path}")
print(f"File size: {ply_path.stat().st_size / 1024 / 1024:.2f} MB")

print("\n=== Test PASSED ===")
