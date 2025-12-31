"""Simple end-to-end test of the offline preprocessing pipeline with a small chunk."""

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check CUDA first
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)
print(f"CUDA: {torch.cuda.get_device_name(0)}")

from offline.preprocess import (
    PreprocessConfig,
    VideoFrameExtractor,
    DA3StreamingProcessor,
    check_cuda_available,
)
from offline.formats import VdzFrame, write_vdz_frame, CameraPose, write_camera_poses

# Setup paths
video_path = Path(r"D:\maxio\Documents\GitHub\vk_gaussian_splatting\_downloaded_resources\BigBuckBunny_320x180.mp4")
output_dir = Path(r"D:\maxio\Documents\GitHub\vk_gaussian_splatting\_downloaded_resources\test_preprocess")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract just 10 frames for quick test
print("\n=== Extracting frames ===")
extractor = VideoFrameExtractor(video_path)
temp_dir = output_dir / "temp_frames"
temp_dir.mkdir(parents=True, exist_ok=True)

import cv2
frame_paths = []
timestamps_ms = []
for i in range(10):
    frame_idx = i * 24  # Every second
    frame, ts = extractor.get_frame(frame_idx)
    path = temp_dir / f"frame_{i:04d}.png"
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    frame_paths.append(path)
    timestamps_ms.append(ts)
    print(f"  Frame {frame_idx} -> {path.name}")
extractor.close()

# Process with DA3
print("\n=== Processing with DA3 ===")
config = PreprocessConfig(
    chunk_size=10,
    overlap=5,
    model_id="depth-anything/DA3-LARGE",
    process_res=504,
)
processor = DA3StreamingProcessor(config)
result = processor.process_chunk(frame_paths, 0, timestamps_ms)

print(f"\nChunk result:")
print(f"  depths shape: {result.depths.shape}")
print(f"  depths range: [{result.depths.min():.2f}, {result.depths.max():.2f}]")
print(f"  confidences shape: {result.confidences.shape}")
print(f"  intrinsics shape: {result.intrinsics.shape}")
print(f"  extrinsics shape: {result.extrinsics.shape}")

# Write VDZ frames
print("\n=== Writing VDZ frames ===")
vdz_path = output_dir / "test_depth.vdz"
with open(vdz_path, "wb") as f:
    for i in range(len(result.depths)):
        depth = result.depths[i]
        z_min = float(depth.min())
        z_max = float(depth.max())
        frame = VdzFrame(
            timestamp_ms=result.timestamps_ms[i],
            width=depth.shape[1],
            height=depth.shape[0],
            depth=depth.astype('float32'),
            z_min=z_min,
            z_max=z_max,
        )
        write_vdz_frame(f, frame, compress=True)
print(f"Wrote {len(result.depths)} VDZ frames to {vdz_path}")
print(f"File size: {vdz_path.stat().st_size / 1024:.1f} KB")

# Write camera poses
print("\n=== Writing camera poses ===")
poses = []
for i in range(len(result.depths)):
    # DA3 outputs (N, 3, 4) extrinsics - need to convert to 4x4 c2w
    ext_34 = result.extrinsics[i]  # (3, 4)
    # This appears to be w2c format, so invert to get c2w
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = ext_34
    c2w = np.linalg.inv(w2c)
    
    poses.append(CameraPose(
        frame_idx=i,
        timestamp_ms=result.timestamps_ms[i],
        intrinsics=result.intrinsics[i],
        extrinsics=c2w,
    ))

write_camera_poses(output_dir, poses)
print(f"Wrote {len(poses)} camera poses")

# Save depth visualization
print("\n=== Saving visualizations ===")
for i in range(min(3, len(result.depths))):
    depth = result.depths[i]
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_vis = (depth_norm * 255).astype('uint8')
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    vis_path = output_dir / f"depth_vis_{i}.png"
    cv2.imwrite(str(vis_path), depth_colored)
    print(f"  Saved {vis_path.name}")

print("\n=== Test PASSED ===")
print(f"Output directory: {output_dir}")
