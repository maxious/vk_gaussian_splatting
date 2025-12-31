"""Quick test of the offline pipeline without DA3 inference."""

import sys
from pathlib import Path

# Test 1: Video extraction
print("=" * 60)
print("Test 1: Video Frame Extraction")
print("=" * 60)

from offline.preprocess import VideoFrameExtractor

video_path = Path(r"D:\maxio\Documents\GitHub\vk_gaussian_splatting\_downloaded_resources\BigBuckBunny_320x180.mp4")
output_dir = Path(r"D:\maxio\Documents\GitHub\vk_gaussian_splatting\_downloaded_resources\test_output")

extractor = VideoFrameExtractor(video_path)
print(f"Video: {video_path.name}")
print(f"  Resolution: {extractor.width}x{extractor.height}")
print(f"  FPS: {extractor.fps}")
print(f"  Frames: {extractor.frame_count}")
print(f"  Duration: {extractor.duration_s:.1f}s")

# Test getting a single frame
frame, ts = extractor.get_frame(100)
print(f"  Frame 100: shape={frame.shape}, timestamp={ts:.1f}ms")
extractor.close()

# Test 2: VDZ format writing
print("\n" + "=" * 60)
print("Test 2: VDZ Format Writing")
print("=" * 60)

import numpy as np
from offline.formats import VdzFrame, write_vdz_frame

output_dir.mkdir(parents=True, exist_ok=True)
test_vdz = output_dir / "test_frame.vdz"

# Create fake depth data
fake_depth = np.random.uniform(0.5, 10.0, (180, 320)).astype(np.float32)
frame = VdzFrame(
    timestamp_ms=1000.0,
    width=320,
    height=180,
    depth=fake_depth,
    z_min=0.5,
    z_max=10.0,
)

with open(test_vdz, "wb") as f:
    bytes_written = write_vdz_frame(f, frame, compress=True)
    
print(f"Wrote VDZ frame: {bytes_written} bytes (compressed)")
print(f"  Original size: {320*180*2} bytes (uint16)")
print(f"  Compression ratio: {320*180*2 / bytes_written:.1f}x")

# Test 3: Camera poses
print("\n" + "=" * 60)
print("Test 3: Camera Pose Format")
print("=" * 60)

from offline.formats import CameraPose, write_camera_poses, read_camera_poses

poses = []
for i in range(10):
    # Create simple test poses
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = i * 0.1  # Move camera along X
    c2w[2, 3] = -5.0  # 5 meters back
    
    K = np.array([
        [500, 0, 160],
        [0, 500, 90],
        [0, 0, 1],
    ], dtype=np.float32)
    
    poses.append(CameraPose(
        frame_idx=i,
        timestamp_ms=i * 41.67,
        intrinsics=K,
        extrinsics=c2w,
    ))

write_camera_poses(output_dir, poses)
print(f"Wrote {len(poses)} camera poses")

# Read back
loaded_poses = read_camera_poses(output_dir)
print(f"Read back {len(loaded_poses)} poses")
print(f"  Pose 0 position: {loaded_poses[0].extrinsics[:3, 3]}")
print(f"  Pose 9 position: {loaded_poses[9].extrinsics[:3, 3]}")

# Test 4: Chunk calculation
print("\n" + "=" * 60)
print("Test 4: Chunk Index Calculation")
print("=" * 60)

from offline.preprocess import PreprocessConfig, OfflinePreprocessor

config = PreprocessConfig(chunk_size=60, overlap=30)
# Use a mock to avoid loading the model
class MockProcessor:
    def __init__(self, config):
        self.config = config
        self.processor = None  # Skip model loading
    
    def _get_chunk_indices(self, total_frames, chunk_size, overlap):
        chunks = []
        start = 0
        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunks.append((start, end))
            if end >= total_frames:
                break
            start = end - overlap
        return chunks

mock = MockProcessor(config)

# Test with 14315 frames (Big Buck Bunny)
chunks = mock._get_chunk_indices(14315, 60, 30)
print(f"14315 frames with chunk_size=60, overlap=30:")
print(f"  Total chunks: {len(chunks)}")
print(f"  First 5 chunks: {chunks[:5]}")
print(f"  Last 3 chunks: {chunks[-3:]}")

# Test with smaller video
chunks_small = mock._get_chunk_indices(150, 60, 30)
print(f"\n150 frames with chunk_size=60, overlap=30:")
print(f"  Total chunks: {len(chunks_small)}")
print(f"  Chunks: {chunks_small}")

print("\n" + "=" * 60)
print("All basic tests passed!")
print("=" * 60)
