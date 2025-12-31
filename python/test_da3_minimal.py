"""Minimal test of DA3 inference with a few frames."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Extract a few frames from video
print("=" * 60)
print("Extracting test frames from video")
print("=" * 60)

video_path = Path(r"D:\maxio\Documents\GitHub\vk_gaussian_splatting\_downloaded_resources\BigBuckBunny_320x180.mp4")
output_dir = Path(r"D:\maxio\Documents\GitHub\vk_gaussian_splatting\_downloaded_resources\test_frames")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
frame_paths = []

# Extract just 10 frames for quick test
for i in range(10):
    frame_idx = i * 24  # Every second
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    path = output_dir / f"frame_{i:04d}.png"
    cv2.imwrite(str(path), frame)
    frame_paths.append(path)
    print(f"  Extracted frame {frame_idx} -> {path.name}")

cap.release()
print(f"Extracted {len(frame_paths)} frames")

# Test DA3 inference
print("\n" + "=" * 60)
print("Testing DA3 inference")
print("=" * 60)

try:
    import torch
    
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("ERROR: CUDA not available. Install CUDA-enabled PyTorch:")
        print("  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    
    from depth_anything_3.api import DepthAnything3
    
    # DA3-LARGE has camera decoder/encoder for pose estimation
    # DA3METRIC-LARGE is depth-only (no camera params)
    model_id = "depth-anything/DA3-LARGE"
    print(f"\nLoading model: {model_id}")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to("cuda").eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Model submodules: {[name for name, _ in model.model.named_children()]}")
    
    print("\nRunning inference on 10 frames...")
    t0 = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            # DA3-LARGE with camera decoder outputs extrinsics/intrinsics
            predictions = model.inference(
                [str(p) for p in frame_paths],
                process_res=504,
                ref_view_strategy="saddle_balanced",
            )
    
    print(f"Inference completed in {time.time() - t0:.1f}s")
    
    # Inspect results
    print("\nResults:")
    print(f"  depth shape: {predictions.depth.shape}")
    print(f"  depth dtype: {predictions.depth.dtype}")
    print(f"  depth range: [{predictions.depth.min():.2f}, {predictions.depth.max():.2f}]")
    
    if predictions.conf is not None:
        print(f"  conf shape: {predictions.conf.shape}")
        print(f"  conf range: [{predictions.conf.min():.2f}, {predictions.conf.max():.2f}]")
    else:
        print("  conf: None")
    
    if predictions.intrinsics is not None:
        print(f"  intrinsics shape: {predictions.intrinsics.shape}")
        print(f"  intrinsics[0]:\n{predictions.intrinsics[0]}")
    else:
        print("  intrinsics: None")
    
    if predictions.extrinsics is not None:
        print(f"  extrinsics shape: {predictions.extrinsics.shape}")
        print(f"  extrinsics[0] (c2w):\n{predictions.extrinsics[0]}")
        # Camera positions
        positions = predictions.extrinsics[:, :3, 3]
        print(f"  camera positions:\n{positions}")
    else:
        print("  extrinsics: None")
    
    # Save a depth visualization
    depth0 = predictions.depth[0]
    depth_norm = (depth0 - depth0.min()) / (depth0.max() - depth0.min() + 1e-6)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output_dir / "depth_vis_0.png"), depth_colored)
    print(f"\nSaved depth visualization to {output_dir / 'depth_vis_0.png'}")
    
    print("\n" + "=" * 60)
    print("DA3 inference test PASSED!")
    print("=" * 60)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
