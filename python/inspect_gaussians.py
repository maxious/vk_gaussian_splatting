"""Inspect DA3 Gaussian output structure."""

from pathlib import Path
from offline.export_gaussian_ply import extract_video_frames, DA3GaussianProcessor

video = Path("../_downloaded_resources/166915-835670849_small.mp4")
out = Path("../_downloaded_resources/test_gs_export/temp")

print("Extracting 3 frames...", flush=True)
frames, ts = extract_video_frames(video, out, frame_skip=15, max_frames=3)
print(f"Extracted {len(frames)} frames", flush=True)

print("\nLoading DA3-GIANT...", flush=True)
import torch
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
model = model.to("cuda").eval()

print("\nRunning inference with infer_gs=True...", flush=True)

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.float16):
        images = [str(p) for p in frames]
        predictions = model.inference(
            images,
            process_res=518,
            ref_view_strategy="saddle_balanced",
            infer_gs=True,
        )

print("\n=== Prediction structure ===")
print(f"depth shape: {predictions.depth.shape}")
print(
    f"extrinsics shape: {predictions.extrinsics.shape if predictions.extrinsics is not None else None}"
)
print(
    f"intrinsics shape: {predictions.intrinsics.shape if predictions.intrinsics is not None else None}"
)

if predictions.gaussians is not None:
    g = predictions.gaussians
    print(f"\n=== Gaussians structure ===")
    print(f"means shape: {g.means.shape}")
    print(f"scales shape: {g.scales.shape}")
    print(f"rotations shape: {g.rotations.shape}")
    print(f"harmonics shape: {g.harmonics.shape}")
    print(f"opacities shape: {g.opacities.shape}")

    # Check if batch dimension corresponds to frames
    if g.means.dim() >= 2:
        print(f"\nBatch size (first dim): {g.means.shape[0]}")
        print(f"Gaussians per batch: {g.means.shape[1] if g.means.dim() > 2 else 'N/A (flat)'}")
else:
    print("No gaussians in prediction!")
