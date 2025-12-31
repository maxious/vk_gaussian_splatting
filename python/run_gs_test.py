"""Quick test for DA3 Gaussian export."""
import sys
from pathlib import Path

print("Starting DA3 Gaussian test...", flush=True)

from offline.export_gaussian_ply import (
    extract_video_frames,
    DA3GaussianProcessor,
    write_static_gaussian_ply,
)

video = Path("../_downloaded_resources/166915-835670849_small.mp4")
out = Path("../_downloaded_resources/test_gs_export")
out.mkdir(exist_ok=True)

print(f"Video: {video}", flush=True)
print(f"Output: {out}", flush=True)

print("Extracting 5 frames (skip=10)...", flush=True)
frames, ts = extract_video_frames(video, out / "temp", frame_skip=10, max_frames=5)
print(f"Extracted {len(frames)} frames", flush=True)

print("Creating DA3GaussianProcessor with DA3-GIANT (for infer_gs support)...", flush=True)
# DA3-GIANT is ~5.4GB - required for Gaussian output
proc = DA3GaussianProcessor(model_id="depth-anything/DA3-GIANT", process_res=518)

print("Processing frames individually (per_frame=True)...", flush=True)
try:
    results = proc.process_frames(frames, ts, per_frame=True)
    print(f"Got {len(results)} GaussianFrames", flush=True)
    
    for i, r in enumerate(results):
        print(f"  Frame {i}: {len(r.means)} gaussians, means shape={r.means.shape}", flush=True)
        ply_path = out / f"da3_frame_{i:04d}.ply"
        write_static_gaussian_ply(ply_path, r.means, r.scales, r.rotations, r.colors, r.opacities)
        print(f"  Wrote: {ply_path}", flush=True)
    
    print("SUCCESS!", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
