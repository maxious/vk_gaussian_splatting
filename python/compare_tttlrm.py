"""Compare tttLRM full vs AR model outputs on the same test scene."""

import os
import time
from pathlib import Path

import torch

from offline.processors.ttt_lrm import TttLRMGaussianProcessor
from offline.ply_io import write_static_gaussian_ply

MANIFEST = Path("/tmp/tttlrm_manifests/006_1_seq1_48cam/006_1_seq1_frame000000.json")
OUTPUT_DIR = Path("/tmp/tttlrm_comparison_48cam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    # (name, autoregressive, num_views, height, width)
    ("tttlrm_full_16v_960", False, 16, 536, 960),
    ("tttlrm_ar_16v_480", True, 16, 264, 480),
    ("tttlrm_ar_48v_480", True, 48, 264, 480),
]

for name, ar, num_views, img_h, img_w in MODELS:
    print(f"\n{'='*60}")
    print(f"Running: {name} (ar={ar}, views={num_views}, res={img_h}x{img_w})")
    print(f"{'='*60}")

    torch.cuda.empty_cache()
    t0 = time.time()

    proc = TttLRMGaussianProcessor(
        device="cuda",
        autoregressive=ar,
        num_input_views=num_views,
        image_size=img_h,
        image_size_x=img_w,
    )
    frames = proc.process_frames([MANIFEST], [0.0])

    elapsed = time.time() - t0
    frame = frames[0]

    ply_path = OUTPUT_DIR / f"{name}.ply"
    write_static_gaussian_ply(
        ply_path,
        frame.means,
        frame.scales,
        frame.rotations,
        frame.colors,
        frame.opacities,
    )

    size_mb = os.path.getsize(ply_path) / 1024 / 1024
    print(f"  Output: {ply_path}")
    print(f"  Gaussians: {len(frame.means):,}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Time: {elapsed:.1f}s")

    # Free model memory before next run
    del proc
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"Comparison PLYs saved to {OUTPUT_DIR}")
print(f"{'='*60}")
