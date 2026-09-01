#!/usr/bin/env python
"""Download InfiniDepth model checkpoints for Gaussian Splatting export.

Run this script to pre-download the model checkpoints before using the export pipeline.
The checkpoints are cached in ~/.cache/huggingface/hub/

Required checkpoints:
- infinidepth.ckpt (~1.5GB) - Main depth model
- infinidepth_gs.ckpt (~600MB) - Gaussian splatting predictor
- MoGe-3 model (~1.5GB) - downloaded from Ruicheng/moge-3-vitl
- skyseg.onnx (~40MB) - Optional sky segmentation

Usage:
    cd python
    source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
    python download_infinidepth.py
"""

import os
import sys

# Disable xet transfer for compatibility
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

print("=" * 60)
print("InfiniDepth Model Downloader")
print("=" * 60)
print()

# Check for CUDA
try:
    import torch

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Model inference requires GPU.")
except ImportError:
    print("WARNING: PyTorch not installed.")

print()
print("Downloading InfiniDepth checkpoints from HuggingFace...")
print("Cache location: ~/.cache/huggingface/hub/")
print()

INFINIDEPTH_REPO = "ritianyu/InfiniDepth"
CHECKPOINTS = [
    ("infinidepth.ckpt", "Main depth model (~1.5GB)"),
    ("infinidepth_gs.ckpt", "Gaussian splatting predictor (~600MB)"),
    ("skyseg.onnx", "Sky segmentation (~40MB, optional)"),
]
MOGE3_REPO = "Ruicheng/moge-3-vitl"

try:
    from huggingface_hub import hf_hub_download
    import huggingface_hub

    print(f"huggingface_hub version: {huggingface_hub.__version__}")
    print()

    for filename, description in CHECKPOINTS:
        print(f"Downloading {filename} ({description})...")
        try:
            path = hf_hub_download(
                repo_id=INFINIDEPTH_REPO,
                filename=filename,
                resume_download=True,
            )
            print(f"  SUCCESS: {path}")
        except Exception as e:
            print(f"  WARNING: Failed to download {filename}: {e}")
            if "skyseg" not in filename:
                print("  This checkpoint is required for InfiniDepth to work.")
                raise

    print("Downloading MoGe-3 model (model.pt)...")
    moge_path = hf_hub_download(repo_id=MOGE3_REPO, filename="model.pt")
    print(f"  SUCCESS: {moge_path}")

    print()
    print("=" * 60)
    print("SUCCESS!")
    print("InfiniDepth and MoGe-3 checkpoints downloaded and cached.")
    print("=" * 60)
    print()
    print("You can now run the Gaussian export pipeline:")
    print("  python -m offline.cli images --input ./images/ --output ./ply_output/ --mode frames")

except KeyboardInterrupt:
    print("\nDownload cancelled by user.")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
