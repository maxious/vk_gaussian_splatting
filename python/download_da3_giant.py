#!/usr/bin/env python
"""Download DA3-GIANT model for Gaussian Splatting export.

Run this script to pre-download the model before using the export pipeline.
The model is ~5.4GB and will be cached in ~/.cache/huggingface/hub/

Usage:
    cd python
    .venv\Scripts\python download_da3_giant.py
"""
import os
import sys

print("=" * 60)
print("DA3-GIANT Model Downloader")
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
print("Downloading depth-anything/DA3-GIANT...")
print("Model size: ~5.4 GB")
print("Cache location: ~/.cache/huggingface/hub/")
print()

try:
    from huggingface_hub import hf_hub_download
    import huggingface_hub
    
    print(f"huggingface_hub version: {huggingface_hub.__version__}")
    
    # Download the model file
    path = hf_hub_download(
        repo_id="depth-anything/DA3-GIANT",
        filename="model.safetensors",
        resume_download=True,
    )
    
    print()
    print("=" * 60)
    print("SUCCESS!")
    print(f"Model cached at: {path}")
    print("=" * 60)
    print()
    print("You can now run the Gaussian export pipeline:")
    print("  python -m offline.export_gaussian_ply -i video.mp4 -o output/ --mode frames")
    
except KeyboardInterrupt:
    print("\nDownload cancelled by user.")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
