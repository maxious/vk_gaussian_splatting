#!/usr/bin/env python3
"""Test script to compress a FreeTimeGS PLY file to SOG format using optimized FAISS K-means."""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from sogs.compression import run_compression, read_ply
import torch


def main():
    parser = argparse.ArgumentParser(description="Compress FreeTimeGS PLY to SOG format")
    parser.add_argument("--input", "-i", required=True, help="Input FreeTimeGS PLY file")
    parser.add_argument("--output", "-o", required=True, help="Output directory for SOG files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PLY file: {input_path}")
    splats = read_ply(str(input_path))

    print(f"Loaded {len(splats['means'])} Gaussians")
    print(f"Available attributes: {list(splats.keys())}")

    # Convert to tensors if needed
    for k, v in splats.items():
        if not isinstance(v, torch.Tensor):
            splats[k] = torch.from_numpy(v).cuda()

    print(f"Compressing to SOG format in {output_dir}")
    try:
        run_compression(str(output_dir), splats, verbose=args.verbose)
        print("✓ Compression completed successfully")

        # Check output files
        files = list(output_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"Output files: {len(files)} files, total size: {total_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"[FAIL] Compression failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
