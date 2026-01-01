"""Export frames from VDZ depth sequences to images for comparison.

Usage:
    python -m offline.vdz_export input.vdz --output ./frames/ --frames 0,50,100
    python -m offline.vdz_export input.vdz --output ./frames/ --every 30
"""

from __future__ import annotations

import argparse
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VDZ_HEADER_STRUCT = struct.Struct("<4sHHIIIfff")  # 32 bytes
VDZ_HEADER_SIZE = VDZ_HEADER_STRUCT.size


@dataclass
class VdzFrame:
    """Single VDZ depth frame."""

    frame_idx: int
    timestamp_ms: int
    width: int
    height: int
    depth: np.ndarray  # float32 metric depth
    z_min: float
    z_max: float


def read_vdz_frames(vdz_path: Path) -> list[VdzFrame]:
    """Read all frames from a VDZ file."""
    frames = []

    with open(vdz_path, "rb") as f:
        frame_idx = 0
        while True:
            header_bytes = f.read(VDZ_HEADER_SIZE)
            if len(header_bytes) < VDZ_HEADER_SIZE:
                break

            (magic, version, dtype, timestamp_ms, width, height, scale, z_min, z_max) = (
                VDZ_HEADER_STRUCT.unpack(header_bytes)
            )

            compressed = magic == b"VDZ2"
            expected_size = width * height * 2  # uint16

            if compressed:
                # Read until we can decompress enough data
                # VDZ doesn't store compressed size, so we read chunks
                chunks = []
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    try:
                        data = zlib.decompress(b"".join(chunks))
                        if len(data) >= expected_size:
                            # Put back excess bytes
                            excess = len(b"".join(chunks)) - len(
                                zlib.compress(data[:expected_size], level=1)
                            )
                            if excess > 0:
                                f.seek(-len(chunk) + (len(chunk) - excess), 1)
                            break
                    except zlib.error:
                        continue

                raw_bytes = zlib.decompress(b"".join(chunks))[:expected_size]
            else:
                raw_bytes = f.read(expected_size)

            encoded = np.frombuffer(raw_bytes, dtype="<u2").reshape(height, width)
            depth = encoded.astype(np.float32) * scale + z_min

            frames.append(
                VdzFrame(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    width=width,
                    height=height,
                    depth=depth,
                    z_min=z_min,
                    z_max=z_max,
                )
            )
            frame_idx += 1

    return frames


def read_vdz_frame_at(vdz_path: Path, target_idx: int) -> VdzFrame | None:
    """Read a specific frame from VDZ (more memory efficient)."""
    with open(vdz_path, "rb") as f:
        frame_idx = 0
        while True:
            header_bytes = f.read(VDZ_HEADER_SIZE)
            if len(header_bytes) < VDZ_HEADER_SIZE:
                return None

            (magic, version, dtype, timestamp_ms, width, height, scale, z_min, z_max) = (
                VDZ_HEADER_STRUCT.unpack(header_bytes)
            )

            compressed = magic == b"VDZ2"
            expected_size = width * height * 2

            if compressed:
                chunks = []
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    try:
                        data = zlib.decompress(b"".join(chunks))
                        if len(data) >= expected_size:
                            break
                    except zlib.error:
                        continue
                raw_bytes = zlib.decompress(b"".join(chunks))[:expected_size]
            else:
                raw_bytes = f.read(expected_size)

            if frame_idx == target_idx:
                encoded = np.frombuffer(raw_bytes, dtype="<u2").reshape(height, width)
                depth = encoded.astype(np.float32) * scale + z_min
                return VdzFrame(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    width=width,
                    height=height,
                    depth=depth,
                    z_min=z_min,
                    z_max=z_max,
                )

            frame_idx += 1

    return None


def depth_to_colormap(depth: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Convert depth to colormap image (turbo colormap)."""
    normalized = (depth - z_min) / (z_max - z_min + 1e-6)
    normalized = np.clip(normalized, 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return colored


def depth_to_grayscale(depth: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Convert depth to 16-bit grayscale (lossless)."""
    normalized = (depth - z_min) / (z_max - z_min + 1e-6)
    normalized = np.clip(normalized, 0, 1)
    return (normalized * 65535).astype(np.uint16)


def export_frames(
    vdz_path: Path,
    output_dir: Path,
    frame_indices: list[int] | None = None,
    every_n: int | None = None,
    colormap: bool = True,
    lossless: bool = True,
) -> None:
    """Export specified frames from VDZ to images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {vdz_path}...")
    frames = read_vdz_frames(vdz_path)
    print(f"Found {len(frames)} frames")

    if frame_indices is None and every_n is None:
        # Export all frames
        indices_to_export = list(range(len(frames)))
    elif every_n is not None:
        indices_to_export = list(range(0, len(frames), every_n))
    else:
        assert frame_indices is not None  # Type guard for mypy/ty
        indices_to_export = [i for i in frame_indices if i < len(frames)]

    print(f"Exporting {len(indices_to_export)} frames...")

    for idx in indices_to_export:
        frame = frames[idx]

        if colormap:
            colored = depth_to_colormap(frame.depth, frame.z_min, frame.z_max)
            color_path = output_dir / f"frame_{idx:06d}_color.png"
            cv2.imwrite(str(color_path), colored)

        if lossless:
            gray16 = depth_to_grayscale(frame.depth, frame.z_min, frame.z_max)
            gray_path = output_dir / f"frame_{idx:06d}_depth.png"
            cv2.imwrite(str(gray_path), gray16)

        print(f"  Frame {idx}: z_range=[{frame.z_min:.2f}, {frame.z_max:.2f}]")

    print(f"Exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export frames from VDZ depth sequence to images")
    parser.add_argument("input", type=Path, help="Input VDZ file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--frames", type=str, default=None, help="Comma-separated frame indices (e.g., 0,50,100)"
    )
    parser.add_argument("--every", type=int, default=None, help="Export every N frames")
    parser.add_argument("--no-colormap", action="store_true", help="Skip colormap output")
    parser.add_argument("--no-lossless", action="store_true", help="Skip 16-bit lossless output")

    args = parser.parse_args()

    frame_indices = None
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(",")]

    export_frames(
        args.input,
        args.output,
        frame_indices=frame_indices,
        every_n=args.every,
        colormap=not args.no_colormap,
        lossless=not args.no_lossless,
    )


if __name__ == "__main__":
    main()
