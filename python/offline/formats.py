"""Output format definitions for offline preprocessing.

Supports multiple output formats:
- VDZ sequence: Per-frame depth for real-time playback (existing format)
- NPZ sequence: Per-frame depth + camera poses for 3D reconstruction
- PLY export: Fused point cloud for static visualization
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


# =============================================================================
# VDZ Format (existing - for real-time playback)
# =============================================================================

VDZ_HEADER_STRUCT = struct.Struct("<4sHHIIIfff")  # 32 bytes
VDZ_HEADER_SIZE = VDZ_HEADER_STRUCT.size


@dataclass(slots=True)
class VdzFrame:
    """Single VDZ depth frame."""

    timestamp_ms: float
    width: int
    height: int
    depth: np.ndarray  # float32 metric depth
    z_min: float
    z_max: float


def write_vdz_frame(f: BinaryIO, frame: VdzFrame, compress: bool = True) -> int:
    """Write a single VDZ frame to file. Returns bytes written."""
    z_min, z_max = frame.z_min, frame.z_max
    if z_max <= z_min:
        z_max = z_min + 1e-3

    scale = (z_max - z_min) / 65535.0
    depth_clipped = np.clip(frame.depth, z_min, z_max)
    encoded = np.rint((depth_clipped - z_min) / scale).astype("<u2")

    raw_bytes = encoded.tobytes()
    if compress:
        payload = zlib.compress(raw_bytes, level=1)
        magic = b"VDZ2"
    else:
        payload = raw_bytes
        magic = b"VDZ1"

    header = VDZ_HEADER_STRUCT.pack(
        magic,
        1,  # version
        1,  # data type (uint16)
        int(frame.timestamp_ms),
        frame.width,
        frame.height,
        scale,
        z_min,
        z_max,
    )

    f.write(header)
    f.write(payload)
    return VDZ_HEADER_SIZE + len(payload)


# =============================================================================
# VDS Format (VDZ Sequence - new container for offline-processed videos)
# =============================================================================

VDS_MAGIC = b"VDS1"
VDS_HEADER_STRUCT = struct.Struct(
    "<4sIIIffI"
)  # magic, version, width, height, fps, duration_s, frame_count


@dataclass(slots=True)
class VdsHeader:
    """VDS sequence header."""

    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int


def write_vds_header(f: BinaryIO, header: VdsHeader) -> None:
    """Write VDS sequence header."""
    data = VDS_HEADER_STRUCT.pack(
        VDS_MAGIC,
        1,  # version
        header.width,
        header.height,
        header.fps,
        header.duration_s,
        header.frame_count,
    )
    f.write(data)


# =============================================================================
# Camera Pose Format
# =============================================================================


@dataclass(slots=True)
class CameraPose:
    """Camera pose for a single frame."""

    frame_idx: int
    timestamp_ms: float
    intrinsics: np.ndarray  # 3x3 K matrix
    extrinsics: np.ndarray  # 4x4 c2w matrix (camera-to-world)


def write_camera_poses(path: Path, poses: list[CameraPose]) -> None:
    """Write camera poses to text files (compatible with DA3-streaming format)."""
    poses_path = path / "camera_poses.txt"
    intrinsics_path = path / "intrinsics.txt"

    with open(poses_path, "w") as f:
        for pose in poses:
            # Flatten 4x4 matrix to 16 values
            values = pose.extrinsics.flatten()
            f.write(" ".join(f"{v:.8f}" for v in values) + "\n")

    with open(intrinsics_path, "w") as f:
        for pose in poses:
            # fx, fy, cx, cy
            K = pose.intrinsics
            f.write(f"{K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")


def read_camera_poses(path: Path) -> list[CameraPose]:
    """Read camera poses from text files."""
    poses_path = path / "camera_poses.txt"
    intrinsics_path = path / "intrinsics.txt"

    poses = []
    extrinsics_lines = poses_path.read_text().strip().split("\n")
    intrinsics_lines = intrinsics_path.read_text().strip().split("\n")

    for i, (ext_line, int_line) in enumerate(zip(extrinsics_lines, intrinsics_lines)):
        ext_values = [float(v) for v in ext_line.split()]
        extrinsics = np.array(ext_values).reshape(4, 4)

        int_values = [float(v) for v in int_line.split()]
        fx, fy, cx, cy = int_values
        intrinsics = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        poses.append(
            CameraPose(
                frame_idx=i,
                timestamp_ms=0.0,  # Will be filled from VDZ sequence
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
        )

    return poses


# =============================================================================
# PLY Export
# =============================================================================


def write_ply_header(f: BinaryIO, num_vertices: int) -> None:
    """Write binary PLY header."""
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {num_vertices}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    f.write(header.encode() + b"\n")


def write_ply_points(f: BinaryIO, points: np.ndarray, colors: np.ndarray) -> None:
    """Write points and colors to PLY file (after header)."""
    structured = np.zeros(
        len(points),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
        ],
    )
    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    structured["red"] = colors[:, 0]
    structured["green"] = colors[:, 1]
    structured["blue"] = colors[:, 2]
    f.write(structured.tobytes())


def write_gs_ply_header(f: BinaryIO, num_vertices: int) -> None:
    """Write binary PLY header for 3D Gaussian Splatting."""
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {num_vertices}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )
    f.write(header.encode() + b"\n")


def write_gs_ply_points(
    f: BinaryIO,
    points: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    """Write Gaussian Splatting points to PLY file.

    Args:
        f: File object
        points: (N, 3) float32 positions
        colors: (N, 3) float32 SH DC coefficients (usually 0-1 RGB converted to SH)
                Note: Standard 3DGS stores (RGB - 0.5) / 0.28209479177387814
        scales: (N, 3) float32 log scales
        opacities: (N, 1) float32 logit opacities (inverse sigmoid)
        quaternions: (N, 4) float32 rotations (w, x, y, z)
    """
    structured = np.zeros(
        len(points),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("nx", np.float32),
            ("ny", np.float32),
            ("nz", np.float32),
            ("f_dc_0", np.float32),
            ("f_dc_1", np.float32),
            ("f_dc_2", np.float32),
            ("opacity", np.float32),
            ("scale_0", np.float32),
            ("scale_1", np.float32),
            ("scale_2", np.float32),
            ("rot_0", np.float32),
            ("rot_1", np.float32),
            ("rot_2", np.float32),
            ("rot_3", np.float32),
        ],
    )

    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]

    # Normals are typically zero
    structured["nx"] = 0
    structured["ny"] = 0
    structured["nz"] = 0

    structured["f_dc_0"] = colors[:, 0]
    structured["f_dc_1"] = colors[:, 1]
    structured["f_dc_2"] = colors[:, 2]

    structured["opacity"] = opacities.flatten()

    structured["scale_0"] = scales[:, 0]
    structured["scale_1"] = scales[:, 1]
    structured["scale_2"] = scales[:, 2]

    # Convention might vary, usually [r, i, j, k] or [w, x, y, z]
    # Standard 3DGS often uses [w, x, y, z] (scalar first)
    structured["rot_0"] = quaternions[:, 0]
    structured["rot_1"] = quaternions[:, 1]
    structured["rot_2"] = quaternions[:, 2]
    structured["rot_3"] = quaternions[:, 3]

    f.write(structured.tobytes())
