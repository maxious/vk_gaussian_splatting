"""PLY file I/O for Gaussian Splatting data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .types import GaussianFrame

logger = logging.getLogger(__name__)


def write_static_gaussian_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    sh_rest: np.ndarray | None = None,
    flip_y: bool = False,
    # Camera metadata (optional - needed for proper viewer positioning)
    intrinsic: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    color_space_index: int | None = None,
) -> None:
    """Write a static 3DGS PLY file using plyfile.

    Args:
        path: Output PLY file path
        means: (N, 3) float32 positions
        scales: (N, 3) float32 log-scales
        rotations: (N, 4) float32 quaternions (wxyz order)
        colors: (N, 3) float32 SH DC coefficients
        opacities: (N,) float32 logit opacities
        sh_rest: Optional (N, 45) float32 higher-order SH coefficients
        flip_y: If True, negate Y coordinates to flip the coordinate system
        intrinsic: Optional (3, 3) camera intrinsic matrix
        extrinsic: Optional (4, 4) camera extrinsic matrix
        image_size: Optional (height, width) of source image
        color_space_index: Optional color space index (0=linear, 1=sRGB)
    """
    n_points = len(means)

    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        n_sh = sh_rest.shape[1]
        for i in range(n_sh):
            dtype_list.append((f"f_rest_{i}", "f4"))

    dtype_list.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )

    elements = np.empty(n_points, dtype=dtype_list)

    elements["x"] = means[:, 0]
    elements["y"] = -means[:, 1] if flip_y else means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["f_dc_0"] = colors[:, 0]
    elements["f_dc_1"] = colors[:, 1]
    elements["f_dc_2"] = colors[:, 2]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        for i in range(sh_rest.shape[1]):
            elements[f"f_rest_{i}"] = sh_rest[:, i]

    # Apply inverse sigmoid to opacity for storage (viewer applies sigmoid when loading)
    opacities_flat = np.clip(
        opacities.squeeze(-1) if opacities.ndim > 1 else opacities, 1e-6, 1 - 1e-6
    )
    opacities_logit = np.log(opacities_flat / (1 - opacities_flat))
    elements["opacity"] = opacities_logit

    # Store log of scales (viewer applies exp when loading)
    elements["scale_0"] = np.log(scales[:, 0])
    elements["scale_1"] = np.log(scales[:, 1])
    elements["scale_2"] = np.log(scales[:, 2])
    elements["rot_0"] = rotations[:, 0]
    elements["rot_1"] = rotations[:, 1]
    elements["rot_2"] = rotations[:, 2]
    elements["rot_3"] = rotations[:, 3]

    ply_elements = [PlyElement.describe(elements, "vertex")]

    # Add camera metadata if provided
    if image_size is not None:
        dtype_image_size = [("image_size", "u4")]
        image_size_array = np.empty(2, dtype=dtype_image_size)
        image_size_array[:] = np.array(
            [image_size[1], image_size[0]], dtype=np.uint32
        )  # (width, height)
        ply_elements.append(PlyElement.describe(image_size_array, "image_size"))

    if intrinsic is not None:
        dtype_intrinsic = [("intrinsic", "f4")]
        intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
        intrinsic_array[:] = intrinsic.flatten().astype(np.float32)
        ply_elements.append(PlyElement.describe(intrinsic_array, "intrinsic"))

    if extrinsic is not None:
        dtype_extrinsic = [("extrinsic", "f4")]
        extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
        extrinsic_array[:] = extrinsic.flatten().astype(np.float32)
        ply_elements.append(PlyElement.describe(extrinsic_array, "extrinsic"))

    if color_space_index is not None:
        dtype_color_space = [("color_space", "u1")]
        color_space_array = np.empty(1, dtype=dtype_color_space)
        color_space_array[:] = np.array([color_space_index], dtype=np.uint8)
        ply_elements.append(PlyElement.describe(color_space_array, "color_space"))

    PlyData(ply_elements, text=False).write(str(path))

    logger.info(f"Wrote {n_points} Gaussians to {path}")


def load_static_gaussian_ply(path: Path) -> GaussianFrame:
    """Load a static 3DGS PLY file and return a GaussianFrame.

    Args:
        path: Path to PLY file

    Returns:
        GaussianFrame with the loaded data (frame_idx=0, timestamp_ms=0)
    """
    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]

    # Extract positions
    means = np.column_stack(
        [
            vertex["x"],
            vertex["y"],
            vertex["z"],
        ]
    ).astype(np.float32)

    # Extract scales
    scales = np.column_stack(
        [
            vertex["scale_0"],
            vertex["scale_1"],
            vertex["scale_2"],
        ]
    ).astype(np.float32)

    # Extract rotations (quaternion wxyz)
    rotations = np.column_stack(
        [
            vertex["rot_0"],
            vertex["rot_1"],
            vertex["rot_2"],
            vertex["rot_3"],
        ]
    ).astype(np.float32)

    # Extract SH DC coefficients (colors)
    colors = np.column_stack(
        [
            vertex["f_dc_0"],
            vertex["f_dc_1"],
            vertex["f_dc_2"],
        ]
    ).astype(np.float32)

    # Extract opacity
    opacities = np.array(vertex["opacity"]).astype(np.float32)

    return GaussianFrame(
        frame_idx=0,
        timestamp_ms=0.0,
        means=means,
        scales=scales,
        rotations=rotations,
        colors=colors,
        opacities=opacities,
    )


def write_freetimegs_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    motion: np.ndarray,
    time_center: np.ndarray,
    time_scale: np.ndarray,
    sh_rest: np.ndarray | None = None,
    flip_y: bool = False,
) -> None:
    """Write a FreeTimeGS PLY file with temporal parameters using plyfile.

    The temporal model is:
        position(t) = mean + motion * (t - time_center)
        opacity(t) = opacity * exp(-0.5 * ((t - time_center) / time_scale)^2)

    Args:
        path: Output PLY file path
        means: (N, 3) float32 positions at time_center
        scales: (N, 3) float32 log-scales
        rotations: (N, 4) float32 quaternions (wxyz order)
        colors: (N, 3) float32 SH DC coefficients
        opacities: (N,) float32 logit opacities
        motion: (N, 3) float32 velocity vectors
        time_center: (N,) float32 temporal center (0-1 normalized)
        time_scale: (N,) float32 temporal width (log-space, will be exp'd by viewer)
        sh_rest: Optional (N, 45) float32 higher-order SH coefficients
        flip_y: If True, negate Y coordinates to flip the coordinate system
    """
    n_points = len(means)

    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        for i in range(sh_rest.shape[1]):
            dtype_list.append((f"f_rest_{i}", "f4"))

    dtype_list.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("motion_0", "f4"),
            ("motion_1", "f4"),
            ("motion_2", "f4"),
            ("t", "f4"),
            ("t_scale", "f4"),
        ]
    )

    elements = np.empty(n_points, dtype=dtype_list)

    elements["x"] = means[:, 0]
    elements["y"] = -means[:, 1] if flip_y else means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["f_dc_0"] = colors[:, 0]
    elements["f_dc_1"] = colors[:, 1]
    elements["f_dc_2"] = colors[:, 2]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        for i in range(sh_rest.shape[1]):
            elements[f"f_rest_{i}"] = sh_rest[:, i]

    elements["opacity"] = opacities.squeeze(-1) if opacities.ndim > 1 else opacities
    elements["scale_0"] = scales[:, 0]
    elements["scale_1"] = scales[:, 1]
    elements["scale_2"] = scales[:, 2]
    elements["rot_0"] = rotations[:, 0]
    elements["rot_1"] = rotations[:, 1]
    elements["rot_2"] = rotations[:, 2]
    elements["rot_3"] = rotations[:, 3]
    elements["motion_0"] = motion[:, 0]
    elements["motion_1"] = motion[:, 1]
    elements["motion_2"] = motion[:, 2]
    elements["t"] = time_center
    elements["t_scale"] = time_scale

    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=False).write(str(path))


def write_delta_compressed_freetimegs_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    deltas: np.ndarray,  # Int16 or Int8 compressed deltas
    time_center: np.ndarray,
    time_scale: np.ndarray,
    compression_scale: float,  # Scale factor to decompress deltas
    use_int8: bool = False,
    sh_rest: np.ndarray | None = None,
    flip_y: bool = False,
) -> None:
    """
    Write a delta-compressed FreeTimeGS PLY file.

    Instead of storing absolute motion vectors, stores compressed temporal deltas.
    The temporal model becomes:
        position(t) = mean + (deltas / compression_scale) * (t - time_center)
        opacity(t) = opacity * exp(-0.5 * ((t - time_center) / time_scale)^2)

    Args:
        path: Output PLY file path
        means: (N, 3) float32 positions at time_center
        scales: (N, 3) float32 log-scales
        rotations: (N, 4) float32 quaternions (wxyz order)
        colors: (N, 3) float32 SH DC coefficients
        opacities: (N,) float32 logit opacities
        deltas: (N, 3) int16 or int8 compressed temporal deltas
        time_center: (N,) float32 temporal center (0-1 normalized)
        time_scale: (N,) float32 temporal width (log-space)
        compression_scale: Scale factor to convert deltas back to float32
        use_int8: Whether deltas are int8 (True) or int16 (False)
        sh_rest: Optional (N, 45) float32 higher-order SH coefficients
        flip_y: If True, negate Y coordinates to flip the coordinate system
    """
    n_points = len(means)

    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        for i in range(sh_rest.shape[1]):
            dtype_list.append((f"f_rest_{i}", "f4"))

    # Use native int types for compressed deltas
    motion_dtype = "i1" if use_int8 else "i2"
    dtype_list.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("motion_0", motion_dtype),
            ("motion_1", motion_dtype),
            ("motion_2", motion_dtype),
            ("t", "f4"),
            ("t_scale", "f4"),
        ]
    )

    elements = np.empty(n_points, dtype=dtype_list)

    elements["x"] = means[:, 0]
    elements["y"] = -means[:, 1] if flip_y else means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["f_dc_0"] = colors[:, 0]
    elements["f_dc_1"] = colors[:, 1]
    elements["f_dc_2"] = colors[:, 2]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        for i in range(sh_rest.shape[1]):
            elements[f"f_rest_{i}"] = sh_rest[:, i]

    elements["opacity"] = opacities.squeeze(-1) if opacities.ndim > 1 else opacities
    elements["scale_0"] = scales[:, 0]
    elements["scale_1"] = scales[:, 1]
    elements["scale_2"] = scales[:, 2]
    elements["rot_0"] = rotations[:, 0]
    elements["rot_1"] = rotations[:, 1]
    elements["rot_2"] = rotations[:, 2]
    elements["rot_3"] = rotations[:, 3]

    elements["motion_0"] = deltas[:, 0]
    elements["motion_1"] = deltas[:, 1]
    elements["motion_2"] = deltas[:, 2]
    elements["t"] = time_center
    elements["t_scale"] = time_scale

    el = PlyElement.describe(elements, "vertex")
    compression_type = "int8" if use_int8 else "int16"
    comments = [
        f"motion_scale {compression_scale:.10f}",
        f"motion_dtype {compression_type}",
    ]
    PlyData([el], text=False, comments=comments).write(str(path))

    logger.info(f"Wrote {n_points} FreeTimeGS Gaussians to {path}")
