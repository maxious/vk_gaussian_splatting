"""
Super Compressed Gaussian Splatting (SOG) format compression.

Based on the official PlayCanvas splat-transform implementation v0.16.1.
"""

import json
import math
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numba import jit, prange
from tqdm import tqdm

# Import FAISS for k-means clustering (mandatory)
import faiss


@jit(nopython=True)
def log_transform(value: float) -> float:
    """Log transform for means (sign(x) * log(|x| + 1))."""
    return math.copysign(math.log(abs(value) + 1), value)


@jit(nopython=True, parallel=True)
def log_transform_vectorized(arr: np.ndarray) -> np.ndarray:
    """Vectorized log transform for means array."""
    result = np.empty_like(arr)
    for i in prange(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            result[i, j] = math.copysign(math.log(abs(val) + 1), val)
    return result


@jit(nopython=True)
def sigmoid(x: float) -> float:
    """Sigmoid function for opacity."""
    return 1 / (1 + math.exp(-x))


@jit(nopython=True, parallel=True)
def sigmoid_vectorized(arr: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid for opacity array."""
    result = np.empty(arr.shape, dtype=np.float64)
    for i in prange(arr.shape[0]):
        result[i] = 1.0 / (1.0 + math.exp(-arr[i]))
    return result


def srgb_to_linear(c: float) -> float:
    """Convert sRGB to linear RGB."""
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    """Convert linear RGB to sRGB."""
    if c <= 0.0031308:
        return c * 12.92
    else:
        return 1.055 * (c ** (1 / 2.4)) - 0.055


@jit(nopython=True, parallel=True)
def pack_quaternions_vectorized(quats: np.ndarray) -> np.ndarray:
    """
    Pack quaternions into RGBA format - vectorized with Numba.

    Args:
        quats: (N, 4) quaternion array

    Returns:
        rgba: (N, 4) uint8 RGBA values
    """
    n = quats.shape[0]
    result = np.zeros((n, 4), dtype=np.uint8)

    for i in prange(n):
        q = quats[i]

        # Normalize quaternion
        norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if norm == 0:
            result[i, 3] = 0
            continue

        q0 = q[0] / norm
        q1 = q[1] / norm
        q2 = q[2] / norm
        q3 = q[3] / norm

        # Find largest component
        abs_q = np.array([abs(q0), abs(q1), abs(q2), abs(q3)])
        max_comp = 0
        max_val = abs_q[0]
        for j in range(1, 4):
            if abs_q[j] > max_val:
                max_val = abs_q[j]
                max_comp = j

        # Create 3-component vector by dropping largest component
        q_arr = np.array([q0, q1, q2, q3])
        abc = np.zeros(3)
        idx = 0
        for j in range(4):
            if j != max_comp:
                abc[idx] = q_arr[j]
                idx += 1

        # Normalize to [-1, 1] and scale to [-127, 127]
        norm_abc = math.sqrt(abc[0] * abc[0] + abc[1] * abc[1] + abc[2] * abc[2])
        if norm_abc > 0:
            abc[0] /= norm_abc
            abc[1] /= norm_abc
            abc[2] /= norm_abc

        # Quantize to 8 bits each, with sign bit
        qa = int(abc[0] * 127) & 0xFF
        qb = int(abc[1] * 127) & 0xFF
        qc = int(abc[2] * 127) & 0xFF

        # Apply octahedral encoding for better precision
        if qa & 0x80:
            qa &= 0x7F
            if qb & 0x80:
                qb = qb | 0x80
            else:
                qb = qb & 0x7F
        if qb & 0x80:
            qb &= 0x7F
            qc |= 0x80

        # Alpha channel stores max component index
        alpha = (max_comp << 6) & 0xFF

        result[i, 0] = qa
        result[i, 1] = qb
        result[i, 2] = qc
        result[i, 3] = alpha

    return result


def morton_order_sort(points: np.ndarray) -> np.ndarray:
    """
    Sort points in Morton (Z-order) curve order.

    Args:
        points: (N, 3) array of 3D points

    Returns:
        indices: Sorted indices
    """
    if len(points) == 0:
        return np.arange(0)

    # Normalize points to [0, 1]
    min_val = points.min(axis=0)
    max_val = points.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1  # Avoid division by zero

    normalized = (points - min_val) / range_val

    # Quantize to 10 bits (0-1023)
    # Note: 10 bits per axis * 3 axes = 30 bits, which fits in 32-bit int
    quantized = (normalized * 1023).astype(np.uint32)

    # Helper to expand bits for Morton encoding
    # Spreads low 10 bits of v to positions: - - 9 - - 8 - - 7 ...
    def expand_bits(v):
        v = (v * 0x00010001) & 0xFF0000FF
        v = (v * 0x00000101) & 0x0F00F00F
        v = (v * 0x00000011) & 0xC30C30C3
        v = (v * 0x00000005) & 0x49249249
        return v

    # Vectorized bit expansion
    # Note: The bit manipulation above is a bit complex to vectorize efficiently in pure numpy
    # without a specific bit-interleaving ufunc.
    # Alternative efficient approach using precomputed lookup or simpler bit shifts:

    x = quantized[:, 0]
    y = quantized[:, 1]
    z = quantized[:, 2]

    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249

    y = (y | (y << 16)) & 0x030000FF
    y = (y | (y << 8)) & 0x0300F00F
    y = (y | (y << 4)) & 0x030C30C3
    y = (y | (y << 2)) & 0x09249249

    z = (z | (z << 16)) & 0x030000FF
    z = (z | (z << 8)) & 0x0300F00F
    z = (z | (z << 4)) & 0x030C30C3
    z = (z | (z << 2)) & 0x09249249

    # Interleave: Z Y X
    codes = (z << 2) | (y << 1) | x

    return np.argsort(codes)


def kmeans_1d(
    data: np.ndarray, n_clusters: int = 256, iterations: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform 1D k-means clustering.

    Args:
        data: (N,) or (N, D) array
        n_clusters: Number of clusters
        iterations: Number of iterations

    Returns:
        centroids: (n_clusters,) centroids
        labels: (N,) cluster labels
    """
    # Use FAISS for k-means (mandatory)
    data_flat = data.flatten() if data.ndim > 1 else data
    data_2d = data_flat.reshape(-1, 1).astype(np.float32)

    kmeans = faiss.Kmeans(
        d=data_2d.shape[1], k=n_clusters, niter=iterations, verbose=False, gpu=False
    )
    kmeans.train(data_2d)
    centroids = kmeans.centroids.flatten()  # type: ignore
    _, labels = kmeans.index.search(data_2d, 1)  # type: ignore
    labels = labels.flatten().astype(np.int32)

    # Sort centroids from smallest to largest
    sort_indices = np.argsort(centroids)
    centroids = centroids[sort_indices]

    # Create inverse mapping for labels
    inv_sort = np.empty_like(sort_indices)
    inv_sort[sort_indices] = np.arange(len(sort_indices))
    labels = inv_sort[labels]

    return centroids, labels


def write_webp_image(filename: str, data: np.ndarray, width: int, height: int) -> None:
    """Write RGBA data as lossless WebP image using OpenCV.

    Uses OpenCV's WebP encoder which is ~50x faster than Pillow for lossless encoding.
    Alpha values are clamped to minimum 1 to ensure truly lossless encoding, as WebP
    optimizes away RGB values for fully transparent pixels (alpha=0).
    """
    if data.dtype != np.uint8:
        raise ValueError("Data must be uint8")

    if data.size != width * height * 4:
        raise ValueError(f"Data size {data.size} doesn't match dimensions {width}x{height}x4")

    # Reshape to image
    rgba = data.reshape(height, width, 4)

    # Clamp alpha to minimum 1 to ensure lossless encoding
    # WebP optimizes away RGB values for alpha=0 pixels, corrupting data
    rgba[:, :, 3] = np.maximum(rgba[:, :, 3], 1)

    # Convert RGBA to BGRA for OpenCV
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

    # Encode as lossless WebP (quality > 100 triggers lossless mode)
    success, encoded = cv2.imencode(".webp", bgra, [cv2.IMWRITE_WEBP_QUALITY, 101])
    if not success:
        raise RuntimeError(f"Failed to encode WebP: {filename}")

    # Write to file
    with open(filename, "wb") as f:
        f.write(encoded.tobytes())


@jit(nopython=True, parallel=True)
def fill_texture_rgba_3channel(
    data: np.ndarray, labels_3d: np.ndarray, alpha_channel: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Fill texture with RGB from labels and alpha from separate array - vectorized."""
    result = np.zeros((height, width, 4), dtype=np.uint8)
    n = labels_3d.shape[0]
    for i in prange(n):
        y = i // width
        x = i % width
        result[y, x, 0] = labels_3d[i, 0]
        result[y, x, 1] = labels_3d[i, 1]
        result[y, x, 2] = labels_3d[i, 2]
        result[y, x, 3] = alpha_channel[i]
    return result


@jit(nopython=True, parallel=True)
def fill_texture_rgba_1channel(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fill texture with single-channel labels in R, rest zeros, alpha=255."""
    result = np.zeros((height, width, 4), dtype=np.uint8)
    n = labels.shape[0]
    for i in prange(n):
        y = i // width
        x = i % width
        result[y, x, 0] = labels[i]
        result[y, x, 3] = 255
    return result


@jit(nopython=True, parallel=True)
def fill_means_textures(
    means_16bit: np.ndarray, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split 16-bit means into low/high byte textures - vectorized."""
    n = means_16bit.shape[0]
    means_l = np.zeros((height, width, 4), dtype=np.uint8)
    means_u = np.zeros((height, width, 4), dtype=np.uint8)

    for i in prange(n):
        y = i // width
        x = i % width
        val = means_16bit[i]
        means_l[y, x, 0] = val[0] & 0xFF
        means_l[y, x, 1] = val[1] & 0xFF
        means_l[y, x, 2] = val[2] & 0xFF
        means_l[y, x, 3] = 255
        means_u[y, x, 0] = (val[0] >> 8) & 0xFF
        means_u[y, x, 1] = (val[1] >> 8) & 0xFF
        means_u[y, x, 2] = (val[2] >> 8) & 0xFF
        means_u[y, x, 3] = 255

    return means_l, means_u


@jit(nopython=True, parallel=True)
def fill_quats_texture(quats_rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fill quaternion texture from packed RGBA - vectorized."""
    n = quats_rgba.shape[0]
    result = np.zeros((height, width, 4), dtype=np.uint8)

    for i in prange(n):
        y = i // width
        x = i % width
        result[y, x, 0] = quats_rgba[i, 0]
        result[y, x, 1] = quats_rgba[i, 1]
        result[y, x, 2] = quats_rgba[i, 2]
        result[y, x, 3] = quats_rgba[i, 3]

    return result


def run_compression(
    output_path: str, splats: Dict[str, torch.Tensor], iterations: int = 10, verbose: bool = False
) -> None:
    """
    Compress Gaussian splats to SOG format.

    Args:
        output_path: Output .sog file path (will be created as zip)
        splats: Dictionary with keys: 'means', 'opacities', 'scales', 'quats', 'sh0'
                Optional: 'motion', 't', 't_scale' for FreeTimeGS support
        iterations: K-means iterations (default: 10)
    """
    if verbose:
        print(f"Compressing {len(splats['means'])} Gaussians to SOG format...")

    # Extract data
    means = splats["means"].cpu().numpy()  # (N, 3)
    opacities = splats["opacities"].cpu().numpy()  # (N,)
    scales = splats["scales"].cpu().numpy()  # (N, 3)
    quats = splats["quats"].cpu().numpy()  # (N, 4)
    sh0 = splats["sh0"].cpu().numpy()  # (N, 1, 3)

    # Extract FreeTimeGS data if present
    motion = splats.get("motion")
    time_center = splats.get("t")
    time_scale = splats.get("t_scale")

    if motion is not None:
        motion = motion.cpu().numpy()  # (N, 3)
    if time_center is not None:
        time_center = time_center.cpu().numpy()  # (N,)
    if time_scale is not None:
        time_scale = time_scale.cpu().numpy()  # (N,)

    num_gaussians = len(means)
    if verbose:
        print(f"Input: {num_gaussians} Gaussians")

    # Sort by Morton order
    indices = morton_order_sort(means)
    means = means[indices]
    opacities = opacities[indices]
    scales = scales[indices]
    quats = quats[indices]
    sh0 = sh0[indices]

    # Apply sorting to optional fields
    if motion is not None:
        motion = motion[indices]
    if time_center is not None:
        time_center = time_center[indices]
    if time_scale is not None:
        time_scale = time_scale[indices]

    # Calculate texture dimensions (square, power of 2 aligned)
    side_len = int(math.ceil(math.sqrt(num_gaussians)))
    side_len = ((side_len + 3) // 4) * 4  # Align to multiple of 4
    width = height = side_len

    print(f"Texture dimensions: {width}x{height}")

    # Create temporary directory for uncompressed files
    temp_dir = output_path.replace(".sog", "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Prepare all WebP write tasks for parallel execution
    webp_tasks: List[Tuple[str, np.ndarray, int, int]] = []

    try:
        # === MEANS: log-transformed, 16-bit split into 8-bit ===
        # Vectorized log transform
        means_log = log_transform_vectorized(means.astype(np.float64))
        means_min = means_log.min(axis=0)
        means_max = means_log.max(axis=0)
        means_range = means_max - means_min
        means_range = np.where(means_range == 0, 1, means_range)

        means_norm = (means_log - means_min) / means_range
        means_16bit = (means_norm * 65535).astype(np.uint16)

        # Vectorized texture fill
        means_l, means_u = fill_means_textures(means_16bit, width, height)
        webp_tasks.append(
            (os.path.join(temp_dir, "means_l.webp"), means_l.flatten(), width, height)
        )
        webp_tasks.append(
            (os.path.join(temp_dir, "means_u.webp"), means_u.flatten(), width, height)
        )

        # === QUATERNIONS: packed into RGBA - vectorized ===
        quats_packed = pack_quaternions_vectorized(quats.astype(np.float64))
        quats_texture = fill_quats_texture(quats_packed, width, height)
        webp_tasks.append(
            (os.path.join(temp_dir, "quats.webp"), quats_texture.flatten(), width, height)
        )

        # === SCALES: k-means clustered ===
        scales_centroids, scales_labels = kmeans_1d(scales.reshape(-1, 3), 256, iterations)
        scales_labels_3d = scales_labels.reshape(-1, 3).astype(np.uint8)
        alpha_255 = np.full(num_gaussians, 255, dtype=np.uint8)
        scales_texture = fill_texture_rgba_3channel(
            scales_labels_3d, scales_labels_3d, alpha_255, width, height
        )
        webp_tasks.append(
            (os.path.join(temp_dir, "scales.webp"), scales_texture.flatten(), width, height)
        )

        # === COLORS + OPACITY: sh0 k-means clustered, opacity sigmoid ===
        colors = sh0.reshape(-1, 3)  # (N, 3)
        colors_centroids, colors_labels = kmeans_1d(colors, 256, iterations)
        colors_labels_3d = colors_labels.reshape(-1, 3).astype(np.uint8)

        # Vectorized sigmoid
        opacity_norm = sigmoid_vectorized(opacities.astype(np.float64))
        opacity_8bit = (opacity_norm * 255).astype(np.uint8)

        sh0_texture = fill_texture_rgba_3channel(
            colors_labels_3d, colors_labels_3d, opacity_8bit, width, height
        )
        webp_tasks.append(
            (os.path.join(temp_dir, "sh0.webp"), sh0_texture.flatten(), width, height)
        )

        # === MOTION VECTORS: k-means clustered (optional) ===
        motion_centroids = None
        if motion is not None:
            motion_centroids, motion_labels = kmeans_1d(motion.reshape(-1, 3), 256, iterations)
            motion_labels_3d = motion_labels.reshape(-1, 3).astype(np.uint8)
            motion_texture = fill_texture_rgba_3channel(
                motion_labels_3d, motion_labels_3d, alpha_255, width, height
            )
            webp_tasks.append(
                (os.path.join(temp_dir, "motion.webp"), motion_texture.flatten(), width, height)
            )

        # === TIME CENTER (t): k-means clustered (optional) ===
        t_centroids = None
        t_labels = None
        if time_center is not None:
            num_t_clusters = min(256, len(np.unique(time_center)))
            t_centroids, t_labels = kmeans_1d(
                time_center.reshape(-1, 1), num_t_clusters, iterations
            )
            t_labels_u8 = t_labels.astype(np.uint8)
            t_texture = fill_texture_rgba_1channel(t_labels_u8, width, height)
            webp_tasks.append(
                (os.path.join(temp_dir, "t.webp"), t_texture.flatten(), width, height)
            )

        # === TIME SCALE (t_scale): k-means clustered (optional) ===
        t_scale_centroids = None
        t_scale_labels = None
        if time_scale is not None:
            num_t_scale_clusters = min(256, len(np.unique(time_scale)))
            t_scale_centroids, t_scale_labels = kmeans_1d(
                time_scale.reshape(-1, 1), num_t_scale_clusters, iterations
            )
            t_scale_labels_u8 = t_scale_labels.astype(np.uint8)
            t_scale_texture = fill_texture_rgba_1channel(t_scale_labels_u8, width, height)
            webp_tasks.append(
                (os.path.join(temp_dir, "t_scale.webp"), t_scale_texture.flatten(), width, height)
            )

        # === PARALLEL WEBP ENCODING ===
        def write_webp_task(task: Tuple[str, np.ndarray, int, int]) -> None:
            filename, data, w, h = task
            write_webp_image(filename, data, w, h)

        with ThreadPoolExecutor(max_workers=min(8, len(webp_tasks))) as executor:
            list(executor.map(write_webp_task, webp_tasks))

        # Create metadata
        metadata = {
            "version": 2,
            "asset": {"generator": "vk_gaussian_splatting sogs v2.0.0"},
            "count": num_gaussians,
            "means": {
                "mins": means_min.tolist(),
                "maxs": means_max.tolist(),
                "files": ["means_l.webp", "means_u.webp"],
            },
            "scales": {"codebook": scales_centroids.tolist(), "files": ["scales.webp"]},
            "quats": {"files": ["quats.webp"]},
            "sh0": {"codebook": colors_centroids.tolist(), "files": ["sh0.webp"]},
        }

        # Add FreeTimeGS fields if present
        if motion_centroids is not None:
            metadata["motion"] = {"codebook": motion_centroids.tolist(), "files": ["motion.webp"]}
        if t_centroids is not None:
            metadata["t"] = {"codebook": t_centroids.tolist(), "files": ["t.webp"]}
        if t_scale_centroids is not None:
            metadata["t_scale"] = {
                "codebook": t_scale_centroids.tolist(),
                "files": ["t_scale.webp"],
            }

        with open(os.path.join(temp_dir, "meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Create zip file (.sog)
        print(f"Creating bundled SOG file: {output_path}")
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in os.listdir(temp_dir):
                zf.write(os.path.join(temp_dir, file), file)

        print(f"SUCCESS: SOG compression complete: {output_path}")

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@torch.no_grad()
def read_ply(path: str) -> Dict[str, torch.Tensor]:
    """
    Reads a .ply file and returns Gaussian splat data.

    Returns:
        dict with keys: 'means', 'opacities', 'scales', 'quats', 'sh0'
    """
    import plyfile

    plydata = plyfile.PlyData.read(path)
    vd = plydata["vertex"].data

    # Extract basic properties
    xyz = np.stack([vd["x"], vd["y"], vd["z"]], axis=-1)
    opacities = vd["opacity"]
    scale = np.stack([vd[f"scale_{i}"] for i in range(3)], axis=-1)
    rotation = np.stack([vd[f"rot_{i}"] for i in range(4)], axis=-1)

    # Extract colors (f_dc)
    f_dc = np.stack([vd[f"f_dc_{i}"] for i in range(3)], axis=-1)

    splats = {}
    splats["means"] = torch.from_numpy(xyz).float().cuda()
    splats["opacities"] = torch.from_numpy(opacities).float().cuda()
    splats["scales"] = torch.from_numpy(scale).float().cuda()
    splats["quats"] = torch.from_numpy(rotation).float().cuda()

    # Reshape colors to (N, 1, 3) for sh0
    sh0_tensor = torch.from_numpy(f_dc).float().unsqueeze(-2)
    splats["sh0"] = sh0_tensor.cuda()

    return splats
