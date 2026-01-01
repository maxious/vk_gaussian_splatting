"""
Super Compressed Gaussian Splatting (SOG) format compression.

Based on the official PlayCanvas splat-transform implementation v0.16.1.
"""

import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Import FAISS for k-means clustering
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. K-means clustering will be limited.")


def log_transform(value: float) -> float:
    """Log transform for means (sign(x) * log(|x| + 1))."""
    return math.copysign(math.log(abs(value) + 1), value)


def sigmoid(x: float) -> float:
    """Sigmoid function for opacity."""
    return 1 / (1 + math.exp(-x))


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


def pack_quaternion(q: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Pack quaternion into RGBA format.

    Returns:
        rgba: RGBA values in [0, 255]
        max_comp: Index of largest component (0-3)
    """
    # Normalize quaternion
    norm = np.linalg.norm(q)
    q = q / norm

    # Find max component and ensure it's positive
    max_comp = int(np.argmax(np.abs(q)))
    if q[max_comp] < 0:
        q = -q

    # Scale by sqrt(2) to fit in [-1, 1] range
    q = q * math.sqrt(2)

    # Reorder components (drop the largest)
    indices = [0, 1, 2, 3]
    indices.remove(max_comp)
    rgb = q[indices]

    # Map from [-1, 1] to [0, 1] to [0, 255]
    rgb = ((rgb * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    # Alpha channel encodes which component was largest
    alpha = 252 + max_comp

    rgba = np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.uint8)
    return rgba, max_comp


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
    if not FAISS_AVAILABLE:
        # Fallback: simple uniform quantization
        print("Warning: FAISS not available, using simple quantization")
        data_flat = data.flatten() if data.ndim > 1 else data
        min_val, max_val = data_flat.min(), data_flat.max()
        centroids = np.linspace(min_val, max_val, n_clusters)
        labels = np.digitize(data_flat, centroids[:-1]) - 1
        labels = np.clip(labels, 0, n_clusters - 1)
        return centroids, labels

    # Use FAISS for k-means
    data_flat = data.flatten() if data.ndim > 1 else data
    data_2d = data_flat.reshape(-1, 1).astype(np.float32)

    if FAISS_AVAILABLE:
        import faiss  # type: ignore

        kmeans = faiss.Kmeans(
            d=data_2d.shape[1], k=n_clusters, niter=iterations, verbose=False, gpu=False
        )
        kmeans.train(data_2d)
        centroids = kmeans.centroids.flatten()  # type: ignore
        _, labels = kmeans.index.search(data_2d, 1)  # type: ignore
        labels = labels.flatten().astype(np.int32)
    else:
        # Fallback to simple uniform quantization
        centroids = np.linspace(data_flat.min(), data_flat.max(), n_clusters).astype(np.float32)
        labels = np.digitize(data_flat, centroids[:-1]).astype(np.int32) - 1
        labels = np.clip(labels, 0, n_clusters - 1)

    # Sort centroids from smallest to largest
    sort_indices = np.argsort(centroids)
    centroids = centroids[sort_indices]

    # Create inverse mapping for labels
    inv_sort = np.empty_like(sort_indices)
    inv_sort[sort_indices] = np.arange(len(sort_indices))
    labels = inv_sort[labels]

    return centroids, labels


def write_webp_image(filename: str, data: np.ndarray, width: int, height: int) -> None:
    """Write RGBA data as lossless WebP image."""
    if data.dtype != np.uint8:
        raise ValueError("Data must be uint8")

    if data.size != width * height * 4:
        raise ValueError(f"Data size {data.size} doesn't match dimensions {width}x{height}x4")

    # Create PIL image from RGBA data
    img = Image.fromarray(data.reshape(height, width, 4), mode="RGBA")
    img.save(filename, format="webp", lossless=True, quality=100, method=6, exact=True)


def run_compression(
    output_path: str, splats: Dict[str, torch.Tensor], iterations: int = 10
) -> None:
    """
    Compress Gaussian splats to SOG format.

    Args:
        output_path: Output .sog file path (will be created as zip)
        splats: Dictionary with keys: 'means', 'opacities', 'scales', 'quats', 'sh0'
        iterations: K-means iterations (default: 10)
    """
    print(f"Compressing {len(splats['means'])} Gaussians to SOG format...")

    # Extract data
    means = splats["means"].cpu().numpy()  # (N, 3)
    opacities = splats["opacities"].cpu().numpy()  # (N,)
    scales = splats["scales"].cpu().numpy()  # (N, 3)
    quats = splats["quats"].cpu().numpy()  # (N, 4)
    sh0 = splats["sh0"].cpu().numpy()  # (N, 1, 3)

    num_gaussians = len(means)
    print(f"Input: {num_gaussians} Gaussians")

    # Sort by Morton order
    indices = morton_order_sort(means)
    means = means[indices]
    opacities = opacities[indices]
    scales = scales[indices]
    quats = quats[indices]
    sh0 = sh0[indices]

    # Calculate texture dimensions (square, power of 2 aligned)
    side_len = int(math.ceil(math.sqrt(num_gaussians)))
    side_len = ((side_len + 3) // 4) * 4  # Align to multiple of 4
    width = height = side_len

    print(f"Texture dimensions: {width}x{height}")

    # Create temporary directory for uncompressed files
    temp_dir = output_path.replace(".sog", "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Write means (log-transformed, 16-bit split into 8-bit)
        means_log = np.array([[log_transform(x) for x in row] for row in means])
        means_min = means_log.min(axis=0)
        means_max = means_log.max(axis=0)
        means_range = means_max - means_min
        means_range = np.where(means_range == 0, 1, means_range)

        means_norm = (means_log - means_min) / means_range
        means_16bit = (means_norm * 65535).astype(np.uint16)

        # Split into low and high bytes
        means_l = np.zeros((height, width, 4), dtype=np.uint8)
        means_u = np.zeros((height, width, 4), dtype=np.uint8)

        for i in range(num_gaussians):
            y, x = divmod(i, width)
            val = means_16bit[i]
            means_l[y, x] = [val[0] & 0xFF, val[1] & 0xFF, val[2] & 0xFF, 255]
            means_u[y, x] = [(val[0] >> 8) & 0xFF, (val[1] >> 8) & 0xFF, (val[2] >> 8) & 0xFF, 255]

        write_webp_image(os.path.join(temp_dir, "means_l.webp"), means_l.flatten(), width, height)
        write_webp_image(os.path.join(temp_dir, "means_u.webp"), means_u.flatten(), width, height)

        # Write quaternions (packed into RGBA)
        quats_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(num_gaussians):
            y, x = divmod(i, width)
            rgba, _ = pack_quaternion(quats[i])
            quats_rgba[y, x] = rgba

        write_webp_image(os.path.join(temp_dir, "quats.webp"), quats_rgba.flatten(), width, height)

        # Write scales (k-means clustered)
        # Flatten to (N*3, 1) to cluster all components together
        scales_centroids, scales_labels = kmeans_1d(scales.reshape(-1, 3), 256, iterations)

        # Reshape labels back to (N, 3) to access x,y,z labels
        scales_labels_3d = scales_labels.reshape(-1, 3)

        scales_data = np.zeros((height, width, 4), dtype=np.uint8)

        for i in range(num_gaussians):
            y, x = divmod(i, width)
            # Store indices for x, y, z in R, G, B channels
            # Alpha is unused (255)
            scales_data[y, x] = [
                scales_labels_3d[i, 0],
                scales_labels_3d[i, 1],
                scales_labels_3d[i, 2],
                255,
            ]

        write_webp_image(
            os.path.join(temp_dir, "scales.webp"), scales_data.flatten(), width, height
        )

        # Write colors + opacity (sh0 + opacity, k-means clustered)
        colors = sh0.reshape(-1, 3)  # (N, 3)
        colors_centroids, colors_labels = kmeans_1d(colors, 256, iterations)

        # Reshape labels back to (N, 3)
        colors_labels_3d = colors_labels.reshape(-1, 3)

        # Add opacity channel
        opacity_norm = np.array([sigmoid(float(o)) for o in opacities])
        opacity_8bit = (opacity_norm * 255).astype(np.uint8)

        sh0_data = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(num_gaussians):
            y, x = divmod(i, width)
            # Store indices for R, G, B in R, G, B channels
            # Store opacity in Alpha channel
            sh0_data[y, x] = [
                colors_labels_3d[i, 0],
                colors_labels_3d[i, 1],
                colors_labels_3d[i, 2],
                opacity_8bit[i],
            ]

        write_webp_image(os.path.join(temp_dir, "sh0.webp"), sh0_data.flatten(), width, height)

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
