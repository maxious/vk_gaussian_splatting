"""Gaussian Splatting generation utilities."""

from __future__ import annotations

import io
import logging
import numpy as np
import cv2
from PIL import Image

from backend.models.depth_model import get_depth_model
from offline.formats import write_gs_ply_header, write_gs_ply_points

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-1) to SH DC coefficients."""
    return (rgb - 0.5) / SH_C0


async def generate_gs_ply(image_bytes: bytes) -> bytes:
    """Generate a 3DGS PLY file from an input image.

    Args:
        image_bytes: Raw bytes of the image file (JPG, PNG, etc.)

    Returns:
        Bytes of the resulting PLY file.
    """
    # 1. Decode image
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image.shape

    # 2. Infer Depth
    depth_model = get_depth_model()
    # Use asynchronous inference
    prediction = await depth_model.infer_depth_async(image)
    depth = prediction.depth

    # 3. Unproject to 3D Points
    # Simple unprojection assuming 60 deg FOV if unknown
    fov_deg = 60.0
    fov_rad = np.radians(fov_deg)
    fx = fy = W / (2 * np.tan(fov_rad / 2))
    cx, cy = W / 2, H / 2

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Filter very small/large depths?
    valid_mask = depth > 0.1

    u = u[valid_mask]
    v = v[valid_mask]
    z = depth[valid_mask]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)

    # 4. Create GS Attributes
    colors_rgb = image[valid_mask].astype(np.float32) / 255.0
    colors_sh = rgb_to_sh(colors_rgb)

    # Opacity: near 1.0. Logit(0.99) ~ 4.6
    opacities = np.full((len(points), 1), 4.6, dtype=np.float32)

    # Scale: heuristic based on distance/pixel size
    # pixel_size at depth z ~ z / fx
    # We want gaussians to cover pixels.
    # Log scale required.
    base_scale = z / fx
    # sqrt(2) to cover diagonals? slightly larger to avoid holes
    scales_linear = np.stack([base_scale, base_scale, base_scale], axis=-1) * 1.5
    scales = np.log(scales_linear + 1e-6)

    # Rotation: Identity (w, x, y, z) = (1, 0, 0, 0)
    quaternions = np.zeros((len(points), 4), dtype=np.float32)
    quaternions[:, 0] = 1.0

    # 5. Write PLY
    with io.BytesIO() as f:
        write_gs_ply_header(f, len(points))
        write_gs_ply_points(f, points, colors_sh, scales, opacities, quaternions)
        return f.getvalue()
