"""MoGe Gaussian Processor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2.
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-8))
    return rotation_matrix


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


class MoGeGaussianProcessor(GaussianProcessor):
    """Process video frames to Gaussian splats using MoGe."""

    def __init__(
        self,
        model_id: str = "Ruicheng/moge-2-vitl-normal",
        device: str = "auto",
        process_res: int = 518,
    ):
        self.model_id = model_id
        self.process_res = process_res
        self.model = None
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = "xpu"
            else:
                self.device = "cpu"
            logger.info(f"Auto-detected device: {self.device}")
        else:
            self.device = device

    def _load_model(self):
        """Lazy load the MoGe model."""
        if self.model is not None:
            return

        # Ensure MoGe is in path - use vendored version in python/moge
        # This file is in python/offline/processors/, so go up to python/
        moge_path = (Path(__file__).parent / ".." / ".." / "moge").resolve()
        if moge_path.exists() and str(moge_path) not in sys.path:
            logger.info(f"Adding {moge_path} to sys.path")
            sys.path.insert(0, str(moge_path))

        try:
            from moge.model import import_model_class_by_version
        except ImportError:
            logger.error("Could not import 'moge'. Please ensure it's vendored in 'python/moge'.")
            raise

        logger.info(f"Loading MoGe model: {self.model_id}")

        # Determine version from model_id
        version = "v2" if "moge-2" in self.model_id else "v1"

        import torch

        self.model = (
            import_model_class_by_version(version)
            .from_pretrained(self.model_id)
            .to(self.device)
            .eval()
        )
        self.dtype = torch.float32  # MoGe usually runs in fp32 or fp16
        logger.info("MoGe model ready")

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """Process frames using MoGe."""
        self._load_model()

        results = []
        import torch

        for i, (path, ts) in enumerate(zip(frame_paths, timestamps_ms)):
            logger.info(f"Processing frame {i + 1}/{len(frame_paths)}: {path.name}")

            # Load image
            img_cv = cv2.imread(str(path))
            if img_cv is None:
                logger.warning(f"Failed to read {path}")
                continue

            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            # Masking
            if masks_dir:
                mask = None
                if mask_first_frame or i > 0:
                    for ext in [path.suffix, ".png", ".jpg", ".jpeg"]:
                        candidate = masks_dir / f"{path.stem}{ext}"
                        if candidate.exists():
                            mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                            break

                if mask is not None:
                    # Apply mask
                    mask_bool = mask > 127
                    img_rgb[~mask_bool] = 0

            # Convert to tensor
            img_tensor = torch.tensor(
                img_rgb / 255.0, dtype=self.dtype, device=self.device
            ).permute(2, 0, 1)

            # Inference
            with torch.no_grad():
                # MoGe infer expects (C, H, W)
                output = self.model.infer(img_tensor, resolution_level=9)

            # Extract results
            points = output["points"].cpu().numpy()  # (H, W, 3)
            mask_valid = output["mask"].cpu().numpy()  # (H, W)

            if "normal" in output:
                normals = output["normal"].cpu().numpy()
            else:
                # Estimate normals from points if not provided (though v2-normal has them)
                # Fallback: simpler to just assume Z-up or compute from cross product of neighbors
                # For now, let's just use [0,0,1] if missing, but typical models have it
                normals = np.zeros_like(points)
                normals[..., 2] = 1.0

            # Flatten
            H, W, _ = points.shape

            # Subsample for point cloud (optional, or use all pixels)
            # Using all pixels might be too heavy (e.g. 1-2M points)
            # Let's use a stride or keep all? 3DGS usually handles 1M+ fine.
            # But let's mask invalid points

            valid_mask = mask_valid > 0.5
            if remove_black_splats:
                brightness = np.max(img_rgb, axis=2)
                valid_mask = valid_mask & (brightness > 5)

            points_flat = points[valid_mask]
            normals_flat = normals[valid_mask]
            colors_flat = img_rgb[valid_mask] / 255.0

            num_points = len(points_flat)
            if num_points == 0:
                continue

            # Compute scales using metric depth and intrinsics
            # Heuristic: splat size should cover ~2 pixels to avoid holes
            # Scale ~ 2 * Depth / Focal_Length

            # Extract focal length from intrinsics
            intrinsics = output["intrinsics"].cpu().numpy()
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            f_avg = (fx + fy) / 2.0

            # Depth is z coordinate of points
            depths_flat = points_flat[:, 2]

            # Base scale factor (tunable, 1.5-2.0 pixels usually good)
            pixel_scale = 2.0

            # Compute metric scale per point
            # Avoid division by zero or negative depths
            depths_safe = np.maximum(depths_flat, 0.1)
            metric_scales = (depths_safe / f_avg) * pixel_scale

            # Expand to (N, 3)
            # We want flat disks aligned with normal
            # x, y = metric_scale, z = metric_scale * 0.1 (thin)
            scales_flat_linear = np.stack(
                [metric_scales, metric_scales, metric_scales * 0.2], axis=1
            )
            scales_flat = np.log(np.maximum(scales_flat_linear, 1e-8))

            # Compute rotations
            # Vectorized normal to quaternion
            # Normal is (N, 3). Target is (0, 0, 1).
            # We can use a simplified "shortest arc" rotation.
            # q = (1 + dot(u, v), cross(u, v)). Normalized.
            # u = (0,0,1). v = normal.
            # dot = nz. cross = (-ny, nx, 0).
            # q = (1 + nz, -ny, nx, 0)
            # Then normalize.

            nz = normals_flat[:, 2]
            nx = normals_flat[:, 0]
            ny = normals_flat[:, 1]

            qw = 1.0 + nz
            qx = -ny
            qy = nx
            qz = np.zeros_like(nx)

            # Handle antiparallel case (nz = -1)
            # If nz is close to -1, we rotate 180 deg around X.
            # q = (0, 1, 0, 0)
            antiparallel = qw < 1e-6
            qw[antiparallel] = 0
            qx[antiparallel] = 1
            qy[antiparallel] = 0
            qz[antiparallel] = 0

            quats = np.stack([qw, qx, qy, qz], axis=1)
            norm = np.linalg.norm(quats, axis=1, keepdims=True)
            quats = quats / (norm + 1e-8)

            # Colors: SH DC (0.282...)
            # SH_C0 = 0.28209479177387814
            # f_dc = (rgb - 0.5) / SH_C0
            SH_C0 = 0.28209479177387814
            colors_sh = (colors_flat - 0.5) / SH_C0

            # Opacities
            opacities_flat = np.ones((num_points,), dtype=np.float32) * 10.0  # High logit -> ~1.0

            results.append(
                GaussianFrame(
                    frame_idx=i,
                    timestamp_ms=ts,
                    means=points_flat.astype(np.float32),
                    scales=scales_flat.astype(np.float32),
                    rotations=quats.astype(np.float32),
                    colors=colors_sh.astype(np.float32),
                    opacities=opacities_flat.astype(np.float32),
                )
            )

        return results
