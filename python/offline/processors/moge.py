"""MoGe Gaussian Processor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)


class MoGeGaussianProcessor(GaussianProcessor):
    """Process video frames to Gaussian splats using MoGe."""

    def __init__(
        self,
        model_id: str = "Ruicheng/moge-2-vitl-normal",
        device: str = "auto",
        debug_output_dir: Path | None = None,
        refine_boundaries: bool = False,
        boundary_min_angle: float = 3.0,
        use_multi_device: bool = False,
        device_spec: str = "auto",
    ):
        self.model_id = model_id
        self.model = None
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        self.refine_boundaries = refine_boundaries
        self.boundary_min_angle = boundary_min_angle
        self.use_multi_device = use_multi_device
        self.device_spec = device_spec
        self._worker_pool = None

        if not use_multi_device:
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
        else:
            self.device = None

    def _load_model(self):
        """Lazy load the MoGe model or create worker pool."""
        if self.use_multi_device:
            if self._worker_pool is not None:
                return

            from common.device_worker_pool import DeviceWorkerPool
            from backend.workers.moge_worker import MoGEDeviceWorker

            logger.info(f"Creating multi-device worker pool for {self.model_id}")
            logger.info(f"Device spec: {self.device_spec}")

            self._worker_pool = DeviceWorkerPool(
                worker_class=MoGEDeviceWorker,
                device_spec=self.device_spec,
                worker_kwargs={"model_id": self.model_id},
            )
            logger.info(f"Worker pool ready with devices: {self._worker_pool.devices}")
            return

        if self.model is not None:
            return

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

        version = "v2" if "moge-2" in self.model_id else "v1"

        model = import_model_class_by_version(version).from_pretrained(self.model_id)

        assert self.device is not None
        self.model = model.to(self.device).eval()
        self.dtype = torch.float32
        logger.info("MoGe model ready")

    def _export_debug_outputs(
        self,
        frame_idx: int,
        frame_name: str,
        points: np.ndarray,
        normals: np.ndarray,
        mask_valid: np.ndarray,
        img_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> None:
        """Export debug PLY point cloud and GLB textured mesh."""
        if self.debug_output_dir is None:
            return

        try:
            import utils3d
            from moge.utils.io import save_glb, save_ply
        except ImportError as e:
            logger.warning(f"Could not import debug export dependencies: {e}")
            return

        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        H, W = mask_valid.shape

        mask_cleaned = mask_valid & ~utils3d.np.depth_map_edge(depth=depth, rtol=0.04)  # type: ignore[misc]

        # Build mesh from depth map
        if normals is not None and "normal" in dir(utils3d.np):
            faces, vertices, vertex_colors, vertex_uvs, vertex_normals = (
                utils3d.np.build_mesh_from_map(
                    points,
                    img_rgb.astype(np.float32) / 255,
                    utils3d.np.uv_map(H, W),
                    normals,
                    mask=mask_cleaned,
                    tri=True,
                )
            )
        else:
            faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
                points,
                img_rgb.astype(np.float32) / 255,
                utils3d.np.uv_map(H, W),
                mask=mask_cleaned,
                tri=True,
            )
            vertex_normals = None

        # OpenGL coordinate conventions: x right, y up, z backward
        vertices_gl = vertices * [1, -1, -1]
        vertex_uvs_gl = vertex_uvs * [1, -1] + [0, 1]
        if vertex_normals is not None:
            vertex_normals_gl = vertex_normals * [1, -1, -1]
        else:
            vertex_normals_gl = None

        # Export GLB (textured mesh)
        glb_path = self.debug_output_dir / f"{frame_name}_mesh.glb"
        save_glb(glb_path, vertices_gl, faces, vertex_uvs_gl, img_rgb, vertex_normals_gl)
        logger.info(f"Exported debug mesh: {glb_path}")

        # Export PLY (point cloud with vertex colors)
        ply_path = self.debug_output_dir / f"{frame_name}_pointcloud.ply"
        empty_faces = np.zeros((0, 3), dtype=np.int32)
        save_ply(ply_path, vertices_gl, empty_faces, vertex_colors, vertex_normals_gl)
        logger.info(f"Exported debug point cloud: {ply_path}")

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

            model = self.model
            assert model is not None

            with torch.no_grad():
                output = model.infer(img_tensor, resolution_level=9)

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

            # Verify shape alignment between MoGe output and input image
            H, W, _ = points.shape
            assert points.shape[:2] == mask_valid.shape, "MoGe output points/mask shape mismatch"
            assert H == img_rgb.shape[0] and W == img_rgb.shape[1], (
                f"MoGe output resolution {H}x{W} differs from input {img_rgb.shape[0]}x{img_rgb.shape[1]}"
            )

            # Extract depth and intrinsics for debug export
            depth = points[:, :, 2]
            intrinsics = output["intrinsics"].cpu().numpy()

            if self.refine_boundaries:
                from ..boundary_refinement import refine_depth_at_boundaries

                logger.info(f"Refining boundaries (min_angle={self.boundary_min_angle}°)...")
                points, boundary_mask = refine_depth_at_boundaries(
                    points=points,
                    intrinsics=intrinsics,
                    min_angle=self.boundary_min_angle,
                )
                depth = points[:, :, 2]

            # Export debug outputs (PLY point cloud and GLB mesh) if enabled
            self._export_debug_outputs(
                frame_idx=i,
                frame_name=path.stem,
                points=points,
                normals=normals,
                mask_valid=mask_valid > 0.5,
                img_rgb=img_rgb,
                depth=depth,
                intrinsics=intrinsics,
            )

            # Build valid mask from MoGe confidence and optionally filter dark pixels
            valid_mask = mask_valid > 0.5
            if remove_black_splats:
                brightness = np.max(img_rgb, axis=2)
                valid_mask = valid_mask & (brightness > 5)

            points_flat = points[valid_mask]
            normals_flat = normals[valid_mask]
            colors_flat = img_rgb[valid_mask] / 255.0

            # Filter out points with non-positive depth (behind camera)
            depths_flat = points_flat[:, 2]
            valid_depth = depths_flat > 0.0
            if not np.all(valid_depth):
                points_flat = points_flat[valid_depth]
                normals_flat = normals_flat[valid_depth]
                colors_flat = colors_flat[valid_depth]
                depths_flat = depths_flat[valid_depth]

            num_points = len(points_flat)
            if num_points == 0:
                logger.warning(f"Frame {i}: no valid points after filtering")
                continue

            # Extract and validate focal length from intrinsics
            intrinsics = output["intrinsics"].cpu().numpy()
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            f_avg = (fx + fy) / 2.0
            if not np.isfinite(f_avg) or f_avg <= 1e-6:
                logger.warning(f"Invalid focal length: fx={fx}, fy={fy}, defaulting to 500")
                f_avg = 500.0

            # Compute scales using metric depth and intrinsics
            # Heuristic: splat size should cover ~2 pixels to avoid holes
            # Scale ~ 2 * Depth / Focal_Length
            pixel_scale = 2.0

            # Clamp depths to avoid ultra-tiny splats
            depths_safe = np.maximum(depths_flat, 0.1)
            metric_scales = (depths_safe / f_avg) * pixel_scale

            # Expand to (N, 3): flat disks aligned with normal
            # x, y = metric_scale, z = metric_scale * 0.2 (thin)
            scales_flat_linear = np.stack(
                [metric_scales, metric_scales, metric_scales * 0.2], axis=1
            )
            scales_flat = np.log(np.maximum(scales_flat_linear, 1e-8))

            # Normalize normals before quaternion construction
            norm_len = np.linalg.norm(normals_flat, axis=1, keepdims=True)
            degenerate = (norm_len < 1e-6).flatten()
            normals_unit = normals_flat / np.maximum(norm_len, 1e-6)
            normals_unit[degenerate] = np.array([0.0, 0.0, 1.0], dtype=normals_unit.dtype)

            # Compute rotations via shortest-arc quaternion from (0,0,1) to normal
            # q = (1 + dot(u, v), cross(u, v)) where u = (0,0,1), v = normal
            # dot = nz, cross = (-ny, nx, 0)
            nx = normals_unit[:, 0]
            ny = normals_unit[:, 1]
            nz = normals_unit[:, 2]

            qw = 1.0 + nz
            qx = -ny
            qy = nx
            qz = np.zeros_like(nx)

            # Handle antiparallel case (nz ≈ -1): rotate 180° around X
            antiparallel = qw < 1e-3
            qw[antiparallel] = 0.0
            qx[antiparallel] = 1.0
            qy[antiparallel] = 0.0
            qz[antiparallel] = 0.0

            quats = np.stack([qw, qx, qy, qz], axis=1)
            quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
            quats = quats / (quat_norm + 1e-8)

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
