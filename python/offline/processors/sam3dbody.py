"""SAM 3D Body Gaussian Processor for human scene reconstruction."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..types import GaussianFrame

from .base import GaussianProcessor

logger = logging.getLogger(__name__)

# Add vendored SAM 3D Body to Python path
# Path(__file__).parent.parent.parent resolves to python/ directory from processors/
sam3d_dir = Path(__file__).parent.parent.parent / "sam_3d_body_git"
if sam3d_dir.exists():
    # Add both main package and tools directory
    sys.path.insert(0, str(sam3d_dir))
    sys.path.insert(1, str(sam3d_dir / "sam_3d_body"))
    logger.info(f"Added SAM 3D Body to path: {sam3d_dir}")
else:
    logger.warning(f"SAM 3D Body directory not found: {sam3d_dir}")


class Sam3DBodyProcessor(GaussianProcessor):
    """Process video frames to Gaussian splats using SAM 3D Body human mesh recovery.

    This processor specializes in human scenes, using Meta's SAM 3D Body to recover
    parametric human meshes. Since SAM 3D Body outputs untextured meshes, we
    sample colors from the original images by projecting 3D vertices back to 2D.

    For best results, use this processor with human subjects and combine with depth-based
    processors (DA3/MoGe/SHARP) for background using HybridProcessor.
    """

    def __init__(
        self,
        hf_repo_id: str = "facebook/sam-3d-body-vith",
        device: str = "cuda",
        points_per_person: int = 10000,
        bbox_threshold: float = 0.8,
        use_mask: bool = False,
    ):
        """Initialize SAM 3D Body processor.

        Args:
            hf_repo_id: HuggingFace repository ID for the model
                      (default: "facebook/sam-3d-body-vith")
                      Alternative: "facebook/sam-3d-body-dinov3"
            device: Device to run inference on (default: "cuda")
            points_per_person: Number of points to sample per detected person
            bbox_threshold: Human detection confidence threshold (0.0-1.0)
            use_mask: Whether to use segmentation masks for refinement
        """
        self.hf_repo_id = hf_repo_id
        self.device = device
        self.points_per_person = points_per_person
        self.bbox_threshold = bbox_threshold
        self.use_mask = use_mask
        self.model = None
        self.estimator = None
        self.faces = None

    def _load_model(self):
        """Lazy load SAM 3D Body model."""
        if self.model is not None:
            return

        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"Loading SAM 3D Body model: {self.hf_repo_id}")
        logger.info("Model files are cached in ~/.cache/huggingface/hub/")

        try:
            # Import SAM 3D Body from vendored location
            from sam_3d_body.sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
            from sam_3d_body.tools.build_detector import HumanDetector

            # Load model
            model, model_cfg = load_sam_3d_body_hf(self.hf_repo_id, device=self.device)
            self.model = model
            self.faces = model_cfg.faces

            # Load human detector (ViTDet for consistency with paper)
            human_detector = HumanDetector(name="vitdet", device=self.device, path="")

            # Create estimator
            self.estimator = SAM3DBodyEstimator(
                sam_3d_body_model=model,
                model_cfg=model_cfg,
                human_detector=human_detector,
                human_segmentor=None,  # Optional: could load SAM2
                fov_estimator=None,  # Optional: could load MoGe2
            )

            logger.info("SAM 3D Body model loaded successfully")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Points per person: {self.points_per_person}")

        except ImportError as e:
            logger.error(f"Failed to import SAM 3D Body: {e}")
            logger.error("Ensure SAM 3D Body is vendored at python/sam_3d_body_git/")
            logger.error("See python/SAM3D_INSTALL.md for setup instructions")
            raise

    def _sample_points_on_mesh(
        self, vertices: np.ndarray, num_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample points uniformly on mesh surface using trimesh.

        Args:
            vertices: (N, 3) mesh vertices
            num_points: Number of points to sample

        Returns:
            points: (num_points, 3) sampled point positions
            normals: (num_points, 3) surface normals at sampled points
        """
        import trimesh

        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)

        # Sample points uniformly on surface
        points, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)

        # Get normals at sampled points
        normals = mesh.face_normals[face_indices]

        return points, normals

    def _sample_colors_from_image(
        self,
        image: np.ndarray,
        points_3d: np.ndarray,
        focal_length: float,
        cam_translation: np.ndarray,
    ) -> np.ndarray:
        """Sample colors from image by projecting 3D points back to 2D.

        This provides texture information to otherwise untextured SAM 3D Body meshes.

        Args:
            image: Input image (H, W, 3) RGB
            points_3d: (N, 3) 3D point positions in camera space
            focal_length: Camera focal length (in pixels)
            cam_translation: Camera translation vector (3,)

        Returns:
            colors: (N, 3) RGB colors for each point
        """
        height, width = image.shape[:2]
        num_points = points_3d.shape[0]

        # Perspective projection: 2D = (focal * 3D) / (Z + translation_z)
        z = points_3d[:, 2] + cam_translation[2]

        # Avoid division by zero or negative Z (behind camera)
        z = np.maximum(z, 1e-6)

        # Project to 2D
        x_2d = focal_length * points_3d[:, 0] / z + width / 2
        y_2d = focal_length * points_3d[:, 1] / z + height / 2

        # Round to integer pixel coordinates
        x_2d = np.round(x_2d).astype(int)
        y_2d = np.round(y_2d).astype(int)

        # Clamp to image bounds
        x_2d = np.clip(x_2d, 0, width - 1)
        y_2d = np.clip(y_2d, 0, height - 1)

        # Sample colors
        colors = image[y_2d, x_2d].astype(np.float32) / 255.0

        return colors

    def _convert_to_gaussian_params(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert point cloud data to Gaussian splat parameters.

        Args:
            points: (N, 3) point positions
            normals: (N, 3) point normals
            colors: (N, 3) RGB colors (0-1 range)

        Returns:
            means: (N, 3) positions
            scales: (N, 3) log-scale
            rotations: (N, 4) quaternion wxyz
            colors_out: (N, 3) SH DC term
            opacities: (N,) logit opacities
        """
        n = len(points)

        # Means: direct from points
        means = points.astype(np.float32)

        # Scales: estimate from normal variance or fixed scale
        # For now, use a fixed scale based on camera distance
        scales = np.full((n, 3), -2.0, dtype=np.float32)  # log scale

        # Rotations: align with normals using shortest rotation
        # Convert normal to quaternion (z-axis aligned)
        z_axis = np.array([0.0, 0.0, 1.0])
        rotations = np.zeros((n, 4), dtype=np.float32)

        for i in range(n):
            # Compute quaternion to rotate z_axis to normal
            normal = normals[i] / (np.linalg.norm(normals[i]) + 1e-6)
            v = np.cross(z_axis, normal)
            s = np.sqrt(2 * (1 + normal[2]))
            rotations[i, 0] = 0.25 * s  # w
            rotations[i, 1] = v[2] / s  # x
            rotations[i, 2] = v[1] / s  # y
            rotations[i, 3] = v[0] / s  # z

        # Colors: convert RGB to SH DC (f_dc)
        # SH DC is just the RGB values scaled by some factor
        # Following 3DGS paper: f_dc = 0.5 * RGB
        colors_out = (colors * 0.5).astype(np.float32)

        # Opacities: fixed high opacity for mesh-based points
        opacities = np.full(n, 2.0, dtype=np.float32)  # logit sigmoid(0.88)

        return means, scales, rotations, colors_out, opacities

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames to extract human Gaussians using SAM 3D Body.

        Args:
            frame_paths: List of frame image paths
            timestamps_ms: Corresponding timestamps in milliseconds
            per_frame: If True, process each frame individually.
                      If False, process all together (mesh-only, so same).

        Returns:
            List of GaussianFrame objects (one per frame)
        """
        from ..types import GaussianFrame

        self._load_model()

        if len(frame_paths) != len(timestamps_ms):
            raise ValueError(
                f"frame_paths ({len(frame_paths)}) and timestamps_ms "
                f"({len(timestamps_ms)}) must have same length"
            )

        frames_data = []

        for frame_idx, (frame_path, timestamp_ms) in enumerate(zip(frame_paths, timestamps_ms)):
            logger.info(f"Processing frame {frame_idx + 1}/{len(frame_paths)}")

            # Load image
            img_bgr = cv2.imread(str(frame_path))
            if img_bgr is None:
                logger.warning(f"Failed to load image: {frame_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Run SAM 3D Body inference
            outputs = self.estimator.process_one_image(
                str(frame_path),
                bbox_thr=self.bbox_threshold,
                use_mask=self.use_mask,
            )

            if not outputs:
                logger.warning(f"No humans detected in {frame_path.name}")
                frames_data.append(
                    GaussianFrame(
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        means=np.empty((0, 3), dtype=np.float32),
                        scales=np.empty((0, 3), dtype=np.float32),
                        rotations=np.empty((0, 4), dtype=np.float32),
                        colors=np.empty((0, 3), dtype=np.float32),
                        opacities=np.empty(0, dtype=np.float32),
                    )
                )
                continue

            logger.info(f"  Detected {len(outputs)} human(s) in frame {frame_idx + 1}")

            # Process each detected person
            all_means = []
            all_scales = []
            all_rotations = []
            all_colors = []
            all_opacities = []

            for person_id, person_output in enumerate(outputs):
                logger.info(f"  Processing person {person_id + 1}")

                # Extract mesh data
                vertices = person_output["pred_vertices"]  # (N_vertices, 3)
                focal_length = person_output["focal_length"]
                cam_translation = person_output["pred_cam_t"]

                # Sample points on mesh surface
                points, normals = self._sample_points_on_mesh(vertices, self.points_per_person)

                # Sample colors from original image
                colors = self._sample_colors_from_image(
                    img_rgb, points, focal_length, cam_translation
                )

                # Convert to Gaussian parameters
                means, scales, rotations, colors_out, opacities = self._convert_to_gaussian_params(
                    points, normals, colors
                )

                all_means.append(means)
                all_scales.append(scales)
                all_rotations.append(rotations)
                all_colors.append(colors_out)
                all_opacities.append(opacities)

            # Combine all people into single frame
            if all_means:
                combined_means = np.vstack(all_means)
                combined_scales = np.vstack(all_scales)
                combined_rotations = np.vstack(all_rotations)
                combined_colors = np.vstack(all_colors)
                combined_opacities = np.hstack(all_opacities)

                logger.info(f"  Total Gaussians: {len(combined_means)} ({len(all_means)} people)")
            else:
                combined_means = np.empty((0, 3), dtype=np.float32)
                combined_scales = np.empty((0, 3), dtype=np.float32)
                combined_rotations = np.empty((0, 4), dtype=np.float32)
                combined_colors = np.empty((0, 3), dtype=np.float32)
                combined_opacities = np.empty(0, dtype=np.float32)

            # Create GaussianFrame
            frame = GaussianFrame(
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                means=combined_means,
                scales=combined_scales,
                rotations=combined_rotations,
                colors=combined_colors,
                opacities=combined_opacities,
            )

            frames_data.append(frame)

        logger.info(f"Processed {len(frames_data)} frames successfully")

        return frames_data
