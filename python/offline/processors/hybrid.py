"""Hybrid processor combining SAM 3D Body (humans) + depth models (background)."""

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
    sys.path.insert(1, str(sam3d_dir / "tools"))
    logger.info(f"Added SAM 3D Body to path: {sam3d_dir}")
else:
    logger.warning(f"SAM 3D Body directory not found: {sam3d_dir}")


class HybridProcessor(GaussianProcessor):
    """Combine SAM 3D Body for humans with depth models for background.

    This processor uses Meta's SAM 3D Body for accurate human mesh recovery,
    while using depth-based models (DA3/MoGe/SHARP) for the rest of the scene.
    The two approaches are complementary:

    - SAM 3D Body: Structured human meshes, handles occlusions, pose-accurate
    - Depth models (DA3/MoGe/SHARP): Textured depth for background objects

    The processor creates human masks from SAM 3D Body outputs and removes
    overlapping depth-based Gaussians in human regions to avoid duplication.

    For best results on mixed human + background scenes.
    """

    def __init__(
        self,
        human_model_id: str = "facebook/sam-3d-body-vith",
        depth_model_id: str = "depth-anything/DA3-GIANT",
        depth_processor_type: str = "da3",  # "da3", "moge", "sharp", or "unisharp"
        device: str = "cuda",
        points_per_person: int = 10000,
        bbox_threshold: float = 0.8,
        depth_process_res: int = 518,
        remove_overlap: bool = True,
        overlap_threshold: float = 0.5,
    ):
        """Initialize hybrid processor.

        Args:
            human_model_id: HuggingFace repo for SAM 3D Body
            depth_model_id: Model ID for depth estimation
            depth_processor_type: Type of depth processor ("da3", "moge", "sharp", "unisharp")
            device: Device for inference
            points_per_person: Points to sample per detected person
            bbox_threshold: Human detection confidence threshold
            depth_process_res: Processing resolution for depth model
            remove_overlap: Whether to remove depth Gaussians overlapping with humans
            overlap_threshold: IoU threshold for overlap detection (0.0-1.0)
        """
        self.human_model_id = human_model_id
        self.depth_model_id = depth_model_id
        self.depth_processor_type = depth_processor_type.lower()
        self.device = device
        self.points_per_person = points_per_person
        self.bbox_threshold = bbox_threshold
        self.depth_process_res = depth_process_res
        self.remove_overlap = remove_overlap
        self.overlap_threshold = overlap_threshold

        self.human_processor = None
        self.depth_processor = None

    def _load_models(self):
        """Lazy load both processors."""
        if self.human_processor is not None and self.depth_processor is not None:
            return

        from .sam3dbody import Sam3DBodyProcessor

        # Load SAM 3D Body for humans
        logger.info("Loading SAM 3D Body processor for humans...")
        self.human_processor = Sam3DBodyProcessor(
            hf_repo_id=self.human_model_id,
            device=self.device,
            points_per_person=self.points_per_person,
            bbox_threshold=self.bbox_threshold,
            use_mask=False,  # Will use bbox-based masks
        )

        # Load depth processor for background
        logger.info(f"Loading {self.depth_processor_type} processor for background...")
        if self.depth_processor_type == "da3":
            from .da3 import DA3GaussianProcessor

            self.depth_processor = DA3GaussianProcessor(
                model_id=self.depth_model_id,
                device=self.device,
                process_res=self.depth_process_res,
            )
        elif self.depth_processor_type == "moge":
            from .moge import MoGeGaussianProcessor

            self.depth_processor = MoGeGaussianProcessor(
                model_id=self.depth_model_id,
                device=self.device,
                process_res=self.depth_process_res,
            )
        elif self.depth_processor_type == "unisharp":
            from .unisharp import UniSHARPGaussianProcessor

            checkpoint_path = self.depth_model_id
            camera_model = "pinhole"

            self.depth_processor = UniSHARPGaussianProcessor(
                checkpoint_path=checkpoint_path,
                device=self.device,
                camera_model=camera_model,
            )
        elif self.depth_processor_type == "sharp":
            from .sharp import SharpGaussianProcessor

            # Sharp expects model_path (custom) or vit_preset (built-in presets)
            # User might pass:
            #   - A preset name: "dinov2l16_384" (default), "dinov2g16", etc.
            #   - Or a custom path: "/path/to/model.pt"
            # We'll map simple IDs to presets or use as path
            model_path = self.depth_model_id
            vit_preset = "dinov2l16_384"  # Default preset

            # If it looks like a preset name, use it directly
            known_presets = ["dinov2", "dinov2g", "dinov2l", "dinov2b", "dinov2s16"]
            if self.depth_model_id.lower() in [p + "_" for p in known_presets]:
                vit_preset = self.depth_model_id
                model_path = None  # Let preset loading use default model

            self.depth_processor = SharpGaussianProcessor(
                model_path=model_path,
                device=self.device,
                vit_preset=vit_preset,
            )
        else:
            raise ValueError(
                f"Unknown depth processor type: {self.depth_processor_type}. "
                f"Must be 'da3', 'moge', 'sharp', or 'unisharp'"
            )

        logger.info("Both processors loaded successfully")

    def _create_human_mask(
        self,
        image: np.ndarray,
        outputs: list[dict],
    ) -> np.ndarray:
        """Create binary mask for human regions from SAM 3D Body outputs.

        Args:
            image: Input image (H, W)
            outputs: List of SAM 3D Body outputs (one per person)

        Returns:
            mask: Binary mask (H, W) where 1=human, 0=background
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for person_output in outputs:
            bbox = person_output["bbox"]
            x1, y1, x2, y2 = bbox.astype(int)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Add bounding box region to mask
            # In a full implementation, we could refine this with the
            # projected mesh silhouette for tighter boundaries
            mask[y1:y2, x1:x2] = 255

        return mask

    def _project_gaussians_to_2d(
        self,
        means: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Project 3D Gaussian centers to 2D for overlap detection.

        Simple orthographic projection assuming Gaussians are in camera space.

        Args:
            means: (N, 3) Gaussian centers in camera space
            image_shape: (height, width) of the image

        Returns:
            positions_2d: (N, 2) projected positions in image space
        """
        height, width = image_shape

        # Simple orthographic projection: x, y map directly to image coordinates
        # Assuming Gaussians are already in normalized camera space
        # This is a simplification - proper implementation would use focal length
        positions_2d = means[:, :2]

        # Normalize to image coordinates
        # For now, assume Gaussians are in [-1, 1] range for x, y
        x_2d = ((positions_2d[:, 0] + 1) * width / 2).astype(int)
        y_2d = ((positions_2d[:, 1] + 1) * height / 2).astype(int)

        # Clamp to bounds
        x_2d = np.clip(x_2d, 0, width - 1)
        y_2d = np.clip(y_2d, 0, height - 1)

        return np.column_stack([x_2d, y_2d])

    def _remove_overlapping_gaussians(
        self,
        depth_means: np.ndarray,
        depth_scales: np.ndarray,
        depth_rotations: np.ndarray,
        depth_colors: np.ndarray,
        depth_opacities: np.ndarray,
        human_mask: np.ndarray,
        human_means_2d: np.ndarray,
        human_radius: float = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Remove depth Gaussians that overlap with human regions.

        Args:
            depth_means: (N, 3) depth Gaussian centers
            depth_scales: (N, 3) depth Gaussian scales
            depth_rotations: (N, 4) depth Gaussian rotations
            depth_colors: (N, 3) depth Gaussian colors
            depth_opacities: (N,) depth Gaussian opacities
            human_mask: (H, W) binary mask for human regions
            human_means_2d: (M, 2) projected 2D positions of human Gaussians

        Returns:
            Filtered depth Gaussian arrays (excluding overlaps)
        """
        if depth_means.shape[0] == 0:
            return (
                depth_means,
                depth_scales,
                depth_rotations,
                depth_colors,
                depth_opacities,
            )

        # Project depth Gaussians to 2D
        depth_means_2d = self._project_gaussians_to_2d(depth_means, human_mask.shape)

        height, width = human_mask.shape

        # Filter out Gaussians that fall within human bounding boxes
        keep_mask = np.ones(len(depth_means), dtype=bool)

        for i, (x, y) in enumerate(depth_means_2d):
            # Check if this point is within any human's radius
            in_human_region = False

            for hx, hy in human_means_2d:
                # Compute distance to nearest human Gaussian center
                dist = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)

                if dist < human_radius:
                    in_human_region = True
                    break

            if in_human_region:
                keep_mask[i] = False

        # Filter arrays
        filtered_means = depth_means[keep_mask]
        filtered_scales = depth_scales[keep_mask]
        filtered_rotations = depth_rotations[keep_mask]
        filtered_colors = depth_colors[keep_mask]
        filtered_opacities = depth_opacities[keep_mask]

        logger.info(
            f"Removed {np.sum(~keep_mask)} depth Gaussians "
            f"overlapping with humans ({np.sum(keep_mask)} remaining)"
        )

        return (
            filtered_means,
            filtered_scales,
            filtered_rotations,
            filtered_colors,
            filtered_opacities,
        )

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames using hybrid approach (humans + background).

        Args:
            frame_paths: List of frame image paths
            timestamps_ms: Corresponding timestamps in milliseconds
            per_frame: If True, process each frame individually.

        Returns:
            List of GaussianFrame objects combining human and background Gaussians
        """
        from ..types import GaussianFrame

        self._load_models()

        if len(frame_paths) != len(timestamps_ms):
            raise ValueError(
                f"frame_paths ({len(frame_paths)}) and timestamps_ms "
                f"({len(timestamps_ms)}) must have same length"
            )

        frames_data = []

        for frame_idx, (frame_path, timestamp_ms) in enumerate(zip(frame_paths, timestamps_ms)):
            logger.info(f"Processing frame {frame_idx + 1}/{len(frame_paths)} (hybrid)")

            # Load image
            img_bgr = cv2.imread(str(frame_path))
            if img_bgr is None:
                logger.warning(f"Failed to load image: {frame_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width = img_rgb.shape[:2]

            # Step 1: Get human Gaussians from SAM 3D Body
            logger.info("  Step 1/3: Extracting human Gaussians...")
            human_frames = self.human_processor.process_frames(
                [frame_path], [timestamp_ms], per_frame=True
            )
            human_frame = human_frames[0]

            # Get SAM 3D Body outputs for mask creation
            # We need to re-run inference to get bounding boxes
            human_outputs = self.human_processor.estimator.process_one_image(
                str(frame_path),
                bbox_thr=self.bbox_threshold,
                use_mask=False,
            )

            # Create human mask
            human_mask = self._create_human_mask(img_rgb, human_outputs)

            # Step 2: Get full depth-based Gaussians
            logger.info("  Step 2/3: Extracting depth Gaussians (full scene)...")
            depth_frames = self.depth_processor.process_frames(
                [frame_path], [timestamp_ms], per_frame=True
            )
            depth_frame = depth_frames[0]

            # Step 3: Combine human and background Gaussians
            logger.info("  Step 3/3: Combining human and background Gaussians...")

            if self.remove_overlap and human_frame.n_gaussians > 0:
                # Remove depth Gaussians that overlap with humans
                # Get human Gaussian 2D positions
                human_means_2d = self._project_gaussians_to_2d(human_frame.means, (height, width))

                (
                    filtered_means,
                    filtered_scales,
                    filtered_rotations,
                    filtered_colors,
                    filtered_opacities,
                ) = self._remove_overlapping_gaussians(
                    depth_frame.means,
                    depth_frame.scales,
                    depth_frame.rotations,
                    depth_frame.colors,
                    depth_frame.opacities,
                    human_mask,
                    human_means_2d,
                )

                background_means = filtered_means
                background_scales = filtered_scales
                background_rotations = filtered_rotations
                background_colors = filtered_colors
                background_opacities = filtered_opacities
            else:
                # No overlap removal
                background_means = depth_frame.means
                background_scales = depth_frame.scales
                background_rotations = depth_frame.rotations
                background_colors = depth_frame.colors
                background_opacities = depth_frame.opacities

            # Combine human and background Gaussians
            combined_means = np.vstack([human_frame.means, background_means])
            combined_scales = np.vstack([human_frame.scales, background_scales])
            combined_rotations = np.vstack([human_frame.rotations, background_rotations])
            combined_colors = np.vstack([human_frame.colors, background_colors])
            combined_opacities = np.hstack([human_frame.opacities, background_opacities])

            logger.info(
                f"  Final result: {len(combined_means)} total Gaussians "
                f"({human_frame.n_gaussians} human + {len(background_means)} background)"
            )

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

        logger.info(f"Processed {len(frames_data)} frames successfully (hybrid)")

        return frames_data
