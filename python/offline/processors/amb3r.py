"""AMB3R + FastGS processor for high-quality Gaussian Splatting."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .base import GaussianProcessor
from .fastgs import FastGSProcessor

if TYPE_CHECKING:
    from ..types import GaussianFrame

logger = logging.getLogger(__name__)


class AMB3RFastGSProcessor(GaussianProcessor):
    """AMB3R + FastGS processor.

    Uses AMB3R for metric-scale 3D reconstruction (depth, poses, point cloud)
    then feeds the result as a synthetic COLMAP dataset to FastGS for
    high-quality Gaussian Splatting training.

    Requirements:
    - AMB3R dependencies: torch, pytorch3d, torch-scatter, spconv, xformers, timm
    - AMB3R checkpoint (auto-downloaded from HuggingFace)
    - FastGS installed and in PYTHONPATH
    - CUDA-capable GPU

    Paper: https://arxiv.org/abs/... (AMB3R: CVPR 2026)
    Repo: https://github.com/HengyiWang/amb3r
    """

    def __init__(
        self,
        device: str = "cuda",
        ckpt_path: str | None = None,
        fastgs_iterations: int = 30_000,
        fastgs_path: str | None = None,
        conf_thresh_percentile: float = 40.0,
        max_points_per_frame: int = 5000,
        process_res: tuple[int, int] = (518, 392),
        window_size: int = 8,
    ):
        """Initialize AMB3R + FastGS processor.

        Args:
            device: PyTorch device
            ckpt_path: Path to AMB3R checkpoint. If None, downloads from HuggingFace.
            fastgs_iterations: Number of FastGS training iterations
            fastgs_path: Path to FastGS installation
            conf_thresh_percentile: Percentile for confidence thresholding
            max_points_per_frame: Max 3D points per frame for COLMAP export
            process_res: (width, height) for AMB3R input resolution
            window_size: Number of frames to process in each AMB3R window
        """
        self.device = device
        self.ckpt_path = ckpt_path
        self.fastgs_iterations = fastgs_iterations
        self.fastgs_path = fastgs_path
        self.conf_thresh_percentile = conf_thresh_percentile
        self.max_points_per_frame = max_points_per_frame
        self.process_res = process_res
        self.window_size = window_size
        self._model = None

    def _load_model(self):
        """Lazy load the AMB3R model."""
        if self._model is not None:
            return self._model

        import torch

        # Add amb3r to path
        amb3r_dir = Path(__file__).parent.parent.parent / "amb3r"
        if str(amb3r_dir) not in sys.path:
            sys.path.insert(0, str(amb3r_dir))

        from amb3r.model import AMB3R

        model = AMB3R(device=self.device, metric_scale=True)

        if self.ckpt_path is not None:
            model.load_weights(self.ckpt_path)
        else:
            # Try to download from HuggingFace
            try:
                from huggingface_hub import hf_hub_download

                ckpt_path = hf_hub_download(
                    repo_id="outsung/amb3r-checkpoint",
                    filename="amb3r.pt",
                )
                model.load_weights(ckpt_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load AMB3R checkpoint: {e}\n"
                    f"Please download manually and pass --ckpt-path, "
                    f"or install huggingface_hub."
                ) from e

        model = model.to(self.device)
        model.eval()
        self._model = model
        logger.info("AMB3R model loaded successfully")
        return model

    def _load_and_preprocess_frames(
        self, frame_paths: list[Path]
    ) -> tuple[np.ndarray, list[str]]:
        """Load frames and preprocess for AMB3R input.

        Args:
            frame_paths: List of image file paths

        Returns:
            images: (T, 3, H, W) array in [-1, 1] range
            image_paths: List of string paths
        """
        from PIL import Image

        target_w, target_h = self.process_res
        images = []

        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((target_w, target_h), Image.BILINEAR)
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = (img_np - 0.5) * 2.0  # Normalize to [-1, 1]
            images.append(img_tensor.transpose(2, 0, 1))  # HWC -> CHW

        images = np.stack(images, axis=0)  # (T, 3, H, W)
        image_paths = [str(p) for p in frame_paths]

        return images, image_paths

    def _run_amb3r_inference(
        self, frame_paths: list[Path]
    ) -> dict:
        """Run AMB3R inference on frames.

        Returns dict with numpy arrays:
            world_points: (T, H, W, 3)
            confidence: (T, H, W)
            extrinsics_w2c: (T, 3, 4)
            intrinsics: (T, 3, 3)
        """
        import torch

        model = self._load_model()
        images_np, image_paths = self._load_and_preprocess_frames(frame_paths)

        # Process in windows
        num_frames = len(frame_paths)
        all_world_points = []
        all_confidence = []
        all_extrinsics = []
        all_intrinsics = []

        for start in range(0, num_frames, self.window_size):
            end = min(start + self.window_size, num_frames)
            window_images = images_np[start:end]

            # AMB3R expects (B, T, 3, H, W)
            images_tensor = torch.from_numpy(window_images).unsqueeze(0).to(self.device)
            frames = {"images": images_tensor}

            with torch.inference_mode():
                res_all = model.forward(frames, iters=1)
                res = res_all[-1]

            # Extract outputs (squeeze batch dim)
            wp = res["world_points"][0].cpu().numpy()  # (T, H, W, 3)
            conf = res["world_points_conf"][0].cpu().numpy()  # (T, H, W, 1)
            # extrinsic is w2c (3, 4)
            ext = res["extrinsic"][0].cpu().numpy()  # (T, 3, 4)
            intr = res["intrinsic"][0].cpu().numpy()  # (T, 3, 3)

            all_world_points.append(wp)
            all_confidence.append(conf[..., 0])  # squeeze last dim
            all_extrinsics.append(ext)
            all_intrinsics.append(intr)

            logger.info(
                f"AMB3R window {start}-{end}: "
                f"conf={conf.mean():.2f}, "
                f"points range=[{wp.min():.2f}, {wp.max():.2f}]"
            )

        return {
            "world_points": np.concatenate(all_world_points, axis=0),
            "confidence": np.concatenate(all_confidence, axis=0),
            "extrinsics_w2c": np.concatenate(all_extrinsics, axis=0),
            "intrinsics": np.concatenate(all_intrinsics, axis=0),
        }

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames with AMB3R + FastGS pipeline.

        1. Run AMB3R inference to get 3D points, poses, intrinsics
        2. Export as synthetic COLMAP dataset
        3. Run FastGS training on the COLMAP dataset
        4. Return trained Gaussians

        Args:
            frame_paths: List of frame image paths
            timestamps_ms: Corresponding timestamps in milliseconds
            per_frame: Ignored — always trains single model

        Returns:
            List of GaussianFrame objects (single frame with trained Gaussians)
        """
        if not frame_paths:
            return []

        logger.info(f"AMB3R+FastGS processing {len(frame_paths)} frames")

        # Step 1: AMB3R inference
        logger.info("Step 1: Running AMB3R inference...")
        amb3r_output = self._run_amb3r_inference(frame_paths)

        # Step 2: Create synthetic COLMAP dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            colmap_dir = temp_path / "sparse" / "0"
            images_dir = temp_path / "images"
            images_dir.mkdir(parents=True)
            colmap_dir.mkdir(parents=True)

            # Symlink images into the dataset
            for fp in frame_paths:
                dst = images_dir / Path(fp).name
                if not dst.exists():
                    os.symlink(str(Path(fp).resolve()), str(dst))

            # Export COLMAP reconstruction
            logger.info("Step 2: Exporting synthetic COLMAP dataset...")
            from ..colmap_export import export_amb3r_to_colmap

            export_amb3r_to_colmap(
                world_points=amb3r_output["world_points"],
                confidence=amb3r_output["confidence"],
                extrinsics_w2c=amb3r_output["extrinsics_w2c"],
                intrinsics=amb3r_output["intrinsics"],
                image_paths=[str(p) for p in frame_paths],
                export_dir=str(colmap_dir),
                conf_thresh_percentile=self.conf_thresh_percentile,
                max_points_per_frame=self.max_points_per_frame,
            )

            # Step 3: Run FastGS
            logger.info("Step 3: Running FastGS training...")
            fastgs = FastGSProcessor(
                device=self.device,
                iterations=self.fastgs_iterations,
                source_path=str(temp_path),
                fastgs_path=self.fastgs_path,
            )

            # FastGS needs frame paths within the dataset's images/ dir
            dataset_frame_paths = [images_dir / Path(fp).name for fp in frame_paths]
            result = fastgs.process_frames(
                dataset_frame_paths, timestamps_ms, per_frame=False
            )

        logger.info(f"AMB3R+FastGS completed, got {len(result)} GaussianFrame(s)")
        return result

    def __repr__(self) -> str:
        return (
            f"AMB3RFastGSProcessor(device={self.device}, "
            f"fastgs_iters={self.fastgs_iterations}, "
            f"window={self.window_size})"
        )
