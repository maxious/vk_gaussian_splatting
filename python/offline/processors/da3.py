"""DA3 Gaussian Processor."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)


class DA3GaussianProcessor(GaussianProcessor):
    """Process video frames to Gaussian splats using DA3."""

    def __init__(
        self,
        model_id: str = "depth-anything/DA3-GIANT",
        device: str = "cuda",
        process_res: int = 518,
    ):
        self.model_id = model_id
        self.device = device
        self.process_res = process_res
        self.model = None

    def _load_model(self):
        """Lazy load the DA3 model."""
        if self.model is not None:
            return

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for DA3 inference")

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            raise RuntimeError(
                "depth-anything-3 not installed. Run: uv pip install depth-anything-3"
            )

        logger.info(f"Loading model: {self.model_id} (this may take a while to download...)")
        logger.info("Model files are cached in ~/.cache/huggingface/hub/")

        import huggingface_hub

        huggingface_hub.logging.set_verbosity_info()

        self.model = DepthAnything3.from_pretrained(self.model_id)
        logger.info(f"Model loaded, moving to {self.device}...")
        self.model = self.model.to(self.device).eval()
        self.dtype = torch.float16
        logger.info("Model ready for inference")

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames to extract Gaussians.

        Args:
            frame_paths: List of frame image paths
            timestamps_ms: Corresponding timestamps in milliseconds
            per_frame: If True, process each frame individually for separate PLYs.
                      If False (default), process all together for merged Gaussians.

        Returns:
            List of GaussianFrame objects (one per frame if per_frame=True,
            otherwise one merged frame)
        """
        self._load_model()

        if per_frame:
            return self._process_frames_individually(frame_paths, timestamps_ms)
        else:
            return self._process_frames_merged(frame_paths, timestamps_ms)

    def _process_frames_merged(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
    ) -> list[GaussianFrame]:
        """Process all frames together, returning merged Gaussians."""
        import torch

        logger.info(f"Processing {len(frame_paths)} frames merged with infer_gs=True")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.dtype):
                images = [str(p) for p in frame_paths]

                predictions = self.model.inference(
                    images,
                    process_res=self.process_res,
                    ref_view_strategy="saddle_balanced",
                    infer_gs=True,
                )

        if predictions.gaussians is None:
            raise RuntimeError(
                "Model did not return Gaussians. "
                "Make sure you're using DA3-GIANT with infer_gs=True"
            )

        gaussians = predictions.gaussians

        means = gaussians.means[0].cpu().numpy()
        scales = gaussians.scales[0].cpu().numpy()

        # Necessary comment: Convert linear scales (0+) to log scales (-inf, +inf) for PLY format compatibility
        if np.all(scales > 0):
            scales = np.log(np.maximum(scales, 1e-10))

        rotations = gaussians.rotations[0].cpu().numpy()
        opacities = gaussians.opacities[0].cpu().numpy()

        harmonics = gaussians.harmonics[0].cpu().numpy()
        colors = harmonics[:, :, 0] if harmonics.ndim == 3 else harmonics

        if opacities.ndim == 2:
            opacities = opacities[:, 0]

        mid_ts = timestamps_ms[len(timestamps_ms) // 2] if timestamps_ms else 0

        return [
            GaussianFrame(
                frame_idx=0,
                timestamp_ms=mid_ts,
                means=means.astype(np.float32),
                scales=scales.astype(np.float32),
                rotations=rotations.astype(np.float32),
                colors=colors.astype(np.float32),
                opacities=opacities.astype(np.float32),
            )
        ]

    def _process_frames_individually(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
    ) -> list[GaussianFrame]:
        """Process each frame individually for per-frame PLY output."""
        import torch

        results = []

        for i, (frame_path, ts) in enumerate(zip(frame_paths, timestamps_ms)):
            logger.info(f"Processing frame {i + 1}/{len(frame_paths)}: {frame_path.name}")

            with torch.no_grad():
                with torch.autocast("cuda", dtype=self.dtype):
                    predictions = self.model.inference(
                        [str(frame_path)],
                        process_res=self.process_res,
                        ref_view_strategy="first",
                        infer_gs=True,
                    )

            if predictions.gaussians is None:
                logger.warning(f"Frame {i} did not return Gaussians, skipping")
                continue

            gaussians = predictions.gaussians

            means = gaussians.means[0].cpu().numpy()
            scales = gaussians.scales[0].cpu().numpy()

            # Necessary comment: Convert linear scales (0+) to log scales (-inf, +inf) for PLY format compatibility
            if np.all(scales > 0):
                scales = np.log(np.maximum(scales, 1e-10))

            rotations = gaussians.rotations[0].cpu().numpy()
            opacities = gaussians.opacities[0].cpu().numpy()

            harmonics = gaussians.harmonics[0].cpu().numpy()
            colors = harmonics[:, :, 0] if harmonics.ndim == 3 else harmonics

            if opacities.ndim == 2:
                opacities = opacities[:, 0]

            results.append(
                GaussianFrame(
                    frame_idx=i,
                    timestamp_ms=ts,
                    means=means.astype(np.float32),
                    scales=scales.astype(np.float32),
                    rotations=rotations.astype(np.float32),
                    colors=colors.astype(np.float32),
                    opacities=opacities.astype(np.float32),
                )
            )

        return results
