"""DA3 Gaussian Processor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import cv2
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

        from depth_anything_3.api import DepthAnything3

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
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """Process frames to extract Gaussians.

        Args:
            frame_paths: List of frame image paths
            timestamps_ms: Corresponding timestamps in milliseconds
            per_frame: If True, process each frame individually for separate PLYs.
                      If False (default), process all together for merged Gaussians.
            masks_dir: Directory containing masks for background removal
            mask_first_frame: Whether to apply mask to the first frame
            remove_black_splats: Whether to remove black/background splats

        Returns:
            List of GaussianFrame objects
        """
        self._load_model()

        images_to_process: list[Union[str, np.ndarray]] = []

        if masks_dir:
            logger.info(f"Applying masks from {masks_dir}")
            for i, path in enumerate(frame_paths):
                img = cv2.imread(str(path))
                if img is None:
                    logger.warning(f"Failed to load image: {path}")
                    images_to_process.append(str(path))
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if mask_first_frame or i > 0:
                    mask_path = None
                    for ext in [path.suffix, ".png", ".jpg", ".jpeg"]:
                        candidate = masks_dir / f"{path.stem}{ext}"
                        if candidate.exists():
                            mask_path = candidate
                            break

                    if mask_path:
                        mask = cv2.imread(str(mask_path))
                        if mask is not None:
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                            black_pixels = np.all(mask == 0, axis=2)
                            img[black_pixels] = 0
                        else:
                            logger.warning(f"Failed to load mask: {mask_path}")

                images_to_process.append(img)
        else:
            images_to_process = [str(p) for p in frame_paths]

        if per_frame:
            return self._process_frames_individually(
                images_to_process, timestamps_ms, frame_paths, remove_black_splats
            )
        else:
            return self._process_frames_merged(
                images_to_process, timestamps_ms, remove_black_splats
            )

    def _process_frames_merged(
        self,
        images: list[Union[str, np.ndarray]],
        timestamps_ms: list[float],
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """Process all frames together, returning merged Gaussians."""
        import torch

        logger.info(f"Processing {len(images)} frames merged with infer_gs=True")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.dtype):
                predictions = self.model.inference(  # type: ignore[attr-defined]
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

        if np.all(scales > 0):
            scales = np.log(np.maximum(scales, 1e-10))

        rotations = gaussians.rotations[0].cpu().numpy()
        opacities = gaussians.opacities[0].cpu().numpy()

        harmonics = gaussians.harmonics[0].cpu().numpy()
        colors = harmonics[:, :, 0] if harmonics.ndim == 3 else harmonics

        if opacities.ndim == 2:
            opacities = opacities[:, 0]

        if remove_black_splats:
            SH_C0 = 0.28209479177387814
            rgb = colors * SH_C0 + 0.5

            brightness = np.max(rgb, axis=1)
            valid_mask = brightness > 0.01

            n_removed = len(means) - np.sum(valid_mask)
            if n_removed > 0:
                logger.info(f"Removed {n_removed} black splats")
                means = means[valid_mask]
                scales = scales[valid_mask]
                rotations = rotations[valid_mask]
                colors = colors[valid_mask]
                opacities = opacities[valid_mask]

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
        images: list[Union[str, np.ndarray]],
        timestamps_ms: list[float],
        frame_paths: list[Path],
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """Process each frame individually for per-frame PLY output."""
        import torch

        results = []

        for i, (img, ts) in enumerate(zip(images, timestamps_ms)):
            name = frame_paths[i].name if i < len(frame_paths) else f"frame_{i}"
            logger.info(f"Processing frame {i + 1}/{len(images)}: {name}")

            with torch.no_grad():
                with torch.autocast("cuda", dtype=self.dtype):
                    predictions = self.model.inference(  # type: ignore[attr-defined]
                        [img],
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

            if np.all(scales > 0):
                scales = np.log(np.maximum(scales, 1e-10))

            rotations = gaussians.rotations[0].cpu().numpy()
            opacities = gaussians.opacities[0].cpu().numpy()

            harmonics = gaussians.harmonics[0].cpu().numpy()
            colors = harmonics[:, :, 0] if harmonics.ndim == 3 else harmonics

            if opacities.ndim == 2:
                opacities = opacities[:, 0]

            if remove_black_splats:
                SH_C0 = 0.28209479177387814
                rgb = colors * SH_C0 + 0.5
                brightness = np.max(rgb, axis=1)
                valid_mask = brightness > 0.01

                if np.sum(valid_mask) < len(means):
                    means = means[valid_mask]
                    scales = scales[valid_mask]
                    rotations = rotations[valid_mask]
                    colors = colors[valid_mask]
                    opacities = opacities[valid_mask]

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
