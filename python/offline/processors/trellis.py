import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Add vendored vkgs_trellis to path before any imports
_VKGS_TRELLIS_PARENT = Path(__file__).parent.parent
if str(_VKGS_TRELLIS_PARENT) not in sys.path:
    sys.path.insert(0, str(_VKGS_TRELLIS_PARENT))

from ..types import GaussianFrame
from .base import GaussianProcessor
from common.datasets import frame_dataset_factory

logger = logging.getLogger(__name__)

try:
    from vkgs_trellis.pipelines import TrellisImageTo3DPipeline

    TRELLIS_AVAILABLE = True
    logger.info("Using vendored vkgs_trellis package")
except ImportError as e:
    import traceback

    logger.warning(f"Failed to import TRELLIS: {e}")
    traceback.print_exc()
    TRELLIS_AVAILABLE = False


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class TrellisProcessor(GaussianProcessor):
    """Processor using Microsoft TRELLIS model."""

    def __init__(
        self,
        model_id: str = "microsoft/TRELLIS-image-large",
        device: str = "cuda",
        low_vram: bool = False,
        preprocess: bool = True,
        bbox_scale: float = 1.2,
    ):
        if not TRELLIS_AVAILABLE:
            raise ImportError(
                "TRELLIS not found. Please ensure it is in PYTHONPATH or installed in python/external/TRELLIS"
            )

        self.device = device
        self.model_id = model_id
        self.preprocess = preprocess
        self.bbox_scale = bbox_scale

        # Configure environment variables as per example
        os.environ["SPCONV_ALGO"] = "native"

        logger.info(f"Loading TRELLIS pipeline: {model_id}")
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
        self.pipeline.to(device)

        # Cast image conditioning model to float16 for compatibility with xformers/flash-attn on newer GPUs
        if "image_cond_model" in self.pipeline.models:
            self.pipeline.models["image_cond_model"] = self.pipeline.models["image_cond_model"].to(
                torch.float16
            )
            logger.info("Cast image_cond_model to float16 for attention compatibility")

        if low_vram:
            pass

    def _crop_to_mask(self, img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Crop image to the bounding box of the mask (if provided).

        Args:
            img: RGB numpy array (H, W, 3)
            mask: Optional grayscale numpy array (H, W)

        Returns:
            Cropped RGB numpy array
        """
        if mask is None:
            return img

        # Find non-zero mask pixels
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            logger.warning("No non-zero mask pixels found, returning original image")
            return img

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Calculate center and size
        center = ((cmin + cmax) / 2, (rmin + rmax) / 2)
        size = max(cmax - cmin, rmax - rmin) * self.bbox_scale

        # Calculate crop bbox
        left = int(center[0] - size / 2)
        top = int(center[1] - size / 2)
        right = int(center[0] + size / 2)
        bottom = int(center[1] + size / 2)

        # Crop with bounds checking
        left = max(0, left)
        top = max(0, top)
        right = min(img.shape[1], right)
        bottom = min(img.shape[0], bottom)

        return img[top:bottom, left:right]

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        if len(frame_paths) == 0:
            return []

        # Create dataset with FrameDataset for parallel loading
        dataset = frame_dataset_factory(
            frame_paths=frame_paths,
            timestamps_ms=timestamps_ms,
            masks_dir=masks_dir,
            mask_first_frame=mask_first_frame,
            image_mode="RGB",
        )

        # Use DataLoader for parallel loading with pinned memory
        pin_memory = self.device in ("cuda", "xpu")
        dataloader = DataLoader(
            dataset,
            batch_size=min(4, len(frame_paths)),  # Process up to 4 frames at once
            num_workers=4,
            pin_memory=pin_memory,
            collate_fn=self._collate_frames,
        )

        results = []

        for batch in dataloader:
            indices, images, batch_timestamps, H, W, masks = batch
            # images: (B, C, H, W) tensor in [0, 1] range
            # Convert to PIL Images for TRELLIS pipeline
            images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            # Apply mask cropping if masks are present
            if masks is not None and masks.sum() > 0:
                masks_np = masks.squeeze(1).cpu().numpy()
                images_np = [
                    self._crop_to_mask(img, mask) for img, mask in zip(images_np, masks_np)
                ]

            # Convert to PIL Images
            pil_images = [Image.fromarray(img).convert("RGB") for img in images_np]

            # Process images through TRELLIS
            if len(pil_images) > 1:
                logger.info(f"Running TRELLIS multi-image inference on {len(pil_images)} frames")
                try:
                    outputs = self.pipeline.run_multi_image(
                        pil_images,
                        seed=1,
                        sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
                        slat_sampler_params={"steps": 25, "cfg_strength": 3},
                        preprocess_image=self.preprocess,
                    )
                    for i, idx in enumerate(indices):
                        self._extract_gaussian(
                            outputs if len(pil_images) == 1 else [outputs["gaussian"][i]],
                            results,
                            frame_idx=int(idx),
                            timestamp=batch_timestamps[i],
                        )
                except AttributeError:
                    logger.warning(
                        "run_multi_image not found (older TRELLIS version?), falling back to single image loop"
                    )
                    for i, pil_img in enumerate(pil_images):
                        self._process_single(pil_img, int(indices[i]), batch_timestamps[i], results)
            else:
                logger.info("Running TRELLIS single-image inference")
                self._process_single(pil_images[0], int(indices[0]), batch_timestamps[0], results)

        return results

    @staticmethod
    def _collate_frames(batch):
        """Custom collate function for FrameDataset items."""
        indices, images, timestamps, H, W, masks = zip(*batch)
        return (
            torch.stack(indices),
            torch.stack(images),
            list(timestamps),
            torch.stack(H),
            torch.stack(W),
            torch.stack(masks) if masks[0] is not None else None,
        )

    def _process_single(self, image, index, timestamp, results):
        outputs = self.pipeline.run(
            image,
            seed=1,
        )
        self._extract_gaussian(outputs, results, index, timestamp)

    def _extract_gaussian(self, outputs, results, frame_idx, timestamp):
        if "gaussian" not in outputs or not outputs["gaussian"]:
            logger.error("No gaussian output from TRELLIS")
            return

        gs = outputs["gaussian"][0]

        # Positions
        means = gs.get_xyz.detach().cpu().numpy().astype(np.float32)

        # Opacities (logit)
        opacities_prob = gs.get_opacity
        opacities_prob = torch.clamp(opacities_prob, 1e-6, 1.0 - 1e-6)
        opacities = (
            torch.log(opacities_prob / (1.0 - opacities_prob))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Scales (log)
        scales_act = gs.get_scaling
        scales = (
            torch.log(torch.clamp(scales_act, min=1e-8)).detach().cpu().numpy().astype(np.float32)
        )

        # Rotations (quaternion)
        rotations = gs.get_rotation.detach().cpu().numpy().astype(np.float32)

        # Colors (SH DC)
        features_dc = gs._features_dc.detach().cpu().numpy()
        colors = (
            np.transpose(features_dc, (0, 2, 1)).reshape(features_dc.shape[0], 3).astype(np.float32)
        )

        results.append(
            GaussianFrame(
                frame_idx=frame_idx,
                timestamp_ms=timestamp,
                means=means,
                scales=scales,
                rotations=rotations,
                colors=colors,
                opacities=opacities,
            )
        )
