import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Add vendored vkgs_trellis to path before any imports
_VKGS_TRELLIS_PARENT = Path(__file__).parent.parent
if str(_VKGS_TRELLIS_PARENT) not in sys.path:
    sys.path.insert(0, str(_VKGS_TRELLIS_PARENT))

from ..types import GaussianFrame
from .base import GaussianProcessor

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
    ):
        if not TRELLIS_AVAILABLE:
            raise ImportError(
                "TRELLIS not found. Please ensure it is in PYTHONPATH or installed in python/external/TRELLIS"
            )

        self.device = device
        self.model_id = model_id

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
            # Optional: Enable some optimizations for low VRAM if available
            # Current TRELLIS implementation doesn't expose much explicit low-vram flags
            # other than standard torch optimizations
            pass

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

        # Load images
        images = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            # Apply mask if available
            if masks_dir:
                mask_path = None
                for ext in [p.suffix, ".png", ".jpg", ".jpeg"]:
                    candidate = masks_dir / f"{p.stem}{ext}"
                    if candidate.exists():
                        mask_path = candidate
                        break

                if mask_path:
                    mask = Image.open(mask_path).convert("L")
                    # Resize mask to match image
                    mask = mask.resize(img.size, Image.NEAREST)
                    # Convert to RGBA and apply mask to alpha channel
                    # This allows Trellis to detect alpha and skip rembg
                    img = img.convert("RGBA")
                    img.putalpha(mask)

            images.append(img)

        results = []

        # Single image processing or Multi-image processing
        if len(images) > 1:
            logger.info(f"Running TRELLIS multi-image inference on {len(images)} frames")
            try:
                outputs = self.pipeline.run_multi_image(
                    images,
                    seed=1,
                    sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": 12, "cfg_strength": 3},
                )
                self._extract_gaussian(
                    outputs,
                    results,
                    frame_idx=0,
                    timestamp=timestamps_ms[0] if timestamps_ms else 0.0,
                )
            except AttributeError:
                logger.warning(
                    "run_multi_image not found (older TRELLIS version?), falling back to single image loop"
                )
                for i, img in enumerate(images):
                    self._process_single(img, i, timestamps_ms[i], results)
        else:
            logger.info("Running TRELLIS single-image inference")
            self._process_single(images[0], 0, timestamps_ms[0], results)

        return results

    def _process_single(self, image, index, timestamp, results):
        outputs = self.pipeline.run(
            image,
            seed=1,
            # Use default sampler params
        )
        self._extract_gaussian(outputs, results, index, timestamp)

    def _extract_gaussian(self, outputs, results, frame_idx, timestamp):
        if "gaussian" not in outputs or not outputs["gaussian"]:
            logger.error("No gaussian output from TRELLIS")
            return

        gs = outputs["gaussian"][0]

        # Extract data from TRELLIS Gaussian object
        # Based on trellis/representations/gaussian/gaussian_model.py

        # Positions
        means = gs.get_xyz.detach().cpu().numpy().astype(np.float32)

        # Opacities (logit)
        # get_opacity returns sigmoid(opacity), so we inverse it.
        # But we can also access _opacity + opacity_bias
        # Safe way using public API:
        opacities_prob = gs.get_opacity
        # Inverse sigmoid: log(p / (1-p))
        opacities_prob = torch.clamp(opacities_prob, 1e-6, 1.0 - 1e-6)
        opacities = (
            torch.log(opacities_prob / (1.0 - opacities_prob))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Scales (log)
        # get_scaling returns scales. We need log(scales)
        scales_act = gs.get_scaling
        scales = (
            torch.log(torch.clamp(scales_act, min=1e-8)).detach().cpu().numpy().astype(np.float32)
        )

        # Rotations (quaternion)
        # TRELLIS returns [w, x, y, z] (standard)
        # GaussianFrame expects [w, x, y, z] (standard)
        rotations = gs.get_rotation.detach().cpu().numpy().astype(np.float32)

        # Colors (SH DC)
        # TRELLIS stores DC features in _features_dc with shape [N, 3, 1]
        # Transform to [N, 3] by transposing and flattening
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
