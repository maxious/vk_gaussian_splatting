import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)

# Add TRELLIS to path if not installed
TRELLIS_PATH = Path("C:/Users/maxious/trel/TRELLIS")
if TRELLIS_PATH.exists() and str(TRELLIS_PATH) not in sys.path:
    sys.path.append(str(TRELLIS_PATH))

try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils.general_utils import inverse_sigmoid

    TRELLIS_AVAILABLE = True
except ImportError:
    TRELLIS_AVAILABLE = False


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
                "TRELLIS not found. Please ensure it is in PYTHONPATH or C:/Users/maxious/trel/TRELLIS"
            )

        self.device = device
        self.model_id = model_id

        # Configure environment variables as per example
        os.environ["SPCONV_ALGO"] = "native"

        logger.info(f"Loading TRELLIS pipeline: {model_id}")
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
        self.pipeline.to(device)

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
                    # Apply mask (make background black)
                    img_np = np.array(img)
                    mask_np = np.array(mask)
                    img_np[mask_np < 128] = 0
                    img = Image.fromarray(img_np)

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
        # get_features returns SH. shape: [N, 3, (sh_degree+1)^2] ?
        # Code says: cat(_features_dc, _features_rest)
        # _features_dc shape: [N, 3, 1] usually
        features = gs.get_features
        # Extract DC (first coeff)
        # Assuming features are [N, 3, K] or [N, K, 3]?
        # gaussian_model.py:
        # _features_dc = features (from_features)
        # save_ply: f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1)
        # This implies _features_dc is [N, 3, 1].
        # Transpose(1,2) -> [N, 1, 3]. Flatten -> [N, 3].
        # So it's RGB order.

        if features.dim() == 3:
            # Take DC component
            # TRELLIS stores as [N, 3, 1] typically for DC
            dc = features[:, :, 0]  # [N, 3]
        else:
            dc = features  # Assuming just DC

        colors = dc.detach().cpu().numpy().astype(np.float32)

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
