"""Sharp Gaussian Processor."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)


class SharpGaussianProcessor(GaussianProcessor):
    """Process images to Gaussian splats using Apple's SHARP model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        vit_preset: str = "dinov2l16_384",
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.vit_preset = vit_preset
        self.predictor = None

    def _load_model(self):
        if self.predictor is not None:
            return

        import torch
        import ssl
        from sharp.models import create_predictor, PredictorParams

        # Disable SSL verification for model download if needed
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]

        logger.info(f"Initializing SHARP model with preset: {self.vit_preset}...")
        params = PredictorParams()

        # Configure model backbone preset
        # Note: These attributes are Literal types in SHARP, but we pass str here.
        # This causes type checker warnings but works at runtime if the string is valid.
        params.monodepth.patch_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.monodepth.image_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.gaussian_decoder.patch_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.gaussian_decoder.image_encoder_preset = self.vit_preset  # type: ignore[assignment]

        self.predictor = create_predictor(params)

        if self.model_path and self.model_path.exists():
            logger.info(f"Loading SHARP weights from local file: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location="cpu")
        else:
            # Check default local resource location
            # Assuming script is run from project root or python dir, try relative paths
            local_resource_path = Path("_downloaded_resources/sharp/sharp_2572gikvuh.pt")
            if not local_resource_path.exists():
                # Try relative to python/ directory if running from there
                local_resource_path = Path("../_downloaded_resources/sharp/sharp_2572gikvuh.pt")

            if local_resource_path.exists():
                logger.info(
                    f"Loading SHARP weights from default local resource: {local_resource_path}"
                )
                state_dict = torch.load(local_resource_path, map_location="cpu")
            else:
                url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
                logger.info(f"Downloading SHARP weights from {url}...")
                state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

        self.predictor.load_state_dict(state_dict)
        self.predictor.eval().to(self.device)
        logger.info("SHARP model ready")

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = True,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """Process frames using SHARP."""
        import torch
        import cv2
        from sharp.cli.predict import predict_image
        from sharp.utils import color_space as cs_utils
        from tqdm import tqdm

        self._load_model()
        results = []

        # Silence SHARP CLI logs
        logging.getLogger("sharp.cli.predict").setLevel(logging.WARNING)

        # Constants from SHARP utils
        SH_C0 = 0.28209479177387814

        pbar = tqdm(
            zip(frame_paths, timestamps_ms),
            total=len(frame_paths),
            desc="SHARP processing",
            unit="frame",
        )
        for i, (path, ts) in enumerate(pbar):
            pbar.set_postfix(file=path.name)

            # Load image
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Failed to load image: {path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]

            # Apply mask if provided (skip first frame if requested)
            if masks_dir is not None and (mask_first_frame or i > 0):
                mask_path = None
                for ext in [path.suffix, ".png", ".jpg", ".jpeg"]:
                    candidate = masks_dir / f"{path.stem}{ext}"
                    if candidate.exists():
                        mask_path = candidate
                        break
                if mask_path is not None:
                    mask = cv2.imread(str(mask_path))
                    if mask is not None:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                        # Where mask is black (background), set image to black
                        black_pixels = np.all(mask == 0, axis=2)
                        img[black_pixels] = 0
                        n_masked = black_pixels.sum()
                        logger.info(f"Masked {n_masked} background pixels in {path.name}")
                    else:
                        logger.warning(f"Failed to load mask: {mask_path}")
                else:
                    logger.warning(f"Mask not found for {path.stem} in {masks_dir}")

            # SHARP expects focal length in pixels.
            # Default to FOV ~60 degrees if unknown (f ~= W) or use 500 like demo
            # Ideally we should estimate from EXIF or use a reasonable default.
            # Using max(H, W) is a safe bet for "normal" lenses.
            f_px = max(H, W) * 0.8

            with torch.no_grad():
                # Note: predict_image expects RGBGaussianPredictor, but self.predictor is
                # initialized as None and loaded later. Type checker will complain.
                # Also device type mismatch (str vs torch.device).
                gaussians = predict_image(self.predictor, img, f_px, device=self.device)  # type: ignore

            # --- Convert to standard 3DGS format ---

            # 1. Means (already in world space relative to camera)
            means = gaussians.mean_vectors.squeeze(0).cpu().numpy()

            # 2. Scales: Linear -> Log
            # SHARP outputs linear scales (singular values)
            scales_linear = gaussians.singular_values.squeeze(0)
            scales = torch.log(torch.clamp(scales_linear, min=1e-8)).cpu().numpy()

            # 3. Rotations: Quaternions (already normalized usually)
            rotations = gaussians.quaternions.squeeze(0).cpu().numpy()

            # 4. Opacities: Prob -> Logit
            # SHARP outputs probabilities (0-1)
            opacities_prob = gaussians.opacities.squeeze(0)
            # Inverse sigmoid: log(p / (1-p))
            opacities_prob = torch.clamp(opacities_prob, 1e-6, 1.0 - 1e-6)
            opacities = torch.log(opacities_prob / (1.0 - opacities_prob)).cpu().numpy()

            # 5. Colors: Linear RGB -> sRGB -> SH DC
            # SHARP predicts Linear RGB.
            # Apple's save_ply converts Linear->sRGB before saving.
            # We follow this to match the "ML-SHARP" color profile in our viewer.
            colors_linear = gaussians.colors.squeeze(0)
            colors_srgb = cs_utils.linearRGB2sRGB(colors_linear)

            # Convert RGB to SH DC (degree 0)
            # sh = (rgb - 0.5) / C0
            colors_sh = (colors_srgb - 0.5) / SH_C0
            colors = colors_sh.cpu().numpy()

            # Filter out black/background splats (SHARP prefers black background)
            if remove_black_splats:
                valid_mask = np.any(colors_linear.cpu().numpy() > 0.01, axis=1)
                n_removed = len(means) - valid_mask.sum()
                if n_removed > 0:
                    logger.info(f"Removed {n_removed} black/background splats from frame {i}")
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
