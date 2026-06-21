"""TripoSplat Gaussian Processor - single image to 3D Gaussians via generative diffusion."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)

TRIPOSPLAT_REPO = "VAST-AI/TripoSplat"
TRIPOSPLAT_FILES = {
    "rmbg": "background_removal/birefnet.safetensors",
    "flow": "diffusion_models/triposplat_fp16.safetensors",
    "dinov3": "clip_vision/dino_v3_vit_h.safetensors",
    "vae": "vae/flux2-vae.safetensors",
    "decoder": "vae/triposplat_vae_decoder_fp16.safetensors",
}

DEFAULT_NUM_GAUSSIANS = 262144
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 3.0


def _download_triposplat_weights(cache_dir: Path | None = None) -> dict[str, str]:
    """Download TripoSplat model weights from HuggingFace Hub.

    Returns dict mapping component names to local file paths.
    """
    from huggingface_hub import hf_hub_download

    cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "hub"
    paths = {}
    for name, filename in TRIPOSPLAT_FILES.items():
        local = hf_hub_download(
            repo_id=TRIPOSPLAT_REPO,
            filename=filename,
            cache_dir=cache_dir,
        )
        paths[name] = local
    return paths


class TripoSplatGaussianProcessor(GaussianProcessor):
    """Generate 3D Gaussians from single images using TripoSplat diffusion model.

    TripoSplat converts a single 2D image into high-quality variable-count
    3D Gaussians via flow-matching diffusion over learned latent codes.
    Output is SH degree 0 (view-independent color).

    Model weights are auto-downloaded from HuggingFace on first use.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        num_gaussians: int = DEFAULT_NUM_GAUSSIANS,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        weights_cache_dir: Path | None = None,
    ):
        self._device = device
        self._num_gaussians = num_gaussians
        self._steps = steps
        self._guidance_scale = guidance_scale
        self._weights_cache_dir = weights_cache_dir
        self._pipeline = None

    def _ensure_pipeline(self):
        """Lazy-load the TripoSplat pipeline on first use."""
        if self._pipeline is not None:
            return

        vendored = Path(__file__).parent.parent.parent / "third_party" / "triposplat"
        if str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))

        from triposplat import TripoSplatPipeline  # type: ignore[import-not-found]

        logger.info("Downloading TripoSplat weights from HuggingFace (%s)...", TRIPOSPLAT_REPO)
        weights = _download_triposplat_weights(cache_dir=self._weights_cache_dir)

        logger.info(
            "Initializing TripoSplat pipeline (device=%s, gaussians=%d, steps=%d, cfg=%.1f)",
            self._device,
            self._num_gaussians,
            self._steps,
            self._guidance_scale,
        )
        self._pipeline = TripoSplatPipeline(
            ckpt_path=weights["flow"],
            decoder_path=weights["decoder"],
            dinov3_path=weights["dinov3"],
            flux2_vae_encoder_path=weights["vae"],
            rmbg_path=weights["rmbg"],
            device=self._device,
        )

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = True,
        **kwargs,
    ) -> list[GaussianFrame]:
        self._ensure_pipeline()

        frames: list[GaussianFrame] = []
        for idx, (path, ts) in enumerate(zip(frame_paths, timestamps_ms)):
            logger.info(
                "TripoSplat processing frame %d/%d: %s",
                idx + 1,
                len(frame_paths),
                path.name,
            )
            gaussian_frame = self._process_single(path, idx, ts)
            frames.append(gaussian_frame)

        return frames

    # Coordinate transform: Y-up (TripoSplat) → Z-up (standard 3DGS viewer).
    # Using a list-of-lists so _get_ply_data can consume it directly.
    _COORD_TRANSFORM = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    def _process_single(
        self, image_path: Path, frame_idx: int, timestamp_ms: float
    ) -> GaussianFrame:
        assert self._pipeline is not None
        image = Image.open(image_path).convert("RGB")
        gaussian, _prepared = self._pipeline.run(
            image,
            num_gaussians=self._num_gaussians,
            steps=self._steps,
            guidance_scale=self._guidance_scale,
        )

        # _get_ply_data handles the coordinate transform for xyz and rotation.
        # It returns log-scale and logit-opacity, but write_static_gaussian_ply
        # applies its own np.log / logit — so extract raw scales and opacities
        # directly from the Gaussian object.
        xyz, _normals, f_dc, _opacities_logit, _scale_log, rotation = gaussian._get_ply_data(
            transform=self._COORD_TRANSFORM,
        )
        scales = gaussian.get_scaling.detach().cpu().numpy().astype(np.float32)
        opacities = gaussian.get_opacity.detach().cpu().numpy().astype(np.float32).ravel()

        return GaussianFrame(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            means=xyz.astype(np.float32),
            scales=scales,
            rotations=rotation.astype(np.float32),
            colors=f_dc.astype(np.float32),
            opacities=opacities,
        )


def parse_triposplat_model_id(model_id: str) -> dict:
    """Parse TripoSplat model_id string into configuration parameters.

    Formats supported:
        triposplat                      → defaults (262144, 20 steps, cfg 3.0)
        triposplat:131072               → 131072 gaussians
        triposplat:65536:steps=10       → 65536 gaussians, 10 steps
        triposplat:131072:cfg=5.0       → 131072 gaussians, cfg 5.0
        triposplat:262144:steps=30:cfg=7.0 → all three

    Returns dict with keys: num_gaussians, steps, guidance_scale
    """
    config = {
        "num_gaussians": DEFAULT_NUM_GAUSSIANS,
        "steps": DEFAULT_STEPS,
        "guidance_scale": DEFAULT_GUIDANCE,
    }

    if ":" not in model_id:
        return config

    parts = model_id.split(":")[1:]
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("steps="):
            config["steps"] = int(part.split("=", 1)[1])
        elif part.startswith("cfg="):
            config["guidance_scale"] = float(part.split("=", 1)[1])
        elif part.isdigit():
            config["num_gaussians"] = int(part)
        else:
            logger.warning("Unknown TripoSplat config part: %s", part)

    return config
