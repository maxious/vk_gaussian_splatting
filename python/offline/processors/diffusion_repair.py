"""Diffusion-based repair processor for low-confidence Gaussian splat regions.

Uses Qwen-Image-Edit-2511 or FLUX.2 Klein with 3D repair LoRA to inpaint
artifacts in rendered Gaussian splat views.

Based on FreeFix paper concepts: per-pixel confidence guidance for iterative refinement.

Supports loading GGUF quantized models from local ComfyUI model directories.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default ComfyUI model paths (can be overridden via env vars or config)
DEFAULT_COMFYUI_ROOT = Path.home() / "ComfyUI"
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", DEFAULT_COMFYUI_ROOT))


class RepairModel(str, Enum):
    """Available diffusion repair models."""

    QWEN_EDIT_2511 = "qwen-edit-2511"
    QWEN_EDIT_2511_GAUSSIAN_SPLASH = "qwen-edit-2511-gaussian-splash"
    FLUX_KLEIN_3D_REPAIR = "flux-klein-3d-repair"


@dataclass
class RepairConfig:
    """Configuration for diffusion repair."""

    model: RepairModel = RepairModel.QWEN_EDIT_2511_GAUSSIAN_SPLASH
    confidence_threshold: float = 0.3
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    true_cfg_scale: float = 4.0
    strength: float = 0.8
    device: str = "cuda"
    dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"  # For GGUF dequantization
    lora_path: Path | None = None
    lora_scale: float = 1.0

    # Local GGUF model paths (auto-detected from ComfyUI if not set)
    gguf_model_path: Path | None = None
    gguf_text_encoder_path: Path | None = None
    gguf_vae_path: Path | None = None

    # ComfyUI model directory (for auto-detection)
    comfyui_root: Path = field(default_factory=lambda: COMFYUI_ROOT)

    def __post_init__(self):
        """Auto-detect GGUF paths from ComfyUI directory if available."""
        if self.gguf_model_path is None:
            self._auto_detect_gguf_paths()

    def _auto_detect_gguf_paths(self):
        """Try to find GGUF models in ComfyUI directory."""
        diffusion_models = self.comfyui_root / "models" / "diffusion_models"
        loras = self.comfyui_root / "models" / "loras"
        vaes = self.comfyui_root / "models" / "vae"
        text_encoders = self.comfyui_root / "models" / "text_encoders"

        if self.model in (RepairModel.QWEN_EDIT_2511, RepairModel.QWEN_EDIT_2511_GAUSSIAN_SPLASH):
            # Look for Qwen-Image-Edit GGUF
            candidates = [
                diffusion_models / "qwen-image-edit-2511-Q4_K_M.gguf",
                diffusion_models / "qwen-image-edit-2511-Q8_0.gguf",
            ]
            for c in candidates:
                if c.exists():
                    self.gguf_model_path = c
                    logger.info(f"Auto-detected Qwen GGUF: {c}")
                    break

            # Look for Gaussian Splash LoRA
            if self.model == RepairModel.QWEN_EDIT_2511_GAUSSIAN_SPLASH and self.lora_path is None:
                splash_lora = loras / "Qwen-Image-Edit-2511-Gaussian-Splash.safetensors"
                if splash_lora.exists():
                    self.lora_path = splash_lora
                    logger.info(f"Auto-detected Gaussian Splash LoRA: {splash_lora}")

            # Qwen VAE
            qwen_vae = vaes / "qwen_image_vae.safetensors"
            if qwen_vae.exists():
                self.gguf_vae_path = qwen_vae

        elif self.model == RepairModel.FLUX_KLEIN_3D_REPAIR:
            # Look for FLUX.2 Klein GGUF
            candidates = [
                diffusion_models / "flux-2-klein-9b-BF16.gguf",
                diffusion_models / "flux-2-klein-9b-Q8_0.gguf",
            ]
            for c in candidates:
                if c.exists():
                    self.gguf_model_path = c
                    logger.info(f"Auto-detected FLUX Klein GGUF: {c}")
                    break

            # Look for 3D repair LoRA
            if self.lora_path is None:
                repair_lora = loras / "flux2-klein9b-lora-mlsharp-3d-repair.safetensors"
                if repair_lora.exists():
                    self.lora_path = repair_lora
                    logger.info(f"Auto-detected 3D repair LoRA: {repair_lora}")

            # FLUX VAE
            flux_vae = vaes / "flux-vae-bf16.safetensors"
            if flux_vae.exists():
                self.gguf_vae_path = flux_vae


@dataclass
class RepairResult:
    """Result from diffusion repair."""

    repaired_image: np.ndarray  # (H, W, 3) RGB uint8
    confidence_mask: np.ndarray  # (H, W) float32 in [0, 1]
    repair_mask: np.ndarray  # (H, W) bool - regions that were repaired
    original_image: np.ndarray  # (H, W, 3) RGB uint8


class BaseDiffusionRepair(ABC):
    """Base class for diffusion-based repair models."""

    def __init__(self, config: RepairConfig):
        self.config = config
        self.pipeline = None
        self._loaded = False

    @abstractmethod
    def _load_pipeline(self) -> None:
        """Load the diffusion pipeline."""
        ...

    @abstractmethod
    def _run_repair(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        reference_image: Image.Image | None = None,
    ) -> Image.Image:
        """Run the repair pipeline on the image."""
        ...

    def load(self) -> None:
        """Load the model if not already loaded."""
        if not self._loaded:
            self._load_pipeline()
            self._loaded = True

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._loaded = False

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

    def repair(
        self,
        rendered_image: np.ndarray,
        confidence_map: np.ndarray,
        reference_image: np.ndarray | None = None,
        prompt: str | None = None,
    ) -> RepairResult:
        """Repair low-confidence regions in a rendered Gaussian splat image.

        Args:
            rendered_image: (H, W, 3) RGB uint8 rendered from Gaussian splats
            confidence_map: (H, W) float32 in [0, 1], higher = more confident
            reference_image: Optional reference image for style/content guidance
            prompt: Optional custom repair prompt

        Returns:
            RepairResult with repaired image and metadata
        """
        self.load()

        # Create repair mask from confidence
        repair_mask = confidence_map < self.config.confidence_threshold

        # If nothing to repair, return original
        if not np.any(repair_mask):
            return RepairResult(
                repaired_image=rendered_image.copy(),
                confidence_mask=confidence_map,
                repair_mask=repair_mask,
                original_image=rendered_image.copy(),
            )

        # Dilate mask slightly for better blending
        from scipy.ndimage import binary_dilation

        kernel = np.ones((5, 5), dtype=bool)
        dilated_mask = binary_dilation(repair_mask, kernel, iterations=2)

        # Convert to PIL
        image_pil = Image.fromarray(rendered_image)
        mask_pil = Image.fromarray((dilated_mask * 255).astype(np.uint8), mode="L")
        ref_pil = Image.fromarray(reference_image) if reference_image is not None else None

        # Default prompt for Gaussian splat repair
        if prompt is None:
            prompt = self._get_default_prompt()

        # Run repair
        repaired_pil = self._run_repair(image_pil, mask_pil, prompt, ref_pil)
        repaired_np = np.array(repaired_pil)

        return RepairResult(
            repaired_image=repaired_np,
            confidence_mask=confidence_map,
            repair_mask=repair_mask,
            original_image=rendered_image.copy(),
        )

    def _get_default_prompt(self) -> str:
        """Get default repair prompt based on model type."""
        if "gaussian" in self.config.model.value.lower():
            return "高斯泼溅,参考图2的场景图，修复图1的场景图透视并修复空白区域"
        return "Fill in the missing regions naturally, maintaining consistent lighting and perspective"


class QwenEditRepair(BaseDiffusionRepair):
    """Repair using Qwen-Image-Edit-2511 with optional Gaussian Splash LoRA.

    Supports loading from:
    - Local GGUF quantized model (fastest, lowest VRAM)
    - HuggingFace full precision model (fallback)
    """

    def _load_pipeline(self) -> None:
        import torch
        from diffusers import QwenImageEditPlusPipeline

        dtype = getattr(torch, self.config.dtype)
        compute_dtype = getattr(torch, self.config.compute_dtype)

        # Try GGUF first if available
        if self.config.gguf_model_path and self.config.gguf_model_path.exists():
            logger.info(f"Loading Qwen-Image-Edit from GGUF: {self.config.gguf_model_path}")
            self._load_from_gguf(dtype, compute_dtype)
        else:
            logger.info(f"Loading Qwen-Image-Edit-2511 from HuggingFace on {self.config.device}...")
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                torch_dtype=dtype,
            )
            self.pipeline.to(self.config.device)

        # Load LoRA if specified
        if self.config.lora_path and self.config.lora_path.exists():
            logger.info(f"Loading LoRA from {self.config.lora_path}")
            self.pipeline.load_lora_weights(
                str(self.config.lora_path),
                adapter_name="gaussian_splash",
            )
            self.pipeline.set_adapters(["gaussian_splash"], [self.config.lora_scale])
        elif self.config.model == RepairModel.QWEN_EDIT_2511_GAUSSIAN_SPLASH:
            # Try to load from HuggingFace
            try:
                logger.info("Loading Gaussian Splash LoRA from HuggingFace...")
                self.pipeline.load_lora_weights(
                    "dx8152/Qwen-Image-Edit-2511-Gaussian-Splash",
                    adapter_name="gaussian_splash",
                )
                self.pipeline.set_adapters(["gaussian_splash"], [self.config.lora_scale])
            except Exception as e:
                logger.warning(f"Could not load Gaussian Splash LoRA: {e}")

        self.pipeline.set_progress_bar_config(disable=True)

    def _load_from_gguf(self, dtype, compute_dtype) -> None:
        """Load transformer from GGUF and build pipeline."""
        import torch
        from diffusers import AutoencoderKL, QwenImageEditPlusPipeline
        from diffusers.models.transformers import QwenImageEditPlusTransformer2DModel
        from diffusers.quantizers import GGUFQuantizationConfig

        gguf_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        # Load transformer from GGUF
        transformer = QwenImageEditPlusTransformer2DModel.from_single_file(
            str(self.config.gguf_model_path),
            quantization_config=gguf_config,
            torch_dtype=dtype,
        )
        transformer.to(self.config.device)

        # Load VAE (from local or HuggingFace)
        if self.config.gguf_vae_path and self.config.gguf_vae_path.exists():
            vae = AutoencoderKL.from_single_file(
                str(self.config.gguf_vae_path),
                torch_dtype=dtype,
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                subfolder="vae",
                torch_dtype=dtype,
            )
        vae.to(self.config.device)

        # Build pipeline with GGUF transformer
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            transformer=transformer,
            vae=vae,
            torch_dtype=dtype,
        )
        self.pipeline.to(self.config.device)

    def _run_repair(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        reference_image: Image.Image | None = None,
    ) -> Image.Image:
        import torch

        # Qwen-Image-Edit supports multi-image input
        images = [image]
        if reference_image is not None:
            images.append(reference_image)

        with torch.inference_mode():
            output = self.pipeline(
                image=images,
                prompt=prompt,
                generator=torch.Generator(device=self.config.device).manual_seed(42),
                true_cfg_scale=self.config.true_cfg_scale,
                negative_prompt="blurry, artifacts, low quality, distorted",
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                num_images_per_prompt=1,
            )

        return output.images[0]


class FluxKlein3DRepair(BaseDiffusionRepair):
    """Repair using FLUX.2 Klein 9B with 3D repair LoRA.

    Supports loading from:
    - Local GGUF quantized model (fastest, lowest VRAM)
    - HuggingFace full precision model (fallback)
    """

    def _load_pipeline(self) -> None:
        import torch
        from diffusers import FluxInpaintPipeline

        dtype = getattr(torch, self.config.dtype)
        compute_dtype = getattr(torch, self.config.compute_dtype)

        # Try GGUF first if available
        if self.config.gguf_model_path and self.config.gguf_model_path.exists():
            logger.info(f"Loading FLUX.2 Klein from GGUF: {self.config.gguf_model_path}")
            self._load_from_gguf(dtype, compute_dtype)
        else:
            logger.info(f"Loading FLUX.2 Klein pipeline from HuggingFace on {self.config.device}...")
            self.pipeline = FluxInpaintPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-base-9B",
                torch_dtype=dtype,
            )
            self.pipeline.to(self.config.device)

        # Load 3D repair LoRA
        if self.config.lora_path and self.config.lora_path.exists():
            logger.info(f"Loading LoRA from {self.config.lora_path}")
            self.pipeline.load_lora_weights(
                str(self.config.lora_path),
                adapter_name="3d_repair",
            )
        else:
            # Try HuggingFace
            try:
                logger.info("Loading ml-sharp 3D repair LoRA from HuggingFace...")
                self.pipeline.load_lora_weights(
                    "cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair",
                    adapter_name="3d_repair",
                )
            except Exception as e:
                logger.warning(f"Could not load 3D repair LoRA: {e}")

        self.pipeline.set_adapters(["3d_repair"], [self.config.lora_scale])
        self.pipeline.set_progress_bar_config(disable=True)

    def _load_from_gguf(self, dtype, compute_dtype) -> None:
        """Load transformer from GGUF and build pipeline."""
        import torch
        from diffusers import AutoencoderKL, FluxInpaintPipeline
        from diffusers.models.transformers import FluxTransformer2DModel
        from diffusers.quantizers import GGUFQuantizationConfig

        gguf_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        # Load transformer from GGUF
        transformer = FluxTransformer2DModel.from_single_file(
            str(self.config.gguf_model_path),
            quantization_config=gguf_config,
            torch_dtype=dtype,
        )
        transformer.to(self.config.device)

        # Load VAE (from local or HuggingFace)
        if self.config.gguf_vae_path and self.config.gguf_vae_path.exists():
            vae = AutoencoderKL.from_single_file(
                str(self.config.gguf_vae_path),
                torch_dtype=dtype,
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.2-klein-base-9B",
                subfolder="vae",
                torch_dtype=dtype,
            )
        vae.to(self.config.device)

        # Build pipeline with GGUF transformer
        self.pipeline = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-base-9B",
            transformer=transformer,
            vae=vae,
            torch_dtype=dtype,
        )
        self.pipeline.to(self.config.device)

    def _run_repair(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        reference_image: Image.Image | None = None,
    ) -> Image.Image:
        import torch

        with torch.inference_mode():
            output = self.pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                height=image.height,
                width=image.width,
                strength=self.config.strength,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=torch.Generator(device=self.config.device).manual_seed(42),
            )

        return output.images[0]


def create_repair_model(config: RepairConfig) -> BaseDiffusionRepair:
    """Factory function to create the appropriate repair model."""
    if config.model in (RepairModel.QWEN_EDIT_2511, RepairModel.QWEN_EDIT_2511_GAUSSIAN_SPLASH):
        return QwenEditRepair(config)
    elif config.model == RepairModel.FLUX_KLEIN_3D_REPAIR:
        return FluxKlein3DRepair(config)
    else:
        raise ValueError(f"Unknown repair model: {config.model}")


def compute_confidence_from_opacity(
    opacities: np.ndarray,
    means: np.ndarray,
    image_size: tuple[int, int],
    focal_length: float | None = None,
) -> np.ndarray:
    """Compute per-pixel confidence map from Gaussian splat attributes.

    Uses opacity accumulation and depth variance as confidence signals.
    Based on FreeFix paper's certainty computation.

    Args:
        opacities: (N,) logit opacities of Gaussians
        means: (N, 3) positions of Gaussians
        image_size: (width, height) of output image
        focal_length: Focal length in pixels (defaults to max(W, H) * 0.8)

    Returns:
        confidence_map: (H, W) float32 in [0, 1]
    """
    import torch
    from scipy.special import expit

    W, H = image_size
    if focal_length is None:
        focal_length = max(W, H) * 0.8

    # Convert logit opacities to probabilities
    opacity_probs = expit(opacities)

    # Get depths
    depths = means[:, 2]
    valid_mask = depths > 0.01

    # Project to image coordinates
    x, y, z = means[:, 0], means[:, 1], means[:, 2]
    z_safe = np.where(valid_mask, z, 1.0)

    u = (x * focal_length / z_safe + W / 2).astype(int)
    v = (y * focal_length / z_safe + H / 2).astype(int)

    # Clamp to image bounds
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    # Accumulate opacity per pixel (simplified splatting)
    confidence = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    valid_indices = np.where(valid_mask)[0]
    for i in valid_indices:
        py, px = v[i], u[i]
        confidence[py, px] += opacity_probs[i]
        counts[py, px] += 1

    # Normalize and clamp
    confidence = np.where(counts > 0, confidence / np.maximum(counts, 1), 0)
    confidence = np.clip(confidence, 0, 1)

    return confidence


def repair_gaussian_render(
    rendered_image: np.ndarray,
    opacities: np.ndarray,
    means: np.ndarray,
    config: RepairConfig | None = None,
    reference_image: np.ndarray | None = None,
) -> RepairResult:
    """Convenience function to repair a rendered Gaussian splat image.

    Args:
        rendered_image: (H, W, 3) RGB uint8 rendered image
        opacities: (N,) logit opacities from Gaussian splats
        means: (N, 3) Gaussian positions
        config: Repair configuration (uses defaults if None)
        reference_image: Optional reference for style guidance

    Returns:
        RepairResult with repaired image
    """
    if config is None:
        config = RepairConfig()

    H, W = rendered_image.shape[:2]

    # Compute confidence from Gaussians
    confidence = compute_confidence_from_opacity(
        opacities=opacities,
        means=means,
        image_size=(W, H),
    )

    # Create and run repair model
    repair_model = create_repair_model(config)
    try:
        result = repair_model.repair(
            rendered_image=rendered_image,
            confidence_map=confidence,
            reference_image=reference_image,
        )
    finally:
        repair_model.unload()

    return result
