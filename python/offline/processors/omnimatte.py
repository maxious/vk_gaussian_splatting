from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_video, export_to_video
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

from ..omnimatte.OmnimatteZero import OmnimatteZero

logger = logging.getLogger(__name__)


def tensor_video_to_pil_images(video_tensor):
    """
    Converts a PyTorch tensor representing a video to a list of PIL Images.
    Handles both (1, T, H, W, 3) and (T, H, W, 3) shapes.
    """
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)
    # Ensure tensor is on CPU and convert to numpy
    video_np = video_tensor.cpu().numpy()
    # Convert each frame (H, W, 3) to PIL Image
    return [Image.fromarray(frame.astype("uint8")) for frame in video_np]


class OmnimatteProcessor:
    """Processor for running OmnimatteZero tasks (Object Removal, Layer Extraction)."""

    def __init__(
        self,
        model_id: str = "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
        gguf_path: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        self.gguf_path = gguf_path
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.vae = None

    def load_model(self):
        """Lazy load the pipeline."""
        if self.pipe is not None:
            return

        logger.info(f"Loading OmnimatteZero pipeline...")
        if self.gguf_path:
            # Load from GGUF quantized checkpoint (much lower VRAM)
            logger.info(f"Loading from GGUF: {self.gguf_path}")
            self.pipe = OmnimatteZero.from_single_file(
                self.gguf_path,
                torch_dtype=self.dtype,
            ).to(self.device)
        else:
            # Load from Diffusers repository (full precision)
            logger.info(f"Loading from Diffusers: {self.model_id}")
            self.pipe = OmnimatteZero.from_pretrained(self.model_id, torch_dtype=self.dtype).to(
                self.device
            )

        logger.info("OmnimatteZero loaded.")

    def remove_object(
        self,
        video_path: Path,
        mask_path: Path,
        output_path: Path,
        num_inference_steps: int = 30,
        height: int = 512,
        width: int = 768,
    ):
        """
        Generate a clean background video by removing the object defined in the mask.

        Args:
            video_path: Path to original video
            mask_path: Path to 'total mask' (object + shadow)
            output_path: Path to save result
        """
        self.load_model()

        # Load video and mask
        video = load_video(str(video_path))
        mask = load_video(str(mask_path))

        # Preprocess
        # Check lengths
        min_len = min(len(video), len(mask))
        video = video[:min_len]
        mask = mask[:min_len]

        # Create conditions
        cond_video = LTXVideoCondition(video=video, frame_index=0)
        cond_mask = LTXVideoCondition(video=mask, frame_index=0)

        logger.info("Running OmnimatteZero inference (Object Removal)...")
        with torch.no_grad():
            result = self.pipe.my_call(
                conditions=[cond_video, cond_mask],
                prompt="Empty",  # Prompt is ignored or set to empty for removal
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                num_frames=len(video),
                frame_rate=24,  # Should match input?
            )

        # Save output
        export_to_video(result.frames[0], str(output_path), fps=24)
        logger.info(f"Background saved to {output_path}")

    def extract_foreground(
        self,
        video_path: Path,
        background_path: Path,
        output_path: Path,
        height: int = 512,
        width: int = 768,
    ):
        """
        Extract foreground layer (object + shadow) using latent difference.

        Args:
            video_path: Original video
            background_path: Generated background video (from remove_object)
            output_path: Path to save foreground video
        """
        self.load_model()

        video_p = load_video(str(video_path))
        video_bg = load_video(str(background_path))

        # Preprocess to tensor
        video_p = self.pipe.video_processor.preprocess_video(
            video_p, width=width, height=height
        ).to(device=self.device, dtype=self.dtype)
        video_bg = self.pipe.video_processor.preprocess_video(
            video_bg, width=width, height=height
        ).to(device=self.device, dtype=self.dtype)

        nframes = min(video_p.shape[2], video_bg.shape[2])
        video_p = video_p[:, :, :nframes]
        video_bg = video_bg[:, :, :nframes]

        logger.info("Computing latent difference...")
        with torch.no_grad():
            # Encode
            posterior_p = self.pipe.vae.encode(video_p).latent_dist
            z_p = posterior_p.mode()

            posterior_bg = self.pipe.vae.encode(video_bg).latent_dist
            z_bg = posterior_bg.mode()

            # Difference
            z_diff = z_p - z_bg

            # Decode
            temb = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            foreground_tensor = self.pipe.vae.decode(z_diff, temb).sample

            # Postprocess
            video_foreground = tensor_video_to_pil_images(
                (
                    (
                        self.pipe.video_processor.postprocess_video(
                            foreground_tensor, output_type="pt"
                        )[0]
                        * 255
                    )
                    .long()
                    .permute(0, 2, 3, 1)
                )
            )

        export_to_video(video_foreground, str(output_path), fps=24)
        logger.info(f"Foreground saved to {output_path}")
