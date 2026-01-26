"""BEN2 Mask Generator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

logger = logging.getLogger(__name__)


class BiRefNetMaskGenerator:
    """Generate masks using BiRefNet model."""

    def __init__(self, model_id: str = "ZhengPeng7/BiRefNet", device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        self.model = None
        self.transform = None

    def load_model(self) -> None:
        """Load the BiRefNet model."""
        if self.model is not None:
            return

        logger.info(f"Loading BiRefNet model: {self.model_id}...")
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model.to(self.device).eval()

            # Standard transform for BiRefNet
            self.transform = transforms.Compose(
                [
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            logger.info("BiRefNet model loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load BiRefNet model: {e}")

    def generate_mask(self, image_path: Path, refine: bool = False) -> np.ndarray | None:
        """Generate binary mask for an image."""
        self.load_model()

        try:
            pil_image = Image.open(image_path).convert("RGB")
            original_size = pil_image.size
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

        input_images = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # BiRefNet returns list of tensors, we want the last one (highest resolution)
            preds = self.model(input_images)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()

        pred_pil = transforms.ToPILImage()(pred)
        mask_pil = pred_pil.resize(original_size, resample=Image.Resampling.BILINEAR)

        mask_np = np.array(mask_pil)

        return mask_np


class Ben2MaskGenerator:
    """Generate masks using PramaLLC's BEN2 model."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load_model(self) -> None:
        """Load the BEN2 model."""
        if self.model is not None:
            return

        try:
            from ben2 import BEN_Base  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "BEN2 is not installed. Please install it with: "
                "uv pip install 'git+https://github.com/PramaLLC/BEN2.git'"
            )

        logger.info("Loading BEN2 model...")
        self.model = BEN_Base.from_pretrained("PramaLLC/BEN2")
        self.model.to(self.device).eval()
        logger.info("BEN2 model loaded.")

    def generate_mask(self, image_path: Path, refine: bool = False) -> np.ndarray | None:
        """Generate binary mask for an image."""
        self.load_model()

        if self.model is None:
            return None

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

        # BEN2 inference returns a PIL Image (foreground with alpha)
        foreground = self.model.inference(pil_image, refine_foreground=refine)

        # Extract alpha channel as mask
        # BEN2 returns RGBA
        if foreground.mode != "RGBA":
            foreground = foreground.convert("RGBA")

        _, _, _, alpha = foreground.split()

        # Convert to numpy
        mask_np = np.array(alpha)

        # Threshold to binary (0 or 255) just in case
        # BEN2 produces soft mattes, so we might want to keep it soft or threshold it
        # For our "black out background" logic, strict thresholding is safer to avoid dark halos
        # But soft edges are nice for blending.
        # However, the current pipeline just sets pixels to black based on exact 0 match in mask?
        # Let's check sharp.py logic:
        # black_pixels = np.all(mask == 0, axis=2)
        # So if ANY channel is not 0, it keeps the pixel.
        # If we save as grayscale, cv2 reads as BGR (3 identical channels).
        # So if we have gray value 1, it's not 0, so pixel kept.
        # So we don't strictly need to threshold, but standardizing to 0/255 is usually cleaner for "masks".
        # Let's keep it as is (0-255 grayscale) to allow for potential soft masking features later,
        # but technically any non-zero value will be treated as "keep" by current simple logic.

        return mask_np


def generate_masks(
    input_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    pattern: str = "*.jpg",
    refine: bool = False,
    model: str = "ben2",
) -> None:
    """Generate masks for all images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = [pattern.split("*")[-1]] if "*" in pattern else [".jpg", ".jpeg", ".png"]
    image_files = []
    if "*" in pattern:
        image_files = list(input_dir.glob(pattern))
    else:
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        logger.warning(f"No images found in {input_dir} matching {pattern}")
        return

    logger.info(f"Found {len(image_files)} images. Initializing {model}...")

    generator = None
    if model.lower() == "ben2":
        generator = Ben2MaskGenerator(device=device)
    elif model.lower() == "birefnet":
        generator = BiRefNetMaskGenerator(device=device)
    elif model.lower() == "birefnet-lite":
        generator = BiRefNetMaskGenerator(model_id="ZhengPeng7/BiRefNet_lite", device=device)
    else:
        raise ValueError(f"Unknown model: {model}. Options: ben2, birefnet, birefnet-lite")

    for img_path in tqdm(image_files, desc=f"Generating masks ({model})"):
        mask = generator.generate_mask(img_path, refine=refine)

        if mask is not None:
            # Save as PNG
            output_path = output_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(output_path), mask)


def main():
    parser = argparse.ArgumentParser(description="Generate masks using BEN2 or BiRefNet")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input image directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output mask directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pattern", type=str, default="*.jpg", help="Glob pattern for input images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ben2",
        choices=["ben2", "birefnet", "birefnet-lite"],
        help="Model to use for mask generation",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Enable foreground refinement (BEN2 only)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    generate_masks(
        args.input,
        args.output,
        device=args.device,
        pattern=args.pattern,
        refine=args.refine,
        model=args.model,
    )


if __name__ == "__main__":
    main()
