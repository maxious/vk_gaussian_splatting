"""Generic frame dataset for loading images from disk with optimized I/O.

This dataset is designed for efficient loading of image sequences for ML pipelines.
Features:
- Background loading with torch DataLoader workers
- Pin-memory support for fast CPU→GPU/XPU transfers
- C-contiguous array conversion for zero-copy tensor creation
- Optional mask loading and resizing
- Timestamp tracking
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """Generic dataset for loading image sequences from disk.

    Supports various image formats (PNG, JPG, etc.) and optional mask loading.
    Designed for use with DataLoader for background loading and pinned memory.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = FrameDataset(image_paths, timestamps_ms)
        >>> dataloader = DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True)
        >>> for batch in dataloader:
        ...     indices, images, timestamps = batch
        ...     # images is (B, C, H, W) tensor, ready for GPU
    """

    def __init__(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        transform: callable | None = None,
        image_mode: str = "RGB",
    ):
        """Initialize FrameDataset.

        Args:
            frame_paths: List of paths to image files
            timestamps_ms: List of timestamps for each frame
            masks_dir: Optional directory containing mask files
            mask_first_frame: Whether to apply mask to first frame
            transform: Optional transform function (img, mask) -> (img, mask)
            image_mode: OpenCV color mode ("RGB", "BGR", "GRAYSCALE")
        """
        self.frame_paths = frame_paths
        self.timestamps_ms = timestamps_ms
        self.masks_dir = masks_dir
        self.mask_first_frame = mask_first_frame
        self.transform = transform
        self.image_mode = image_mode

        # Map image_mode to cv2 constants
        mode_map = {
            "RGB": cv2.COLOR_BGR2RGB,
            "BGR": None,  # Keep as BGR
            "GRAYSCALE": cv2.COLOR_BGR2GRAY,
        }
        self.color_convert = mode_map.get(image_mode)

    def __len__(self) -> int:
        return len(self.frame_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load a single image from disk."""
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")

        if self.color_convert is not None:
            img = cv2.cvtColor(img, self.color_convert)

        return img

    def _load_mask(self, image_path: Path, H: int, W: int) -> np.ndarray | None:
        """Load and resize mask for an image."""
        if self.masks_dir is None:
            return None

        if not self.mask_first_frame and image_path == self.frame_paths[0]:
            return None

        mask_path = None
        for ext in [image_path.suffix, ".png", ".jpg", ".jpeg"]:
            candidate = self.masks_dir / f"{image_path.stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            return None

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return mask

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, torch.Tensor, float, int, int, torch.Tensor | None]:
        """Get a single frame.

        Returns:
            Tuple of (index, image_tensor, timestamp_ms, height, width, mask_tensor or None)
            - image_tensor: (C, H, W) float tensor in [0, 1] range
            - mask_tensor: (1, H, W) float tensor or None
        """
        path = self.frame_paths[idx]
        ts = self.timestamps_ms[idx]

        # Load image
        img = self._load_image(path)
        H, W = img.shape[:2]

        # Load mask
        mask = self._load_mask(path, H, W)

        # Apply transforms
        if self.transform:
            img, mask = self.transform(img, mask)

        # Convert to tensor with C-contiguity for zero-copy
        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).permute(2, 0, 1)

        if mask is not None:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros(1, H, W, dtype=torch.float32)

        return idx, img_tensor, ts, H, W, mask_tensor


def frame_dataset_factory(
    frame_paths: list[Path],
    timestamps_ms: list[float],
    masks_dir: Path | None = None,
    mask_first_frame: bool = False,
    transform: callable | None = None,
    image_mode: str = "RGB",
) -> FrameDataset:
    """Factory function for creating FrameDataset with optimal defaults.

    This is the recommended way to create a FrameDataset as it provides
    a consistent interface across the codebase.
    """
    return FrameDataset(
        frame_paths=frame_paths,
        timestamps_ms=timestamps_ms,
        masks_dir=masks_dir,
        mask_first_frame=mask_first_frame,
        transform=transform,
        image_mode=image_mode,
    )
