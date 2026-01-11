"""MoGe depth inference worker."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

from common.device_worker_pool import DeviceWorker

logger = logging.getLogger(__name__)


class MoGEDeviceWorker(
    DeviceWorker[np.ndarray, tuple[np.ndarray, float, float, np.ndarray | None]]
):
    def __init__(self, device: str, worker_id: int, model_id: str = "Ruicheng/moge-2-vitl-normal"):
        super().__init__(device, worker_id)
        self.model_id = model_id
        self.model = None
        self.resolution_level = 9  # Default resolution level

    def load_model(self):
        try:
            from moge.model import import_model_class_by_version
        except ImportError:
            # Fallback if moge is vendored in python/moge but not installed as package
            import sys
            from pathlib import Path

            # Assume python/moge is available in sys.path or relative
            # If running from python/ directory, it should be importable as moge
            try:
                from moge.model import import_model_class_by_version
            except ImportError:
                raise RuntimeError("Could not import moge. Please install dependencies.")

        # Determine version
        version = "v2" if "moge-2" in self.model_id else "v1"

        logger.info(f"Loading MoGe model {self.model_id} on {self.device}")

        # Load model
        ModelClass = import_model_class_by_version(version)
        self.model = ModelClass.from_pretrained(self.model_id).to(self.device).eval()

        # Use fp16 for speed if on CUDA
        if "cuda" in self.device:
            self.model.half()

        logger.info(f"MoGe model loaded on {self.device}")

    def process_item(self, item: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray | None]:
        """Process a single frame.

        Returns:
            (depth_map, z_min, z_max, normal_map)
        """
        if self.model is None:
            self.load_model()

        # MoGe expects RGB float tensor (0..1)
        if item.dtype == np.uint8:
            img_tensor = torch.from_numpy(item).float() / 255.0
        else:
            img_tensor = torch.from_numpy(item).float()

        # (H, W, 3) -> (3, H, W)
        img_tensor = img_tensor.permute(2, 0, 1).to(self.device)

        if "cuda" in self.device:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            output = self.model.infer(img_tensor, resolution_level=self.resolution_level)

        depth = output["depth"].cpu().float().numpy()
        mask = output["mask"].cpu().float().numpy()

        # Normals if available
        normals = None
        if "normal" in output:
            # (H, W, 3) -1..1
            normals = output["normal"].cpu().float().numpy()

        # MoGe metric depth can have outliers or be zero
        # Mask valid pixels
        valid_mask = mask > 0.5

        if valid_mask.sum() > 0:
            z_min = float(depth[valid_mask].min())
            z_max = float(depth[valid_mask].max())
        else:
            z_min = 0.1
            z_max = 10.0

        return depth, z_min, z_max, normals
