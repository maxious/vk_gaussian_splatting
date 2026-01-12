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

        logger.info(f"Worker {self.worker_id}: Loading MoGe model {self.model_id} on {self.device}")

        # Load model
        ModelClass = import_model_class_by_version(version)
        self.model = ModelClass.from_pretrained(self.model_id).to(self.device).eval()

        # XPU requires explicit dtype conversions
        # MoGe models have specific dtype requirements per component
        if "xpu" in self.device:
            logger.info(f"Worker {self.worker_id}: Applying XPU dtype optimizations...")
            self._apply_xpu_dtype_fixes()
        elif "cuda" in self.device:
            # Use fp16 for speed if on CUDA
            self.model.half()

        logger.info(f"Worker {self.worker_id}: MoGe model ready on {self.device}")

    def _apply_xpu_dtype_fixes(self):
        if self.model is None:
            return

        import torch

        for name, module in self.model.named_modules():
            if hasattr(module, "_xpu_dtype_fixed"):
                continue

            if any(layer_type in name for layer_type in ["LayerNorm", "BatchNorm", "GroupNorm"]):
                module.to(torch.float32)
            elif any(layer_type in name for layer_type in ["encoder", "decoder", "transformer"]):
                try:
                    module.to(torch.bfloat16)
                except Exception:
                    try:
                        module.to(torch.float16)
                    except Exception:
                        logger.warning(
                            f"Could not convert {name} to reduced precision, keeping fp32"
                        )

            object.__setattr__(module, "_xpu_dtype_fixed", True)

        logger.info(f"Worker {self.worker_id}: XPU dtype optimizations applied")

    def process_item(self, item: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray | None]:
        if self.model is None:
            self.load_model()

        model = self.model
        assert model is not None

        if item.dtype == np.uint8:
            img_tensor = torch.from_numpy(item).float() / 255.0
        else:
            img_tensor = torch.from_numpy(item).float()

        img_tensor = img_tensor.permute(2, 0, 1).to(self.device)

        if "cuda" in self.device:
            img_tensor = img_tensor.half()
        elif "xpu" in self.device:
            img_tensor = img_tensor.to(torch.bfloat16)

        with torch.no_grad():
            output = model.infer(img_tensor, resolution_level=self.resolution_level)

        depth = output["depth"].cpu().float().numpy()
        mask = output["mask"].cpu().float().numpy()

        normals = None
        if "normal" in output:
            normals = output["normal"].cpu().float().numpy()

        valid_mask = mask > 0.5

        if valid_mask.sum() > 0:
            z_min = float(depth[valid_mask].min())
            z_max = float(depth[valid_mask].max())
        else:
            z_min = 0.1
            z_max = 10.0

        return depth, z_min, z_max, normals
