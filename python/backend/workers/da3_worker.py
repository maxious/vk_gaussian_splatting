"""DA3 device worker for multi-GPU/XPU depth inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from common.device_worker_pool import DeviceWorker as BaseDeviceWorker

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    DepthAnything3 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DA3DeviceWorker(BaseDeviceWorker[np.ndarray, tuple[np.ndarray, float, float]]):
    INTERNAL_MODEL = None

    def __init__(
        self,
        device: str,
        worker_id: int,
        model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        cache_dir: str | Path | None = None,
        process_res: int = 518,
    ):
        super().__init__(
            device, worker_id, model_id=model_id, cache_dir=cache_dir, process_res=process_res
        )
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else Path("checkpoints")
        self.process_res = process_res

    def load_model(self) -> None:
        if DepthAnything3 is None:
            raise RuntimeError(
                'depth-anything-3 package is not installed; run `uv pip install "videodepthviewer3d[inference]"`.'
            )

        torch.set_float32_matmul_precision("high")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Worker {self.worker_id}: Loading DA3 model {self.model_id} on {self.device}")
        self.model = DepthAnything3.from_pretrained(self.model_id, cache_dir=str(self.cache_dir))
        self.model = self.model.to(self.device).eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        logger.info(f"Worker {self.worker_id}: DA3 model ready on {self.device}")

    def process_item(self, item: np.ndarray) -> tuple[np.ndarray, float, float]:
        pil = Image.fromarray(item, mode="RGB")

        prediction = self.model.inference(
            [pil],
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )

        depth = np.array(prediction.depth[0], dtype=np.float32, copy=True)
        depth = np.nan_to_num(depth, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        z_min = float(np.percentile(depth, 1))
        z_max = float(np.percentile(depth, 99))

        if z_max <= z_min + 1e-6:
            z_min = 0.5
            z_max = 10.0

        return (depth, z_min, z_max)

    def process_batch(self, items: list[np.ndarray]) -> list[tuple[np.ndarray, float, float]]:
        return [self.process_item(item) for item in items]
