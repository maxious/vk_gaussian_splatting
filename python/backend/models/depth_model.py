"""Depth inference helper wrapping Depth Anything 3 metric models."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
import cv2
import torch

from backend.config import get_settings
from common.device_worker_pool import DeviceWorkerPool

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    DepthAnything3 = None  # type: ignore[assignment]


@dataclass(slots=True)
class DepthPrediction:
    """Container for a depth map in meters."""

    depth: np.ndarray
    z_min: float
    z_max: float


class DepthModel:
    """Lazy-loading wrapper around Video Depth Anything / Depth Anything 3."""

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.depth_model_id
        self.device = torch.device(device or settings.device)
        self.process_res = settings.depth_process_res
        self.cache_dir = settings.data_root.parent / "checkpoints"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = None
        self._semaphore: asyncio.Semaphore | None = None
        self._max_workers = settings.inference_worker_count

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if DepthAnything3 is None:
            raise RuntimeError(
                'depth-anything-3 package is not installed; run `uv pip install "videodepthviewer3d[inference]"`.'
            )
        model = DepthAnything3.from_pretrained(self.model_id, cache_dir=str(self.cache_dir))
        self._model = model.to(self.device).eval()
        return self._model

    async def infer_depth_async(
        self,
        frame: np.ndarray,
        process_res: Optional[int] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> DepthPrediction:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_workers)
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self.infer_depth, frame, process_res, target_size
            )

    @property
    def inflight_count(self) -> int:
        if self._semaphore is None:
            return 0
        # semaphore.value is the number of available slots.
        # inflight = max_workers - available
        return self._max_workers - self._semaphore._value

    def infer_depth(
        self,
        frame: np.ndarray,
        process_res: Optional[int] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> DepthPrediction:
        model = self._ensure_model()
        pil = Image.fromarray(frame, mode="RGB")

        # Use provided process_res or default to self.process_res
        res = process_res if process_res is not None else self.process_res

        # Debug: Check device and res
        # param_device = next(model.parameters()).device
        # print(f"[DepthModel] Inferring on {param_device} with res={res}. Frame: {pil.size}")

        prediction = model.inference(
            [pil],
            process_res=res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        depth = np.array(prediction.depth[0], dtype=np.float32, copy=True)

        # Resize to target size if provided, otherwise to original frame size
        tgt_w, tgt_h = target_size if target_size else (pil.width, pil.height)
        depth = self._resize_depth(depth, tgt_h, tgt_w)

        depth = np.nan_to_num(depth, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        z_min = float(np.percentile(depth, 1))
        z_max = float(np.percentile(depth, 99))

        # If all values are 0 (NaN/Inf replaced), use fallback range
        if z_max <= z_min + 1e-6:
            z_min = 0.5
            z_max = 10.0

        return DepthPrediction(depth=depth, z_min=z_min, z_max=z_max)

    @staticmethod
    def _resize_depth(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if depth.shape == (target_h, target_w):
            return depth
        # Use OpenCV for resizing to release GIL (PIL often holds it)
        # cv2.resize expects (width, height)
        # Use INTER_AREA if downscaling, INTER_CUBIC/LINEAR if upscaling
        # But here we just use INTER_LINEAR for speed/quality balance or INTER_AREA for downsampling
        # The original code used INTER_CUBIC.
        # If we are resizing from inference size (small) to target size (medium/large),
        # we should use CUBIC or LINEAR.
        # If we are resizing from inference size (large) to target size (small), AREA is better.

        # Simple heuristic:
        interpolation = cv2.INTER_CUBIC
        if target_w < depth.shape[1] and target_h < depth.shape[0]:
            interpolation = cv2.INTER_AREA

        resized = cv2.resize(depth, (target_w, target_h), interpolation=interpolation)
        return resized.astype(np.float32, copy=False)


_depth_model: DepthModel | None = None
_multi_device_depth_model: MultiDeviceDepthModel | None = None


def get_depth_model(use_multi_device: bool = False) -> DepthModel | MultiDeviceDepthModel:
    global _depth_model, _multi_device_depth_model

    if use_multi_device:
        if _multi_device_depth_model is None:
            _multi_device_depth_model = MultiDeviceDepthModel()
        return _multi_device_depth_model
    else:
        if _depth_model is None:
            _depth_model = DepthModel()
        return _depth_model


class MultiDeviceDepthModel:
    def __init__(self, model_id: Optional[str] = None, device_spec: str = "auto") -> None:
        settings = get_settings()
        self.model_id = model_id or settings.depth_model_id
        self.device_spec = device_spec
        self.process_res = settings.depth_process_res
        self.cache_dir = settings.data_root.parent / "checkpoints"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._worker_pool: DeviceWorkerPool | None = None
        self._max_workers = settings.inference_worker_count

    def _ensure_worker_pool(self) -> DeviceWorkerPool:
        if self._worker_pool is not None:
            return self._worker_pool

        from backend.workers.da3_worker import DA3DeviceWorker

        self._worker_pool = DeviceWorkerPool(
            worker_class=DA3DeviceWorker,
            device_spec=self.device_spec,
            worker_kwargs={
                "model_id": self.model_id,
                "cache_dir": self.cache_dir,
                "process_res": self.process_res,
            },
        )
        return self._worker_pool

    async def infer_depth_async(
        self,
        frame: np.ndarray,
        process_res: Optional[int] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> DepthPrediction:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.infer_depth, frame, process_res, target_size)

    @property
    def inflight_count(self) -> int:
        return 0

    def infer_depth(
        self,
        frame: np.ndarray,
        process_res: Optional[int] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> DepthPrediction:
        pool = self._ensure_worker_pool()

        future, device = pool.submit(frame)
        depth, z_min, z_max = future.result()

        tgt_w, tgt_h = target_size if target_size else (frame.shape[1], frame.shape[0])
        depth = self._resize_depth(depth, tgt_h, tgt_w)

        return DepthPrediction(depth=depth, z_min=z_min, z_max=z_max)

    @staticmethod
    def _resize_depth(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if depth.shape == (target_h, target_w):
            return depth

        interpolation = cv2.INTER_CUBIC
        if target_w < depth.shape[1] and target_h < depth.shape[0]:
            interpolation = cv2.INTER_AREA

        resized = cv2.resize(depth, (target_w, target_h), interpolation=interpolation)
        return resized.astype(np.float32, copy=False)

    def shutdown(self):
        if self._worker_pool is not None:
            self._worker_pool.shutdown()
            self._worker_pool = None
