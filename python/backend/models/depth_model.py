"""Depth inference helper wrapping Depth Anything 3 metric models."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
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
        # Use HuggingFace default cache (~/.cache/huggingface/hub/)
        model = DepthAnything3.from_pretrained(self.model_id)
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

        # Use provided process_res or default to self.process_res
        res = process_res if process_res is not None else self.process_res

        # Debug: Check device and res
        # param_device = next(model.parameters()).device
        # print(f"[DepthModel] Inferring on {param_device} with res={res}. Frame shape: {frame.shape}")

        # Ensure frame is numpy array (already should be)
        if isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame

        prediction = model.inference(
            [frame_np],
            process_res=res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        depth = np.array(prediction.depth[0], dtype=np.float32, copy=True)

        # Resize to target size if provided, otherwise to original frame size
        tgt_w, tgt_h = target_size if target_size else (frame_np.shape[1], frame_np.shape[0])
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
        self._worker_pool: DeviceWorkerPool | None = None
        self._max_workers = settings.inference_worker_count
        self._executor: ThreadPoolExecutor | None = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self._executor

    def _ensure_worker_pool(self) -> DeviceWorkerPool:
        if self._worker_pool is not None:
            return self._worker_pool

        from backend.workers.da3_worker import DA3DeviceWorker

        self._worker_pool = DeviceWorkerPool(
            worker_class=DA3DeviceWorker,
            device_spec=self.device_spec,
            worker_kwargs={
                "model_id": self.model_id,
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
        executor = self._get_executor()
        return await loop.run_in_executor(
            executor, self.infer_depth, frame, process_res, target_size
        )

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

    def infer_depth_batch(
        self,
        frames: list[np.ndarray],
        target_sizes: list[tuple[int, int]] | None = None,
        process_res: int | None = None,
        batch_size: int | None = None,
    ) -> list[DepthPrediction]:
        """Process multiple frames in parallel across devices.

        Uses DeviceWorkerPool.map() for efficient parallel processing.
        Supports per-worker batch inference for better throughput.

        Args:
            frames: List of input frames
            target_sizes: Optional list of target sizes (W, H) for each frame output
            process_res: Optional processing resolution override
            batch_size: Optional batch size for per-worker batch inference.
                       If None, processes one frame per worker call.

        Returns:
            List of DepthPrediction objects
        """
        pool = self._ensure_worker_pool()

        # Process in parallel using pool.map with optional batch processing
        results: list[DepthPrediction] = []
        for i, (depth, z_min, z_max) in enumerate(pool.map(frames, batch_size=batch_size)):
            tgt_w, tgt_h = (
                target_sizes[i] if target_sizes else (frames[i].shape[1], frames[i].shape[0])
            )
            depth = self._resize_depth(depth, tgt_h, tgt_w)
            results.append(DepthPrediction(depth=depth, z_min=z_min, z_max=z_max))

        return results

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
