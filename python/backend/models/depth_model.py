from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from backend.config import get_settings
from common.device_worker_pool import DeviceWorkerPool


def _get_device_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    elif device.type == "xpu":
        return torch.xpu.device(device)
    return device


@dataclass(slots=True)
class DepthPrediction:
    depth: np.ndarray
    z_min: float
    z_max: float
    normals: np.ndarray | None = None


class DepthModel:
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.depth_model_id
        self.device = torch.device(device or settings.device)
        self.process_res = settings.depth_process_res
        self._model: Any = None
        self._semaphore: asyncio.Semaphore | None = None
        self._max_workers = settings.inference_worker_count

    @staticmethod
    def _dense_query_coord(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        ys = (
            (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / max(float(h), 1.0)
        ) * 2.0 - 1.0
        xs = (
            (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / max(float(w), 1.0)
        ) * 2.0 - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        query = torch.stack([grid_y, grid_x], dim=-1).reshape(1, -1, 2)
        return query.expand(batch, -1, -1).contiguous()

    @staticmethod
    def _resize_for_inference(frame: np.ndarray, process_res: int) -> np.ndarray:
        if process_res <= 0:
            return frame

        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim <= process_res:
            return frame

        scale = float(process_res) / float(max_dim)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _depth_tensor_to_numpy(depth_tensor: torch.Tensor, h: int, w: int) -> np.ndarray:
        depth_tensor = depth_tensor.detach().float().cpu()

        if depth_tensor.ndim == 4 and depth_tensor.shape[1] == 1:
            depth = depth_tensor[0, 0].numpy()
        elif depth_tensor.ndim == 3 and depth_tensor.shape[0] == 1:
            if depth_tensor.shape[1] == h and depth_tensor.shape[2] == w:
                depth = depth_tensor[0].numpy()
            else:
                depth = depth_tensor[0].reshape(h, w).numpy()
        elif depth_tensor.ndim == 2:
            depth = depth_tensor.reshape(h, w).numpy()
        else:
            depth = depth_tensor.reshape(h, w).numpy()

        return np.asarray(depth, dtype=np.float32)

    @staticmethod
    def _compute_depth_range(depth: np.ndarray) -> tuple[float, float]:
        valid = np.isfinite(depth) & (depth > 0.0)
        if np.any(valid):
            z_min = float(np.percentile(depth[valid], 1.0))
            z_max = float(np.percentile(depth[valid], 99.0))
        else:
            z_min = 0.5
            z_max = 10.0

        if z_max <= z_min + 1e-6:
            z_min = 0.5
            z_max = 10.0
        return z_min, z_max

    def _resolve_model_path(self) -> str | None:
        raw = (self.model_id or "").strip()
        if raw.lower() in {"", "infinidepth", "infini_depth", "default"}:
            return None

        candidate = Path(raw).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(
                f"InfiniDepth checkpoint not found at '{candidate}'. "
                "Set VIDEO_DEPTH_MODEL_ID to a local checkpoint .pth path."
            )
        return str(candidate)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        if self.device.type not in ("cuda", "xpu"):
            raise RuntimeError(
                f"InfiniDepth requires CUDA or XPU, got device '{self.device}'. "
                "Set VIDEO_DEPTH_DEVICE_SPEC=cuda or xpu."
            )

        try:
            from InfiniDepth.model import InfiniDepth
        except ImportError as exc:
            raise RuntimeError(
                "InfiniDepth package is not available. Install python extras with `--extra infini_depth`."
            ) from exc

        model_path = self._resolve_model_path()
        with _get_device_context(self.device):
            self._model = InfiniDepth(model_path=model_path)
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
                None,
                self.infer_depth,
                frame,
                process_res,
                target_size,
            )

    @property
    def inflight_count(self) -> int:
        if self._semaphore is None:
            return 0
        return self._max_workers - self._semaphore._value

    def infer_depth(
        self,
        frame: np.ndarray,
        process_res: Optional[int] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> DepthPrediction:
        model = self._ensure_model()
        res = process_res if process_res is not None else self.process_res

        frame_np = frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
        frame_proc = self._resize_for_inference(frame_np, res)
        h, w = frame_proc.shape[:2]

        if frame_proc.dtype == np.uint8:
            image = torch.from_numpy(frame_proc).to(torch.float32) / 255.0
        else:
            image = torch.from_numpy(frame_proc).to(torch.float32)
            image = image.clamp(0.0, 1.0)

        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        query_coord = self._dense_query_coord(batch=1, h=h, w=w, device=image.device)

        with torch.no_grad():
            with _get_device_context(self.device):
                output: Any = model.inference(
                    image=image,
                    query_coord=query_coord,
                    use_batch_infer=True,
                )
        pred_depth = output[0]

        depth = self._depth_tensor_to_numpy(pred_depth, h=h, w=w)
        depth = np.nan_to_num(depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        z_min, z_max = self._compute_depth_range(depth)

        tgt_w, tgt_h = target_size if target_size else (frame_np.shape[1], frame_np.shape[0])
        depth = self._resize_depth(depth, tgt_h, tgt_w)

        return DepthPrediction(depth=depth, z_min=z_min, z_max=z_max, normals=None)

    @staticmethod
    def _resize_depth(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if depth.shape == (target_h, target_w):
            return depth

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
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self._executor

    def _ensure_worker_pool(self) -> DeviceWorkerPool:
        if self._worker_pool is not None:
            return self._worker_pool

        from backend.workers.infini_depth_worker import InfiniDepthDeviceWorker

        self._worker_pool = DeviceWorkerPool(
            worker_class=InfiniDepthDeviceWorker,
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
        future, _ = pool.submit(frame)
        depth, z_min, z_max, normals = future.result()

        tgt_w, tgt_h = target_size if target_size else (frame.shape[1], frame.shape[0])
        depth = self._resize_depth(depth, tgt_h, tgt_w)

        if normals is not None:
            normals = cv2.resize(normals, (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)
            normals = np.clip(normals, -1.0, 1.0)

        return DepthPrediction(depth=depth, z_min=z_min, z_max=z_max, normals=normals)

    def infer_depth_batch(
        self,
        frames: list[np.ndarray],
        target_sizes: list[tuple[int, int]] | None = None,
        process_res: int | None = None,
        batch_size: int | None = None,
    ) -> list[DepthPrediction]:
        pool = self._ensure_worker_pool()

        results: list[DepthPrediction] = []
        for i, (depth, z_min, z_max, normals) in enumerate(pool.map(frames, batch_size=batch_size)):
            tgt_w, tgt_h = (
                target_sizes[i] if target_sizes else (frames[i].shape[1], frames[i].shape[0])
            )
            depth = self._resize_depth(depth, tgt_h, tgt_w)

            if normals is not None:
                normals = cv2.resize(normals, (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)
                normals = np.clip(normals, -1.0, 1.0)

            results.append(DepthPrediction(depth=depth, z_min=z_min, z_max=z_max, normals=normals))

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
