from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from common.device_worker_pool import DeviceWorker
from InfiniDepth.utils.moge_utils import estimate_metric_depth_and_intrinsics_with_moge2


def _depth_to_disparity(depth: torch.Tensor) -> torch.Tensor:
    disp = depth.clone()
    valid = disp > 0
    disp[valid] = 1.0 / disp[valid]
    return disp


def _get_device_context(device: str):
    """Get device context manager for CUDA or XPU."""
    if device.startswith("cuda"):
        return torch.cuda.device(torch.device(device))
    elif device.startswith("xpu"):
        return torch.xpu.device(torch.device(device))
    else:
        return torch.device(device)


logger = logging.getLogger(__name__)


class InfiniDepthDeviceWorker(DeviceWorker[np.ndarray, tuple[np.ndarray, float, float, None]]):
    def __init__(
        self,
        device: str,
        worker_id: int,
        model_id: Optional[str] = None,
        process_res: int = 640,
    ) -> None:
        super().__init__(device, worker_id, model_id=model_id, process_res=process_res)
        self.model_id = model_id
        self.process_res = int(process_res)

    def _resolve_model_path(self) -> tuple[Optional[str], Optional[str]]:
        if self.model_id and self.model_id.strip().lower() not in {
            "infinidepth",
            "infini_depth",
            "default",
            "",
            None,
        }:
            candidate = Path(self.model_id).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(
                    f"InfiniDepth checkpoint not found at '{candidate}'. "
                    "Set VIDEO_DEPTH_MODEL_ID to a local checkpoint .pth path."
                )
            return str(candidate), None

        from huggingface_hub import snapshot_download

        try:
            cache_dir = snapshot_download("ritianyu/InfiniDepth", allow_patterns=["*.ckpt"])
            model_path = Path(cache_dir) / "infinidepth.ckpt"
            moge_path = Path(cache_dir) / "moge2.pt"
            return str(model_path), str(moge_path) if moge_path.exists() else None
        except Exception:
            return None, None

    def load_model(self) -> None:
        if not (self.device.startswith("cuda") or self.device.startswith("xpu")):
            raise RuntimeError(
                f"InfiniDepth worker requires CUDA or XPU device, got '{self.device}'. "
                "Set VIDEO_DEPTH_DEVICE_SPEC=cuda or xpu."
            )

        try:
            from InfiniDepth.model import InfiniDepth
        except ImportError as exc:
            raise RuntimeError(
                "InfiniDepth package is not available. Install python extras with `--extra infini_depth`."
            ) from exc

        model_path, self.moge_path = self._resolve_model_path()

        with _get_device_context(self.device):
            self.model = InfiniDepth(model_path=model_path)
        logger.info(
            "InfiniDepth worker %s loaded model on %s (checkpoint=%s, moge=%s)",
            self.worker_id,
            self.device,
            model_path or "<none>",
            self.moge_path or "<default>",
        )

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
            # Scale down to process_res while maintaining aspect ratio
            # Round to nearest multiple of 16 for DINOv3 compatibility
            scale = float(process_res) / float(max_dim)
            new_w = max(16, int(round(w * scale / 16) * 16))
            new_h = max(16, int(round(h * scale / 16) * 16))
            if (new_w, new_h) != (w, h):
                return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return frame

        # Scale up if needed, also round to nearest multiple of 16
        new_w = max(16, int(round(w / 16) * 16))
        new_h = max(16, int(round(h / 16) * 16))
        if (new_w, new_h) != (w, h):
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

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

    def process_item(self, item: np.ndarray) -> tuple[np.ndarray, float, float, None]:
        if self.model is None:
            raise RuntimeError("InfiniDepth worker model is not loaded")

        frame = np.asarray(item)
        frame = self._resize_for_inference(frame, self.process_res)
        h, w = frame.shape[:2]

        if frame.dtype == np.uint8:
            image = torch.from_numpy(frame).to(torch.float32) / 255.0
        else:
            image = torch.from_numpy(frame).to(torch.float32)
            image = image.clamp(0.0, 1.0)

        target_device = torch.device(self.device)
        image = image.permute(2, 0, 1).unsqueeze(0).to(target_device, non_blocking=True)

        moge_path = self.moge_path if self.moge_path else "Ruicheng/moge-2-vitl-normal"
        pred_depth_moge, gt_depth_mask, _ = estimate_metric_depth_and_intrinsics_with_moge2(
            image=image,
            pretrained_model_name_or_path=moge_path,
        )
        gt_disp = _depth_to_disparity(pred_depth_moge)
        prompt_disp = _depth_to_disparity(pred_depth_moge)

        query_coord = self._dense_query_coord(batch=1, h=h, w=w, device=image.device)

        with torch.no_grad():
            with _get_device_context(self.device):
                output: Any = self.model.inference(
                    image=image,
                    query_coord=query_coord,
                    gt_depth=gt_disp,
                    gt_depth_mask=gt_depth_mask,
                    prompt_depth=prompt_disp,
                    prompt_mask=gt_depth_mask > 0,
                    use_batch_infer=True,
                )
        pred_disp, _ = output
        pred_depth = 1.0 / torch.clamp(pred_disp, min=1e-3)

        depth = self._depth_tensor_to_numpy(pred_depth, h=h, w=w)
        depth = np.nan_to_num(depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        z_min, z_max = self._compute_depth_range(depth)
        return depth, z_min, z_max, None

    def process_batch(self, items: list[np.ndarray]) -> list[tuple[np.ndarray, float, float, None]]:
        return [self.process_item(item) for item in items]
