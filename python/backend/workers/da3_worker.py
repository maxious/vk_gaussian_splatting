"""DA3 device worker for multi-GPU/XPU depth inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.datasets import FrameDataset, frame_dataset_factory
from common.device_worker_pool import DeviceWorker as BaseDeviceWorker

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    DepthAnything3 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DA3DeviceWorker(
    BaseDeviceWorker[np.ndarray, tuple[np.ndarray, float, float, np.ndarray | None]]
):
    INTERNAL_MODEL = None

    def __init__(
        self,
        device: str,
        worker_id: int,
        model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        process_res: int = 518,
    ):
        super().__init__(device, worker_id, model_id=model_id, process_res=process_res)
        self.model_id = model_id
        self.process_res = process_res

    def load_model(self) -> None:
        if DepthAnything3 is None:
            raise RuntimeError(
                'depth-anything-3 package is not installed; run `uv pip install "videodepthviewer3d[inference]"`.'
            )

        torch.set_float32_matmul_precision("high")

        logger.info(f"Worker {self.worker_id}: Loading DA3 model {self.model_id} on {self.device}")
        # Use HuggingFace default cache (~/.cache/huggingface/hub/)
        self.model = DepthAnything3.from_pretrained(self.model_id)
        self.model = self.model.to(self.device).eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        logger.info(f"Worker {self.worker_id}: DA3 model ready on {self.device}")

    def _convert_tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert CHW tensor to HWC numpy array in [0, 255] range."""
        tensor = tensor.cpu()
        # Denormalize from ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device, dtype=tensor.dtype).view(
            3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device, dtype=tensor.dtype).view(
            3, 1, 1
        )
        tensor = tensor * std + mean
        tensor = tensor.clamp(0, 1)
        # Convert to HWC and uint8
        arr = tensor.permute(1, 2, 0).numpy()
        return (arr * 255).astype(np.uint8)

    def process_item(self, item: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray | None]:
        prediction = self.model.inference(
            [item],
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

        # Return 4-tuple for compatibility with MoGe worker
        return (depth, z_min, z_max, None)

    def process_batch(
        self, items: list[np.ndarray]
    ) -> list[tuple[np.ndarray, float, float, np.ndarray | None]]:
        """Process batch of frames using model batch inference.

        Passes all frames to model.inference() at once for better throughput.
        """
        if not items:
            return []

        # Process all frames in a single model call (true batch inference)
        prediction = self.model.inference(
            items,
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )

        results = []
        for i, depth in enumerate(prediction.depth):
            depth = np.array(depth, dtype=np.float32, copy=True)
            depth = np.nan_to_num(depth, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

            z_min = float(np.percentile(depth, 1))
            z_max = float(np.percentile(depth, 99))

            if z_max <= z_min + 1e-6:
                z_min = 0.5
                z_max = 10.0

            results.append((depth, z_min, z_max, None))

        return results

    def process_batch_dataloader(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> list[tuple[np.ndarray, float, float, np.ndarray | None]]:
        """Process frames using DataLoader for parallel loading and pinned memory.

        Uses TRUE batch inference - all frames in a batch are processed at once
        via model.inference() for optimal throughput.
        """
        dataset = frame_dataset_factory(
            frame_paths=frame_paths,
            timestamps_ms=timestamps_ms,
            image_mode="RGB",
        )

        # Determine pin_memory based on device
        pin_memory = self.device in ("cuda", "xpu")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_frames,
        )

        results = []
        for batch in dataloader:
            indices, frames, batch_timestamps, H, W, masks = batch
            # frames: (B, C, H, W) tensor in [0, 1] range

            # Convert tensor batch to numpy for DA3 model
            # True batching - all frames at once, not loop-per-frame
            batch_np = self._convert_tensor_batch_to_numpy(frames)

            # TRUE BATCH INFERENCE - process entire batch at once
            batch_results = self.process_batch(batch_np)
            results.extend(batch_results)

        return results

    def _convert_tensor_batch_to_numpy(self, frames: torch.Tensor) -> list[np.ndarray]:
        """Convert a batch of float32 tensors to numpy arrays efficiently.

        Args:
            frames: (B, C, H, W) float32 tensor in [0, 1] range

        Returns:
            List of (H, W, C) uint8 numpy arrays
        """
        # Handle float32 tensors from DataLoader (denormalize from ImageNet)
        if frames.dtype == torch.float32:
            # Denormalize from ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device, dtype=frames.dtype)
            std = torch.tensor([0.229, 0.224, 0.225], device=frames.device, dtype=frames.dtype)
            frames = frames * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
            frames = frames.clamp(0, 1)

        # Convert to numpy: (B, C, H, W) -> [(H, W, C), ...]
        frames_np = frames.cpu().numpy()
        frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

        # Permute to HWC for each frame in batch
        return [frames_np[i].transpose(1, 2, 0) for i in range(frames_np.shape[0])]

    @staticmethod
    def _collate_frames(
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom collate function for FrameDataset items."""
        indices, images, timestamps, H, W, masks = zip(*batch)
        return (
            torch.stack(indices),
            torch.stack(images),
            list(timestamps),
            torch.stack(H),
            torch.stack(W),
            torch.stack(masks),
        )
