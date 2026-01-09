"""Sharp Gaussian Processor with optimized I/O and GPU preprocessing."""

from __future__ import annotations

import logging
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterator

import numpy as np

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)


@dataclass
class ProfilingStats:
    io_time: float = 0.0
    preprocess_time: float = 0.0
    inference_time: float = 0.0
    postprocess_time: float = 0.0
    total_frames: int = 0
    frame_times: list[float] = field(default_factory=list)

    def log_summary(self):
        if self.total_frames == 0:
            return
        logger.info("=== SHARP Processing Profile ===")
        logger.info(f"Total frames: {self.total_frames}")
        logger.info(
            f"I/O (load+mask):     {self.io_time:.2f}s ({self.io_time / self.total_frames * 1000:.1f}ms/frame)"
        )
        logger.info(
            f"Preprocess (resize): {self.preprocess_time:.2f}s ({self.preprocess_time / self.total_frames * 1000:.1f}ms/frame)"
        )
        logger.info(
            f"Inference (GPU):     {self.inference_time:.2f}s ({self.inference_time / self.total_frames * 1000:.1f}ms/frame)"
        )
        logger.info(
            f"Postprocess (CPU):   {self.postprocess_time:.2f}s ({self.postprocess_time / self.total_frames * 1000:.1f}ms/frame)"
        )
        total = self.io_time + self.preprocess_time + self.inference_time + self.postprocess_time
        logger.info(
            f"Total measured:      {total:.2f}s ({total / self.total_frames * 1000:.1f}ms/frame)"
        )
        if self.frame_times:
            avg = sum(self.frame_times) / len(self.frame_times)
            logger.info(f"Avg wall-clock:      {avg * 1000:.1f}ms/frame")


@dataclass
class PreloadedFrame:
    index: int
    path: Path
    timestamp_ms: float
    image: np.ndarray
    height: int
    width: int
    mask_applied: bool = False
    mask: np.ndarray | None = None


class DeviceWorker:
    """Worker that processes frames on a specific device."""

    INTERNAL_SIZE = (1536, 1536)
    SH_C0 = 0.28209479177387814

    def __init__(self, device: str, vit_preset: str = "dinov2l16_384"):
        self.device = device
        self.vit_preset = vit_preset
        self.predictor = None
        self._load_model()

    def _load_model(self):
        import ssl
        import torch
        from sharp.models import create_predictor, PredictorParams

        torch.set_float32_matmul_precision("high")
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]

        params = PredictorParams()
        params.monodepth.patch_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.monodepth.image_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.gaussian_decoder.patch_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.gaussian_decoder.image_encoder_preset = self.vit_preset  # type: ignore[assignment]

        self.predictor = create_predictor(params)

        local_resource_paths = [
            Path("_downloaded_resources/sharp/sharp_2572gikvuh.pt"),
            Path("../_downloaded_resources/sharp/sharp_2572gikvuh.pt"),
        ]
        state_dict = None
        for resource_path in local_resource_paths:
            if resource_path.exists():
                state_dict = torch.load(resource_path, map_location="cpu")
                break

        if state_dict is None:
            url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
            state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

        self.predictor.load_state_dict(state_dict)
        self.predictor.eval().to(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    def process_frame(
        self,
        img: np.ndarray,
        H: int,
        W: int,
        mask: np.ndarray | None = None,
    ) -> GaussianFrame:
        import torch
        import torch.nn.functional as F
        from sharp.utils.gaussians import unproject_gaussians

        img_pt = torch.from_numpy(img).to(self.device, non_blocking=True).float()
        img_pt = img_pt.permute(2, 0, 1).unsqueeze(0) / 255.0

        img_gpu = F.interpolate(
            img_pt,
            size=self.INTERNAL_SIZE,
            mode="bilinear",
            align_corners=True,
        )

        f_px = max(H, W) * 0.8
        disparity_factor = torch.tensor([f_px / W]).float().to(self.device)

        with torch.no_grad():
            gaussians_ndc = self.predictor(img_gpu, disparity_factor)

        intrinsics = (
            torch.tensor(
                [
                    [f_px, 0, W / 2, 0],
                    [0, f_px, H / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .to(self.device)
        )

        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= self.INTERNAL_SIZE[0] / W
        intrinsics_resized[1] *= self.INTERNAL_SIZE[1] / H

        gaussians = unproject_gaussians(
            gaussians_ndc,
            torch.eye(4).to(self.device),
            intrinsics_resized,
            self.INTERNAL_SIZE,
        )

        means_tensor = gaussians.mean_vectors.squeeze(0)
        scales_linear = gaussians.singular_values.squeeze(0)
        rotations_tensor = gaussians.quaternions.squeeze(0)
        opacities_prob = gaussians.opacities.squeeze(0)
        colors_linear = gaussians.colors.squeeze(0)

        scales_tensor = torch.log(torch.clamp(scales_linear, min=1e-8))
        opacities_prob = torch.clamp(opacities_prob, 1e-6, 1.0 - 1e-6)
        opacities_tensor = torch.log(opacities_prob / (1.0 - opacities_prob))
        colors_sh = (colors_linear - 0.5) / self.SH_C0

        valid_mask = torch.ones(means_tensor.shape[0], dtype=torch.bool, device=self.device)

        if mask is not None:
            mask_tensor = (
                torch.from_numpy(mask.astype(np.float32)).to(self.device).unsqueeze(0).unsqueeze(0)
            )

            x = means_tensor[:, 0]
            y = means_tensor[:, 1]
            z = means_tensor[:, 2]

            valid_z = z > 1e-3
            valid_mask = valid_mask & valid_z

            z_safe = torch.where(valid_z, z, torch.ones_like(z))
            u_px = (x * f_px / z_safe) + W / 2.0
            v_px = (y * f_px / z_safe) + H / 2.0

            u_norm = 2.0 * (u_px / W) - 1.0
            v_norm = 2.0 * (v_px / H) - 1.0

            grid_coords = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)

            mask_sampled = torch.nn.functional.grid_sample(
                mask_tensor,
                grid_coords,
                mode="nearest",
                align_corners=False,
                padding_mode="zeros",
            )

            geometric_mask = mask_sampled.reshape(-1) > 0.5
            valid_mask = valid_mask & geometric_mask

        means_tensor = means_tensor[valid_mask]
        scales_tensor = scales_tensor[valid_mask]
        rotations_tensor = rotations_tensor[valid_mask]
        opacities_tensor = opacities_tensor[valid_mask]
        colors_sh = colors_sh[valid_mask]

        means = means_tensor.cpu().numpy()
        scales = scales_tensor.cpu().numpy()
        rotations = rotations_tensor.cpu().numpy()
        opacities = opacities_tensor.cpu().numpy()
        colors = colors_sh.cpu().numpy()

        return GaussianFrame(
            frame_idx=0,
            timestamp_ms=0.0,
            means=means.astype(np.float32),
            scales=scales.astype(np.float32),
            rotations=rotations.astype(np.float32),
            colors=colors.astype(np.float32),
            opacities=opacities.astype(np.float32),
        )


class AsyncImageLoader:
    def __init__(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        masks_dir: Path | None,
        mask_first_frame: bool,
        prefetch_count: int = 4,
        num_workers: int = 2,
    ):
        self.frame_paths = frame_paths
        self.timestamps_ms = timestamps_ms
        self.masks_dir = masks_dir
        self.mask_first_frame = mask_first_frame
        self.prefetch_count = prefetch_count
        self.num_workers = num_workers
        self.queue: Queue[PreloadedFrame | None] = Queue(maxsize=prefetch_count)
        self._executor: ThreadPoolExecutor | None = None
        self._loader_thread: Thread | None = None

    def _load_single_frame(self, index: int) -> PreloadedFrame | None:
        import cv2

        path = self.frame_paths[index]
        ts = self.timestamps_ms[index]

        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to load image: {path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        mask_applied = False
        valid_mask = None

        if self.masks_dir is not None and (self.mask_first_frame or index > 0):
            mask_path = None
            for ext in [path.suffix, ".png", ".jpg", ".jpeg"]:
                candidate = self.masks_dir / f"{path.stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is not None:
                mask = cv2.imread(str(mask_path))
                if mask is not None:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    black_pixels = np.all(mask == 0, axis=2)

                    valid_mask = ~black_pixels
                    mask_applied = True

                    img[black_pixels] = 0

        return PreloadedFrame(
            index=index,
            path=path,
            timestamp_ms=ts,
            image=img,
            height=H,
            width=W,
            mask_applied=mask_applied,
            mask=valid_mask if mask_applied else None,
        )

    def _loader_worker(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            next_to_submit = 0
            next_to_yield = 0
            n_frames = len(self.frame_paths)

            while next_to_yield < n_frames:
                while len(futures) < self.prefetch_count and next_to_submit < n_frames:
                    future = executor.submit(self._load_single_frame, next_to_submit)
                    futures.append((next_to_submit, future))
                    next_to_submit += 1

                if futures:
                    idx, future = futures.pop(0)
                    result = future.result()
                    self.queue.put(result)
                    next_to_yield += 1

        self.queue.put(None)

    def start(self):
        self._loader_thread = Thread(target=self._loader_worker, daemon=True)
        self._loader_thread.start()

    def __iter__(self) -> Iterator[PreloadedFrame]:
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item

    def stop(self):
        if self._loader_thread:
            self._loader_thread.join(timeout=5.0)


class SharpGaussianProcessor(GaussianProcessor):
    """Process images to Gaussian splats using Apple's SHARP model with optimized pipeline."""

    INTERNAL_SIZE = (1536, 1536)

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        vit_preset: str = "dinov2l16_384",
        prefetch_count: int = 4,
        num_io_workers: int = 2,
        enable_profiling: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.vit_preset = vit_preset
        self.prefetch_count = prefetch_count
        self.num_io_workers = num_io_workers
        self.enable_profiling = enable_profiling
        self.predictor = None
        self._torch = None
        self._F = None

    @staticmethod
    def _discover_devices(device_spec: str = "auto") -> list[str]:
        """Discover all available GPU devices based on device_spec.

        Args:
            device_spec: Device specification like 'auto', 'cuda', 'xpu', 'cpu', 'cuda:0,1', 'xpu:0,1'

        Returns:
            List of device strings (e.g., ['cuda:0', 'cuda:1'] or ['xpu:0'])
        """
        import torch

        if ":" in device_spec:
            device_type, device_indices = device_spec.split(":", 1)
            indices = [int(i.strip()) for i in device_indices.split(",")]
            return [f"{device_type}:{i}" for i in indices]

        if device_spec == "auto":
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                devices = [f"cuda:{i}" for i in range(num_devices)]
                logger.info(f"Auto-detected {num_devices} CUDA device(s): {devices}")
                return devices

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                num_devices = torch.xpu.device_count()
                devices = [f"xpu:{i}" for i in range(num_devices)]
                logger.info(f"Auto-detected {num_devices} XPU device(s): {devices}")
                return devices

            if torch.backends.mps.is_available():
                logger.info("Auto-detected MPS device (Apple Silicon)")
                return ["mps"]

            logger.info("No GPU detected, using CPU")
            return ["cpu"]

        if device_spec == "cuda":
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                devices = [f"cuda:{i}" for i in range(num_devices)]
                logger.info(f"Using {num_devices} CUDA device(s): {devices}")
                return devices
            logger.warning("CUDA requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "xpu":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                num_devices = torch.xpu.device_count()
                devices = [f"xpu:{i}" for i in range(num_devices)]
                logger.info(f"Using {num_devices} XPU device(s): {devices}")
                return devices
            logger.warning("XPU requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "mps":
            if torch.backends.mps.is_available():
                logger.info("Using MPS device")
                return ["mps"]
            logger.warning("MPS requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "cpu":
            logger.info("Using CPU")
            return ["cpu"]

        logger.warning(f"Unknown device spec '{device_spec}', falling back to CPU")
        return ["cpu"]

    def _load_model(self):
        if self.predictor is not None:
            return

        import torch
        import torch.nn.functional as F
        import ssl
        from sharp.models import create_predictor, PredictorParams

        self._torch = torch
        self._F = F

        # Enable TensorFloat32 for better performance on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")

        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]

        logger.info(
            f"Initializing SHARP model with preset: {self.vit_preset} on device {self.device}..."
        )
        params = PredictorParams()

        params.monodepth.patch_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.monodepth.image_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.gaussian_decoder.patch_encoder_preset = self.vit_preset  # type: ignore[assignment]
        params.gaussian_decoder.image_encoder_preset = self.vit_preset  # type: ignore[assignment]

        self.predictor = create_predictor(params)

        if self.model_path and self.model_path.exists():
            logger.info(f"Loading SHARP weights from local file: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location="cpu")
        else:
            local_resource_path = Path("_downloaded_resources/sharp/sharp_2572gikvuh.pt")
            if not local_resource_path.exists():
                local_resource_path = Path("../_downloaded_resources/sharp/sharp_2572gikvuh.pt")

            if local_resource_path.exists():
                logger.info(f"Loading SHARP weights from: {local_resource_path}")
                state_dict = torch.load(local_resource_path, map_location="cpu")
            else:
                url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
                logger.info(f"Downloading SHARP weights from {url}...")
                state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

        self.predictor.load_state_dict(state_dict)
        self.predictor.eval().to(self.device)

        # torch.compile was tested and found to slow down performance - removed

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        logger.info(f"SHARP model ready on {self.device}")

    def _preprocess_on_gpu(self, img: np.ndarray):
        import torch
        import torch.nn.functional as F

        img_pt = torch.from_numpy(img).to(self.device, non_blocking=True).float()
        img_pt = img_pt.permute(2, 0, 1).unsqueeze(0) / 255.0

        img_resized = F.interpolate(
            img_pt,
            size=self.INTERNAL_SIZE,
            mode="bilinear",
            align_corners=True,
        )

        return img_resized

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = True,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames using SHARP with multi-device parallel processing."""
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        logging.getLogger("sharp.cli.predict").setLevel(logging.WARNING)

        devices = self._discover_devices(self.device)
        logger.info(f"Using {len(devices)} device(s): {devices}")

        if len(devices) == 1:
            return self._process_single_device(
                frame_paths, timestamps_ms, masks_dir, mask_first_frame
            )

        return self._process_multi_device(
            frame_paths, timestamps_ms, masks_dir, mask_first_frame, devices
        )

    def _process_single_device(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        masks_dir: Path | None,
        mask_first_frame: bool,
    ) -> list[GaussianFrame]:
        """Process frames on a single device (original async I/O pipeline)."""
        import torch.nn.functional as F
        from sharp.utils.gaussians import unproject_gaussians
        from tqdm import tqdm

        self._load_model()

        SH_C0 = 0.28209479177387814
        stats = ProfilingStats() if self.enable_profiling else None

        loader = AsyncImageLoader(
            frame_paths=frame_paths,
            timestamps_ms=timestamps_ms,
            masks_dir=masks_dir,
            mask_first_frame=mask_first_frame,
            prefetch_count=self.prefetch_count,
            num_workers=self.num_io_workers,
        )
        loader.start()

        results = []
        pbar = tqdm(total=len(frame_paths), desc="SHARP processing", unit="frame")

        try:
            for preloaded in loader:
                frame_start = time.perf_counter()

                if stats:
                    io_end = time.perf_counter()

                pbar.set_postfix(file=preloaded.path.name)

                preprocess_start = time.perf_counter()
                img_gpu = self._preprocess_on_gpu(preloaded.image)
                H, W = preloaded.height, preloaded.width
                f_px = max(H, W) * 0.8
                disparity_factor = self._torch.tensor([f_px / W]).float().to(self.device)

                if stats:
                    stats.preprocess_time += time.perf_counter() - preprocess_start

                inference_start = time.perf_counter()
                with self._torch.no_grad():
                    gaussians_ndc = self.predictor(img_gpu, disparity_factor)

                intrinsics = (
                    self._torch.tensor(
                        [
                            [f_px, 0, W / 2, 0],
                            [0, f_px, H / 2, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    )
                    .float()
                    .to(self.device)
                )

                intrinsics_resized = intrinsics.clone()
                intrinsics_resized[0] *= self.INTERNAL_SIZE[0] / W
                intrinsics_resized[1] *= self.INTERNAL_SIZE[1] / H

                gaussians = unproject_gaussians(
                    gaussians_ndc,
                    self._torch.eye(4).to(self.device),
                    intrinsics_resized,
                    self.INTERNAL_SIZE,
                )

                if stats:
                    stats.inference_time += time.perf_counter() - inference_start

                postprocess_start = time.perf_counter()

                means_tensor = gaussians.mean_vectors.squeeze(0)
                scales_linear = gaussians.singular_values.squeeze(0)
                rotations_tensor = gaussians.quaternions.squeeze(0)
                opacities_prob = gaussians.opacities.squeeze(0)
                colors_linear = gaussians.colors.squeeze(0)

                scales_tensor = self._torch.log(self._torch.clamp(scales_linear, min=1e-8))
                opacities_prob = self._torch.clamp(opacities_prob, 1e-6, 1.0 - 1e-6)
                opacities_tensor = self._torch.log(opacities_prob / (1.0 - opacities_prob))
                colors_sh = (colors_linear - 0.5) / SH_C0

                valid_mask = self._torch.ones(
                    means_tensor.shape[0], dtype=self._torch.bool, device=self.device
                )

                if preloaded.mask is not None:
                    mask_np = preloaded.mask.astype(np.float32)
                    mask_tensor = (
                        self._torch.from_numpy(mask_np).to(self.device).unsqueeze(0).unsqueeze(0)
                    )

                    x = means_tensor[:, 0]
                    y = means_tensor[:, 1]
                    z = means_tensor[:, 2]

                    valid_z = z > 1e-3
                    valid_mask = valid_mask & valid_z

                    z_safe = self._torch.where(valid_z, z, self._torch.ones_like(z))

                    u_px = (x * f_px / z_safe) + W / 2.0
                    v_px = (y * f_px / z_safe) + H / 2.0

                    u_norm = 2.0 * (u_px / W) - 1.0
                    v_norm = 2.0 * (v_px / H) - 1.0

                    grid_coords = (
                        self._torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
                    )

                    mask_sampled = self._torch.nn.functional.grid_sample(
                        mask_tensor,
                        grid_coords,
                        mode="nearest",
                        align_corners=False,
                        padding_mode="zeros",
                    )

                    geometric_mask = mask_sampled.reshape(-1) > 0.5
                    valid_mask = valid_mask & geometric_mask

                means_tensor = means_tensor[valid_mask]
                scales_tensor = scales_tensor[valid_mask]
                rotations_tensor = rotations_tensor[valid_mask]
                opacities_tensor = opacities_tensor[valid_mask]
                colors_sh = colors_sh[valid_mask]

                means = means_tensor.cpu().numpy()
                scales = scales_tensor.cpu().numpy()
                rotations = rotations_tensor.cpu().numpy()
                opacities = opacities_tensor.cpu().numpy()
                colors = colors_sh.cpu().numpy()

                if stats:
                    stats.postprocess_time += time.perf_counter() - postprocess_start

                results.append(
                    GaussianFrame(
                        frame_idx=preloaded.index,
                        timestamp_ms=preloaded.timestamp_ms,
                        means=means.astype(np.float32),
                        scales=scales.astype(np.float32),
                        rotations=rotations.astype(np.float32),
                        colors=colors.astype(np.float32),
                        opacities=opacities.astype(np.float32),
                    )
                )

                frame_time = time.perf_counter() - frame_start
                if stats:
                    stats.frame_times.append(frame_time)
                    stats.total_frames += 1

                pbar.update(1)

        finally:
            loader.stop()
            pbar.close()

        if stats:
            stats.log_summary()

        return results

    def _process_multi_device(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        masks_dir: Path | None,
        mask_first_frame: bool,
        devices: list[str],
    ) -> list[GaussianFrame]:
        """Process frames across multiple devices using process pool (spawn)."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        n_devices = len(devices)
        n_frames = len(frame_paths)
        batch_size = 4  # Frames per batch per device

        logger.info(
            f"Distributing {n_frames} frames across {n_devices} device(s) "
            f"(~{n_frames // n_devices} frames per device, batch size {batch_size})"
        )

        # PyTorch requires 'spawn' context for multiprocessing with CUDA/XPU
        ctx = multiprocessing.get_context("spawn")

        # Increased prefetch count to keep CPU buffer full
        # 10 batches per device is a reasonable buffer
        prefetch_count = max(50, n_devices * batch_size * 4)

        loader = AsyncImageLoader(
            frame_paths=frame_paths,
            timestamps_ms=timestamps_ms,
            masks_dir=masks_dir,
            mask_first_frame=mask_first_frame,
            prefetch_count=prefetch_count,
            num_workers=self.num_io_workers,
        )
        loader.start()

        frame_queue: list[tuple[int, PreloadedFrame]] = []
        for idx, preloaded in enumerate(loader):
            if preloaded is not None:
                frame_queue.append((idx, preloaded))

        results: list[tuple[int, GaussianFrame] | None] = [None] * n_frames
        pbar = tqdm(total=n_frames, desc="Multi-device SHARP processing", unit="frame")

        # Create one executor per device to ensure affinity and persistent model loading
        executors = {}
        for device in devices:
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(device, self.vit_preset),
            )
            executors[device] = executor

        try:
            futures = {}
            current_frame_idx = 0

            # Distribute batches to devices
            while current_frame_idx < len(frame_queue):
                for device in devices:
                    if current_frame_idx >= len(frame_queue):
                        break

                    # Create batch
                    batch_end = min(current_frame_idx + batch_size, len(frame_queue))
                    batch = frame_queue[current_frame_idx:batch_end]
                    current_frame_idx = batch_end

                    if not batch:
                        continue

                    # Submit batch to specific device executor
                    future = executors[device].submit(_worker_process_batch, batch)
                    futures[future] = device

            # Collect results
            for future in as_completed(futures):
                device = futures[future]
                try:
                    batch_results = future.result()
                    for idx, result in batch_results:
                        results[idx] = (idx, result)
                        pbar.update(1)
                        pbar.set_postfix(device=f"{device}", file=f"frame_{idx:06d}.png")
                except Exception as e:
                    logger.error(f"Error processing batch on {device}: {e}")

        finally:
            pbar.close()
            loader.stop()
            for executor in executors.values():
                executor.shutdown()

        sorted_results = sorted(
            [r for r in results if r is not None and r[1] is not None], key=lambda x: x[0]
        )

        return [r[1] for r in sorted_results]


_GLOBAL_WORKER = None


def _init_worker(device: str, vit_preset: str):
    """Initialize global worker instance."""
    global _GLOBAL_WORKER
    _GLOBAL_WORKER = DeviceWorker(device, vit_preset)


def _worker_process_batch(
    batch: list[tuple[int, PreloadedFrame]],
) -> list[tuple[int, GaussianFrame]]:
    """Process a batch of frames using the global worker."""
    results = []
    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialized!")

    for frame_idx, preloaded in batch:
        try:
            result = _GLOBAL_WORKER.process_frame(
                preloaded.image,
                preloaded.height,
                preloaded.width,
                preloaded.mask if preloaded.mask_applied else None,
            )
            result.frame_idx = frame_idx
            result.timestamp_ms = preloaded.timestamp_ms
            results.append((frame_idx, result))
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
            # Continue processing batch, missing frames will be handled by main process

    return results
