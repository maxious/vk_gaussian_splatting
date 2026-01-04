"""Sharp Gaussian Processor with optimized I/O and GPU preprocessing."""

from __future__ import annotations

import logging
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
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
                    # Use pure green for masking (traditional chroma key approach)
                    img[black_pixels] = [0, 255, 0]  # Pure green chroma key
                    mask_applied = True

        return PreloadedFrame(
            index=index,
            path=path,
            timestamp_ms=ts,
            image=img,
            height=H,
            width=W,
            mask_applied=mask_applied,
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

        logger.info(f"Initializing SHARP model with preset: {self.vit_preset}...")
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

        logger.info("SHARP model ready")

    def _preprocess_on_gpu(self, img: np.ndarray) -> "torch.Tensor":
        torch = self._torch
        F = self._F

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
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """Process frames using SHARP with async I/O and GPU preprocessing."""
        import torch
        from sharp.utils import color_space as cs_utils
        from sharp.utils.gaussians import unproject_gaussians
        from tqdm import tqdm

        self._load_model()

        logging.getLogger("sharp.cli.predict").setLevel(logging.WARNING)

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
                disparity_factor = torch.tensor([f_px / W]).float().to(self.device)

                if stats:
                    torch.cuda.synchronize()
                    stats.preprocess_time += time.perf_counter() - preprocess_start

                inference_start = time.perf_counter()
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

                if stats:
                    torch.cuda.synchronize()
                    stats.inference_time += time.perf_counter() - inference_start

                postprocess_start = time.perf_counter()

                # Batch GPU operations before CPU transfer for better performance
                means_tensor = gaussians.mean_vectors.squeeze(0)
                scales_linear = gaussians.singular_values.squeeze(0)
                rotations_tensor = gaussians.quaternions.squeeze(0)
                opacities_prob = gaussians.opacities.squeeze(0)
                colors_linear = gaussians.colors.squeeze(0)

                # Compute all transformations on GPU
                scales_tensor = torch.log(torch.clamp(scales_linear, min=1e-8))
                opacities_prob = torch.clamp(opacities_prob, 1e-6, 1.0 - 1e-6)
                opacities_tensor = torch.log(opacities_prob / (1.0 - opacities_prob))

                # Move color conversion to GPU
                colors_srgb = cs_utils.linearRGB2sRGB(colors_linear)
                colors_sh = (colors_srgb - 0.5) / SH_C0

                # Filter out green screen Gaussians at SH level before RGB conversion
                if remove_black_splats:
                    # Pure green [0,255,0] becomes SH coefficients ≈ [-1.77, 1.77, -1.77]
                    # Filter directly on SH coefficients for maximum accuracy
                    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))  # ≈ 0.282
                    expected_sh_green = torch.tensor(
                        [-1.77, 1.77, -1.77], device=colors_sh.device, dtype=colors_sh.dtype
                    )

                    # Check if SH coefficients match green screen pattern (with tolerance)
                    green_sh_mask = torch.all(
                        torch.abs(colors_sh - expected_sh_green) < 0.5,  # Allow some variation
                        dim=1,
                    )

                    # Keep only non-green screen Gaussians
                    valid_mask = ~green_sh_mask
                    means_tensor = means_tensor[valid_mask]
                    scales_tensor = scales_tensor[valid_mask]
                    rotations_tensor = rotations_tensor[valid_mask]
                    opacities_tensor = opacities_tensor[valid_mask]
                    colors_sh = colors_sh[valid_mask]

                # Single batched CPU transfer (only valid Gaussians)
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
