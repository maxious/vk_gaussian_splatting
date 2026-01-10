#!/usr/bin/env python3
"""Benchmark multi-XPU DA3 performance: single vs multi-device inference.

This script measures the speedup achieved by distributing DA3 depth inference
across multiple XPU devices using the DeviceWorkerPool.

Usage:
    python benchmark_da3_multi_xpu.py [--num-frames 20] [--warmup 2] [--iterations 3]
    python benchmark_da3_multi_xpu.py --device-spec xpu:0        # Single XPU
    python benchmark_da3_multi_xpu.py --device-spec xpu:0,1      # Dual XPU
    python benchmark_da3_multi_xpu.py --device-spec auto         # Auto-detect
    python benchmark_da3_multi_xpu.py --help

Requirements:
    - Intel XPU backend: `uv sync --extra xpu`
    - Or CUDA backend: `uv sync --extra cuda`
    - Or CPU only: `uv sync --extra cpu`
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence, cast

import cv2
import numpy as np
import torch

# Add python/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backend.models.depth_model import DepthModel, MultiDeviceDepthModel, get_depth_model
from common.device_worker_pool import DeviceWorkerPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    device_spec: str
    num_frames: int
    total_time_ms: float
    per_frame_time_ms: float
    fps: float
    num_devices: int
    iterations: int = 1


@dataclass
class BenchmarkResults:
    """Collection of benchmark results for comparison."""

    single_device: BenchmarkResult | None = None
    multi_device: BenchmarkResult | None = None
    results: list[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
        if result.num_devices == 1:
            self.single_device = result
        else:
            self.multi_device = result

    def print_summary(self) -> None:
        """Print a summary of all benchmark results."""
        print()
        print("=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)

        for result in self.results:
            print()
            print(f"Device Spec: {result.device_spec}")
            print(f"  Frames: {result.num_frames}")
            print(f"  Devices: {result.num_devices}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Total Time: {result.total_time_ms:.1f} ms")
            print(f"  Per Frame: {result.per_frame_time_ms:.1f} ms")
            print(f"  Throughput: {result.fps:.2f} fps")

        if self.single_device and self.multi_device:
            speedup = self.single_device.total_time_ms / self.multi_device.total_time_ms
            print()
            print("-" * 80)
            print("SPEEDUP ANALYSIS")
            print("-" * 80)
            print(
                f"Single Device ({self.single_device.device_spec}): {self.single_device.fps:.2f} fps"
            )
            print(
                f"Multi Device ({self.multi_device.device_spec}): {self.multi_device.fps:.2f} fps"
            )
            print(f"Speedup: {speedup:.2f}x")
            print(f"Expected: ~{self.multi_device.num_devices}x (ideal linear scaling)")

            efficiency = speedup / self.multi_device.num_devices * 100
            print(f"Parallel Efficiency: {efficiency:.1f}%")

        print("=" * 80)


def create_test_frame(width: int = 640, height: int = 360, seed: int = 42) -> np.ndarray:
    """Create a synthetic test frame with realistic image content.

    Args:
        width: Frame width
        height: Frame height
        seed: Random seed for reproducibility

    Returns:
        RGB image as numpy array (H, W, 3) in [0, 255] range
    """
    rng = np.random.default_rng(seed)

    # Create a gradient background
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    background = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.uint8)

    # Add some random shapes to simulate real content
    for _ in range(10):
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        r = rng.integers(10, 50)
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        cv2.circle(background, (x, y), r, color.tolist(), -1)

    # Add some noise
    noise = rng.integers(-10, 10, size=background.shape, dtype=np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return background


def load_test_frames(
    num_frames: int,
    width: int = 640,
    height: int = 360,
    data_dir: Path | None = None,
    video_path: Path | None = None,
) -> list[np.ndarray]:
    """Load or create test frames from video, directory, or synthetic data.

    Args:
        num_frames: Number of frames to load
        width: Frame width for resizing/synthetic frames
        height: Frame height
        data_dir: Optional directory containing image files (frame_XXXXX.png, etc.)
        video_path: Optional video file to decode frames from

    Returns:
        List of RGB frames as numpy arrays
    """
    # Priority: video_path > data_dir > synthetic
    if video_path and video_path.exists():
        logger.info(f"Loading frames from video: {video_path}")
        return load_frames_from_video(video_path, num_frames, width, height)

    if data_dir and data_dir.exists():
        logger.info(f"Loading frames from directory: {data_dir}")
        frames = []

        # Try multiple filename formats: 6-digit (frame_000000.png), 5-digit (frame_00000.png), 4-digit (frame_0000.png)
        for i in range(num_frames):
            frame_path = None
            for fmt in ["06d", "05d", "04d"]:
                candidate = data_dir / f"frame_{i:{fmt}}.png"
                if candidate.exists():
                    frame_path = candidate
                    break
                candidate = data_dir / f"frame_{i:{fmt}}.jpg"
                if candidate.exists():
                    frame_path = candidate
                    break

            if frame_path is None:
                logger.warning(f"Frame {i} not found, falling back to synthetic frames")
                break

            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to load {frame_path}, falling back to synthetic frames")
                break

            # Resize to target if needed
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        if len(frames) == num_frames:
            logger.info(f"Loaded {len(frames)} frames from {data_dir}")
            return frames

    logger.info(f"Creating {num_frames} synthetic test frames...")
    return [create_test_frame(width, height, seed=i) for i in range(num_frames)]


def load_frames_from_video(
    video_path: Path, num_frames: int, target_width: int, target_height: int
) -> list[np.ndarray]:
    """Decode frames directly from video file using OpenCV.

    This is the standard approach. For better performance with torchcodec,
    use the VideoDataset class directly.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to decode
        target_width: Target frame width
        target_height: Target frame height

    Returns:
        List of RGB frames as numpy arrays
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {video_path.name}")
    logger.info(f"  Resolution: {original_width}x{original_height}")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Target: {num_frames} frames at {target_width}x{target_height}")

    # Calculate frame skip to get approximately num_frames
    if total_frames >= num_frames:
        # Decode every Nth frame
        frame_skip = max(1, total_frames // num_frames)
    else:
        # Decode all frames and pad with duplicates if needed
        frame_skip = 1

    frames = []
    decoded = 0
    frame_idx = 0

    while decoded < num_frames and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Resize to target resolution
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(
                    frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
                )

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            decoded += 1

            if decoded % 20 == 0:
                logger.info(f"  Decoded {decoded}/{num_frames} frames...")

        frame_idx += 1

    cap.release()

    # Pad with last frame if we didn't get enough
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    logger.info(f"Decoded {len(frames)} frames from video")
    return frames


def benchmark_single_device(
    frames: list[np.ndarray],
    model_id: str,
    process_res: int,
    warmup: int = 2,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark single-device DA3 inference.

    Args:
        frames: List of input frames
        model_id: DA3 model identifier
        process_res: Processing resolution
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    logger.info("=" * 60)
    logger.info("SINGLE DEVICE BENCHMARK")
    logger.info("=" * 60)

    # Determine device string for single-device model
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_str = "xpu"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"

    # Import DepthAnything3 directly
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        logger.error(
            "depth_anything_3 package not found. Install with: pip install -e depth_anything_3/"
        )
        raise RuntimeError("depth_anything-3 package not installed")

    logger.info(f"Using device: {device_str}")

    # Load model
    logger.info(f"Loading DA3 model {model_id}...")
    cache_dir = Path("checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = DepthAnything3.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = model.to(device_str).eval()
    logger.info("Model loaded")

    # Warmup
    logger.info(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        _ = model.inference(
            [frames[i % len(frames)]],
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        # Sync for accurate timing
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark iterations
    logger.info(f"Running benchmark ({iterations} iterations)...")
    times = []

    for iteration in range(iterations):
        start = time.perf_counter()

        for frame in frames:
            _ = model.inference(
                [frame],
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
            # Sync after each frame for accurate per-frame timing
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({len(frames)} frames)")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    total_time_ms = avg_time * 1000
    per_frame_ms = total_time_ms / len(frames)
    fps = 1000 / per_frame_ms

    result = BenchmarkResult(
        device_spec="single",
        num_frames=len(frames),
        total_time_ms=total_time_ms,
        per_frame_time_ms=per_frame_ms,
        fps=fps,
        num_devices=1,
        iterations=iterations,
    )

    logger.info(f"Single Device Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

    return result


def benchmark_single_device_batch(
    frames: list[np.ndarray],
    model_id: str,
    process_res: int,
    batch_size: int = 4,
    warmup: int = 2,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark single-device DA3 inference with batch processing.

    Tests if processing multiple frames at once improves throughput on a single device.

    Args:
        frames: List of input frames
        model_id: DA3 model identifier
        process_res: Processing resolution
        batch_size: Number of frames to process at once
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    logger.info("=" * 60)
    logger.info(f"SINGLE DEVICE BATCH BENCHMARK (batch_size={batch_size})")
    logger.info("=" * 60)

    # Determine device string
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_str = "xpu"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"

    from depth_anything_3.api import DepthAnything3

    logger.info(f"Using device: {device_str}")

    # Load model
    logger.info(f"Loading DA3 model {model_id}...")
    cache_dir = Path("checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = DepthAnything3.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = model.to(device_str).eval()
    logger.info("Model loaded")

    # Prepare batches
    num_complete_batches = len(frames) // batch_size
    remainder = len(frames) % batch_size

    # Warmup
    logger.info(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        # Use first batch_size frames for warmup
        warmup_batch = frames[:batch_size] if len(frames) >= batch_size else frames
        target_sizes = [(f.shape[1], f.shape[0]) for f in warmup_batch]
        _ = model.inference(
            warmup_batch,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        # Sync for accurate timing
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark iterations
    logger.info(f"Running benchmark ({iterations} iterations)...")
    times = []

    for iteration in range(iterations):
        start = time.perf_counter()

        # Process complete batches
        for b in range(num_complete_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            batch_frames = frames[start_idx:end_idx]
            _ = model.inference(
                batch_frames,
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )

        # Process remainder frames
        if remainder > 0:
            remainder_frames = frames[num_complete_batches * batch_size :]
            if remainder_frames:
                _ = model.inference(
                    remainder_frames,
                    process_res=process_res,
                    process_res_method="upper_bound_resize",
                    export_dir=None,
                )

        # Sync after all frames
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({len(frames)} frames)")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    total_time_ms = avg_time * 1000
    per_frame_ms = total_time_ms / len(frames)
    fps = 1000 / per_frame_ms

    result = BenchmarkResult(
        device_spec=f"single+batch{batch_size}",
        num_frames=len(frames),
        total_time_ms=total_time_ms,
        per_frame_time_ms=per_frame_ms,
        fps=fps,
        num_devices=1,
        iterations=iterations,
    )

    logger.info(f"Single Device Batch Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

    return result


def benchmark_multi_device(
    frames: list[np.ndarray],
    model_id: str,
    process_res: int,
    device_spec: str,
    batch_size: int = 4,
    warmup: int = 2,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark multi-device DA3 inference using DeviceWorkerPool.

    Uses DeviceWorkerPool.map() with batch_size for efficient parallel processing.

    Args:
        frames: List of input frames
        model_id: DA3 model identifier
        process_res: Processing resolution
        device_spec: Device specification (e.g., "xpu:0,1", "auto")
        batch_size: Batch size for per-worker batch inference
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    logger.info("=" * 60)
    logger.info(f"MULTI DEVICE BENCHMARK (device_spec={device_spec}, batch={batch_size})")
    logger.info("=" * 60)

    # Determine number of devices from device_spec
    if ":" in device_spec:
        device_type, device_indices = device_spec.split(":", 1)
        num_devices = len([int(i.strip()) for i in device_indices.split(",")])
    elif device_spec == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            num_devices = torch.xpu.device_count()
        elif torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
        else:
            num_devices = 1
    elif device_spec in ("xpu", "cuda"):
        # Handle generic device type spec (e.g., "xpu" means all available XPU devices)
        if device_spec == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            num_devices = torch.xpu.device_count()
        elif device_spec == "cuda" and torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
        else:
            num_devices = 1
    else:
        num_devices = 1

    logger.info(f"Detected {num_devices} device(s)")

    # Create multi-device model
    cache_dir = Path("checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)

    multi_model = MultiDeviceDepthModel(
        model_id=model_id,
        device_spec=device_spec,
    )
    multi_model.process_res = process_res
    multi_model.cache_dir = cache_dir

    # Warmup
    logger.info(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        # Use batch inference for warmup
        warmup_frames = frames[: min(4, len(frames))]  # Small batch for warmup
        target_sizes = [(f.shape[1], f.shape[0]) for f in warmup_frames]
        _ = multi_model.infer_depth_batch(
            warmup_frames, target_sizes=target_sizes, batch_size=batch_size
        )
        # Sync for accurate timing
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark iterations
    logger.info(f"Running benchmark ({iterations} iterations)...")
    times = []

    for iteration in range(iterations):
        start = time.perf_counter()

        # Use batch inference for true parallel multi-device processing
        target_sizes = [(frame.shape[1], frame.shape[0]) for frame in frames]
        _ = multi_model.infer_depth_batch(frames, target_sizes=target_sizes, batch_size=batch_size)

        # Sync after all frames
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({len(frames)} frames)")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    total_time_ms = avg_time * 1000
    per_frame_ms = total_time_ms / len(frames)
    fps = 1000 / per_frame_ms

    result = BenchmarkResult(
        device_spec=device_spec,
        num_frames=len(frames),
        total_time_ms=total_time_ms,
        per_frame_time_ms=per_frame_ms,
        fps=fps,
        num_devices=num_devices,
        iterations=iterations,
    )

    logger.info(f"Multi Device Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

    # Shutdown the worker pool
    multi_model.shutdown()

    return result


def benchmark_multi_device_torchcodec(
    video_path: Path,
    model_id: str,
    process_res: int,
    device_spec: str,
    batch_size: int = 4,
    warmup: int = 2,
    iterations: int = 3,
) -> BenchmarkResult | None:
    """Benchmark multi-device with torchcodec decoding (accurate timestamps).

    Uses torchcodec to decode frames with accurate PTS timestamps,
    then feeds frames to DeviceWorkerPool for parallel multi-XPU inference.

    Args:
        video_path: Path to video file
        model_id: DA3 model identifier
        process_res: Processing resolution
        device_spec: Device specification
        batch_size: Batch size for per-worker inference
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics, or None if torchcodec unavailable
    """
    from pathlib import Path

    logger.info("=" * 60)
    logger.info(f"TORCHCODEC MULTI-DEVICE: {video_path.name}")
    logger.info("=" * 60)

    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        logger.warning(f"torchcodec not available: {e}")
        return None

    # Open decoder once and decode all frames efficiently
    decoder = VideoDecoder(str(video_path))
    total_frames = decoder.metadata.num_frames
    if total_frames is None:
        raise ValueError(f"Cannot determine frame count for video: {video_path}")
    average_fps = decoder.metadata.average_fps
    if average_fps is None:
        raise ValueError(f"Cannot determine FPS for video: {video_path}")

    num_frames = total_frames
    logger.info(f"  Frames: {num_frames}")
    logger.info(f"  Batch size: {batch_size}")

    # Decode ALL frames in chunks (torchcodec has limits on get_frames_in_range)
    # Must re-create decoder for each chunk due to torchcodec state issues
    logger.info("Decoding all frames with torchcodec...")
    frames: list[np.ndarray] = []
    target_sizes: list[tuple[int, int]] = []
    all_pts: list[float] = []

    chunk_size = 200  # torchcodec can handle ~350 frames in one call safely
    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        # Re-create decoder for each chunk (torchcodec has state issues with reuse)
        # Use num_ffmpeg_threads=0 to let FFmpeg decide optimal thread count
        chunk_decoder = VideoDecoder(str(video_path), num_ffmpeg_threads=0)
        frame_batch = chunk_decoder.get_frames_in_range(start, end)
        frames_tensor = frame_batch.data  # (B, C, H, W) uint8
        pts_batch = [ts * 1000.0 for ts in frame_batch.pts_seconds.tolist()]
        del chunk_decoder

        for i in range(len(frames_tensor)):
            frame_tensor = frames_tensor[i]
            frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
            frames.append(frame_np)
            target_sizes.append((frame_tensor.shape[2], frame_tensor.shape[1]))  # (W, H)
            all_pts.append(pts_batch[i])

    del decoder
    logger.info(f"  Decoded {len(frames)} frames with timestamps")

    # Create multi-device model
    cache_dir = Path("checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)

    multi_model = MultiDeviceDepthModel(
        model_id=model_id,
        device_spec=device_spec,
    )
    multi_model.process_res = process_res
    multi_model.cache_dir = cache_dir

    # Warmup (using cached frames)
    logger.info(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        warmup_frames = frames[: min(batch_size, len(frames))]
        warmup_sizes = target_sizes[: min(batch_size, len(target_sizes))]
        _ = multi_model.infer_depth_batch(
            warmup_frames, target_sizes=warmup_sizes, batch_size=len(warmup_frames)
        )
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark iterations (using cached frames)
    logger.info(f"Running benchmark ({iterations} iterations)...")
    times: list[float] = []

    for iteration in range(iterations):
        start = time.perf_counter()

        # Multi-device inference with cached frames
        _ = multi_model.infer_depth_batch(frames, target_sizes=target_sizes, batch_size=batch_size)

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({len(frames)} frames)")

    multi_model.shutdown()

    # Calculate statistics
    avg_time = sum(times) / len(times)
    total_time_ms = avg_time * 1000
    per_frame_ms = total_time_ms / len(frames)
    fps = 1000 / per_frame_ms

    # Verify timestamp precision
    if all_pts:
        sorted_pts = sorted(all_pts)
        max_deviation = max(float(abs(ts - sorted_pts[i])) for i, ts in enumerate(all_pts))
        logger.info(f"  Timestamp precision: max deviation {max_deviation:.3f} ms")

    num_devices = len([d for d in device_spec.split(",") if d]) if "," in device_spec else 1

    result = BenchmarkResult(
        device_spec=f"torchcodec+{device_spec}",
        num_frames=len(frames),
        total_time_ms=total_time_ms,
        per_frame_time_ms=per_frame_ms,
        fps=fps,
        num_devices=num_devices,
        iterations=iterations,
    )

    logger.info(f"Torchcodec Multi-Device Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

    return result


def benchmark_dataloader_true_batch(
    frames: list[np.ndarray],
    model_id: str,
    process_res: int,
    device_spec: str,
    batch_size: int = 4,
    num_workers: int = 4,
    parallel_preload: int = 0,
    warmup: int = 1,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark with TRUE batch processing via DataLoader + process_batch().

    This tests the optimized path where:
    1. DataLoader loads frames in parallel
    2. Frames are converted to numpy batch
    3. process_batch() is called ONCE for the entire batch (true batching)

    Args:
        frames: List of input frames
        model_id: DA3 model identifier
        process_res: Processing resolution
        device_spec: Device specification
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        parallel_preload: Number of workers for parallel preload (0=disabled)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from common.datasets import FrameDataset
    from backend.workers.da3_worker import DA3DeviceWorker
    from common.device_worker_pool import DeviceWorkerPool, _init_worker

    logger.info("=" * 60)
    logger.info(
        f"TRUE BATCH DATALOADER (device={device_spec}, batch={batch_size}, preload={parallel_preload})"
    )
    logger.info("=" * 60)

    # Create temporary directory with test frames
    with TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(frames):
            path = Path(tmpdir) / f"frame_{i:04d}.png"
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), frame_bgr)
            frame_paths.append(path)

        timestamps_ms = [i * 33.33 for i in range(len(frames))]

        # Create dataset with parallel preload
        dataset = FrameDataset(
            frame_paths=frame_paths,
            timestamps_ms=timestamps_ms,
            image_mode="RGB",
            parallel_preload=parallel_preload if parallel_preload > 0 else 0,
        )

        # Create worker pool with single worker for true batch testing
        cache_dir = Path("checkpoints")
        cache_dir.mkdir(parents=True, exist_ok=True)

        pool = DeviceWorkerPool(
            worker_class=DA3DeviceWorker,
            device_spec=device_spec,
            worker_kwargs={
                "model_id": model_id,
                "cache_dir": cache_dir,
                "process_res": process_res,
            },
        )

        # Get the worker for batch processing
        devices = pool.devices
        logger.info(f"Using device: {devices[0]}")

        # Warmup
        logger.info(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            # Process first batch
            warmup_batch = frames[:batch_size] if len(frames) >= batch_size else frames
            warmup_results = pool.map(warmup_batch, batch_size=len(warmup_batch))
            _ = list(warmup_results)  # Consume generator

            # Sync
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

        # Benchmark iterations
        logger.info(f"Running benchmark ({iterations} iterations)...")
        times = []

        for iteration in range(iterations):
            start = time.perf_counter()

            # Process all frames with true batching
            results = pool.map(frames, batch_size=batch_size)
            _ = list(results)  # Consume generator

            # Sync
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed)
            logger.info(
                f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({len(frames)} frames)"
            )

        # Calculate statistics
        avg_time = sum(times) / len(times)
        total_time_ms = avg_time * 1000
        per_frame_ms = total_time_ms / len(frames)
        fps = 1000 / per_frame_ms

        num_devices = len(devices) if devices else 1

        result = BenchmarkResult(
            device_spec=f"{device_spec}+true_batch",
            num_frames=len(frames),
            total_time_ms=total_time_ms,
            per_frame_time_ms=per_frame_ms,
            fps=fps,
            num_devices=num_devices,
            iterations=iterations,
        )

        logger.info(f"True Batch Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

        pool.shutdown()

        return result


def benchmark_parallel_preload(
    frames: list[np.ndarray],
    model_id: str,
    process_res: int,
    device_spec: str,
    num_preload_workers: int = 4,
    batch_size: int = 4,
    warmup: int = 1,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark with parallel frame preloading using joblib.

    Tests the parallel_preload option in FrameDataset which uses
    joblib with threading backend for parallel I/O.

    Args:
        frames: List of input frames
        model_id: DA3 model identifier
        process_res: Processing resolution
        device_spec: Device specification
        num_preload_workers: Number of workers for parallel preload
        batch_size: Batch size for processing
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from common.datasets import FrameDataset
    from backend.workers.da3_worker import DA3DeviceWorker
    from common.device_worker_pool import DeviceWorkerPool

    logger.info("=" * 60)
    logger.info(f"PARALLEL PRELOAD BENCHMARK (workers={num_preload_workers})")
    logger.info("=" * 60)

    # Create temporary directory with test frames
    with TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(frames):
            path = Path(tmpdir) / f"frame_{i:04d}.png"
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), frame_bgr)
            frame_paths.append(path)

        timestamps_ms = [i * 33.33 for i in range(len(frames))]

        # First pass: WITHOUT parallel preload (baseline)
        logger.info("Baseline (no preload)...")
        dataset_no_preload = FrameDataset(
            frame_paths=frame_paths,
            timestamps_ms=timestamps_ms,
            image_mode="RGB",
            parallel_preload=0,
        )

        # Measure preload time for comparison
        logger.info(f"Measuring parallel preload time ({num_preload_workers} workers)...")
        preload_start = time.perf_counter()
        dataset_with_preload = FrameDataset(
            frame_paths=frame_paths,
            timestamps_ms=timestamps_ms,
            image_mode="RGB",
            parallel_preload=num_preload_workers,
        )
        preload_time = time.perf_counter() - preload_start
        logger.info(f"Parallel preload time: {preload_time * 1000:.1f} ms")

        # Create worker pool
        cache_dir = Path("checkpoints")
        cache_dir.mkdir(parents=True, exist_ok=True)

        pool = DeviceWorkerPool(
            worker_class=DA3DeviceWorker,
            device_spec=device_spec,
            worker_kwargs={
                "model_id": model_id,
                "cache_dir": cache_dir,
                "process_res": process_res,
            },
        )

        # Benchmark WITH parallel preload
        logger.info(f"Running benchmark with parallel preload...")
        times = []

        for iteration in range(iterations):
            start = time.perf_counter()

            results = pool.map(frames, batch_size=batch_size)
            _ = list(results)

            # Sync
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed)
            logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms")

        avg_time = sum(times) / len(times)
        total_time_ms = avg_time * 1000
        per_frame_ms = total_time_ms / len(frames)
        fps = 1000 / per_frame_ms

        num_devices = len(pool.devices) if pool.devices else 1

        result = BenchmarkResult(
            device_spec=f"{device_spec}+preload{num_preload_workers}",
            num_frames=len(frames),
            total_time_ms=total_time_ms,
            per_frame_time_ms=per_frame_ms,
            fps=fps,
            num_devices=num_devices,
            iterations=iterations,
        )

        logger.info(f"Parallel Preload Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

        pool.shutdown()

        return result


def print_system_info() -> None:
    """Print system and device information."""
    print()
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)

    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")

    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  - CUDA:{i} {name} ({mem:.1f} GB)")
    else:
        print("CUDA Available: No")

    # XPU info
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"XPU Available: Yes")
        print(f"XPU Devices: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            name = torch.xpu.get_device_name(i)
            mem = torch.xpu.get_device_properties(i).total_memory / 1e9
            print(f"  - XPU:{i} {name} ({mem:.1f} GB)")
    else:
        print("XPU Available: No")

    # CPU info
    print(f"CPU Cores: {torch.get_num_threads()} threads")

    print("=" * 80)


def benchmark_torchcodec_video(
    video_path: Path,
    model_id: str,
    process_res: int,
    device_spec: str,
    batch_size: int = 4,
    warmup: int = 1,
    iterations: int = 3,
) -> BenchmarkResult | None:
    """Benchmark using torchcodec for direct video decoding (no PNG intermediate)."""
    from pathlib import Path

    logger.info("=" * 60)
    logger.info(f"TORCHCODEC VIDEO BENCHMARK: {video_path.name}")
    logger.info("=" * 60)

    try:
        from common.datasets import VideoDataset
    except ImportError:
        logger.warning("torchcodec not available, skipping torchcodec benchmark")
        return None

    # Determine device string
    if "xpu" in device_spec.lower():
        device_str = "xpu"
    elif "cuda" in device_spec.lower() or device_spec.lower() == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_spec

    from depth_anything_3.api import DepthAnything3
    from torchcodec.decoders import VideoDecoder

    logger.info(f"Using device: {device_str}")

    # Load model
    logger.info(f"Loading DA3 model {model_id}...")
    cache_dir = Path("checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = DepthAnything3.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = model.to(device_str).eval()
    logger.info("Model loaded")

    # Get video info
    temp_decoder = VideoDecoder(str(video_path))
    total_frames = temp_decoder.metadata.num_frames
    if total_frames is None:
        raise ValueError(f"Cannot determine frame count for video: {video_path}")
    num_frames: int = total_frames
    fps = temp_decoder.metadata.average_fps
    del temp_decoder

    logger.info(f"Video: {video_path.name}")
    logger.info(f"  Frames: {num_frames}")
    logger.info(f"  FPS: {fps:.2f}")

    # Warmup
    logger.info(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        # Create fresh decoder for warmup
        decoder = VideoDecoder(str(video_path))
        warmup_batch = decoder.get_frames_in_range(0, min(batch_size, num_frames))
        warmup_frames = [frame.permute(1, 2, 0).cpu().numpy() for frame in warmup_batch.data]
        del decoder

        _ = model.inference(
            warmup_frames,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        # Sync
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark iterations
    logger.info(f"Running benchmark ({iterations} iterations)...")
    times = []

    for iteration in range(iterations):
        start = time.perf_counter()

        # Process all frames - create fresh decoder for each batch to avoid state issues
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)

            # Create fresh decoder for each batch (torchcodec has state issues with reuse)
            decoder = VideoDecoder(str(video_path))
            frame_batch = decoder.get_frames_in_range(batch_start, batch_end)
            frame_tensors = frame_batch.data  # (B, C, H, W) uint8

            # Convert to numpy arrays for DA3
            frames_np = [frame.permute(1, 2, 0).cpu().numpy() for frame in frame_tensors]

            _ = model.inference(
                frames_np,
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )

            del decoder

        # Sync
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({num_frames} frames)")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    total_time_ms = avg_time * 1000
    per_frame_ms = total_time_ms / num_frames
    fps = 1000 / per_frame_ms

    num_devices = len([d for d in device_spec.split(",") if d]) if "," in device_spec else 1

    result = BenchmarkResult(
        device_spec=f"torchcodec+{device_str}",
        num_frames=num_frames,
        total_time_ms=total_time_ms,
        per_frame_time_ms=per_frame_ms,
        fps=fps,
        num_devices=num_devices,
        iterations=iterations,
    )

    logger.info(f"Torchcodec Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

    return result


def benchmark_torchcodec_streaming(
    video_path: Path,
    model_id: str,
    process_res: int,
    device_spec: str,
    batch_size: int = 4,
    warmup: int = 1,
    iterations: int = 3,
    max_frames: int | None = None,
) -> BenchmarkResult | None:
    """Benchmark torchcodec with forward-only streaming (optimal for real-time backend).

    This benchmark tests the optimal path for the real backend:
    - Opens decoder ONCE and streams forward sequentially
    - Uses get_frames_in_range() for efficient forward-only decoding
    - Captures precise timestamps from FrameBatch.pts_seconds
    - NO decoder recreation overhead (major performance improvement)

    Args:
        video_path: Path to video file
        model_id: DA3 model identifier
        process_res: Processing resolution
        device_spec: Device specification
        batch_size: Number of frames per batch
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        max_frames: Maximum frames to process (None = all frames)

    Returns:
        BenchmarkResult with timing statistics, or None if torchcodec unavailable
    """
    logger.info("=" * 60)
    logger.info(f"TORCHCODEC STREAMING BENCHMARK: {video_path.name}")
    logger.info("=" * 60)

    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        logger.warning("torchcodec not available, skipping streaming benchmark")
        return None

    # Determine device string
    if "xpu" in device_spec.lower():
        device_str = "xpu"
    elif "cuda" in device_spec.lower() or device_spec.lower() == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_spec

    from depth_anything_3.api import DepthAnything3

    logger.info(f"Using device: {device_str}")

    # Load model
    logger.info(f"Loading DA3 model {model_id}...")
    cache_dir = Path("checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = DepthAnything3.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = model.to(device_str).eval()
    logger.info("Model loaded")

    # Get video info
    temp_decoder = VideoDecoder(str(video_path))
    total_frames = temp_decoder.metadata.num_frames
    if total_frames is None:
        raise ValueError(f"Cannot determine frame count for video: {video_path}")
    num_frames: int = min(total_frames, max_frames) if max_frames else total_frames
    fps = temp_decoder.metadata.average_fps
    del temp_decoder

    logger.info(f"Video: {video_path.name}")
    logger.info(f"  Total frames: {total_frames}, processing: {num_frames}")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Batch size: {batch_size}")

    # Warmup - create decoder, decode batch_size frames, close
    logger.info(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        decoder = VideoDecoder(str(video_path))
        warmup_batch = decoder.get_frames_in_range(0, min(batch_size, num_frames))
        warmup_frames = [frame.permute(1, 2, 0).cpu().numpy() for frame in warmup_batch.data]
        del decoder

        _ = model.inference(
            warmup_frames,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        # Sync
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark iterations
    logger.info(f"Running benchmark ({iterations} iterations)...")
    times: list[float] = []
    all_timestamps: list[float] = []

    for iteration in range(iterations):
        start = time.perf_counter()

        # Open decoder ONCE for the entire video
        decoder = VideoDecoder(str(video_path))
        frame_idx = 0

        while frame_idx < num_frames:
            end_idx = min(frame_idx + batch_size, num_frames)
            frame_batch = decoder.get_frames_in_range(frame_idx, end_idx)
            frame_tensors = frame_batch.data  # (B, C, H, W) uint8

            # Capture precise timestamps from FrameBatch (this is the key!)
            batch_ts = [ts * 1000.0 for ts in frame_batch.pts_seconds.tolist()]
            all_timestamps.extend(batch_ts)

            # Convert to numpy arrays for DA3
            frames_np = [frame.permute(1, 2, 0).cpu().numpy() for frame in frame_tensors]

            _ = model.inference(
                frames_np,
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )

            frame_idx += len(frames_np)

        del decoder

        # Sync
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({num_frames} frames)")

    # Verify timestamp precision
    if all_timestamps:
        # Check that timestamps are monotonically increasing and precise
        sorted_ts = sorted(all_timestamps)
        max_deviation = max(float(abs(ts - sorted_ts[i])) for i, ts in enumerate(all_timestamps))
        logger.info(
            f"  Timestamp precision: max deviation from expected order: {max_deviation:.3f} ms"
        )

    # Calculate statistics
    avg_time = sum(times) / len(times)
    total_time_ms = avg_time * 1000
    per_frame_ms = total_time_ms / num_frames
    fps = 1000 / per_frame_ms

    num_devices = len([d for d in device_spec.split(",") if d]) if "," in device_spec else 1

    result = BenchmarkResult(
        device_spec=f"torchcodec+stream+{device_str}",
        num_frames=num_frames,
        total_time_ms=total_time_ms,
        per_frame_time_ms=per_frame_ms,
        fps=fps,
        num_devices=num_devices,
        iterations=iterations,
    )

    logger.info(f"Torchcodec Streaming Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

    return result


def benchmark_cv2_video(
    video_path: Path,
    model_id: str,
    process_res: int,
    device_spec: str,
    batch_size: int = 4,
    warmup: int = 1,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark using OpenCV for video decoding (baseline for comparison).

    This benchmark provides a baseline comparison for the torchcodec approach:
    - OpenCV video decoding (standard approach)
    - No intermediate PNG storage (just decode on-the-fly)

    Args:
        video_path: Path to video file
        model_id: DA3 model identifier
        process_res: Processing resolution
        device_spec: Device specification
        batch_size: Batch size for DA3 inference
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    logger.info("=" * 60)
    logger.info(f"CV2 VIDEO BENCHMARK: {video_path.name}")
    logger.info("=" * 60)

    # Determine device string
    if "xpu" in device_spec.lower():
        device_str = "xpu"
    elif "cuda" in device_spec.lower() or device_spec.lower() == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_spec

    from depth_anything_3.api import DepthAnything3

    logger.info(f"Using device: {device_str}")

    # Load model
    logger.info(f"Loading DA3 model {model_id}...")
    cache_dir = Path("checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = DepthAnything3.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = model.to(device_str).eval()
    logger.info("Model loaded")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Video: {video_path.name}")
    logger.info(f"  Frames: {num_frames}")
    logger.info(f"  FPS: {fps:.2f}")

    # Warmup
    logger.info(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        warmup_frames = []
        for _ in range(min(batch_size, num_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            warmup_frames.append(frame_rgb)
        _ = model.inference(
            warmup_frames,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        # Sync
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark iterations
    logger.info(f"Running benchmark ({iterations} iterations)...")
    times = []

    for iteration in range(iterations):
        start = time.perf_counter()

        # Process all frames in batches
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        batch_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame_rgb)

            if len(batch_frames) == batch_size:
                _ = model.inference(
                    batch_frames,
                    process_res=process_res,
                    process_res_method="upper_bound_resize",
                    export_dir=None,
                )
                batch_frames = []

        # Process remaining frames
        if batch_frames:
            _ = model.inference(
                batch_frames,
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )

        # Sync
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({num_frames} frames)")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    total_time_ms = avg_time * 1000
    per_frame_ms = total_time_ms / num_frames
    fps = 1000 / per_frame_ms

    num_devices = len([d for d in device_spec.split(",") if d]) if "," in device_spec else 1

    result = BenchmarkResult(
        device_spec=f"cv2+{device_spec}",
        num_frames=num_frames,
        total_time_ms=total_time_ms,
        per_frame_time_ms=per_frame_ms,
        fps=fps,
        num_devices=num_devices,
        iterations=iterations,
    )

    logger.info(f"CV2 Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

    cap.release()

    return result


def main() -> None:
    """Run the multi-XPU DA3 benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark multi-XPU DA3 performance: single vs multi-device inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect devices and run benchmark
  python benchmark_da3_multi_xpu.py

  # Single XPU (xpu:0 only)
  python benchmark_da3_multi_xpu.py --device-spec xpu:0

  # Dual XPU with 30 frames
  python benchmark_da3_multi_xpu.py --device-spec xpu:0,1 --num-frames 30

  # Compare single vs multi-device with 5 iterations
  python benchmark_da3_multi_xpu.py --iterations 5

  # Use DataLoader batch processing
  python benchmark_da3_multi_xpu.py --batch-size 8 --num-workers 8

  # Load frames from video file
  python benchmark_da3_multi_xpu.py --video ~/Downloads/lake.mp4

  # Load frames from PNG directory
  python benchmark_da3_multi_xpu.py --input-dir ~/Downloads/lake/temp_frames
        """,
    )

    parser.add_argument("--num-frames", type=int, default=20, help="Number of frames to process")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=360, help="Frame height")
    parser.add_argument(
        "--model-id", type=str, default="depth-anything/DA3METRIC-LARGE", help="DA3 model ID"
    )
    parser.add_argument("--process-res", type=int, default=640, help="DA3 processing resolution")
    parser.add_argument("--device-spec", type=str, default="auto", help="Device specification")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for DataLoader benchmark"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument(
        "--skip-single",
        action="store_true",
        help="Skip single-device benchmark",
    )
    parser.add_argument(
        "--skip-multi",
        action="store_true",
        help="Skip multi-device benchmark",
    )
    parser.add_argument(
        "--skip-dataloader",
        action="store_true",
        help="Skip true batch DataLoader benchmark",
    )
    parser.add_argument(
        "--skip-true-batch",
        action="store_true",
        help="Skip true batch processing benchmark",
    )
    parser.add_argument(
        "--skip-parallel-preload",
        action="store_true",
        help="Skip parallel preload benchmark",
    )
    parser.add_argument(
        "--skip-cv2-video",
        action="store_true",
        help="Skip CV2 video decoding benchmark",
    )
    parser.add_argument(
        "--skip-torchcodec",
        action="store_true",
        help="Skip torchcodec video benchmark (batched with decoder recreation)",
    )
    parser.add_argument(
        "--skip-torchcodec-streaming",
        action="store_true",
        help="Skip torchcodec streaming benchmark (forward-only, optimal for real backend)",
    )
    parser.add_argument(
        "--skip-streaming-multi",
        action="store_true",
        help="Skip multi-device streaming benchmark (interleaved decoders)",
    )
    parser.add_argument(
        "--skip-torchcodec-multi",
        action="store_true",
        help="Skip torchcodec multi-device benchmark (accurate timestamps + multi-XPU)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input frames (frame_0000.png, etc.)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Video file to decode frames from (use instead of --input-dir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print system info
    print_system_info()

    # Load test frames (video > input-dir > synthetic)
    frames = load_test_frames(
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        data_dir=args.input_dir,
        video_path=args.video,
    )

    logger.info(f"Loaded {len(frames)} frames of size {args.width}x{args.height}")

    # Check device availability and set device_spec
    device_spec = args.device_spec
    if device_spec == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_spec = "xpu"
            num_detected = torch.xpu.device_count()
            detected_type = "XPU"
        elif torch.cuda.is_available():
            device_spec = "cuda"
            num_detected = torch.cuda.device_count()
            detected_type = "CUDA"
        else:
            device_spec = "cpu"
            num_detected = 1
            detected_type = "CPU"
        logger.info(
            f"Auto-detected {num_detected} {detected_type} device(s), using device_spec='{device_spec}'"
        )

    # Run benchmarks
    results = BenchmarkResults()

    # Single device benchmark
    if not args.skip_single:
        try:
            result = benchmark_single_device(
                frames=frames,
                model_id=args.model_id,
                process_res=args.process_res,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.add_result(result)
        except Exception as e:
            logger.error(f"Single device benchmark failed: {e}")

    # Single device batch inference benchmark (batch_size=8)
    try:
        result = benchmark_single_device_batch(
            frames=frames,
            model_id=args.model_id,
            process_res=args.process_res,
            batch_size=8,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results.add_result(result)
    except Exception as e:
        logger.error(f"Single device batch benchmark failed: {e}")

    # Multi-device benchmark
    if not args.skip_multi:
        try:
            result = benchmark_multi_device(
                frames=frames,
                model_id=args.model_id,
                process_res=args.process_res,
                device_spec=device_spec,
                batch_size=2,  # Optimal for multi-device parallelism
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.add_result(result)
        except Exception as e:
            logger.error(f"Multi-device benchmark failed: {e}")

    # True batch processing benchmark (new optimized path)
    if not args.skip_true_batch and device_spec != "cpu":
        try:
            result = benchmark_dataloader_true_batch(
                frames=frames,
                model_id=args.model_id,
                process_res=args.process_res,
                device_spec=device_spec,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                parallel_preload=0,  # Test without preload first
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.add_result(result)
        except Exception as e:
            logger.error(f"True batch benchmark failed: {e}")

    # Parallel preload benchmark
    if not args.skip_parallel_preload and device_spec != "cpu":
        try:
            result = benchmark_parallel_preload(
                frames=frames,
                model_id=args.model_id,
                process_res=args.process_res,
                device_spec=device_spec,
                num_preload_workers=args.num_workers,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.add_result(result)
        except Exception as e:
            logger.error(f"Parallel preload benchmark failed: {e}")

    # Video benchmarks (if video file provided)
    if args.video and args.video.exists():
        logger.info("\n" + "=" * 60)
        logger.info("VIDEO BENCHMARKS")
        logger.info("=" * 60)

        # CV2 baseline benchmark
        if not args.skip_cv2_video:
            try:
                result = benchmark_cv2_video(
                    video_path=args.video,
                    model_id=args.model_id,
                    process_res=args.process_res,
                    device_spec=device_spec,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                results.add_result(result)
            except Exception as e:
                logger.error(f"CV2 video benchmark failed: {e}")

        # Torchcodec benchmark
        if not args.skip_torchcodec:
            try:
                result = benchmark_torchcodec_video(
                    video_path=args.video,
                    model_id=args.model_id,
                    process_res=args.process_res,
                    device_spec=device_spec,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                if result:
                    results.add_result(result)
            except Exception as e:
                logger.error(f"Torchcodec video benchmark failed: {e}")

        # Torchcodec streaming benchmark (optimal for real backend)
        if not args.skip_torchcodec_streaming:
            try:
                result = benchmark_torchcodec_streaming(
                    video_path=args.video,
                    model_id=args.model_id,
                    process_res=args.process_res,
                    device_spec=device_spec,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    max_frames=args.num_frames,
                )
                if result:
                    results.add_result(result)
            except Exception as e:
                logger.error(f"Torchcodec streaming benchmark failed: {e}")

        # Torchcodec multi-device benchmark (accurate timestamps + multi-XPU)
        if not args.skip_torchcodec_multi:
            try:
                result = benchmark_multi_device_torchcodec(
                    video_path=args.video,
                    model_id=args.model_id,
                    process_res=args.process_res,
                    device_spec=device_spec,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                if result:
                    results.add_result(result)
            except Exception as e:
                logger.error(f"Torchcodec multi-device benchmark failed: {e}")

    # Print summary
    results.print_summary()

    # Save results to JSON if requested
    if args.output:
        import json

        output_data = {
            "config": {
                "num_frames": args.num_frames,
                "width": args.width,
                "height": args.height,
                "model_id": args.model_id,
                "process_res": args.process_res,
                "device_spec": device_spec,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
            "results": [
                {
                    "device_spec": r.device_spec,
                    "num_frames": r.num_frames,
                    "num_devices": r.num_devices,
                    "total_time_ms": r.total_time_ms,
                    "per_frame_time_ms": r.per_frame_time_ms,
                    "fps": r.fps,
                    "iterations": r.iterations,
                }
                for r in results.results
            ],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
