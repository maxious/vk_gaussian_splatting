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
from typing import Callable

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
    num_frames: int, width: int = 640, height: int = 360, data_dir: Path | None = None
) -> list[np.ndarray]:
    """Load or create test frames.

    Args:
        num_frames: Number of frames to load
        width: Frame width
        height: Frame height
        data_dir: Optional directory containing image files (frame_XXXXX.png, etc.)

    Returns:
        List of RGB frames as numpy arrays
    """
    if data_dir and data_dir.exists():
        logger.info(f"Loading frames from {data_dir}...")
        frames = []

        # Try 5-digit format first (frame_00000.png), then 4-digit (frame_0000.png)
        for i in range(num_frames):
            frame_path = data_dir / f"frame_{i:05d}.png"
            if not frame_path.exists():
                frame_path = data_dir / f"frame_{i:04d}.png"
            if not frame_path.exists():
                frame_path = data_dir / f"frame_{i:04d}.jpg"
            if not frame_path.exists():
                logger.warning(f"Frame {frame_path} not found, falling back to synthetic frames")
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


def benchmark_multi_device(
    frames: list[np.ndarray],
    model_id: str,
    process_res: int,
    device_spec: str,
    warmup: int = 2,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark multi-device DA3 inference using DeviceWorkerPool.

    Args:
        frames: List of input frames
        model_id: DA3 model identifier
        process_res: Processing resolution
        device_spec: Device specification (e.g., "xpu:0,1", "auto")
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    logger.info("=" * 60)
    logger.info(f"MULTI DEVICE BENCHMARK (device_spec={device_spec})")
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
        _ = multi_model.infer_depth_batch(warmup_frames, target_sizes=target_sizes)
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
        _ = multi_model.infer_depth_batch(frames, target_sizes=target_sizes)

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


def benchmark_batch_with_dataloader(
    frames: list[np.ndarray],
    model_id: str,
    process_res: int,
    device_spec: str,
    batch_size: int = 4,
    num_workers: int = 4,
    warmup: int = 1,
    iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark multi-device DA3 with batch processing via DataLoader.

    This tests the FrameDataset + DataLoader path with pinned memory,
    which was added to improve multi-device throughput.

    Args:
        frames: List of input frames
        model_id: DA3 model identifier
        process_res: Processing resolution
        device_spec: Device specification
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from torch.utils.data import DataLoader

    from common.datasets import FrameDataset

    logger.info("=" * 60)
    logger.info(f"BATCH DATALOADER BENCHMARK (device_spec={device_spec}, batch={batch_size})")
    logger.info("=" * 60)

    # Create temporary directory with test frames
    with TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(frames):
            path = Path(tmpdir) / f"frame_{i:04d}.png"
            # Convert RGB to BGR for cv2.imwrite
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), frame_bgr)
            frame_paths.append(path)

        timestamps_ms = [i * 33.33 for i in range(len(frames))]  # ~30 fps

        # Create dataset
        dataset = FrameDataset(
            frame_paths=frame_paths,
            timestamps_ms=timestamps_ms,
            image_mode="RGB",
        )

        # Determine pin_memory based on device
        pin_memory = "cuda" in device_spec or "xpu" in device_spec

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda batch: batch,  # Pass through
        )

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
            for batch in dataloader:
                for item in batch:
                    _, frame_tensor, _, H, W, _ = item
                    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                    H_val = int(H.item()) if hasattr(H, "item") else int(H)
                    W_val = int(W.item()) if hasattr(W, "item") else int(W)
                    _ = multi_model.infer_depth(frame_np, target_size=(W_val, H_val))
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

            for batch in dataloader:
                for item in batch:
                    _, frame_tensor, _, H, W, _ = item
                    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                    H_val = int(H.item()) if hasattr(H, "item") else int(H)
                    W_val = int(W.item()) if hasattr(W, "item") else int(W)
                    _ = multi_model.infer_depth(frame_np, target_size=(W_val, H_val))

            # Sync after all frames
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

        result = BenchmarkResult(
            device_spec=f"{device_spec}+dataloader",
            num_frames=len(frames),
            total_time_ms=total_time_ms,
            per_frame_time_ms=per_frame_ms,
            fps=fps,
            num_devices=1 if device_spec == "cpu" else 2,
            iterations=iterations,
        )

        logger.info(f"Batch DataLoader Result: {per_frame_ms:.1f} ms/frame ({fps:.2f} fps)")

        multi_model.shutdown()

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
        help="Skip DataLoader batch benchmark",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input frames (frame_0000.png, etc.)",
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

    # Load test frames
    frames = load_test_frames(
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        data_dir=args.input_dir,
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

    # Multi-device benchmark
    if not args.skip_multi:
        try:
            result = benchmark_multi_device(
                frames=frames,
                model_id=args.model_id,
                process_res=args.process_res,
                device_spec=device_spec,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.add_result(result)
        except Exception as e:
            logger.error(f"Multi-device benchmark failed: {e}")

    # DataLoader batch benchmark
    if not args.skip_dataloader and device_spec != "cpu":
        try:
            result = benchmark_batch_with_dataloader(
                frames=frames,
                model_id=args.model_id,
                process_res=args.process_res,
                device_spec=device_spec,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.add_result(result)
        except Exception as e:
            logger.error(f"DataLoader batch benchmark failed: {e}")

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
