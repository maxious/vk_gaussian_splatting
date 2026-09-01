#!/usr/bin/env python3
"""Benchmark multi-XPU MoGe performance: single vs multi-device inference.

Usage:
    python benchmark_moge_multi_xpu.py --video ~/Downloads/15650007-uhd_1440_2560_30fps.mp4
    python benchmark_moge_multi_xpu.py --video ~/Downloads/15650007-uhd_1440_2560_30fps.mp4 --device-spec xpu:0
    python benchmark_moge_multi_xpu.py --video ~/Downloads/15650007-uhd_1440_2560_30fps.mp4 --device-spec xpu:0,1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from backend.workers.moge_worker import MoGEDeviceWorker
from common.device_worker_pool import DeviceWorkerPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    device_spec: str
    num_frames: int
    total_time_ms: float
    per_frame_time_ms: float
    fps: float
    num_devices: int
    iterations: int = 1


def load_video_frames(
    video_path: Path, max_frames: int = 50, target_width: int = 640, target_height: int = 480
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {video_path.name}")
    logger.info(
        f"  Original: {original_width}x{original_height}, {fps:.2f} fps, {total_frames} frames"
    )
    logger.info(f"  Target: {max_frames} frames at {target_width}x{target_height}")

    frames = []
    decoded = 0
    frame_skip = max(1, total_frames // max_frames)

    frame_idx = 0
    while decoded < max_frames and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(
                    frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            decoded += 1

        frame_idx += 1

    cap.release()
    logger.info(f"Loaded {len(frames)} frames from video")
    return frames


def benchmark_single_device(
    frames: list[np.ndarray],
    model_id: str,
    warmup: int = 2,
    iterations: int = 3,
) -> BenchmarkResult:
    logger.info("=" * 60)
    logger.info("SINGLE DEVICE BENCHMARK")
    logger.info("=" * 60)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_str = "xpu:0"
    elif torch.cuda.is_available():
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    logger.info(f"Using device: {device_str}")

    worker = MoGEDeviceWorker(device=device_str, worker_id=0, model_id=model_id)
    worker.load_model()

    logger.info(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        _ = worker.process_item(frames[i % len(frames)])
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    logger.info(f"Running benchmark ({iterations} iterations)...")
    times = []

    for iteration in range(iterations):
        start = time.perf_counter()

        for frame in frames:
            _ = worker.process_item(frame)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({len(frames)} frames)")

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
    device_spec: str,
    batch_size: int = 4,
    warmup: int = 2,
    iterations: int = 3,
) -> BenchmarkResult:
    logger.info("=" * 60)
    logger.info(f"MULTI DEVICE BENCHMARK (device_spec={device_spec})")
    logger.info("=" * 60)

    num_devices = len([d for d in device_spec.split(",") if d]) if "," in device_spec else 1

    pool = DeviceWorkerPool(
        worker_class=MoGEDeviceWorker,
        device_spec=device_spec,
        worker_kwargs={"model_id": model_id},
    )

    logger.info(f"Worker pool ready with {len(pool.devices)} devices: {pool.devices}")

    logger.info(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        warmup_frames = frames[: min(batch_size, len(frames))]
        results = pool.map(warmup_frames, batch_size=len(warmup_frames))
        _ = list(results)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    logger.info(f"Running benchmark ({iterations} iterations)...")
    times = []

    for iteration in range(iterations):
        start = time.perf_counter()

        results = pool.map(frames, batch_size=batch_size)
        _ = list(results)

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logger.info(f"  Iteration {iteration + 1}: {elapsed * 1000:.1f} ms ({len(frames)} frames)")

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

    pool.shutdown()

    return result


def print_comparison(single: BenchmarkResult, multi: BenchmarkResult):
    print()
    print("=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"Single Device: {single.per_frame_time_ms:.1f} ms/frame ({single.fps:.2f} fps)")
    print(
        f"Multi Device ({multi.num_devices}x): {multi.per_frame_time_ms:.1f} ms/frame ({multi.fps:.2f} fps)"
    )

    speedup = single.total_time_ms / multi.total_time_ms
    print(f"Speedup: {speedup:.2f}x")
    print(f"Expected (ideal): {multi.num_devices}x")

    efficiency = speedup / multi.num_devices * 100
    print(f"Parallel Efficiency: {efficiency:.1f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoGe multi-XPU performance")
    parser.add_argument("--video", type=Path, required=True, help="Path to video file")
    parser.add_argument("--model-id", default="Ruicheng/moge-3-vitl", help="MoGe-3 model ID")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum frames to process")
    parser.add_argument("--target-width", type=int, default=640, help="Target frame width")
    parser.add_argument("--target-height", type=int, default=480, help="Target frame height")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for multi-device")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=3, help="Benchmark iterations")
    parser.add_argument(
        "--device-spec", default="auto", help="Device spec (auto, xpu:0, xpu:0,1, etc.)"
    )
    parser.add_argument("--skip-single", action="store_true", help="Skip single-device benchmark")

    args = parser.parse_args()

    if not args.video.exists():
        logger.error(f"Video file not found: {args.video}")
        return 1

    logger.info("Loading video frames...")
    frames = load_video_frames(
        args.video,
        max_frames=args.max_frames,
        target_width=args.target_width,
        target_height=args.target_height,
    )

    single_result = None
    if not args.skip_single:
        single_result = benchmark_single_device(
            frames,
            args.model_id,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    multi_result = benchmark_multi_device(
        frames,
        args.model_id,
        device_spec=args.device_spec,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    if single_result:
        print_comparison(single_result, multi_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
