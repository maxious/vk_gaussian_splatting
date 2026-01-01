#!/usr/bin/env python3
"""Benchmark for WebP encoding performance using OpenCV.

Tests sequential vs parallel WebP encoding for SOG compression.
Supports both lossy (quality 0-100) and lossless (quality 101) modes.

Usage:
    python benchmark_webp.py [--size 1024] [--num-images 8] [--iterations 3]
    python benchmark_webp.py --lossless  # Test lossless mode (for SOG)
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    method: str
    total_time_ms: float
    per_image_time_ms: float
    throughput_mpix_per_sec: float


def create_test_image(width: int, height: int, seed: int = 42) -> np.ndarray:
    """Create a test RGBA image with random data."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)


def encode_single_webp(img_array: np.ndarray, quality: int = 90) -> bytes:
    """Encode a single RGBA image to WebP using OpenCV.

    Args:
        img_array: RGBA image as numpy array
        quality: 0-100 for lossy, 101 for lossless
    """
    # OpenCV expects BGR(A), input is RGBA - convert to BGRA
    bgra = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
    # Note: quality > 100 triggers lossless mode in OpenCV
    success, encoded = cv2.imencode(".webp", bgra, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not success:
        raise RuntimeError("OpenCV WebP encoding failed")
    return encoded.tobytes()


def encode_single_webp_pillow(img_array: np.ndarray, lossless: bool = False) -> bytes:
    """Encode a single RGBA image to WebP using Pillow (current SOG implementation)."""
    img = Image.fromarray(img_array, mode="RGBA")
    buffer = io.BytesIO()
    if lossless:
        # Match SOG exporter settings exactly
        img.save(buffer, format="webp", lossless=True, quality=100, method=6, exact=True)
    else:
        img.save(buffer, format="webp", quality=90)
    return buffer.getvalue()


def encode_webp_sequential(images: list[np.ndarray], quality: int = 90) -> list[bytes]:
    """Encode images to WebP sequentially."""
    return [encode_single_webp(img, quality) for img in images]


def encode_webp_parallel(
    images: list[np.ndarray], quality: int = 90, max_workers: int | None = None
) -> list[bytes]:
    """Encode images to WebP in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(encode_single_webp, img, quality) for img in images]
        return [f.result() for f in futures]


def encode_webp_pillow_parallel(
    images: list[np.ndarray], lossless: bool = False, max_workers: int | None = None
) -> list[bytes]:
    """Encode images to WebP in parallel using Pillow."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(encode_single_webp_pillow, img, lossless) for img in images]
        return [f.result() for f in futures]


def run_benchmark(
    images: list[np.ndarray],
    parallel: bool,
    iterations: int = 3,
    max_workers: int | None = None,
    quality: int = 90,
    use_pillow: bool = False,
    lossless: bool = False,
) -> BenchmarkResult:
    """Run benchmark for sequential or parallel encoding."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        if use_pillow:
            if parallel:
                _ = encode_webp_pillow_parallel(images, lossless=lossless, max_workers=max_workers)
            else:
                _ = [encode_single_webp_pillow(img, lossless=lossless) for img in images]
        else:
            if parallel:
                _ = encode_webp_parallel(images, quality=quality, max_workers=max_workers)
            else:
                _ = encode_webp_sequential(images, quality=quality)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    total_pixels = sum(img.shape[0] * img.shape[1] for img in images)

    method_name = "Pillow" if use_pillow else "OpenCV"
    if parallel:
        method_name += " parallel"
    else:
        method_name += " sequential"

    return BenchmarkResult(
        method=method_name,
        total_time_ms=avg_time * 1000,
        per_image_time_ms=(avg_time / len(images)) * 1000,
        throughput_mpix_per_sec=total_pixels / avg_time / 1e6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark WebP encoding performance")
    parser.add_argument("--size", type=int, default=1024, help="Image size (width=height)")
    parser.add_argument("--num-images", type=int, default=8, help="Number of images to encode")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument(
        "--workers", type=int, default=None, help="Max parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--lossless", action="store_true", help="Test lossless encoding (quality=101)"
    )
    parser.add_argument(
        "--quality", type=int, default=90, help="Lossy quality (0-100, ignored if --lossless)"
    )
    args = parser.parse_args()

    # Quality 101 = lossless in OpenCV
    quality = 101 if args.lossless else args.quality
    mode_str = "lossless" if args.lossless else f"lossy (q={quality})"

    print("WebP Encoding Benchmark (OpenCV)")
    print("=" * 60)
    print(f"Mode: {mode_str}")
    print(f"Image size: {args.size}x{args.size} ({args.size * args.size / 1e6:.2f} Mpix each)")
    print(f"Number of images: {args.num_images}")
    print(f"Total pixels: {args.size * args.size * args.num_images / 1e6:.2f} Mpix")
    print(f"Iterations: {args.iterations}")
    print(f"OpenCV version: {cv2.__version__}")
    print()

    # Create test images
    print("Creating test images...", end=" ", flush=True)
    images = [create_test_image(args.size, args.size, seed=i) for i in range(args.num_images)]
    print("done")
    print()

    # Warmup
    print("Warming up...", end=" ", flush=True)
    _ = encode_webp_sequential([images[0]], quality=quality)
    _ = encode_webp_parallel([images[0]], quality=quality)
    _ = encode_single_webp_pillow(images[0], lossless=args.lossless)
    print("done")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    print("-" * 60)

    results = []

    # Pillow sequential
    print("  Pillow sequential...", end=" ", flush=True)
    r = run_benchmark(
        images, parallel=False, iterations=args.iterations, use_pillow=True, lossless=args.lossless
    )
    results.append(r)
    print(f"{r.total_time_ms:.1f}ms")

    # Pillow parallel
    print("  Pillow parallel...", end=" ", flush=True)
    r = run_benchmark(
        images,
        parallel=True,
        iterations=args.iterations,
        max_workers=args.workers,
        use_pillow=True,
        lossless=args.lossless,
    )
    results.append(r)
    print(f"{r.total_time_ms:.1f}ms")

    # OpenCV sequential
    print("  OpenCV sequential...", end=" ", flush=True)
    r = run_benchmark(images, parallel=False, iterations=args.iterations, quality=quality)
    results.append(r)
    print(f"{r.total_time_ms:.1f}ms")

    # OpenCV parallel
    print("  OpenCV parallel...", end=" ", flush=True)
    r = run_benchmark(
        images, parallel=True, iterations=args.iterations, max_workers=args.workers, quality=quality
    )
    results.append(r)
    print(f"{r.total_time_ms:.1f}ms")

    # Print results
    print()
    print("Results:")
    print("-" * 70)
    print(f"{'Method':<20} {'Total (ms)':<12} {'Per Image (ms)':<15} {'Throughput':<15}")
    print("-" * 70)

    for r in results:
        print(
            f"{r.method:<20} {r.total_time_ms:>8.1f}     "
            f"{r.per_image_time_ms:>10.1f}      "
            f"{r.throughput_mpix_per_sec:>7.2f} Mpix/s"
        )

    print("-" * 70)

    # Calculate speedups
    pillow_seq = results[0]
    pillow_par = results[1]
    opencv_seq = results[2]
    opencv_par = results[3]

    print(f"\nSpeedups:")
    print(
        f"  Pillow parallel vs sequential: {pillow_seq.total_time_ms / pillow_par.total_time_ms:.2f}x"
    )
    print(
        f"  OpenCV parallel vs sequential: {opencv_seq.total_time_ms / opencv_par.total_time_ms:.2f}x"
    )
    print(
        f"  OpenCV parallel vs Pillow parallel: {pillow_par.total_time_ms / opencv_par.total_time_ms:.2f}x"
    )
    print(
        f"  OpenCV sequential vs Pillow sequential: {pillow_seq.total_time_ms / opencv_seq.total_time_ms:.2f}x"
    )

    # Verify output
    print("\nVerifying output...")
    result_opencv = encode_webp_sequential(images[:1], quality=quality)
    result_pillow = [encode_single_webp_pillow(images[0], lossless=args.lossless)]
    print(f"  OpenCV output size: {len(result_opencv[0])} bytes")
    print(f"  Pillow output size: {len(result_pillow[0])} bytes")
    compression_ratio = (args.size * args.size * 4) / len(result_opencv[0])
    print(f"  Compression ratio (OpenCV): {compression_ratio:.1f}x")
    print("  [OK] Encoding successful")


if __name__ == "__main__":
    main()
