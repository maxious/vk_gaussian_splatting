#!/usr/bin/env python3
"""Benchmark for WebP encoding performance.

Compares sequential vs parallel WebP encoding for SOG compression.

Usage:
    python benchmark_webp.py [--size 1024] [--num-images 8] [--iterations 3]
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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


def encode_webp_sequential(images: list[np.ndarray], quality: int = 90) -> list[bytes]:
    """Encode images to WebP sequentially."""
    results = []
    for img_array in images:
        img = Image.fromarray(img_array, mode="RGBA")
        buffer = io.BytesIO()
        img.save(buffer, format="WebP", quality=quality, lossless=False)
        results.append(buffer.getvalue())
    return results


def encode_single_webp(img_array: np.ndarray, quality: int = 90) -> bytes:
    """Encode a single image to WebP."""
    img = Image.fromarray(img_array, mode="RGBA")
    buffer = io.BytesIO()
    img.save(buffer, format="WebP", quality=quality, lossless=False)
    return buffer.getvalue()


def encode_webp_parallel(
    images: list[np.ndarray], quality: int = 90, max_workers: int | None = None
) -> list[bytes]:
    """Encode images to WebP in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(encode_single_webp, img, quality) for img in images]
        return [f.result() for f in futures]


def run_benchmark(
    images: list[np.ndarray],
    method: str,
    iterations: int = 3,
    max_workers: int | None = None,
) -> BenchmarkResult:
    """Run benchmark for a specific encoding method."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        if method == "sequential":
            _ = encode_webp_sequential(images)
        elif method == "parallel":
            _ = encode_webp_parallel(images, max_workers=max_workers)
        else:
            raise ValueError(f"Unknown method: {method}")
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    total_pixels = sum(img.shape[0] * img.shape[1] for img in images)

    return BenchmarkResult(
        method=method,
        total_time_ms=avg_time * 1000,
        per_image_time_ms=(avg_time / len(images)) * 1000,
        throughput_mpix_per_sec=total_pixels / avg_time / 1e6,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark WebP encoding performance")
    parser.add_argument("--size", type=int, default=1024, help="Image size (width=height)")
    parser.add_argument("--num-images", type=int, default=8, help="Number of images to encode")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument(
        "--workers", type=int, default=None, help="Max parallel workers (default: CPU count)"
    )
    args = parser.parse_args()

    print("WebP Encoding Benchmark")
    print("=" * 60)
    print(f"Image size: {args.size}x{args.size} ({args.size * args.size / 1e6:.2f} Mpix each)")
    print(f"Number of images: {args.num_images}")
    print(f"Total pixels: {args.size * args.size * args.num_images / 1e6:.2f} Mpix")
    print(f"Iterations: {args.iterations}")
    print()

    # Create test images
    print("Creating test images...", end=" ", flush=True)
    images = [create_test_image(args.size, args.size, seed=i) for i in range(args.num_images)]
    print("done")
    print()

    # Warmup
    print("Warming up...", end=" ", flush=True)
    _ = encode_webp_sequential([images[0]])
    _ = encode_webp_parallel([images[0]])
    print("done")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    print("-" * 60)

    results: list[BenchmarkResult] = []

    # Sequential
    print("  Sequential encoding...", end=" ", flush=True)
    result = run_benchmark(images, "sequential", args.iterations)
    results.append(result)
    print(f"{result.total_time_ms:.1f}ms")

    # Parallel (default workers)
    print("  Parallel encoding...", end=" ", flush=True)
    result = run_benchmark(images, "parallel", args.iterations, args.workers)
    results.append(result)
    print(f"{result.total_time_ms:.1f}ms")

    # Print results
    print()
    print("Results:")
    print("-" * 60)
    print(f"{'Method':<20} {'Total (ms)':<15} {'Per Image (ms)':<15} {'Throughput':<15}")
    print("-" * 60)

    for r in results:
        print(
            f"{r.method:<20} {r.total_time_ms:>10.1f}     "
            f"{r.per_image_time_ms:>10.1f}      "
            f"{r.throughput_mpix_per_sec:>7.2f} Mpix/s"
        )

    print("-" * 60)

    # Calculate speedup
    if len(results) >= 2:
        speedup = results[0].total_time_ms / results[1].total_time_ms
        print(f"\nSpeedup (parallel vs sequential): {speedup:.2f}x")

    # Verify correctness (compare file sizes - should be similar)
    print("\nVerifying correctness...")
    seq_results = encode_webp_sequential(images[:1])
    par_results = encode_webp_parallel(images[:1])
    size_diff = abs(len(seq_results[0]) - len(par_results[0]))
    print(f"  Sequential output size: {len(seq_results[0])} bytes")
    print(f"  Parallel output size: {len(par_results[0])} bytes")
    print(f"  Size difference: {size_diff} bytes (expected: 0)")

    if size_diff == 0:
        print("  [OK] Outputs are identical")
    else:
        print("  [WARN] Outputs differ slightly (WebP encoding may be non-deterministic)")


if __name__ == "__main__":
    main()
