import time
import numpy as np
import logging
from offline.motion_tracking_cuda import match_gaussians_batch_cupy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def match_gaussians_batch_faiss_cpu(
    all_means: list[np.ndarray],
    max_distance: float,
) -> list[list[tuple[int, int]]]:
    """Local implementation of FAISS CPU matching for benchmarking."""
    import faiss

    if len(all_means) < 2:
        return []

    all_matches = []

    for i in range(len(all_means) - 1):
        means_a = np.ascontiguousarray(all_means[i], dtype=np.float32)
        means_b = np.ascontiguousarray(all_means[i + 1], dtype=np.float32)

        index_a = faiss.IndexFlatL2(3)
        index_b = faiss.IndexFlatL2(3)

        index_a.add(means_a)
        index_b.add(means_b)

        dist_a_to_b, idx_a_to_b = index_b.search(means_a, 1)
        dist_b_to_a, idx_b_to_a = index_a.search(means_b, 1)

        dist_a_to_b = np.sqrt(dist_a_to_b[:, 0])
        idx_a_to_b = idx_a_to_b[:, 0]
        idx_b_to_a = idx_b_to_a[:, 0]

        matches = []
        for a_idx in range(len(means_a)):
            b_idx = idx_a_to_b[a_idx]
            if dist_a_to_b[a_idx] < max_distance and idx_b_to_a[b_idx] == a_idx:
                matches.append((a_idx, b_idx))

        all_matches.append(matches)

    return all_matches


def run_benchmark(N, D=3, max_dist=0.1):
    print(f"\nBenchmarking N={N} points...")

    # Frame A
    means_a = np.random.rand(N, D).astype(np.float32)
    # Frame B: Shift A slightly + noise
    shift = np.random.normal(0, 0.01, (N, D)).astype(np.float32)
    means_b = means_a + shift

    all_means = [means_a, means_b]

    # --- FAISS CPU ---
    start = time.time()
    matches_faiss = match_gaussians_batch_faiss_cpu(all_means, max_dist)
    faiss_time = time.time() - start
    print(f"FAISS (CPU) Time: {faiss_time:.4f}s")

    # --- CuPy GPU ---
    import cupy as cp

    cp.cuda.Stream.null.synchronize()
    start = time.time()
    matches_cupy = match_gaussians_batch_cupy(all_means, max_dist)
    cp.cuda.Stream.null.synchronize()
    cp_time = time.time() - start
    print(f"CuPy (GPU) Time:  {cp_time:.4f}s")

    print(f"Speedup vs FAISS: {faiss_time / cp_time:.2f}x")

    return faiss_time, cp_time


def benchmark():
    # Warmup
    print("Warming up...")
    run_benchmark(1000)

    sizes = [10_000, 50_000, 100_000, 500_000]
    results = []

    for N in sizes:
        try:
            res = run_benchmark(N)
            results.append((N, *res))
        except Exception as e:
            print(f"Failed for N={N}: {e}")

    print("\n--- Summary ---")
    print(f"{'N':<10} | {'FAISS CPU':<10} | {'CuPy GPU':<10} | {'Speedup':<10}")
    print("-" * 50)
    for N, ft, ct in results:
        print(f"{N:<10} | {ft:<10.4f} | {ct:<10.4f} | {ft / ct:<10.2f}x")


if __name__ == "__main__":
    benchmark()
