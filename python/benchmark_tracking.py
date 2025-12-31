import time
import numpy as np
import logging
import os
from pathlib import Path
from offline.motion_tracking_cuda import match_gaussians_batch_cupy
from plyfile import PlyData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gaussians_from_ply(ply_path):
    """Load Gaussian means from PLY file."""
    plydata = PlyData.read(ply_path)
    vertex_data = plydata["vertex"]
    means = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(
        np.float32
    )
    return means


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


def run_benchmark_with_real_data(ply_folder, max_dist=0.1, num_frames=2):
    """Run benchmark using real Gaussian data from PLY files."""
    ply_files = sorted(Path(ply_folder).glob("*.ply"))
    if len(ply_files) < 2:
        print(f"Need at least 2 PLY files, found {len(ply_files)}")
        return

    # Load specified number of frames
    all_means = []
    for ply_file in ply_files[:num_frames]:
        means = load_gaussians_from_ply(ply_file)
        all_means.append(means)
        print(f"Loaded {len(means)} Gaussians from {ply_file.name}")

    if len(all_means) < 2:
        return

    # Use a subset for benchmarking
    N = min(100000, min(len(m) for m in all_means))  # Cap at 100k for realistic testing
    all_means = [m[:N] for m in all_means]

    print(f"\nBenchmarking with {N} points per frame, {len(all_means)} frames...")

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

    # --- RAPIDS cuML GPU ---
    try:
        from cuml.neighbors import NearestNeighbors
        import cudf
        import cupy as cp

        start = time.time()
        matches_cuml = match_gaussians_batch_cuml(all_means, max_dist)
        cp.cuda.Stream.null.synchronize()
        cuml_time = time.time() - start
        print(f"cuML (GPU) Time:  {cuml_time:.4f}s")
        speedup_cuml = faiss_time / cuml_time
    except ImportError:
        print("cuML not available")
        cuml_time = None
        speedup_cuml = None

    if cuml_time is not None:
        print(f"Speedup vs FAISS: CuPy {faiss_time / cp_time:.2f}x, cuML {speedup_cuml:.2f}x")
        return faiss_time, cp_time, cuml_time
    else:
        print(f"Speedup vs FAISS: {faiss_time / cp_time:.2f}x")
        return faiss_time, cp_time


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


def benchmark(use_real_data=False, ply_folder=None, num_frames=2):
    if use_real_data and ply_folder:
        print("Benchmarking with real data...")
        start_total = time.time()
        run_benchmark_with_real_data(ply_folder, num_frames=num_frames)
        total_time = time.time() - start_total
        print(f"Total benchmark time: {total_time:.4f}s")
        return

    # Warmup
    print("Warming up...")
    run_benchmark(1000)

    sizes = [5_000, 10_000, 20_000]  # Smaller sizes for testing
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


def match_gaussians_batch_cuml(
    all_means: list[np.ndarray],
    max_distance: float,
) -> list[list[tuple[int, int]]]:
    """Match Gaussians between all consecutive frame pairs using RAPIDS cuML."""
    from cuml.neighbors import NearestNeighbors
    import cudf
    import cupy as cp

    all_matches = []

    for i in range(len(all_means) - 1):
        means_a = all_means[i].astype(np.float32)
        means_b = all_means[i + 1].astype(np.float32)

        # Use cuML NearestNeighbors for KNN
        nn_a_to_b = NearestNeighbors(n_neighbors=1, algorithm="brute")
        nn_a_to_b.fit(cudf.DataFrame(means_b))
        dist_a_to_b, idx_a_to_b = nn_a_to_b.kneighbors(cudf.DataFrame(means_a))

        nn_b_to_a = NearestNeighbors(n_neighbors=1, algorithm="brute")
        nn_b_to_a.fit(cudf.DataFrame(means_a))
        dist_b_to_a, idx_b_to_a = nn_b_to_a.kneighbors(cudf.DataFrame(means_b))

        # Convert to numpy
        dist_a_to_b = dist_a_to_b.to_numpy().flatten()
        idx_a_to_b = idx_a_to_b.to_numpy().flatten()
        idx_b_to_a = idx_b_to_a.to_numpy().flatten()

        matches = []
        for a_idx in range(len(means_a)):
            b_idx = idx_a_to_b[a_idx]
            if dist_a_to_b[a_idx] < max_distance and idx_b_to_a[b_idx] == a_idx:
                matches.append((a_idx, b_idx))

        all_matches.append(matches)

        logger.debug(f"Frame {i}->{i + 1}: {len(matches)} matches (cuML)")

    return all_matches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Gaussian motion tracking libraries")
    parser.add_argument(
        "--real-data", type=str, help="Folder containing PLY files for real data benchmark"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=2,
        help="Number of frames to use for benchmark (default: 2)",
    )
    args = parser.parse_args()

    if args.real_data:
        benchmark(use_real_data=True, ply_folder=args.real_data)
    else:
        benchmark()
