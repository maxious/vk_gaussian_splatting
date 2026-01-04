"""Benchmark motion tracking implementations: FAISS (CPU) vs cuTile (GPU).

Also validates CUDA results against FAISS as ground truth.
"""

import logging
import numpy as np
from pathlib import Path
import time

import torch
from plyfile import PlyData

from offline.motion_tracking_cpu import match_gaussians_sliding_window_faiss
from offline.motion_tracking_cuda import (
    match_gaussians_sliding_window_cuda,
    knn_search_cutile,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_means_from_ply(ply_path: Path) -> np.ndarray:
    """Load Gaussian means from PLY file."""
    plydata = PlyData.read(ply_path)
    vertex_data = plydata["vertex"]
    return np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(
        np.float32
    )


def load_data(data_dir: Path, num_frames: int = 10) -> list[np.ndarray]:
    """Load sequence of PLY files."""
    ply_files = sorted(data_dir.glob("*.ply"))[:num_frames]
    if not ply_files:
        raise ValueError(f"No PLY files found in {data_dir}")

    logger.info(f"Loading {len(ply_files)} frames from {data_dir}...")
    return [load_means_from_ply(p) for p in ply_files]


def analyze_knn_differences(
    means_a: np.ndarray,
    means_b: np.ndarray,
) -> dict:
    """Compare KNN results at the raw level (before mutual matching)."""
    import faiss

    # FAISS KNN
    means_a_f = np.ascontiguousarray(means_a, dtype=np.float32)
    means_b_f = np.ascontiguousarray(means_b, dtype=np.float32)

    index_b = faiss.IndexFlatL2(3)
    index_b.add(means_b_f)  # type: ignore
    dist_faiss, idx_faiss = index_b.search(means_a_f, 1)  # type: ignore
    dist_faiss = dist_faiss[:, 0]
    idx_faiss = idx_faiss[:, 0]

    # cuTile KNN
    a_gpu = torch.from_numpy(means_a).cuda().float()
    b_gpu = torch.from_numpy(means_b).cuda().float()
    dist_cuda, idx_cuda = knn_search_cutile(a_gpu, b_gpu)
    dist_cuda = dist_cuda.cpu().numpy()
    idx_cuda = idx_cuda.cpu().numpy()

    # Compare
    idx_match = idx_faiss == idx_cuda
    idx_agree_pct = idx_match.mean() * 100

    # For disagreements, check if distances are very close (tie-breaking)
    disagree_mask = ~idx_match
    if disagree_mask.any():
        # Get the FAISS-chosen neighbor distance
        faiss_chosen_dist = dist_faiss[disagree_mask]
        # Get the CUDA-chosen neighbor distance
        cuda_chosen_dist = dist_cuda[disagree_mask]
        # Relative difference
        rel_diff = np.abs(faiss_chosen_dist - cuda_chosen_dist) / (faiss_chosen_dist + 1e-10)

        return {
            "idx_agreement_pct": idx_agree_pct,
            "n_disagreements": disagree_mask.sum(),
            "rel_dist_diff_mean": rel_diff.mean(),
            "rel_dist_diff_max": rel_diff.max(),
            "near_ties": (rel_diff < 0.01).sum(),  # <1% distance difference
        }
    else:
        return {
            "idx_agreement_pct": 100.0,
            "n_disagreements": 0,
            "rel_dist_diff_mean": 0.0,
            "rel_dist_diff_max": 0.0,
            "near_ties": 0,
        }


def validate_cuda_vs_faiss(
    all_means: list[np.ndarray],
    max_distance: float,
    window_size: int,
) -> dict:
    """Compare CUDA results against FAISS ground truth."""
    logger.info("Running FAISS matching...")
    faiss_matches = match_gaussians_sliding_window_faiss(all_means, max_distance, window_size)
    faiss_set = set(faiss_matches)

    logger.info("Running CUDA matching...")
    cuda_matches = match_gaussians_sliding_window_cuda(all_means, max_distance, window_size)
    cuda_set = set(cuda_matches)

    common = faiss_set & cuda_set
    faiss_only = faiss_set - cuda_set
    cuda_only = cuda_set - faiss_set
    union = faiss_set | cuda_set

    agreement_ratio = len(common) / len(union) if union else 1.0

    result = {
        "total_faiss": len(faiss_matches),
        "total_cuda": len(cuda_matches),
        "common": len(common),
        "faiss_only": len(faiss_only),
        "cuda_only": len(cuda_only),
        "agreement_ratio": agreement_ratio,
    }

    logger.info("Validation results:")
    logger.info(f"  FAISS matches: {result['total_faiss']}")
    logger.info(f"  CUDA matches:  {result['total_cuda']}")
    logger.info(f"  Common:        {result['common']}")
    logger.info(f"  FAISS only:    {result['faiss_only']}")
    logger.info(f"  CUDA only:     {result['cuda_only']}")
    logger.info(f"  Agreement:     {result['agreement_ratio']:.2%}")

    return result


def benchmark_function(name: str, func, warmup: int = 1, repeat: int = 3):
    """Simple benchmark without pyperf overhead."""
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)
    logger.info(f"{name}: {mean_time * 1000:.1f} ms ± {std_time * 1000:.1f} ms")
    return mean_time


def run_benchmark():
    # Load data
    data_dir = Path("tests/data/sharp_sequence")
    if not data_dir.exists():
        data_dir = Path("python/tests/data/sharp_sequence")

    if not data_dir.exists():
        logger.warning("Test data not found. Using synthetic data.")
        N, n_frames = 50000, 5
        all_means = [np.random.rand(N, 3).astype(np.float32) for _ in range(n_frames)]
    else:
        all_means = load_data(data_dir, num_frames=5)

    max_distance = 0.05
    window_size = 3

    # === Raw KNN Analysis ===
    logger.info("=" * 60)
    logger.info("RAW KNN ANALYSIS (first frame pair)")
    logger.info("=" * 60)

    knn_analysis = analyze_knn_differences(all_means[0], all_means[1])
    logger.info(f"  Index agreement: {knn_analysis['idx_agreement_pct']:.2f}%")
    logger.info(f"  Disagreements: {knn_analysis['n_disagreements']}")
    logger.info(f"  Near-ties (within 1% dist): {knn_analysis['near_ties']}")
    logger.info(f"  Rel dist diff (mean): {knn_analysis['rel_dist_diff_mean']:.6f}")
    logger.info(f"  Rel dist diff (max): {knn_analysis['rel_dist_diff_max']:.6f}")

    # === Validation ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION: Comparing CUDA vs FAISS results")
    logger.info("=" * 60)

    validation = validate_cuda_vs_faiss(all_means, max_distance, window_size)
    if validation["agreement_ratio"] < 0.95:
        logger.warning(f"Low agreement: {validation['agreement_ratio']:.2%}")
    else:
        logger.info(f"Validation PASSED: {validation['agreement_ratio']:.2%} agreement")

    # === Benchmarks ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARKS")
    logger.info("=" * 60)

    faiss_time = benchmark_function(
        "FAISS (CPU)",
        lambda: match_gaussians_sliding_window_faiss(all_means, max_distance, window_size),
    )

    cuda_time = benchmark_function(
        "cuTile (GPU)",
        lambda: match_gaussians_sliding_window_cuda(all_means, max_distance, window_size),
    )

    speedup = faiss_time / cuda_time
    logger.info(f"\nSpeedup: {speedup:.1f}x")


if __name__ == "__main__":
    run_benchmark()
