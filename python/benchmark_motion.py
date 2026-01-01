import logging
import numpy as np
from pathlib import Path
import pyperf
from plyfile import PlyData
from offline.motion_tracking_cpu import match_gaussians_sliding_window_faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_means_from_ply(ply_path):
    """Load Gaussian means from PLY file."""
    plydata = PlyData.read(ply_path)
    vertex_data = plydata["vertex"]
    means = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(
        np.float32
    )
    return means


def load_data(data_dir, num_frames=10):
    """Load sequence of PLY files."""
    ply_files = sorted(Path(data_dir).glob("*.ply"))
    if not ply_files:
        raise ValueError(f"No PLY files found in {data_dir}")

    if len(ply_files) > num_frames:
        ply_files = ply_files[:num_frames]

    logger.info(f"Loading {len(ply_files)} frames from {data_dir}...")
    all_means = []
    for p in ply_files:
        means = load_means_from_ply(p)
        all_means.append(means)

    return all_means


def run_benchmark():
    # Use minimal runs for quick benchmarking
    runner = pyperf.Runner(loops=1, warmups=1, processes=1)

    # Setup data
    # Use relative path assuming running from python/ directory or project root
    data_dir = Path("tests/data/sharp_sequence")
    if not data_dir.exists():
        # Try relative to python/
        data_dir = Path("python/tests/data/sharp_sequence")

    if not data_dir.exists():
        logger.warning(f"Test data not found at {data_dir}. Using synthetic data.")
        # Synthetic data fallback
        N = 50000
        n_frames = 5
        all_means = [np.random.rand(N, 3).astype(np.float32) for _ in range(n_frames)]
    else:
        try:
            all_means = load_data(data_dir, num_frames=5)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return

    # Benchmark - pyperf handles warmup internally
    logger.info("Starting benchmark...")
    runner.bench_func(
        "match_gaussians_sliding_window_faiss",
        lambda: match_gaussians_sliding_window_faiss(all_means, max_distance=0.05, window_size=3),
    )


if __name__ == "__main__":
    run_benchmark()
