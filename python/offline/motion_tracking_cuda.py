"""CUDA-accelerated Gaussian motion tracking using FAISS and CuPy.

Provides GPU-accelerated alternatives for:
- Bidirectional nearest-neighbor matching (FAISS)
- Trajectory building and motion fitting (CuPy)

Install dependencies:
    pip install faiss-gpu cupy-cuda12x  # or cupy-cuda11x for CUDA 11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cupy as cp

logger = logging.getLogger(__name__)


def check_cuda_available() -> bool:
    """Check if CUDA acceleration is available."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        test_arr = cp.array([1, 2, 3])
        _ = test_arr.sum()
        return True
    except Exception as e:
        logger.debug(f"CuPy not available: {e}")
        return False


def assert_cuda_available() -> None:
    """Assert that CuPy CUDA is working, with detailed error message."""
    try:
        import cupy as cp
    except ImportError as e:
        raise RuntimeError(
            f"CuPy not installed. Install with: uv pip install cupy-cuda13x\nError: {e}"
        )

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            raise RuntimeError("No CUDA devices found")

        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        logger.info(f"CuPy CUDA device: {device_name}")

        test_arr = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        result = float(test_arr.sum())
        assert result == 6.0, f"CuPy compute test failed: expected 6.0, got {result}"
        logger.info("CuPy CUDA verified working")

    except Exception as e:
        raise RuntimeError(
            f"CuPy CUDA initialization failed: {e}\n"
            f"Make sure CUDA toolkit is installed and matches cupy-cuda13x"
        )


def check_faiss_gpu_available() -> bool:
    """Check if FAISS GPU is available (Deprecated)."""
    return False


def check_faiss_available() -> bool:
    """Check if FAISS is available (Deprecated)."""
    return False


def check_faiss_available() -> bool:
    """Check if FAISS (CPU or GPU) is available."""
    try:
        import faiss

        return True
    except ImportError:
        return False


@dataclass
class TrajectoryDataGPU:
    """Trajectory data stored on GPU for batch processing."""

    trajectory_ids: np.ndarray  # (total_observations,) trajectory ID for each observation
    frame_indices: np.ndarray  # (total_observations,) frame index
    times_normalized: np.ndarray  # (total_observations,) normalized time
    positions: np.ndarray  # (total_observations, 3)
    scales: np.ndarray  # (total_observations, 3)
    rotations: np.ndarray  # (total_observations, 4)
    colors: np.ndarray  # (total_observations, 3)
    opacities: np.ndarray  # (total_observations,)
    n_trajectories: int


def match_gaussians_faiss(
    means_a: np.ndarray,
    means_b: np.ndarray,
    max_distance: float,
    use_gpu: bool = False,
) -> list[tuple[int, int]]:
    """Match Gaussians using FAISS for fast nearest-neighbor search.

    Uses bidirectional matching with mutual best match filter.
    Can use GPU or CPU FAISS depending on availability.
    """
    import faiss

    means_a = np.ascontiguousarray(means_a, dtype=np.float32)
    means_b = np.ascontiguousarray(means_b, dtype=np.float32)

    if use_gpu and check_faiss_gpu_available():
        res = faiss.StandardGpuResources()
        index_a = faiss.GpuIndexFlatL2(res, 3)
        index_b = faiss.GpuIndexFlatL2(res, 3)
    else:
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
    for i in range(len(means_a)):
        j = idx_a_to_b[i]
        if dist_a_to_b[i] < max_distance and idx_b_to_a[j] == i:
            matches.append((i, j))

    return matches


def match_gaussians_batch_faiss(
    all_means: list[np.ndarray],
    max_distance: float,
    use_gpu: bool = False,
) -> list[list[tuple[int, int]]]:
    """Match Gaussians between all consecutive frame pairs using FAISS.

    Deprecated: Use match_gaussians_batch_cupy instead.
    """
    raise NotImplementedError("FAISS support has been removed.")


def match_gaussians_batch_cupy(
    all_means: list[np.ndarray],
    max_distance: float,
) -> list[list[tuple[int, int]]]:
    """Match Gaussians between all consecutive frame pairs using CuPy.

    Implements batched brute-force KNN on GPU without FAISS.
    Appropriate for when FAISS GPU is not available but CuPy is.
    """
    import cupy as cp

    all_matches = []
    chunk_size = 4096  # Process A in chunks to limit memory usage

    for i in range(len(all_means) - 1):
        # Convert host arrays to device arrays
        means_a = cp.asarray(all_means[i], dtype=cp.float32)
        means_b = cp.asarray(all_means[i + 1], dtype=cp.float32)

        N = len(means_a)
        M = len(means_b)

        # We need to find:
        # 1. Nearest neighbor in B for each A
        # 2. Nearest neighbor in A for each B

        idx_a_to_b = cp.empty(N, dtype=cp.int32)
        dist_a_to_b_sq = cp.empty(N, dtype=cp.float32)

        # --- Step 1: A -> B ---
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>
        b_sq_norm = cp.sum(means_b**2, axis=1)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            a_chunk = means_a[start:end]

            # (B_S, 3) @ (3, M) -> (B_S, M)
            dist_matrix = -2 * cp.dot(a_chunk, means_b.T)
            dist_matrix += cp.sum(a_chunk**2, axis=1)[:, None]
            dist_matrix += b_sq_norm[None, :]

            min_dist_sq = cp.min(dist_matrix, axis=1)
            min_indices = cp.argmin(dist_matrix, axis=1)

            idx_a_to_b[start:end] = min_indices
            dist_a_to_b_sq[start:end] = min_dist_sq

        # --- Step 2: B -> A ---
        idx_b_to_a = cp.empty(M, dtype=cp.int32)
        a_sq_norm = cp.sum(means_a**2, axis=1)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            b_chunk = means_b[start:end]

            dist_matrix = -2 * cp.dot(b_chunk, means_a.T)
            dist_matrix += cp.sum(b_chunk**2, axis=1)[:, None]
            dist_matrix += a_sq_norm[None, :]

            idx_b_to_a[start:end] = cp.argmin(dist_matrix, axis=1)

        # --- Step 3: Filter matches on GPU ---
        max_dist_sq = max_distance * max_distance

        # Check mutual nearest neighbor condition and distance threshold
        # We process on CPU or GPU? Let's do it on GPU then copy result
        # i is index in A, j is index in B

        # Array of indices 0..N-1
        a_indices = cp.arange(N, dtype=cp.int32)

        # j = idx_a_to_b[i]
        potential_matches_b = idx_a_to_b

        # Check 1: Distance within threshold
        valid_dist = dist_a_to_b_sq < max_dist_sq

        # Check 2: Mutual match: idx_b_to_a[j] == i
        # We need to gather idx_b_to_a values at indices potential_matches_b
        reciprocal_matches_a = idx_b_to_a[potential_matches_b]
        valid_mutual = reciprocal_matches_a == a_indices

        valid_mask = valid_dist & valid_mutual

        valid_indices_a = a_indices[valid_mask]
        valid_indices_b = potential_matches_b[valid_mask]

        # Transfer to CPU
        valid_a_cpu = cp.asnumpy(valid_indices_a)
        valid_b_cpu = cp.asnumpy(valid_indices_b)

        # Zip into tuples
        matches = list(zip(valid_a_cpu, valid_b_cpu))
        all_matches.append(matches)

        logger.debug(f"Frame {i}->{i + 1}: {len(matches)} matches (CuPy)")

    return all_matches


def build_trajectories_gpu(
    frames: list,  # list[GaussianFrame]
    max_distance: float,
) -> TrajectoryDataGPU:
    """Build trajectories using accelerated matching.

    Uses union-find on CPU (fast enough) with CuPy for matching.
    Falls back to scipy KDTree if CuPy is unavailable.
    """
    if len(frames) == 0:
        raise ValueError("No frames provided")

    t_start = frames[0].timestamp_ms
    t_end = frames[-1].timestamp_ms
    t_range = max(t_end - t_start, 1e-6)

    total_gaussians = sum(len(f.means) for f in frames)
    logger.info(f"Building trajectories for {total_gaussians} total Gaussian observations")

    all_means = [f.means for f in frames]

    if check_cuda_available():
        logger.info("Using CuPy KNN for matching")
        all_matches = match_gaussians_batch_cupy(all_means, max_distance)
    else:
        logger.info("CuPy not available, using scipy KDTree")
        from offline.export_gaussian_ply import match_gaussians_bidirectional

        all_matches = []
        for i in range(len(frames) - 1):
            matches = match_gaussians_bidirectional(
                frames[i].means, frames[i + 1].means, max_distance
            )
            all_matches.append(matches)

    frame_offsets = [0]
    for f in frames:
        frame_offsets.append(frame_offsets[-1] + len(f.means))

    parent = list(range(total_gaussians))
    rank = [0] * total_gaussians

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    for frame_idx, matches in enumerate(all_matches):
        offset_a = frame_offsets[frame_idx]
        offset_b = frame_offsets[frame_idx + 1]
        for idx_a, idx_b in matches:
            union(offset_a + idx_a, offset_b + idx_b)

    root_to_traj_id: dict[int, int] = {}
    trajectory_ids = np.zeros(total_gaussians, dtype=np.int32)

    for i in range(total_gaussians):
        root = find(i)
        if root not in root_to_traj_id:
            root_to_traj_id[root] = len(root_to_traj_id)
        trajectory_ids[i] = root_to_traj_id[root]

    n_trajectories = len(root_to_traj_id)
    logger.info(f"Found {n_trajectories} unique trajectories")

    frame_indices = np.zeros(total_gaussians, dtype=np.int32)
    times_normalized = np.zeros(total_gaussians, dtype=np.float32)
    positions = np.zeros((total_gaussians, 3), dtype=np.float32)
    scales = np.zeros((total_gaussians, 3), dtype=np.float32)
    rotations = np.zeros((total_gaussians, 4), dtype=np.float32)
    colors = np.zeros((total_gaussians, 3), dtype=np.float32)
    opacities = np.zeros(total_gaussians, dtype=np.float32)

    idx = 0
    for frame_idx, frame in enumerate(frames):
        n = len(frame.means)
        t_norm = (frame.timestamp_ms - t_start) / t_range

        frame_indices[idx : idx + n] = frame_idx
        times_normalized[idx : idx + n] = t_norm
        positions[idx : idx + n] = frame.means
        scales[idx : idx + n] = frame.scales
        rotations[idx : idx + n] = frame.rotations
        colors[idx : idx + n] = frame.colors
        opacities[idx : idx + n] = frame.opacities
        idx += n

    return TrajectoryDataGPU(
        trajectory_ids=trajectory_ids,
        frame_indices=frame_indices,
        times_normalized=times_normalized,
        positions=positions,
        scales=scales,
        rotations=rotations,
        colors=colors,
        opacities=opacities,
        n_trajectories=n_trajectories,
    )


def fit_trajectories_cupy(
    traj_data: TrajectoryDataGPU,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Fit motion parameters to all trajectories using CuPy GPU acceleration.

    Returns:
        (means, scales, rotations, colors, opacities, motion, time_center, time_scale)
    """
    import cupy as cp
    import cupyx

    n_traj = traj_data.n_trajectories
    n_obs = len(traj_data.trajectory_ids)

    traj_ids_gpu = cp.asarray(traj_data.trajectory_ids)
    times_gpu = cp.asarray(traj_data.times_normalized)
    positions_gpu = cp.asarray(traj_data.positions)
    scales_gpu = cp.asarray(traj_data.scales)
    rotations_gpu = cp.asarray(traj_data.rotations)
    colors_gpu = cp.asarray(traj_data.colors)
    opacities_gpu = cp.asarray(traj_data.opacities)

    time_sum = cp.zeros(n_traj, dtype=cp.float32)
    time_sq_sum = cp.zeros(n_traj, dtype=cp.float32)
    obs_count = cp.zeros(n_traj, dtype=cp.int32)

    cupyx.scatter_add(time_sum, traj_ids_gpu, times_gpu)
    cupyx.scatter_add(time_sq_sum, traj_ids_gpu, times_gpu**2)
    cupyx.scatter_add(obs_count, traj_ids_gpu, cp.ones(n_obs, dtype=cp.int32))

    obs_count_safe = cp.maximum(obs_count, 1).astype(cp.float32)
    time_center = time_sum / obs_count_safe
    time_var = time_sq_sum / obs_count_safe - time_center**2
    time_scale = cp.sqrt(cp.maximum(time_var, 0.0025)) * 2.0
    time_scale = cp.maximum(time_scale, 0.05)
    time_scale_log = cp.log(time_scale)

    dt = times_gpu - time_center[traj_ids_gpu]

    pos_sum = cp.zeros((n_traj, 3), dtype=cp.float32)
    pos_dt_sum = cp.zeros((n_traj, 3), dtype=cp.float32)
    dt_sq_sum = cp.zeros(n_traj, dtype=cp.float32)

    for dim in range(3):
        cupyx.scatter_add(pos_sum[:, dim], traj_ids_gpu, positions_gpu[:, dim])
        cupyx.scatter_add(pos_dt_sum[:, dim], traj_ids_gpu, positions_gpu[:, dim] * dt)
    cupyx.scatter_add(dt_sq_sum, traj_ids_gpu, dt**2)

    dt_sq_sum_safe = cp.maximum(dt_sq_sum, 1e-8)
    velocity = pos_dt_sum / dt_sq_sum_safe[:, None]

    single_obs_mask = obs_count <= 1
    velocity[single_obs_mask] = 0.0

    pos_center = pos_sum / obs_count_safe[:, None]

    weights = cp.maximum(opacities_gpu, 0.01)
    weight_sum = cp.zeros(n_traj, dtype=cp.float32)
    cupyx.scatter_add(weight_sum, traj_ids_gpu, weights)
    weight_sum_safe = cp.maximum(weight_sum, 1e-8)

    weighted_scales = cp.zeros((n_traj, 3), dtype=cp.float32)
    weighted_rotations = cp.zeros((n_traj, 4), dtype=cp.float32)
    weighted_colors = cp.zeros((n_traj, 3), dtype=cp.float32)
    weighted_opacities = cp.zeros(n_traj, dtype=cp.float32)

    for dim in range(3):
        cupyx.scatter_add(weighted_scales[:, dim], traj_ids_gpu, scales_gpu[:, dim] * weights)
        cupyx.scatter_add(weighted_colors[:, dim], traj_ids_gpu, colors_gpu[:, dim] * weights)
    for dim in range(4):
        cupyx.scatter_add(weighted_rotations[:, dim], traj_ids_gpu, rotations_gpu[:, dim] * weights)
    cupyx.scatter_add(weighted_opacities, traj_ids_gpu, opacities_gpu * weights)

    weighted_scales /= weight_sum_safe[:, None]
    weighted_rotations /= weight_sum_safe[:, None]
    weighted_colors /= weight_sum_safe[:, None]
    weighted_opacities /= weight_sum_safe

    rot_norms = cp.linalg.norm(weighted_rotations, axis=1, keepdims=True)
    weighted_rotations /= cp.maximum(rot_norms, 1e-8)

    results = (
        cp.asnumpy(pos_center).astype(np.float32),
        cp.asnumpy(weighted_scales).astype(np.float32),
        cp.asnumpy(weighted_rotations).astype(np.float32),
        cp.asnumpy(weighted_colors).astype(np.float32),
        cp.asnumpy(weighted_opacities).astype(np.float32),
        cp.asnumpy(velocity).astype(np.float32),
        cp.asnumpy(time_center).astype(np.float32),
        cp.asnumpy(time_scale_log).astype(np.float32),
    )

    vel_mag = np.linalg.norm(results[5], axis=1)
    logger.info(
        f"Motion stats: velocity magnitude mean={vel_mag.mean():.4f}, max={vel_mag.max():.4f}"
    )
    logger.info(f"Time center: min={results[6].min():.3f}, max={results[6].max():.3f}")
    logger.info(f"Time scale (log): min={results[7].min():.3f}, max={results[7].max():.3f}")

    return results


def compute_motion_vectors_gpu(
    frames: list,  # list[GaussianFrame]
    fps: float,
    max_match_distance: float | None = None,
    match_distance_ratio: float = 0.02,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """GPU-accelerated motion vector computation.

    Drop-in replacement for compute_motion_vectors() that uses FAISS GPU
    for matching and CuPy for trajectory fitting.
    """
    from offline.export_gaussian_ply import compute_scene_scale

    if len(frames) < 2:
        frame = frames[0]
        n = len(frame.means)
        return (
            frame.means,
            frame.scales,
            frame.rotations,
            frame.colors,
            frame.opacities,
            np.zeros((n, 3), dtype=np.float32),
            np.full(n, 0.5, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    if max_match_distance is None:
        scene_scale = compute_scene_scale(frames)
        max_match_distance = scene_scale * match_distance_ratio
        logger.info(
            f"Scene scale: {scene_scale:.2f}, using match distance: {max_match_distance:.2f}"
        )

    traj_data = build_trajectories_gpu(frames, max_match_distance)

    if check_cuda_available():
        logger.info("Using CuPy for trajectory fitting")
        return fit_trajectories_cupy(traj_data)
    else:
        logger.warning("CuPy not available, falling back to CPU fitting")
        from offline.export_gaussian_ply import (
            GaussianTrajectory,
            fit_trajectory_motion,
        )

        trajectories: dict[int, GaussianTrajectory] = {}
        for i in range(len(traj_data.trajectory_ids)):
            traj_id = traj_data.trajectory_ids[i]
            if traj_id not in trajectories:
                trajectories[traj_id] = GaussianTrajectory(
                    frame_indices=[],
                    times_normalized=[],
                    positions=[],
                    scales=[],
                    rotations=[],
                    colors=[],
                    opacities=[],
                )
            traj = trajectories[traj_id]
            traj.frame_indices.append(traj_data.frame_indices[i])
            traj.times_normalized.append(traj_data.times_normalized[i])
            traj.positions.append(traj_data.positions[i])
            traj.scales.append(traj_data.scales[i])
            traj.rotations.append(traj_data.rotations[i])
            traj.colors.append(traj_data.colors[i])
            traj.opacities.append(traj_data.opacities[i])

        n_traj = len(trajectories)
        all_means = np.zeros((n_traj, 3), dtype=np.float32)
        all_scales = np.zeros((n_traj, 3), dtype=np.float32)
        all_rotations = np.zeros((n_traj, 4), dtype=np.float32)
        all_colors = np.zeros((n_traj, 3), dtype=np.float32)
        all_opacities = np.zeros(n_traj, dtype=np.float32)
        all_motion = np.zeros((n_traj, 3), dtype=np.float32)
        all_time_center = np.zeros(n_traj, dtype=np.float32)
        all_time_scale = np.zeros(n_traj, dtype=np.float32)

        for i, traj in enumerate(trajectories.values()):
            pos_center, velocity, t_center, t_scale_log = fit_trajectory_motion(traj)
            all_means[i] = pos_center
            all_motion[i] = velocity
            all_time_center[i] = t_center
            all_time_scale[i] = t_scale_log

            weights = np.maximum(np.array(traj.opacities), 0.01)
            weights /= weights.sum()
            all_scales[i] = np.average(traj.scales, axis=0, weights=weights)
            all_rotations[i] = np.average(traj.rotations, axis=0, weights=weights)
            all_colors[i] = np.average(traj.colors, axis=0, weights=weights)
            all_opacities[i] = np.average(traj.opacities, weights=weights)

        rot_norms = np.linalg.norm(all_rotations, axis=1, keepdims=True)
        all_rotations /= np.maximum(rot_norms, 1e-8)

        return (
            all_means,
            all_scales,
            all_rotations,
            all_colors,
            all_opacities,
            all_motion,
            all_time_center,
            all_time_scale,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("CUDA acceleration status:")
    print(f"  CuPy available: {check_cuda_available()}")
    print(f"  FAISS available: {check_faiss_available()}")
    print(f"  FAISS GPU available: {check_faiss_gpu_available()}")

    if check_cuda_available():
        import cupy as cp

        print(f"  CuPy version: {cp.__version__}")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"  CUDA device: {name}")

        print("\nRunning CuPy verification...")
        assert_cuda_available()
        print("CuPy is working correctly!")

    if check_faiss_available():
        import faiss

        print(
            f"\nFAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}"
        )
        if check_faiss_gpu_available():
            print(f"  FAISS GPUs: {faiss.get_num_gpus()}")
