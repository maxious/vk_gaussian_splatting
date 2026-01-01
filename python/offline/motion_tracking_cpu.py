"""CPU-accelerated Gaussian motion tracking using FAISS.

Provides CPU-accelerated alternatives for:
- Bidirectional nearest-neighbor matching (FAISS)
- Trajectory building and motion fitting (NumPy)

Install dependencies:
    pip install faiss-cpu
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def check_faiss_available() -> bool:
    """Check if FAISS is available."""
    try:
        import faiss

        return True
    except ImportError:
        return False


@dataclass
class TrajectoryData:
    """Trajectory data stored in memory for batch processing."""

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
) -> list[tuple[int, int]]:
    """Match Gaussians using FAISS for fast nearest-neighbor search.

    Uses bidirectional matching with mutual best match filter.
    """
    import faiss

    means_a = np.ascontiguousarray(means_a, dtype=np.float32)
    means_b = np.ascontiguousarray(means_b, dtype=np.float32)

    index_a = faiss.IndexFlatL2(3)
    index_b = faiss.IndexFlatL2(3)

    index_a.add(means_a)  # type: ignore
    index_b.add(means_b)  # type: ignore

    dist_a_to_b, idx_a_to_b = index_b.search(means_a, 1)  # type: ignore
    dist_b_to_a, idx_b_to_a = index_a.search(means_b, 1)  # type: ignore

    dist_a_to_b = np.sqrt(dist_a_to_b[:, 0])
    idx_a_to_b = idx_a_to_b[:, 0]
    idx_b_to_a = idx_b_to_a[:, 0]

    matches = []
    for i in range(len(means_a)):
        j = idx_a_to_b[i]
        if dist_a_to_b[i] < max_distance and idx_b_to_a[j] == i:
            matches.append((i, j))

    return matches


def match_gaussians_sliding_window_faiss(
    all_means: list[np.ndarray],
    max_distance: float,
    window_size: int = 3,
) -> list[tuple[int, int, int, int]]:
    """Match Gaussians using sliding window for gap-bridging.

    For each point in frame i, finds the best match in frames i+1 to i+window_size.
    Prioritizes closer frames - if a good match is found in i+1, doesn't look further.

    Args:
        all_means: List of (N_i, 3) arrays of Gaussian positions per frame
        max_distance: Maximum distance for valid match
        window_size: How many future frames to search (default 3)

    Returns:
        List of (frame_a, idx_a, frame_b, idx_b) tuples representing matches
    """
    import faiss

    n_frames = len(all_means)
    if n_frames < 2:
        return []

    # Pre-build all FAISS indices
    indices = []
    for means in all_means:
        means = np.ascontiguousarray(means, dtype=np.float32)
        index = faiss.IndexFlatL2(3)
        index.add(means)
        indices.append(index)

    all_matches: list[tuple[int, int, int, int]] = []
    max_dist_sq = max_distance * max_distance

    for frame_a in range(n_frames - 1):
        means_a = np.ascontiguousarray(all_means[frame_a], dtype=np.float32)
        n_a = len(means_a)

        # Track which points in frame_a have been matched
        matched_a = np.zeros(n_a, dtype=bool)
        # Best match info for each point: (frame_b, idx_b, distance_sq)
        best_match_frame = np.full(n_a, -1, dtype=np.int32)
        best_match_idx = np.full(n_a, -1, dtype=np.int32)
        best_match_dist = np.full(n_a, np.inf, dtype=np.float32)

        # Search in window, prioritizing closer frames
        for offset in range(1, min(window_size + 1, n_frames - frame_a)):
            frame_b = frame_a + offset
            means_b = np.ascontiguousarray(all_means[frame_b], dtype=np.float32)
            n_b = len(means_b)

            # Forward search: A -> B
            dist_a_to_b, idx_a_to_b = indices[frame_b].search(means_a, 1)
            dist_a_to_b = dist_a_to_b[:, 0]  # Squared distances from FAISS
            idx_a_to_b = idx_a_to_b[:, 0]

            # Backward search: B -> A (for mutual match check)
            dist_b_to_a, idx_b_to_a = indices[frame_a].search(means_b, 1)
            idx_b_to_a = idx_b_to_a[:, 0]

            # Find valid mutual matches
            # Optimization: Vectorized check

            # Candidates for A
            j_indices = idx_a_to_b

            # Check mutual match
            # idx_b_to_a[j] == i
            # idx_b_to_a is (N_b,), we need to index it with j_indices (N_a,)
            # j_indices values are < N_b
            is_mutual = idx_b_to_a[j_indices] == np.arange(n_a)

            # Check distance
            is_close = dist_a_to_b < max_dist_sq

            # Check not already matched
            is_new = ~matched_a

            valid_mask = is_mutual & is_close & is_new

            matched_indices = np.where(valid_mask)[0]

            if len(matched_indices) > 0:
                matched_a[matched_indices] = True
                best_match_frame[matched_indices] = frame_b
                best_match_idx[matched_indices] = j_indices[matched_indices]
                best_match_dist[matched_indices] = dist_a_to_b[matched_indices]

        # Collect matches for this source frame
        # (frame_a, i, best_match_frame[i], best_match_idx[i])
        valid_matches = np.where(best_match_frame >= 0)[0]
        for i in valid_matches:
            all_matches.append(
                (
                    frame_a,
                    i,
                    int(best_match_frame[i]),
                    int(best_match_idx[i]),
                )
            )

    return all_matches


def match_gaussians_batch_faiss(
    all_means: list[np.ndarray],
    max_distance: float,
    window_size: int = 1,
) -> list[list[tuple[int, int]]]:
    """Match Gaussians between consecutive frame pairs using FAISS.

    This is a compatibility wrapper that returns matches in the old format.
    For new code, use match_gaussians_sliding_window_faiss directly.
    """
    if window_size == 1:
        # Fast path: just do pairwise matching
        all_matches = []
        for i in range(len(all_means) - 1):
            matches = match_gaussians_faiss(all_means[i], all_means[i + 1], max_distance)
            all_matches.append(matches)
        return all_matches
    else:
        # Use sliding window and convert format
        sw_matches = match_gaussians_sliding_window_faiss(all_means, max_distance, window_size)

        # Group by consecutive frame pairs for compatibility
        n_frames = len(all_means)
        all_matches: list[list[tuple[int, int]]] = [[] for _ in range(n_frames - 1)]

        for frame_a, idx_a, frame_b, idx_b in sw_matches:
            # Only include consecutive matches in compatibility mode
            if frame_b == frame_a + 1:
                all_matches[frame_a].append((idx_a, idx_b))

        return all_matches


def build_trajectories(
    frames: list,  # list[GaussianFrame]
    max_distance: float,
    window_size: int = 3,
) -> TrajectoryData:
    """Build trajectories using accelerated matching with sliding window.

    Args:
        frames: List of GaussianFrame objects
        max_distance: Maximum distance for valid matches
        window_size: How many future frames to search for matches (default 3)
    """
    if len(frames) == 0:
        raise ValueError("No frames provided")

    t_start = frames[0].timestamp_ms
    t_end = frames[-1].timestamp_ms
    t_range = max(t_end - t_start, 1e-6)

    total_gaussians = sum(len(f.means) for f in frames)
    logger.info(f"Building trajectories for {total_gaussians} total Gaussian observations")

    all_means = [f.means for f in frames]

    # Choose best available matching method
    sliding_matches = None
    all_matches = None
    if check_faiss_available():
        logger.info(f"Using FAISS sliding window matching (window_size={window_size})")
        sliding_matches = match_gaussians_sliding_window_faiss(all_means, max_distance, window_size)
        use_sliding_format = True
    else:
        logger.info("FAISS not available, using scipy KDTree (pairwise only)")
        all_matches = []
        for i in range(len(frames) - 1):
            matches = match_gaussians_faiss(frames[i].means, frames[i + 1].means, max_distance)
            all_matches.append(matches)
        use_sliding_format = False

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

    # Apply matches to union-find
    if use_sliding_format:
        assert sliding_matches is not None  # Type guard for mypy/ty
        # Sliding window format: (frame_a, idx_a, frame_b, idx_b)
        gap_bridged = 0
        for frame_a, idx_a, frame_b, idx_b in sliding_matches:
            global_a = frame_offsets[frame_a] + idx_a
            global_b = frame_offsets[frame_b] + idx_b
            union(global_a, global_b)
            if frame_b > frame_a + 1:
                gap_bridged += 1
        logger.info(f"Total matches: {len(sliding_matches)}, gap-bridged: {gap_bridged}")
    else:
        assert all_matches is not None  # Type guard for mypy/ty
        # Pairwise format: list of lists
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

    return TrajectoryData(
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


def fit_trajectories(
    traj_data: TrajectoryData,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Fit motion parameters to all trajectories using NumPy.

    Returns:
        (means, scales, rotations, colors, opacities, motion, time_center, time_scale)
    """
    n_traj = traj_data.n_trajectories
    n_obs = len(traj_data.trajectory_ids)

    # All data is already on CPU (numpy)
    traj_ids = traj_data.trajectory_ids
    times = traj_data.times_normalized
    positions = traj_data.positions
    scales = traj_data.scales
    rotations = traj_data.rotations
    colors = traj_data.colors
    opacities = traj_data.opacities

    time_sum = np.zeros(n_traj, dtype=np.float32)
    time_sq_sum = np.zeros(n_traj, dtype=np.float32)
    obs_count = np.zeros(n_traj, dtype=np.int32)

    np.add.at(time_sum, traj_ids, times)
    np.add.at(time_sq_sum, traj_ids, times**2)
    np.add.at(obs_count, traj_ids, np.ones(n_obs, dtype=np.int32))

    obs_count_safe = np.maximum(obs_count, 1).astype(np.float32)
    time_center = time_sum / obs_count_safe
    time_var = time_sq_sum / obs_count_safe - time_center**2
    time_scale = np.sqrt(np.maximum(time_var, 0.0025)) * 2.0
    time_scale = np.maximum(time_scale, 0.05)
    time_scale_log = np.log(time_scale)

    dt = times - time_center[traj_ids]

    pos_sum = np.zeros((n_traj, 3), dtype=np.float32)
    pos_dt_sum = np.zeros((n_traj, 3), dtype=np.float32)
    dt_sq_sum = np.zeros(n_traj, dtype=np.float32)

    for dim in range(3):
        np.add.at(pos_sum[:, dim], traj_ids, positions[:, dim])
        np.add.at(pos_dt_sum[:, dim], traj_ids, positions[:, dim] * dt)
    np.add.at(dt_sq_sum, traj_ids, dt**2)

    dt_sq_sum_safe = np.maximum(dt_sq_sum, 1e-8)
    velocity = pos_dt_sum / dt_sq_sum_safe[:, None]

    single_obs_mask = obs_count <= 1
    velocity[single_obs_mask] = 0.0

    pos_center = pos_sum / obs_count_safe[:, None]

    weights = np.maximum(opacities, 0.01)
    weight_sum = np.zeros(n_traj, dtype=np.float32)
    np.add.at(weight_sum, traj_ids, weights)
    weight_sum_safe = np.maximum(weight_sum, 1e-8)

    weighted_scales = np.zeros((n_traj, 3), dtype=np.float32)
    weighted_rotations = np.zeros((n_traj, 4), dtype=np.float32)
    weighted_colors = np.zeros((n_traj, 3), dtype=np.float32)
    weighted_opacities = np.zeros(n_traj, dtype=np.float32)

    for dim in range(3):
        np.add.at(weighted_scales[:, dim], traj_ids, scales[:, dim] * weights)
        np.add.at(weighted_colors[:, dim], traj_ids, colors[:, dim] * weights)
    for dim in range(4):
        np.add.at(weighted_rotations[:, dim], traj_ids, rotations[:, dim] * weights)
    np.add.at(weighted_opacities, traj_ids, opacities * weights)

    weighted_scales /= weight_sum_safe[:, None]
    weighted_rotations /= weight_sum_safe[:, None]
    weighted_colors /= weight_sum_safe[:, None]
    weighted_opacities /= weight_sum_safe

    rot_norms = np.linalg.norm(weighted_rotations, axis=1, keepdims=True)
    weighted_rotations /= np.maximum(rot_norms, 1e-8)

    results = (
        pos_center.astype(np.float32),
        weighted_scales.astype(np.float32),
        weighted_rotations.astype(np.float32),
        weighted_colors.astype(np.float32),
        weighted_opacities.astype(np.float32),
        velocity.astype(np.float32),
        time_center.astype(np.float32),
        time_scale_log.astype(np.float32),
    )

    vel_mag = np.linalg.norm(results[5], axis=1)
    logger.info(
        f"Motion stats: velocity magnitude mean={vel_mag.mean():.4f}, max={vel_mag.max():.4f}"
    )
    logger.info(f"Time center: min={results[6].min():.3f}, max={results[6].max():.3f}")
    logger.info(f"Time scale (log): min={results[7].min():.3f}, max={results[7].max():.3f}")

    return results


def compute_motion_vectors(
    frames: list,  # list[GaussianFrame]
    fps: float,
    max_match_distance: float | None = None,
    match_distance_ratio: float = 0.02,
    window_size: int = 3,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """CPU-accelerated motion vector computation.

    Uses FAISS for matching and NumPy for trajectory fitting.
    """
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

    def compute_scene_scale(frames) -> float:
        """Compute approximate scene scale from Gaussian positions."""
        all_means = np.vstack([f.means for f in frames])
        bbox_min = all_means.min(axis=0)
        bbox_max = all_means.max(axis=0)
        diagonal = np.linalg.norm(bbox_max - bbox_min)
        return float(diagonal)

    if max_match_distance is None:
        scene_scale = compute_scene_scale(frames)
        max_match_distance = scene_scale * match_distance_ratio
        logger.info(
            f"Scene scale: {scene_scale:.2f}, using match distance: {max_match_distance:.2f}"
        )

    traj_data = build_trajectories(frames, max_match_distance, window_size=window_size)

    return fit_trajectories(traj_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Motion Tracking CPU status:")
    print(f"  FAISS available: {check_faiss_available()}")

    if check_faiss_available():
        import faiss

        print(
            f"  FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}"
        )
