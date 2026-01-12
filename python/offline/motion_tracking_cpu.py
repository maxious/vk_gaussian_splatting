"""CPU-accelerated Gaussian motion tracking using FAISS.

Provides CPU-accelerated alternatives for:
- Bidirectional nearest-neighbor matching (FAISS)
- Trajectory building and motion fitting (NumPy)

Install dependencies:
    pip install faiss-cpu
"""

# type: ignore  # Suppress FAISS type checking issues

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numba import jit

from .union_find import (
    union_find_components,
    union_find_find,
    union_find_init,
    union_find_union,
)

# Suppress verbose Numba debug logging
logging.getLogger("numba").setLevel(logging.WARNING)

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


@jit(nopython=True)
def apply_matches_to_union_find(
    parent: np.ndarray,
    rank: np.ndarray,
    matches: np.ndarray,
    frame_offsets: np.ndarray,
) -> int:
    """Apply sliding window matches to union-find structure.

    Args:
        parent: Union-find parent array
        rank: Union-find rank array
        matches: (M, 4) array of (frame_a, idx_a, frame_b, idx_b)
        frame_offsets: Cumulative offsets for each frame

    Returns:
        Number of gap-bridged matches
    """
    gap_bridged = 0
    n_matches = matches.shape[0]

    # Note: Can't parallelize union-find operations (race conditions)
    # but the find operations with path compression are fast
    for i in range(n_matches):
        frame_a = matches[i, 0]
        idx_a = matches[i, 1]
        frame_b = matches[i, 2]
        idx_b = matches[i, 3]

        global_a = frame_offsets[frame_a] + idx_a
        global_b = frame_offsets[frame_b] + idx_b
        union_find_union(parent, rank, global_a, global_b)

        if frame_b > frame_a + 1:
            gap_bridged += 1

    return gap_bridged


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

    index_a = faiss.IndexFlatL2(3)  # type: ignore[possibly-missing-attribute]
    index_b = faiss.IndexFlatL2(3)  # type: ignore[possibly-missing-attribute]

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
        index = faiss.IndexFlatL2(3)  # type: ignore[attr-defined]
        index.add(means)  # type: ignore
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
    frame_offsets_arr = np.array(frame_offsets, dtype=np.int_)

    # Initialize optimized Union-Find
    parent, rank = union_find_init(total_gaussians)

    # Apply matches to union-find
    if use_sliding_format:
        assert sliding_matches is not None  # Type guard for mypy/ty
        # Convert to numpy array for Numba
        if len(sliding_matches) > 0:
            matches_arr = np.array(sliding_matches, dtype=np.int_)
            gap_bridged = apply_matches_to_union_find(parent, rank, matches_arr, frame_offsets_arr)
        else:
            gap_bridged = 0
        logger.info(f"Total matches: {len(sliding_matches)}, gap-bridged: {gap_bridged}")
    else:
        assert all_matches is not None  # Type guard for mypy/ty
        # Pairwise format: list of lists - convert to flat array
        flat_matches = []
        for frame_idx, matches in enumerate(all_matches):
            for idx_a, idx_b in matches:
                flat_matches.append((frame_idx, idx_a, frame_idx + 1, idx_b))
        if len(flat_matches) > 0:
            matches_arr = np.array(flat_matches, dtype=np.int_)
            apply_matches_to_union_find(parent, rank, matches_arr, frame_offsets_arr)

    # Finalize trajectory IDs using optimized function
    trajectory_ids, n_trajectories = union_find_components(parent)
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


def fit_trajectories_delta_compression(
    traj_data: TrajectoryData,
    compression_ratio_target: float = 51.0,
    use_int8: bool = False,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Fit temporal compression using delta encoding with Int16/Int8 quantization.

    Inspired by P-4DGS predictive coding. Instead of storing absolute motion vectors,
    stores compressed temporal deltas for ~51x compression ratio.

    Returns:
        (means, scales, rotations, colors, opacities, deltas, time_center, time_scale)
        deltas: compressed temporal deltas (Int16 or Int8)
    """
    n_traj = traj_data.n_trajectories
    n_obs = len(traj_data.trajectory_ids)

    traj_ids = traj_data.trajectory_ids
    times = traj_data.times_normalized
    positions = traj_data.positions
    scales = traj_data.scales
    rotations = traj_data.rotations
    colors = traj_data.colors
    opacities = traj_data.opacities

    # Compute time centers (same as FreeTimeGS)
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

    # For SPAG-4D: Compute temporal deltas instead of linear motion
    # Sort observations by trajectory and time
    sort_idx = np.lexsort((times, traj_ids))
    sorted_traj_ids = traj_ids[sort_idx]
    sorted_times = times[sort_idx]
    sorted_positions = positions[sort_idx]
    sorted_scales = scales[sort_idx]
    sorted_rotations = rotations[sort_idx]
    sorted_colors = colors[sort_idx]
    sorted_opacities = opacities[sort_idx]

    # Compute deltas for each trajectory
    deltas = np.zeros((n_traj, 3), dtype=np.float32)  # Will be compressed to Int16/Int8
    pos_center = np.zeros((n_traj, 3), dtype=np.float32)

    for traj_id in range(n_traj):
        mask = sorted_traj_ids == traj_id
        if np.sum(mask) == 0:
            continue

        traj_positions = sorted_positions[mask]
        traj_times = sorted_times[mask]

        # Use first position as reference
        pos_center[traj_id] = traj_positions[0]

        if len(traj_positions) > 1:
            # Compute average delta per unit time
            time_diffs = np.diff(traj_times)
            pos_diffs = np.diff(traj_positions, axis=0)

            # Avoid division by zero
            valid_mask = time_diffs > 1e-8
            if np.any(valid_mask):
                avg_delta = np.mean(pos_diffs[valid_mask] / time_diffs[valid_mask, None], axis=0)
                deltas[traj_id] = avg_delta
            else:
                deltas[traj_id] = 0.0
        else:
            deltas[traj_id] = 0.0

    # Compress deltas to Int16/Int8 based on target compression ratio
    # First, find the scale factor to fit deltas into Int16 range
    delta_magnitudes = np.linalg.norm(deltas, axis=1)
    max_delta = np.max(delta_magnitudes) if len(delta_magnitudes) > 0 else 1.0

    scale_factor = 1.0  # Default scale factor
    if max_delta > 0:
        # Scale to fit in Int16 range (-32768 to 32767)
        # Apply compression ratio target
        scale_factor = (32767.0 / max_delta) / np.sqrt(compression_ratio_target)

    deltas_compressed = deltas * scale_factor

    if use_int8:
        # For higher compression, use Int8 range (-127 to 127)
        deltas_compressed = np.clip(deltas_compressed, -127, 127).astype(np.int8)
    else:
        # Use Int16 for better precision
        deltas_compressed = np.clip(deltas_compressed, -32767, 32767).astype(np.int16)

    # Compute weighted averages for other properties (same as FreeTimeGS)
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
        deltas_compressed.astype(np.int16),  # Compressed deltas
        time_center.astype(np.float32),
        time_scale_log.astype(np.float32),
        scale_factor,  # Return compression scale for decompression
    )

    delta_mag = np.linalg.norm(results[5].astype(np.float32) / scale_factor, axis=1)
    logger.info(
        f"Delta compression stats: magnitude mean={delta_mag.mean():.6f}, max={delta_mag.max():.6f}"
    )
    logger.info(
        f"Delta compression: scale_factor={scale_factor:.2f}, target_ratio={compression_ratio_target:.1f}x"
    )
    logger.info(f"Time center: min={results[6].min():.3f}, max={results[6].max():.3f}")

    return results


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
