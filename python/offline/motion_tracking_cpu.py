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


def smart_sample_trajectories(
    positions: np.ndarray,
    velocities: np.ndarray,
    opacities: np.ndarray,
    target_count: int,
    voxel_size: float = 0.05,
    velocity_weight: float = 5.0,
    center_weight: float = 2.0,
) -> np.ndarray:
    """Sample trajectories preserving sparse background, moving objects, and foreground.

    Args:
        positions: (N, 3) trajectory center positions
        velocities: (N, 3) velocity vectors
        opacities: (N,) opacity values
        target_count: Number of trajectories to sample
        voxel_size: Size of voxels for density estimation
        velocity_weight: Weight for velocity magnitude boosting
        center_weight: Weight for center distance weighting

    Returns:
        indices of selected trajectories
    """
    n_traj = len(positions)
    if n_traj <= target_count:
        return np.arange(n_traj, dtype=np.int32)

    # 1. Voxel-based density weighting
    voxel_coords = (positions / voxel_size).astype(np.int32)
    voxel_hashes = (
        voxel_coords[:, 0] * 73856093
        ^ voxel_coords[:, 1] * 19349663
        ^ voxel_coords[:, 2] * 83492791
    )
    unique_voxels, voxel_counts = np.unique(voxel_hashes, return_counts=True)
    voxel_count_map = dict(zip(unique_voxels, voxel_counts))
    voxel_counts_per_traj = np.array([voxel_count_map.get(h, 1) for h in voxel_hashes])
    w_density = 1.0 / np.sqrt(voxel_counts_per_traj)

    # 2. Velocity magnitude boosting
    vel_mag = np.linalg.norm(velocities, axis=1)
    vel_mag_norm = vel_mag / (np.max(vel_mag) + 1e-8)
    w_velocity = 1.0 + (vel_mag_norm * velocity_weight)

    # 3. Center distance weighting
    centroid = positions.mean(axis=0)
    dist_from_center = np.linalg.norm(positions - centroid, axis=1)
    max_dist = np.max(dist_from_center) + 1e-8
    dist_norm = dist_from_center / max_dist
    w_center = 1.0 + ((1.0 - dist_norm) * center_weight)

    # Combined weight
    weights = w_density * w_velocity * w_center
    weights = weights / weights.sum()

    # Sample proportional to weights
    selected_indices = np.random.choice(n_traj, size=target_count, replace=False, p=weights)

    return selected_indices.astype(np.int32)


def _compute_trajectory_attributes(
    traj_data: TrajectoryData,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute trajectory attributes: center position, weighted scales/rotations/colors/opacities,
    velocity, time_center, time_scale_log.

    Returns 8 float32 arrays used by both fit_trajectories and
    fit_trajectories_delta_compression.
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

    # Duration from track support (observation span)
    # Option A: Use observation span (max_t - min_t for this trajectory)
    time_min = np.zeros(n_traj, dtype=np.float32)
    time_max = np.zeros(n_traj, dtype=np.float32)
    np.minimum.at(time_min, traj_ids, times)
    np.maximum.at(time_max, traj_ids, times)
    time_span = time_max - time_min

    # Duration = span * multiplier (for overlap), clamped
    # Use 1.5x multiplier for overlap between trajectories
    time_scale = np.clip(time_span * 1.5, 0.05, 0.5)

    # This prevents splats from disappearing abruptly when time scale is too small
    time_scale = np.maximum(time_scale, 0.05)
    time_scale_log = np.log(time_scale)

    dt = times - time_center[traj_ids]

    # Track-quality-weighted trajectory fitting
    # Factor 1: Opacity (high opacity = more reliable)
    track_weights = np.maximum(opacities, 0.1)

    # Factor 2: Trajectory length (longer tracks = more confident)
    obs_count_weight = np.sqrt(obs_count[traj_ids])
    track_weights *= obs_count_weight

    # Normalize weights
    track_weights = track_weights / (track_weights.sum() + 1e-8)

    pos_sum = np.zeros((n_traj, 3), dtype=np.float32)
    pos_dt_sum = np.zeros((n_traj, 3), dtype=np.float32)
    dt_sq_sum = np.zeros(n_traj, dtype=np.float32)

    for dim in range(3):
        np.add.at(pos_sum[:, dim], traj_ids, positions[:, dim] * track_weights)
        np.add.at(pos_dt_sum[:, dim], traj_ids, positions[:, dim] * dt * track_weights)
        np.add.at(dt_sq_sum, traj_ids, dt**2 * track_weights)

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


def fit_trajectories_delta_compression(
    traj_data: TrajectoryData,
    compression_ratio_target: float = 51.0,
    use_int8: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Fit temporal compression using delta encoding with Int16/Int8 quantization.

    Inspired by P-4DGS predictive coding. Instead of storing absolute motion vectors,
    stores compressed temporal deltas for ~51x compression ratio.

    Returns:
        (means, scales, rotations, colors, opacities, deltas, time_center, time_scale,
         compression_scale)
        deltas: compressed temporal deltas (Int16 or Int8)
        compression_scale: float, scale factor for dequantization:
            velocity = deltas.astype(float32) / compression_scale
    """
    pos_center, scales, rotations, colors, opacities, velocity, time_center, time_scale = (
        _compute_trajectory_attributes(traj_data)
    )

    dtype = np.int8 if use_int8 else np.int16
    int_max = np.iinfo(dtype).max
    int_min = np.iinfo(dtype).min

    max_abs_vel = float(np.max(np.abs(velocity))) + 1e-12

    if max_abs_vel < 1e-12:
        # Edge case: all-zero velocity
        deltas = np.zeros_like(velocity, dtype=dtype)
        compression_scale = 1.0
    else:
        # Safety margin: 0.99 to reduce outlier saturation
        compression_scale = (int_max * 0.99) / max_abs_vel
        deltas = np.clip(
            np.round(velocity * compression_scale), int_min, int_max
        ).astype(dtype)

    return (
        pos_center,
        scales,
        rotations,
        colors,
        opacities,
        deltas,
        time_center,
        time_scale,
        compression_scale,
    )


def fit_trajectories(
    traj_data: TrajectoryData,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit trajectories to obtain float32 motion vectors. This is the non-delta-compression path."""
    return _compute_trajectory_attributes(traj_data)


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
