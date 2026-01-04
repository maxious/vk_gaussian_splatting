"""GPU-accelerated Gaussian motion tracking using cuTile."""

import logging
from math import ceil
from typing import List, Tuple, Optional

import numpy as np
import torch
import cuda.tile as ct

from .motion_tracking_cpu import TrajectoryData, fit_trajectories, apply_matches_to_union_find
from .union_find import union_find_components, union_find_init

logger = logging.getLogger(__name__)

# Tile size for cuTile kernel (power of 2, tuned for RTX 5080)
DEFAULT_TILE_SIZE = 128
# Bucket size to reduce kernel recompilation (pad M to multiple of this)
COMPILATION_BUCKET_SIZE = 4096
DIM_PADDED = 4  # Pad 3D coords to 4D for power-of-2 requirement

# Epsilon for tie-breaking (relative to typical squared distances)
TIE_EPSILON = 1e-6


@ct.kernel
def knn_kernel(
    A, B, OutIdx, OutDist, Tm: ct.Constant[int], Tn: ct.Constant[int], M: ct.Constant[int]
):
    """
    Find Nearest Neighbor in B for each point in A with deterministic tie-breaking.

    When distances are equal (within epsilon), prefers lower index for reproducibility.

    A: (N_padded, 4) - 3D coords + optional 4th attribute (opacity/luminance)
    B: (M_padded, 4)
    OutIdx: (N_padded, 1)
    OutDist: (N_padded, 1)
    """
    pid = ct.bid(0)

    # Load A tile (Tm, 4) - all dimensions now power of 2
    a = ct.load(A, index=(pid, 0), shape=(Tm, 4))
    a_col = a.reshape((Tm, 1, 4))

    # Initialize accumulators
    best_dist = ct.full((Tm, 1), 1e30, dtype=ct.float32)  # Large value as "infinity"
    best_idx = ct.full((Tm, 1), -1, dtype=ct.int32)

    num_b_tiles = ct.cdiv(M, Tn)

    # Loop over all tiles of B
    for j in range(num_b_tiles):
        # Load B tile (Tn, 4)
        b = ct.load(B, index=(j, 0), shape=(Tn, 4))
        b_row = b.reshape((1, Tn, 4))

        # Compute squared Euclidean distance: sum((a - b)^2, axis=2)
        # 4th dimension contributes to distance if non-zero (opacity/luminance matching)
        diff = a_col - b_row  # (Tm, Tn, 4)
        dist_sq = ct.sum(diff * diff, axis=2)  # (Tm, Tn)

        # Find min distance and index in this B-tile
        curr_min = ct.min(dist_sq, axis=1, keepdims=True)  # (Tm, 1)
        curr_idx = ct.argmin(dist_sq, axis=1, keepdims=True)  # (Tm, 1)

        # Convert local index to global index
        curr_idx = curr_idx + (j * Tn)

        # Deterministic tie-breaking: update if strictly closer,
        # OR if equal distance but lower index (for reproducibility)
        is_closer = curr_min < best_dist
        is_tie_lower_idx = (curr_min < best_dist + 1e-6) & (curr_idx < best_idx)
        update_mask = is_closer | is_tie_lower_idx

        best_dist = ct.where(update_mask, curr_min, best_dist)
        best_idx = ct.where(update_mask, curr_idx, best_idx)

    # Store results
    ct.store(OutIdx, index=(pid, 0), tile=best_idx)
    ct.store(OutDist, index=(pid, 0), tile=best_dist)


def _prepare_tensors(
    means_a: torch.Tensor,
    means_b: torch.Tensor,
    tile_size: int,
    attr_a: Optional[torch.Tensor] = None,
    attr_b: Optional[torch.Tensor] = None,
    attr_weight: float = 0.0,
):
    """
    Prepare tensors with proper padding for cuTile.

    Args:
        means_a: (N, 3) positions for set A
        means_b: (M, 3) positions for set B
        tile_size: Tile size for cuTile (must be power of 2)
        attr_a: Optional (N,) attribute for 4th dimension (e.g., opacity)
        attr_b: Optional (M,) attribute for 4th dimension
        attr_weight: Weight for 4th dimension relative to spatial distance.
                     The attribute is scaled by: attr * attr_weight * spatial_scale
                     where spatial_scale is the std of positions.

    Returns:
        A_padded, B_padded, N, M
    """
    N, D = means_a.shape
    M, _ = means_b.shape
    assert D == 3

    # Compute spatial scale for attribute weighting
    if attr_a is not None and attr_b is not None and attr_weight > 0:
        # Use position std as spatial scale reference
        all_pos = torch.cat([means_a, means_b], dim=0)
        spatial_scale = all_pos.std()

        # Scale attribute to be comparable to spatial distances
        # attr in [0,1] * weight * spatial_scale -> comparable to position diffs
        scaled_attr_a = attr_a * attr_weight * spatial_scale
        scaled_attr_b = attr_b * attr_weight * spatial_scale

        # Concatenate position + scaled attribute
        A_4d = torch.cat([means_a, scaled_attr_a.unsqueeze(1)], dim=1)  # (N, 4)
        B_4d = torch.cat([means_b, scaled_attr_b.unsqueeze(1)], dim=1)  # (M, 4)
    else:
        # Pad 3D -> 4D with zeros (cuTile requires power-of-2 dimensions)
        A_4d = torch.nn.functional.pad(means_a, (0, 1), value=0)  # (N, 4)
        B_4d = torch.nn.functional.pad(means_b, (0, 1), value=0)  # (M, 4)

    # Pad row count to multiple of tile_size
    pad_a = (tile_size - (N % tile_size)) % tile_size

    # Optimization: Pad B to bucket size to stabilize M for caching
    # This prevents recompiling the kernel for every frame when point count changes slightly
    target_M = (
        (M + COMPILATION_BUCKET_SIZE - 1) // COMPILATION_BUCKET_SIZE * COMPILATION_BUCKET_SIZE
    )
    pad_b = target_M - M

    if pad_a > 0:
        A_padded = torch.nn.functional.pad(A_4d, (0, 0, 0, pad_a), value=0)
    else:
        A_padded = A_4d

    if pad_b > 0:
        # Pad with large value so padded points are never selected
        B_padded = torch.nn.functional.pad(B_4d, (0, 0, 0, pad_b), value=1e10)
    else:
        B_padded = B_4d

    return A_padded, B_padded, N, M


def knn_search_cutile(
    means_a: torch.Tensor,
    means_b: torch.Tensor,
    tile_size: int = DEFAULT_TILE_SIZE,
    attr_a: Optional[torch.Tensor] = None,
    attr_b: Optional[torch.Tensor] = None,
    attr_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find nearest neighbors in B for each point in A using cuTile.

    Args:
        means_a: (N, 3) positions for set A
        means_b: (M, 3) positions for set B
        tile_size: Tile size for cuTile kernel
        attr_a: Optional (N,) attribute for 4th dimension (e.g., opacity)
        attr_b: Optional (M,) attribute for 4th dimension
        attr_weight: Weight for attribute dimension (0 = disabled, 0.1-0.5 typical)

    Returns:
        (distances_squared, indices)
    """
    A_padded, B_padded, N, M = _prepare_tensors(
        means_a, means_b, tile_size, attr_a, attr_b, attr_weight
    )

    N_pad = A_padded.shape[0]
    M_pad = B_padded.shape[0]

    OutIdx = torch.empty((N_pad, 1), dtype=torch.int32, device=means_a.device)
    OutDist = torch.empty((N_pad, 1), dtype=torch.float32, device=means_a.device)

    grid = (ceil(N_pad / tile_size), 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        knn_kernel,
        (A_padded, B_padded, OutIdx, OutDist, tile_size, tile_size, M_pad),
    )

    # Slice to original size
    indices = OutIdx[:N, 0]
    dists = OutDist[:N, 0]

    return dists, indices


def match_gaussians_sliding_window_cuda(
    all_means: List[np.ndarray],
    max_distance: float,
    window_size: int = 3,
    all_opacities: Optional[List[np.ndarray]] = None,
    opacity_weight: float = 0.0,
) -> List[Tuple[int, int, int, int]]:
    """
    Match Gaussians using GPU-accelerated KNN.

    Args:
        all_means: List of (N_i, 3) position arrays per frame
        max_distance: Maximum Euclidean distance for valid match
        window_size: How many future frames to search (gap-bridging)
        all_opacities: Optional list of (N_i,) opacity arrays per frame.
                       When provided with opacity_weight > 0, uses opacity
                       as 4th dimension to improve visual consistency.
        opacity_weight: Weight for opacity in distance calculation.
                        0 = disabled (default), 0.1-0.5 = typical values.
                        Higher values favor matching points with similar opacity.

    Returns:
        List of (frame_a, idx_a, frame_b, idx_b) tuples
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Move all data to GPU
    gpu_means = [torch.from_numpy(m).cuda().float() for m in all_means]

    # Prepare opacity tensors if provided
    gpu_opacities: Optional[List[torch.Tensor]] = None
    if all_opacities is not None and opacity_weight > 0:
        gpu_opacities = [torch.from_numpy(o).cuda().float() for o in all_opacities]
        logger.info(f"Using opacity matching with weight={opacity_weight}")

    n_frames = len(all_means)
    all_matches = []
    max_dist_sq = max_distance * max_distance

    for frame_a in range(n_frames - 1):
        means_a = gpu_means[frame_a]
        n_a = len(means_a)
        opacity_a = gpu_opacities[frame_a] if gpu_opacities else None

        # Track matches for frame_a
        matched_a = torch.zeros(n_a, dtype=torch.bool, device="cuda")
        best_match_frame = torch.full((n_a,), -1, dtype=torch.int32, device="cuda")
        best_match_idx = torch.full((n_a,), -1, dtype=torch.int32, device="cuda")

        for offset in range(1, min(window_size + 1, n_frames - frame_a)):
            frame_b = frame_a + offset
            means_b = gpu_means[frame_b]
            opacity_b = gpu_opacities[frame_b] if gpu_opacities else None

            # Forward search: A -> B
            dist_ab, idx_ab = knn_search_cutile(
                means_a, means_b, attr_a=opacity_a, attr_b=opacity_b, attr_weight=opacity_weight
            )

            # Backward search: B -> A (for mutual check)
            dist_ba, idx_ba = knn_search_cutile(
                means_b, means_a, attr_a=opacity_b, attr_b=opacity_a, attr_weight=opacity_weight
            )

            # Check mutual match: idx_ba[idx_ab[i]] == i
            idx_ba_mapped = torch.gather(idx_ba, 0, idx_ab.long())

            is_mutual = idx_ba_mapped == torch.arange(n_a, device="cuda")
            is_close = dist_ab < max_dist_sq
            is_new = ~matched_a

            valid_mask = is_mutual & is_close & is_new

            if valid_mask.any():
                matched_indices = torch.nonzero(valid_mask).squeeze(1)

                matched_a[matched_indices] = True
                best_match_frame[matched_indices] = frame_b
                best_match_idx[matched_indices] = idx_ab[matched_indices].int()

        # Collect matches
        valid_indices = torch.nonzero(best_match_frame >= 0).squeeze(1)
        if valid_indices.numel() > 0:
            v_idx = valid_indices.cpu().numpy()
            v_frame = best_match_frame[valid_indices].cpu().numpy()
            v_match_idx = best_match_idx[valid_indices].cpu().numpy()

            for i in range(len(v_idx)):
                all_matches.append((frame_a, int(v_idx[i]), int(v_frame[i]), int(v_match_idx[i])))

    return all_matches


def compute_motion_vectors_cuda(
    frames: list,
    fps: float,
    max_match_distance: Optional[float] = None,
    match_distance_ratio: float = 0.02,
    window_size: int = 3,
    opacity_weight: float = 0.1,
):
    """
    Compute motion vectors using CUDA for matching.

    Args:
        frames: List of frame data with means, scales, rotations, colors, opacities
        fps: Frames per second
        max_match_distance: Maximum distance for matching (auto-computed if None)
        match_distance_ratio: Ratio of scene diagonal for auto max_distance
        window_size: Number of frames to search forward
        opacity_weight: Weight for opacity in matching (0=disabled, 0.1-0.5 typical)
    """
    if len(frames) < 2:
        # Fallback for single frame
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

    # Compute scene scale on CPU (fast enough)
    all_means_np = [f.means for f in frames]
    all_opacities_np = [f.opacities for f in frames]

    if max_match_distance is None:
        stacked = np.vstack(all_means_np)
        bbox_min = stacked.min(axis=0)
        bbox_max = stacked.max(axis=0)
        diag = np.linalg.norm(bbox_max - bbox_min)
        max_match_distance = float(diag * match_distance_ratio)
        logger.info(f"Scene scale: {diag:.2f}, match dist: {max_match_distance:.2f}")

    # 1. Match on GPU
    matches = match_gaussians_sliding_window_cuda(
        all_means_np,
        max_distance=max_match_distance,
        window_size=window_size,
        all_opacities=all_opacities_np,
        opacity_weight=opacity_weight,
    )
    logger.info(f"Found {len(matches)} matches using cuTile")

    # 2. Build Union-Find (CPU)
    total_gaussians = sum(len(m) for m in all_means_np)
    frame_offsets = [0]
    for m in all_means_np:
        frame_offsets.append(frame_offsets[-1] + len(m))
    frame_offsets_arr = np.array(frame_offsets, dtype=np.int_)

    parent, rank = union_find_init(total_gaussians)
    matches_arr = np.array(matches, dtype=np.int_)
    if len(matches) > 0:
        apply_matches_to_union_find(parent, rank, matches_arr, frame_offsets_arr)

    trajectory_ids, n_trajectories = union_find_components(parent)

    # 3. Fit Trajectories (CPU/NumPy)
    frame_indices = np.zeros(total_gaussians, dtype=np.int32)
    times_normalized = np.zeros(total_gaussians, dtype=np.float32)
    positions = np.zeros((total_gaussians, 3), dtype=np.float32)
    scales = np.zeros((total_gaussians, 3), dtype=np.float32)
    rotations = np.zeros((total_gaussians, 4), dtype=np.float32)
    colors = np.zeros((total_gaussians, 3), dtype=np.float32)
    opacities = np.zeros(total_gaussians, dtype=np.float32)

    t_start = frames[0].timestamp_ms
    t_end = frames[-1].timestamp_ms
    t_range = max(t_end - t_start, 1e-6)

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

    traj_data = TrajectoryData(
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

    return fit_trajectories(traj_data)
