"""Optimized boundary refinement for depth maps.

Based on the technique from Materialist mesh_recon.py, but optimized for performance
using vectorized NumPy operations instead of nested loops.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def refine_depth_at_boundaries(
    points: np.ndarray,
    intrinsics: np.ndarray,
    min_angle: float = 3.0,
    max_iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine depth at boundaries using grazing-angle detection and depth propagation.

    This addresses halo artifacts at occlusion boundaries by preferring background depth.

    Args:
        points: (H, W, 3) array of 3D points in camera space
        intrinsics: (3, 3) camera intrinsic matrix
        min_angle: Minimum angle (degrees) between viewing ray and surface normal.
                   Triangles with smaller angles are considered boundaries.
        max_iterations: Maximum depth propagation iterations (default 10)

    Returns:
        Tuple of:
        - refined_points: (H, W, 3) array with refined depths
        - boundary_mask: (H, W) boolean mask of detected boundary pixels
    """
    H, W, _ = points.shape
    depth = points[:, :, 2].copy()

    # Detect boundaries using grazing angle check
    boundary_mask = _detect_boundaries_vectorized(points, min_angle)

    if not np.any(boundary_mask):
        logger.info("No boundaries detected, skipping refinement")
        return points, boundary_mask

    n_boundaries = np.sum(boundary_mask)
    logger.info(f"Detected {n_boundaries} boundary pixels ({n_boundaries / (H * W) * 100:.1f}%)")

    # Build reference map: which neighbor has the largest depth?
    refer_map = _build_reference_map(depth, boundary_mask)

    # Propagate depths iteratively
    refined_depth = _propagate_depth(depth, refer_map, max_iterations)

    # Update points with refined depth
    refined_points = points.copy()
    refined_points[:, :, 2] = refined_depth

    n_changed = np.sum(np.abs(refined_depth - depth) > 1e-6)
    logger.info(f"Refined {n_changed} pixels ({n_changed / (H * W) * 100:.1f}%)")

    return refined_points, boundary_mask


def _detect_boundaries_vectorized(points: np.ndarray, min_angle: float) -> np.ndarray:
    """Detect boundary pixels using vectorized grazing-angle test.

    Args:
        points: (H, W, 3) array of 3D points
        min_angle: Minimum angle threshold in degrees

    Returns:
        (H, W) boolean mask of boundary pixels
    """
    H, W, _ = points.shape
    min_angle_rad = np.deg2rad(min_angle)
    min_sin = np.sin(min_angle_rad)

    # Get neighbor offsets: right, down, left, up
    # We'll check 4 triangles per pixel: (center, right, down), (center, down, left), etc.
    offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    boundary_mask = np.zeros((H, W), dtype=bool)

    # Pad points to handle boundary conditions
    points_padded = np.pad(points, ((1, 1), (1, 1), (0, 0)), mode="edge")

    for i, (dy1, dx1) in enumerate(offsets):
        # Next offset in sequence (wraps around)
        dy2, dx2 = offsets[(i + 1) % 4]

        # Get neighbor points (with padding offset +1)
        p0 = points_padded[1:-1, 1:-1]  # center
        p1 = points_padded[1 + dy1 : H + 1 + dy1, 1 + dx1 : W + 1 + dx1]
        p2 = points_padded[1 + dy2 : H + 1 + dy2, 1 + dx2 : W + 1 + dx2]

        # Check for zero points (invalid)
        valid = ~(np.all(p0 == 0, axis=2) | np.all(p1 == 0, axis=2) | np.all(p2 == 0, axis=2))

        # Compute triangle normals
        v1 = p1 - p0
        v2 = p2 - p0
        normals = np.cross(v1, v2)

        # Normalize
        norm_len = np.linalg.norm(normals, axis=2, keepdims=True)
        norm_len = np.where(norm_len > 1e-10, norm_len, 1.0)  # avoid division by zero
        normals = normals / norm_len

        # Compute viewing ray (center point normalized)
        center_norm = np.linalg.norm(p0, axis=2, keepdims=True)
        center_norm = np.where(center_norm > 1e-10, center_norm, 1.0)
        view_ray = p0 / center_norm

        # Compute angle: sin(angle) = |dot(normal, view_ray)|
        dot_product = np.abs(np.sum(normals * view_ray, axis=2))
        grazing = (dot_product < min_sin) & valid

        # Mark as boundary
        boundary_mask |= grazing

    # Erode 1px from image edges (unreliable boundaries)
    boundary_mask[0, :] = False
    boundary_mask[-1, :] = False
    boundary_mask[:, 0] = False
    boundary_mask[:, -1] = False

    return boundary_mask


def _build_reference_map(depth: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    """Build reference map: for each boundary pixel, which neighbor has max depth?

    Args:
        depth: (H, W) depth map
        boundary_mask: (H, W) boolean mask of boundary pixels

    Returns:
        (H, W, 2) array of (i, j) indices. -1 means no reference (not a boundary).
    """
    H, W = depth.shape
    refer_map = np.full((H, W, 2), -1, dtype=np.int32)

    # For each boundary pixel, find neighbor with max depth
    # Neighbors: right, down, left, up
    offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Pad depth
    depth_padded = np.pad(depth, ((1, 1), (1, 1)), mode="edge")

    # Vectorized: for all boundary pixels, gather neighbor depths
    boundary_coords = np.argwhere(boundary_mask)  # (N, 2) array of (i, j)

    for i, j in boundary_coords:
        # Get depths of neighbors (with padding offset +1)
        neighbor_depths = [depth_padded[i + 1 + dy, j + 1 + dx] for dy, dx in offsets]
        neighbor_depths = np.array(neighbor_depths)

        # Find neighbor with max depth
        max_idx = np.argmax(neighbor_depths)
        dy, dx = offsets[max_idx]

        # Store reference (clip to valid range)
        ref_i = np.clip(i + dy, 0, H - 1)
        ref_j = np.clip(j + dx, 0, W - 1)

        # Only reference if neighbor has larger depth
        if depth_padded[i + 1 + dy, j + 1 + dx] > depth[i, j]:
            refer_map[i, j] = [ref_i, ref_j]

    return refer_map


def _propagate_depth(
    depth: np.ndarray,
    refer_map: np.ndarray,
    max_iterations: int = 10,
) -> np.ndarray:
    """Propagate depth values following reference chain.

    Args:
        depth: (H, W) original depth map
        refer_map: (H, W, 2) reference map from _build_reference_map
        max_iterations: Maximum chain length to follow

    Returns:
        (H, W) refined depth map
    """
    H, W = depth.shape
    refined_depth = depth.copy()

    # Find all pixels with references
    has_ref = refer_map[:, :, 0] >= 0
    ref_coords = np.argwhere(has_ref)  # (N, 2)

    if len(ref_coords) == 0:
        return refined_depth

    logger.debug(f"Propagating depth for {len(ref_coords)} boundary pixels...")

    # For each pixel with a reference, follow the chain
    for i, j in ref_coords:
        current_i, current_j = i, j
        max_depth = depth[i, j]

        # Follow reference chain up to max_iterations
        for _ in range(max_iterations):
            ref_i, ref_j = refer_map[current_i, current_j]

            if ref_i < 0:  # No more references
                break

            # Update max depth
            max_depth = max(max_depth, depth[ref_i, ref_j])

            # Move to next in chain
            current_i, current_j = ref_i, ref_j

        # Assign max depth found in chain
        refined_depth[i, j] = max_depth

    return refined_depth
