"""Tests for temporal color correction (mined from FreeTimeGS++)."""

import numpy as np
import pytest

from offline.color_correction import (
    DC_MAX,
    DC_MIN,
    apply_temporal_color_correction,
    correct_trajectory_colors,
)
from offline.motion_tracking_cpu import TrajectoryData


def _make_colored_traj_data(
    n_traj: int = 300,
    n_frames: int = 10,
    gains: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[TrajectoryData, np.ndarray]:
    """Build TrajectoryData with per-frame affine color distortion.

    Returns (traj_data, base_colors).
    """
    rng = np.random.RandomState(seed)
    n_obs = n_traj * n_frames
    traj_ids = np.repeat(np.arange(n_traj), n_frames)
    frame_indices = np.tile(np.arange(n_frames), n_traj)

    base = rng.uniform(-1.0, 1.0, size=(n_traj, 3)).astype(np.float32)
    if gains is None:
        gains = rng.uniform(0.7, 1.3, size=n_frames)
    if offsets is None:
        offsets = rng.uniform(-0.15, 0.15, size=(n_frames, 3))

    colors = base[traj_ids] * gains[frame_indices, None] + offsets[frame_indices]
    colors = colors.astype(np.float32)

    td = TrajectoryData(
        trajectory_ids=traj_ids,
        frame_indices=frame_indices,
        times_normalized=(frame_indices / max(n_frames - 1, 1)).astype(np.float32),
        positions=rng.randn(n_obs, 3).astype(np.float32),
        scales=np.ones((n_obs, 3), dtype=np.float32) * 0.1,
        rotations=np.tile([1, 0, 0, 0], (n_obs, 1)).astype(np.float32),
        colors=colors,
        opacities=np.ones(n_obs, dtype=np.float32) * 0.5,
        n_trajectories=n_traj,
    )
    return td, base


def _per_frame_mean_spread(colors: np.ndarray, frame_indices: np.ndarray) -> float:
    """Mean per-channel std of the frame means (temporal flicker proxy)."""
    means = [colors[frame_indices == f].mean(axis=0) for f in np.unique(frame_indices)]
    return float(np.std(np.stack(means), axis=0).mean())


def test_reduces_per_frame_color_flicker():
    td, _ = _make_colored_traj_data()
    before = _per_frame_mean_spread(td.colors, td.frame_indices)

    corrected, stats = apply_temporal_color_correction(
        td.colors, td.trajectory_ids, td.frame_indices, td.n_trajectories
    )
    after = _per_frame_mean_spread(corrected, td.frame_indices)

    assert stats.num_fitted_frames > 0
    assert after < before * 0.25


def test_no_op_when_colors_already_consistent():
    rng = np.random.RandomState(1)
    n_traj, n_frames = 200, 8
    traj_ids = np.repeat(np.arange(n_traj), n_frames)
    frame_indices = np.tile(np.arange(n_frames), n_traj)
    base = rng.uniform(-1.0, 1.0, size=(n_traj, 3)).astype(np.float32)
    colors = base[traj_ids].copy()

    corrected, _ = apply_temporal_color_correction(
        colors, traj_ids, frame_indices, n_traj
    )
    np.testing.assert_allclose(corrected, colors, atol=1e-3)


def test_identity_when_too_few_pairs():
    td, _ = _make_colored_traj_data(n_traj=3, n_frames=4)
    corrected, stats = apply_temporal_color_correction(
        td.colors, td.trajectory_ids, td.frame_indices, td.n_trajectories, min_pairs=50
    )
    assert stats.num_fitted_frames == 0
    np.testing.assert_array_equal(corrected, td.colors)


def test_blend_zero_is_identity():
    td, _ = _make_colored_traj_data(n_traj=100, n_frames=5)
    corrected, _ = apply_temporal_color_correction(
        td.colors, td.trajectory_ids, td.frame_indices, td.n_trajectories, blend=0.0
    )
    np.testing.assert_array_equal(corrected, td.colors)


def test_output_clipped_to_valid_dc_range():
    # Extreme per-frame distortions push corrected values out of range.
    td, _ = _make_colored_traj_data(
        n_traj=100,
        n_frames=6,
        gains=np.array([5.0, 0.1, 5.0, 0.1, 5.0, 0.1], dtype=np.float32),
        offsets=np.array([[2.0, -2.0, 2.0]] * 6, dtype=np.float32),
    )
    corrected, _ = apply_temporal_color_correction(
        td.colors, td.trajectory_ids, td.frame_indices, td.n_trajectories, clip_dc=True
    )
    assert corrected.min() >= DC_MIN - 1e-6
    assert corrected.max() <= DC_MAX + 1e-6


def test_handles_single_observation_trajectories():
    rng = np.random.RandomState(2)
    n_obs = 60
    td = TrajectoryData(
        trajectory_ids=np.arange(n_obs),
        frame_indices=np.arange(n_obs) % 5,
        times_normalized=np.zeros(n_obs, dtype=np.float32),
        positions=rng.randn(n_obs, 3).astype(np.float32),
        scales=np.ones((n_obs, 3), dtype=np.float32) * 0.1,
        rotations=np.tile([1, 0, 0, 0], (n_obs, 1)).astype(np.float32),
        colors=rng.uniform(-1, 1, (n_obs, 3)).astype(np.float32),
        opacities=np.ones(n_obs, dtype=np.float32) * 0.5,
        n_trajectories=n_obs,
    )
    corrected, stats = apply_temporal_color_correction(
        td.colors, td.trajectory_ids, td.frame_indices, td.n_trajectories
    )
    assert np.all(np.isfinite(corrected))
    assert stats.mean_gain == pytest.approx(1.0, abs=1e-2)


def test_correct_trajectory_colors_preserves_other_fields():
    td, _ = _make_colored_traj_data(n_traj=128, n_frames=6)
    out = correct_trajectory_colors(td)

    assert out.n_trajectories == td.n_trajectories
    np.testing.assert_array_equal(out.positions, td.positions)
    np.testing.assert_array_equal(out.trajectory_ids, td.trajectory_ids)
    assert out.colors.shape == td.colors.shape
    # Corrected colors should actually change for distorted input.
    assert not np.allclose(out.colors, td.colors)


def test_empty_input_is_safe():
    colors = np.zeros((0, 3), dtype=np.float32)
    corrected, stats = apply_temporal_color_correction(
        colors, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0
    )
    assert corrected.shape == (0, 3)
    assert stats.num_fitted_frames == 0
