"""Tests for the FreeTimeGS++ gated-marginalization approximation."""

import numpy as np

from offline.motion_tracking_cpu import (
    PERSISTENT_SIGMA,
    TrajectoryData,
    fit_trajectories,
    fit_trajectories_delta_compression,
    persistent_gate_from_time_scale,
)


def _mixed_coverage_data(n_full: int = 50, n_short: int = 50, n_frames: int = 10):
    """Trajectories either observed in every frame or in only two frames."""
    traj_ids = np.concatenate(
        [
            np.repeat(np.arange(n_full), n_frames),
            np.repeat(np.arange(n_full, n_full + n_short), 2),
        ]
    )
    frame_indices = np.concatenate(
        [
            np.tile(np.arange(n_frames), n_full),
            np.tile(np.array([4, 5]), n_short),
        ]
    )
    n_obs = len(traj_ids)
    rng = np.random.RandomState(7)
    return TrajectoryData(
        trajectory_ids=traj_ids,
        frame_indices=frame_indices,
        times_normalized=(frame_indices / (n_frames - 1)).astype(np.float32),
        positions=rng.randn(n_obs, 3).astype(np.float32),
        scales=np.ones((n_obs, 3), dtype=np.float32) * 0.1,
        rotations=np.tile([1, 0, 0, 0], (n_obs, 1)).astype(np.float32),
        colors=np.zeros((n_obs, 3), dtype=np.float32),
        opacities=np.ones(n_obs, dtype=np.float32) * 0.5,
        n_trajectories=n_full + n_short,
    ), n_full


def test_gating_lifts_full_coverage_sigma():
    td, n_full = _mixed_coverage_data()
    sigma = np.exp(fit_trajectories(td, temporal_gating=True)[7])

    np.testing.assert_allclose(sigma[:n_full], PERSISTENT_SIGMA, rtol=1e-5)
    # Short tracks keep their span-based duration (well below persistent).
    assert np.all(sigma[n_full:] < 0.5)


def test_gating_is_opt_in():
    td, n_full = _mixed_coverage_data()
    sigma = np.exp(fit_trajectories(td, temporal_gating=False)[7])
    # Historical behaviour: clipped to 0.5 even for full coverage.
    np.testing.assert_allclose(sigma[:n_full], 0.5, rtol=1e-5)


def test_gating_threads_through_delta_path():
    td, n_full = _mixed_coverage_data()
    result = fit_trajectories_delta_compression(td, temporal_gating=True)
    assert len(result) == 9
    sigma = np.exp(result[7])
    np.testing.assert_allclose(sigma[:n_full], PERSISTENT_SIGMA, rtol=1e-5)


def test_persistent_gate_from_time_scale():
    log_sigma = np.log(np.array([0.1, 0.5, PERSISTENT_SIGMA], dtype=np.float32))
    gate = persistent_gate_from_time_scale(log_sigma)
    np.testing.assert_array_equal(gate, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_write_freetimegs_ply_with_gate_roundtrip(tmp_path):
    from plyfile import PlyData

    from offline.ply_io import write_freetimegs_ply

    n = 5
    gate = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    out = tmp_path / "gated.ply"
    write_freetimegs_ply(
        out,
        means=np.zeros((n, 3), dtype=np.float32),
        scales=np.zeros((n, 3), dtype=np.float32),
        rotations=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        colors=np.zeros((n, 3), dtype=np.float32),
        opacities=np.zeros(n, dtype=np.float32),
        motion=np.zeros((n, 3), dtype=np.float32),
        time_center=np.full(n, 0.5, dtype=np.float32),
        time_scale=np.zeros(n, dtype=np.float32),
        gate=gate,
    )
    vertex = PlyData.read(str(out))["vertex"]
    assert "t_gate" in vertex.data.dtype.names
    np.testing.assert_array_equal(vertex["t_gate"], gate)
