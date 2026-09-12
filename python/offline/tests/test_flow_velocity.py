"""Tests for the flow-guided velocity prior (FreeTimeGS++ velocity distillation)."""

import numpy as np

from offline.motion_tracking_cpu import TrajectoryData, fit_trajectories


def _make_flow_data(n_traj: int = 100, n_frames: int = 5, static: np.ndarray | None = None):
    """Static trajectories (zero LS velocity) with a known per-frame 3D flow.

    `static` selects which trajectories have valid flow; the rest are all-NaN.
    """
    traj_ids = np.repeat(np.arange(n_traj), n_frames)
    frame_indices = np.tile(np.arange(n_frames), n_traj)
    n_obs = len(traj_ids)
    rng = np.random.RandomState(3)

    base = rng.randn(n_traj, 3).astype(np.float32)
    positions = base[traj_ids]  # constant over time -> LS velocity == 0

    flows = np.full((n_obs, 3), np.nan, dtype=np.float32)
    if static is None:
        static = np.ones(n_traj, dtype=bool)
    displacement = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    valid = static[traj_ids] & (frame_indices < n_frames - 1)
    flows[valid] = displacement

    td = TrajectoryData(
        trajectory_ids=traj_ids,
        frame_indices=frame_indices,
        times_normalized=(frame_indices / (n_frames - 1)).astype(np.float32),
        positions=positions,
        scales=np.ones((n_obs, 3), dtype=np.float32) * 0.1,
        rotations=np.tile([1, 0, 0, 0], (n_obs, 1)).astype(np.float32),
        colors=np.zeros((n_obs, 3), dtype=np.float32),
        opacities=np.ones(n_obs, dtype=np.float32) * 0.5,
        n_trajectories=n_traj,
        flows=flows,
    )
    return td


def test_flow_prior_sets_velocity_when_ls_is_zero():
    td = _make_flow_data()
    velocity = fit_trajectories(td, flow_prior_weight=1.0)[5]
    # dt per frame = 0.25, displacement 1.0 -> velocity 4.0
    np.testing.assert_allclose(velocity[:, 0], 4.0, rtol=1e-4)
    np.testing.assert_allclose(velocity[:, 1:], 0.0, atol=1e-5)


def test_flow_prior_is_opt_in():
    td = _make_flow_data()
    velocity = fit_trajectories(td, flow_prior_weight=0.0)[5]
    np.testing.assert_allclose(velocity, 0.0, atol=1e-6)


def test_flow_prior_blend_weight():
    td = _make_flow_data()
    velocity = fit_trajectories(td, flow_prior_weight=0.5)[5]
    np.testing.assert_allclose(velocity[:, 0], 2.0, rtol=1e-4)


def test_trajectories_without_flow_keep_ls_velocity():
    static = np.zeros(100, dtype=bool)
    static[:50] = True
    td = _make_flow_data(static=static)
    velocity = fit_trajectories(td, flow_prior_weight=1.0)[5]

    np.testing.assert_allclose(velocity[:50, 0], 4.0, rtol=1e-4)
    # The other half has no flow support and stays at the LS velocity (0).
    np.testing.assert_allclose(velocity[50:], 0.0, atol=1e-6)


def test_no_flows_is_safe():
    td = _make_flow_data()
    td.flows = None
    velocity = fit_trajectories(td, flow_prior_weight=1.0)[5]
    np.testing.assert_allclose(velocity, 0.0, atol=1e-6)
