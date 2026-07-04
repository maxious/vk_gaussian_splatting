"""Tests for FreeTimeGS motion tracking and delta compression."""
import numpy as np

from offline.motion_tracking_cpu import (
    TrajectoryData,
    fit_trajectories,
    fit_trajectories_delta_compression,
)


def _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=True):
    """Create deterministic synthetic TrajectoryData for testing.

    Args:
        n_traj: Number of trajectories
        seed: Random seed for reproducibility
        moving: If True, positions vary by frame (producing velocity).
               If False, all positions per trajectory are identical (static).
    """
    rng = np.random.RandomState(seed)
    obs_per_traj = 3
    n_obs = n_traj * obs_per_traj
    traj_ids = np.repeat(np.arange(n_traj), obs_per_traj)
    frame_indices = np.tile(np.arange(obs_per_traj), n_traj)
    times_normalized = np.linspace(0, 1, n_obs, dtype=np.float32)

    if moving:
        # Positions that actually move (different per observation)
        positions = rng.randn(n_obs, 3).astype(np.float32)
    else:
        # Static: all same position per trajectory
        base_pos = (
            rng.randn(n_traj, 3).astype(np.float32)
            if n_traj > 0
            else np.zeros((1, 3), dtype=np.float32)
        )
        positions = np.repeat(base_pos, obs_per_traj, axis=0)[:n_obs]

    return TrajectoryData(
        trajectory_ids=traj_ids,
        frame_indices=frame_indices,
        times_normalized=times_normalized,
        positions=positions,
        scales=np.ones((n_obs, 3), dtype=np.float32) * 0.1,
        rotations=np.tile([1, 0, 0, 0], (n_obs, 1)).astype(np.float32),
        colors=np.zeros((n_obs, 3), dtype=np.float32),
        opacities=np.ones(n_obs, dtype=np.float32) * 0.5,
        n_trajectories=n_traj,
    )


# 1. Import test
def test_motion_tracking_cpu_imports():
    """Verify fit_trajectories is importable from CPU module."""
    assert fit_trajectories is not None
    assert fit_trajectories_delta_compression is not None
    assert TrajectoryData is not None


# 2. Return tuple shape test
def test_fit_trajectories_returns_8_tuple():
    """fit_trajectories returns 8 values, correct shapes."""
    td = _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=True)
    result = fit_trajectories(td)

    assert len(result) == 8

    # pos_center: (n_traj, 3)
    assert result[0].shape == (3, 3)
    # scales: (n_traj, 3)
    assert result[1].shape == (3, 3)
    # rotations: (n_traj, 4)
    assert result[2].shape == (3, 4)
    # colors: (n_traj, 3)
    assert result[3].shape == (3, 3)
    # opacities: (n_traj,)
    assert result[4].shape == (3,)
    # velocity: (n_traj, 3)
    assert result[5].shape == (3, 3)
    # time_center: (n_traj,)
    assert result[6].shape == (3,)
    # time_scale_log: (n_traj,)
    assert result[7].shape == (3,)


# 3. Delta returns 9-tuple test
def test_fit_trajectories_delta_compression_returns_9_tuple():
    """Delta compression returns 9 values, 6th is int16, 9th is float."""
    td = _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=True)
    result = fit_trajectories_delta_compression(td)

    assert len(result) == 9
    # 6th value (index 5) is int16 deltas
    assert result[5].dtype == np.int16
    # 9th value (index 8) is float
    assert isinstance(result[8], float)


# 4. Default dtype is int16
def test_deltas_are_int16_by_default():
    """result[5].dtype == np.int16"""
    td = _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=True)
    result = fit_trajectories_delta_compression(td)
    assert result[5].dtype == np.int16


# 5. int8 mode
def test_deltas_are_int8_when_use_int8_true():
    """result[5].dtype == np.int8"""
    td = _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=True)
    result = fit_trajectories_delta_compression(td, use_int8=True)
    assert result[5].dtype == np.int8


# 6. compression_scale is float
def test_compression_scale_is_float():
    """isinstance(result[8], float)"""
    td = _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=True)
    result = fit_trajectories_delta_compression(td)
    assert isinstance(result[8], float)


# 7. Int16 round-trip accuracy
def test_round_trip_int16_lossless_within_tolerance():
    """L_inf < 1e-4 * max_abs"""
    td = _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=True)

    velocity = fit_trajectories(td)[5]
    result = fit_trajectories_delta_compression(td)
    deltas = result[5]
    compression_scale = result[8]

    reconstructed = deltas.astype(np.float32) / compression_scale
    max_abs_vel = float(np.max(np.abs(velocity)))
    l_inf = float(np.max(np.abs(reconstructed - velocity)))

    assert l_inf < 1e-4 * max(max_abs_vel, 1e-8), (
        f"L_inf={l_inf}, max_abs={max_abs_vel}"
    )


# 8. Int8 round-trip accuracy
def test_round_trip_int8_lossless_within_tolerance():
    """L_inf < 1e-2 * max_abs"""
    td = _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=True)

    velocity = fit_trajectories(td)[5]
    result = fit_trajectories_delta_compression(td, use_int8=True)
    deltas = result[5]
    compression_scale = result[8]

    reconstructed = deltas.astype(np.float32) / compression_scale
    max_abs_vel = float(np.max(np.abs(velocity)))
    l_inf = float(np.max(np.abs(reconstructed - velocity)))

    assert l_inf < 1e-2 * max(max_abs_vel, 1e-8), (
        f"L_inf={l_inf}, max_abs={max_abs_vel}"
    )


# 9. Zero velocity handling
def test_zero_velocity_handled():
    """Static positions => deltas all near-zero, no NaN."""
    td = _make_synthetic_trajectory_data(n_traj=3, seed=42, moving=False)
    result = fit_trajectories_delta_compression(td)

    deltas = result[5]

    # With static positions, velocity should be near zero.
    # Due to float32 noise, max_abs_vel may be ~1e-12, giving a tiny compression_scale
    # that produces deltas with at most 1 quantization level of noise.
    max_delta = int(np.max(np.abs(deltas)))
    assert max_delta <= 1, (
        f"static positions should produce near-zero deltas, got max={max_delta}"
    )
    assert not np.any(np.isnan(deltas.astype(np.float32))), "no NaN in deltas"


# 10. Extreme motion clamping
def test_extreme_motion_clamps_to_int_range():
    """Very large velocity => deltas within int16 range."""
    n_traj = 3
    rng = np.random.RandomState(9999)
    # Very large positions => extreme velocity
    positions = rng.randn(9, 3).astype(np.float32) * 1000.0

    traj_ids = np.repeat(np.arange(n_traj), 3)
    td = TrajectoryData(
        trajectory_ids=traj_ids,
        frame_indices=np.tile(np.arange(3), n_traj),
        times_normalized=np.linspace(0, 1, 9, dtype=np.float32),
        positions=positions,
        scales=np.ones((9, 3), dtype=np.float32) * 0.1,
        rotations=np.tile([1, 0, 0, 0], (9, 1)).astype(np.float32),
        colors=np.zeros((9, 3), dtype=np.float32),
        opacities=np.ones(9, dtype=np.float32) * 0.5,
        n_trajectories=n_traj,
    )

    result = fit_trajectories_delta_compression(td)
    deltas = result[5]

    assert np.all(deltas >= np.iinfo(np.int16).min), (
        f"min delta={deltas.min()}"
    )
    assert np.all(deltas <= np.iinfo(np.int16).max), (
        f"max delta={deltas.max()}"
    )


# 11. PLY round-trip
def test_write_delta_compressed_ply_round_trip(tmp_path):
    """Write PLY, read back with plyfile, verify values."""
    from plyfile import PlyData

    from offline.ply_io import write_delta_compressed_freetimegs_ply

    n = 5
    rng = np.random.RandomState(42)
    means = rng.randn(n, 3).astype(np.float32)
    scales = np.ones((n, 3), dtype=np.float32) * 0.1
    rotations = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
    colors = np.zeros((n, 3), dtype=np.float32)
    opacities = np.ones(n, dtype=np.float32) * 0.5
    deltas = np.array([[100, -200, 30000]] * n, dtype=np.int16)
    time_center = np.full(n, 0.5, dtype=np.float32)
    time_scale = np.full(n, -2.0, dtype=np.float32)  # log-space

    path = tmp_path / "test.ply"
    write_delta_compressed_freetimegs_ply(
        path,
        means,
        scales,
        rotations,
        colors,
        opacities,
        deltas,
        time_center,
        time_scale,
        compression_scale=0.001,
        use_int8=False,
    )
    pd = PlyData.read(str(path))
    m0 = pd["vertex"]["motion_0"]
    assert m0.dtype == np.int16
    assert m0[0] == 100

    # Verify other values round-trip
    x = pd["vertex"]["x"]
    assert np.allclose(x, means[:, 0])


# 12. PLY header comment
def test_ply_has_compression_scale_comment(tmp_path):
    """Header contains comment motion_scale."""
    from offline.ply_io import write_delta_compressed_freetimegs_ply

    n = 3
    rng = np.random.RandomState(42)
    means = rng.randn(n, 3).astype(np.float32)
    scales = np.ones((n, 3), dtype=np.float32) * 0.1
    rotations = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
    colors = np.zeros((n, 3), dtype=np.float32)
    opacities = np.ones(n, dtype=np.float32) * 0.5
    deltas = np.array([[100, -200, 300]] * n, dtype=np.int16)
    time_center = np.full(n, 0.5, dtype=np.float32)
    time_scale = np.full(n, -2.0, dtype=np.float32)

    path = tmp_path / "test.ply"
    write_delta_compressed_freetimegs_ply(
        path,
        means,
        scales,
        rotations,
        colors,
        opacities,
        deltas,
        time_center,
        time_scale,
        compression_scale=0.001,
        use_int8=False,
    )

    with open(path, "rb") as f:
        raw = f.read()
    end_idx = raw.find(b"end_header")
    header_text = raw[: end_idx + len(b"end_header")].decode("ascii")

    assert "comment motion_scale" in header_text
    assert "0.001" in header_text


# 13. PLY motion property
def test_ply_has_int16_motion_properties(tmp_path):
    """Header contains property short motion_0."""
    from offline.ply_io import write_delta_compressed_freetimegs_ply

    n = 3
    rng = np.random.RandomState(42)
    means = rng.randn(n, 3).astype(np.float32)
    scales = np.ones((n, 3), dtype=np.float32) * 0.1
    rotations = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
    colors = np.zeros((n, 3), dtype=np.float32)
    opacities = np.ones(n, dtype=np.float32) * 0.5
    deltas = np.array([[100, -200, 300]] * n, dtype=np.int16)
    time_center = np.full(n, 0.5, dtype=np.float32)
    time_scale = np.full(n, -2.0, dtype=np.float32)

    path = tmp_path / "test.ply"
    write_delta_compressed_freetimegs_ply(
        path,
        means,
        scales,
        rotations,
        colors,
        opacities,
        deltas,
        time_center,
        time_scale,
        compression_scale=0.001,
        use_int8=False,
    )

    with open(path, "rb") as f:
        raw = f.read()
    end_idx = raw.find(b"end_header")
    header_text = raw[: end_idx + len(b"end_header")].decode("ascii")

    assert "property short motion_0" in header_text


# 14. Single trajectory with static positions => zero motion
def test_fit_trajectories_single_frame():
    """Single trajectory with static positions => zero motion."""
    td = _make_synthetic_trajectory_data(n_traj=1, moving=False)
    result = fit_trajectories(td)
    assert len(result) == 8
    motion = result[5]
    assert np.allclose(motion, 0), "static positions should produce zero motion"


# 15. Multiple trajectories with moving data => non-zero motion
def test_fit_trajectories_two_frames():
    """Multiple trajectories with moving data => non-zero motion."""
    td = _make_synthetic_trajectory_data(n_traj=3, moving=True)
    result = fit_trajectories(td)
    assert len(result) == 8
    motion = result[5]
    assert np.any(np.abs(motion) > 0), (
        "moving positions should produce non-zero motion"
    )


# 16. Agreement between fit_trajectories and delta compression
def test_fit_trajectories_delta_compression_and_fit_trajectories_agree_on_float32():
    """Motion from fit_trajectories approx equals deltas / compression_scale."""
    for seed in [42, 123, 256]:
        td = _make_synthetic_trajectory_data(n_traj=5, seed=seed, moving=True)

        velocity_float = fit_trajectories(td)[5]
        result_delta = fit_trajectories_delta_compression(td)
        deltas = result_delta[5]
        compression_scale = result_delta[8]

        reconstructed = deltas.astype(np.float32) / compression_scale

        max_abs = float(np.max(np.abs(velocity_float)))
        l_inf = float(np.max(np.abs(reconstructed - velocity_float)))

        assert l_inf < 1e-4 * max(max_abs, 1e-8), (
            f"seed={seed}, L_inf={l_inf}"
        )


# 17. Single-observation trajectory edge case
def test_fit_trajectories_single_observation():
    """Single-observation trajectories => zero motion (obs_count <= 1 case)."""
    rng = np.random.RandomState(42)
    n_traj = 2
    td = TrajectoryData(
        trajectory_ids=np.array([0, 1], dtype=np.int32),
        frame_indices=np.array([0, 0], dtype=np.int32),
        times_normalized=np.array([0.0, 0.0], dtype=np.float32),
        positions=rng.randn(2, 3).astype(np.float32),
        scales=np.ones((2, 3), dtype=np.float32) * 0.1,
        rotations=np.tile([1, 0, 0, 0], (2, 1)).astype(np.float32),
        colors=np.zeros((2, 3), dtype=np.float32),
        opacities=np.ones(2, dtype=np.float32) * 0.5,
        n_trajectories=n_traj,
    )
    # fit_trajectories: single-obs trajectories have velocity forced to 0
    result = fit_trajectories(td)
    assert len(result) == 8
    assert result[0].shape == (2, 3)
    motion = result[5]
    assert np.allclose(motion, 0), (
        "single-observation trajectories should have zero motion"
    )

    # fit_trajectories_delta_compression should not crash
    result_delta = fit_trajectories_delta_compression(td)
    assert len(result_delta) == 9
    assert not np.any(np.isnan(result_delta[5].astype(np.float32)))
