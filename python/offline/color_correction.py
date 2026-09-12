"""Temporal color/exposure correction for per-frame Gaussian sequences.

Mined from the FreeTimeGS++ per-view affine ``ColorCorrector``
(arXiv:2605.03337).  FreeTimeGS++ learns a ``rgb @ (I + M) + v`` correction per
view against the photometric loss.  The feed-forward export path used here
(DA3 / SHARP / ZipSplat / 4DAnyone) cannot train such a corrector, so we fit the
same affine model in closed form against a trajectory-consensus reference.

Each Gaussian is tracked across frames by the motion tracker, which gives us the
correspondence between a per-frame observation and its trajectory.  For every
frame we solve a ridge-regularised least-squares problem mapping that frame's
SH-DC colors onto a per-trajectory mean reference.  This removes per-frame
global gain/offset flicker (exposure and white-balance drift) while leaving
high-frequency, content-specific color intact.

Because ``rgb = 0.5 + SH_C0 * f_dc`` is affine, fitting an affine map in SH-DC
space is equivalent to fitting one in RGB space, so no colorspace conversion is
needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .motion_tracking_cpu import TrajectoryData

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814
# SH-DC values that correspond to the valid linear-RGB range [0, 1].
DC_MIN = (0.0 - 0.5) / SH_C0
DC_MAX = (1.0 - 0.5) / SH_C0

DEFAULT_RIDGE = 1e-3
DEFAULT_MIN_PAIRS = 8


@dataclass
class ColorCorrectionStats:
    """Summary of a temporal color-correction pass."""

    num_frames: int
    num_fitted_frames: int
    mean_gain: float
    mean_shift: float
    blend: float


def _fit_affine(
    src: np.ndarray,  # (n, 3)
    dst: np.ndarray,  # (n, 3)
    ridge: float,
) -> np.ndarray | None:
    """Closed-form affine fit ``src @ A.T + b ~= dst``.

    Returns a ``(4, 3)`` matrix ``W`` such that ``[src, 1] @ W`` is the corrected
    color, or ``None`` if the normal equations are singular.
    """
    n = len(src)
    if n == 0:
        return None

    x = np.concatenate([src, np.ones((n, 1), dtype=src.dtype)], axis=1)  # (n, 4)
    xtx = x.astype(np.float64).T @ x.astype(np.float64)
    # Do not regularise the bias term.
    xtx = xtx + np.diag([ridge, ridge, ridge, 0.0])
    xty = x.astype(np.float64).T @ dst.astype(np.float64)

    try:
        return np.linalg.solve(xtx, xty)  # (4, 3)
    except np.linalg.LinAlgError:
        return None


def apply_temporal_color_correction(
    colors: np.ndarray,  # (n_obs, 3)
    trajectory_ids: np.ndarray,  # (n_obs,)
    frame_indices: np.ndarray,  # (n_obs,)
    n_trajectories: int,
    *,
    ridge: float = DEFAULT_RIDGE,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    blend: float = 1.0,
    clip_dc: bool = True,
) -> tuple[np.ndarray, ColorCorrectionStats]:
    """Fit and apply a per-frame affine color correction.

    Args:
        colors: per-observation SH-DC coefficients.
        trajectory_ids: trajectory id for each observation.
        frame_indices: frame index for each observation.
        n_trajectories: total number of trajectories.
        ridge: L2 regularisation on the linear part of the affine map.
        min_pairs: minimum matched observations required to fit a frame.  Frames
            with fewer correspondences are left unchanged (avoids unstable fits
            for sparsely matched or independently generated frames).
        blend: interpolation factor in ``[0, 1]`` applied to the correction.
        clip_dc: clamp corrected SH-DC to the valid linear-RGB range.

    Returns:
        ``(corrected_colors, stats)``.
    """
    colors = np.asarray(colors, dtype=np.float32)
    trajectory_ids = np.asarray(trajectory_ids)
    frame_indices = np.asarray(frame_indices)

    blend = float(np.clip(blend, 0.0, 1.0))
    corrected = colors.copy()

    if blend <= 0.0 or n_trajectories <= 0 or len(colors) == 0:
        stats = ColorCorrectionStats(
            num_frames=0,
            num_fitted_frames=0,
            mean_gain=1.0,
            mean_shift=0.0,
            blend=blend,
        )
        return corrected, stats

    # Per-trajectory consensus color: the target every frame is aligned to.
    # A constant target (rather than a leave-one-out mean) is what removes
    # temporal flicker, because each frame is mapped onto the same reference.
    color_sum = np.zeros((n_trajectories, 3), dtype=np.float64)
    counts = np.zeros(n_trajectories, dtype=np.int64)
    np.add.at(color_sum, trajectory_ids, colors.astype(np.float64))
    np.add.at(counts, trajectory_ids, 1)

    reference = color_sum[trajectory_ids] / np.maximum(counts[trajectory_ids], 1)[:, None]
    reference = reference.astype(np.float32)

    num_frames = int(frame_indices.max()) + 1 if len(frame_indices) else 0
    fitted_frames = 0
    gains: list[float] = []
    shifts: list[float] = []

    for frame in range(num_frames):
        mask = frame_indices == frame
        n_pairs = int(mask.sum())
        if n_pairs < min_pairs:
            continue

        src = colors[mask]
        dst = reference[mask]

        w = _fit_affine(src, dst, ridge)
        if w is None or not np.all(np.isfinite(w)):
            continue

        x = np.concatenate([src, np.ones((n_pairs, 1), dtype=src.dtype)], axis=1)
        fitted = (x.astype(np.float64) @ w).astype(np.float32)
        corrected[mask] = (1.0 - blend) * src + blend * fitted

        # Diagnostics: average diagonal gain and bias magnitude.
        gains.append(float(np.mean(np.diag(w[:3, :3]))))
        shifts.append(float(np.linalg.norm(w[3, :])))
        fitted_frames += 1

    if clip_dc:
        np.clip(corrected, DC_MIN, DC_MAX, out=corrected)

    stats = ColorCorrectionStats(
        num_frames=num_frames,
        num_fitted_frames=fitted_frames,
        mean_gain=float(np.mean(gains)) if gains else 1.0,
        mean_shift=float(np.mean(shifts)) if shifts else 0.0,
        blend=blend,
    )
    return corrected, stats


def correct_trajectory_colors(
    traj_data: TrajectoryData,
    *,
    ridge: float = DEFAULT_RIDGE,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    blend: float = 1.0,
    clip_dc: bool = True,
) -> TrajectoryData:
    """Return a copy of ``traj_data`` with temporally corrected colors.

    This is the pipeline entry point: call it after trajectories have been built
    and before fitting velocity/attributes.
    """
    corrected, stats = apply_temporal_color_correction(
        traj_data.colors,
        traj_data.trajectory_ids,
        traj_data.frame_indices,
        traj_data.n_trajectories,
        ridge=ridge,
        min_pairs=min_pairs,
        blend=blend,
        clip_dc=clip_dc,
    )

    logger.info(
        "Temporal color correction: %d/%d frames fitted (gain=%.4f shift=%.4f blend=%.2f)",
        stats.num_fitted_frames,
        stats.num_frames,
        stats.mean_gain,
        stats.mean_shift,
        stats.blend,
    )

    if stats.num_fitted_frames == 0:
        return traj_data

    return replace(traj_data, colors=corrected)
