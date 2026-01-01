"""Core data types for Gaussian Splatting export pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianFrame:
    """Gaussian splat data for a single frame.

    Attributes:
        frame_idx: Frame index in the sequence
        timestamp_ms: Timestamp in milliseconds
        means: (N, 3) float32 positions
        scales: (N, 3) float32 log-scales
        rotations: (N, 4) float32 quaternions (wxyz order)
        colors: (N, 3) float32 SH DC coefficients
        opacities: (N,) float32 logit opacities
    """

    frame_idx: int
    timestamp_ms: float
    means: np.ndarray  # (N, 3) positions
    scales: np.ndarray  # (N, 3) log-scale
    rotations: np.ndarray  # (N, 4) quaternion wxyz
    colors: np.ndarray  # (N, 3) SH DC term (f_dc)
    opacities: np.ndarray  # (N,) logit opacity

    def __len__(self) -> int:
        """Return the number of Gaussians in this frame."""
        return len(self.means)

    @property
    def n_gaussians(self) -> int:
        """Return the number of Gaussians in this frame."""
        return len(self.means)
