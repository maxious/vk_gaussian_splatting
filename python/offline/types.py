"""Core data types for Gaussian Splatting export pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianFrame:
    frame_idx: int
    timestamp_ms: float
    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    flow: np.ndarray | None = None
    intrinsic: np.ndarray | None = None
    extrinsic: np.ndarray | None = None
    image_size: tuple[int, int] | None = None
    color_space_index: int | None = None

    def __len__(self) -> int:
        """Return the number of Gaussians in this frame."""
        return len(self.means)

    @property
    def n_gaussians(self) -> int:
        """Return the number of Gaussians in this frame."""
        return len(self.means)
