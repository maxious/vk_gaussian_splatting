"""Base class for Gaussian processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import GaussianFrame


class GaussianProcessor(ABC):
    """Abstract base class for Gaussian Splatting processors.

    All processors must implement the process_frames method to convert
    images/video frames into GaussianFrame objects.
    """

    @abstractmethod
    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames to extract Gaussians.

        Args:
            frame_paths: List of frame image paths
            timestamps_ms: Corresponding timestamps in milliseconds
            per_frame: If True, process each frame individually for separate PLYs.
                      If False (default), process all together for merged Gaussians.

        Returns:
            List of GaussianFrame objects
        """
        ...
