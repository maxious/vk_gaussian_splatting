"""Matrix3D Gaussian Processor."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np

from ..ply_io import load_static_gaussian_ply
from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)


class Matrix3DGaussianProcessor(GaussianProcessor):
    """Process images to Gaussian splats using Apple's Matrix3D model."""

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str | None = None,
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path or "checkpoints/matrix3d_512.pt"
        self.matrix3d_path = Path(__file__).parent.parent.parent / "matrix3d"

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = True,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames using Matrix3D.

        For single frame: use single-view to 3D
        For multiple frames: use unposed few-shot to 3D
        """
        if len(frame_paths) == 1:
            return self._process_single_view(frame_paths[0], timestamps_ms[0])
        else:
            return self._process_unposed_few_shot(frame_paths, timestamps_ms)

    def _process_single_view(self, image_path: Path, timestamp_ms: float) -> list[GaussianFrame]:
        """Process single image using Matrix3D single-view to 3D."""
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "input.png"
            shutil.copy2(image_path, temp_path)

            # Run Matrix3D single-view pipeline
            cmd = [
                "python",
                str(self.matrix3d_path / "pipeline_single_to_3d.py"),
                "--data_path",
                str(temp_path),
                "--exp_name",
                "single-to-3d",
                "--gpu",
                "0" if self.device == "cuda" else "cpu",
            ]

            logger.info(f"Running Matrix3D single-view: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.matrix3d_path, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Matrix3D failed: {result.stderr}")
                return []

            # Find the output PLY file
            results_dir = self.matrix3d_path / "results" / "single-to-3d" / "input"
            ply_files = list(results_dir.glob("*.ply"))
            if not ply_files:
                logger.error("No PLY file found in Matrix3D output")
                return []

            ply_path = ply_files[0]
            logger.info(f"Loading PLY from {ply_path}")

            # Load the PLY as GaussianFrame
            frame = load_static_gaussian_ply(ply_path)
            frame.timestamp_ms = timestamp_ms
            return [frame]

    def _process_unposed_few_shot(
        self, frame_paths: list[Path], timestamps_ms: list[float]
    ) -> list[GaussianFrame]:
        """Process multiple images using Matrix3D unposed few-shot to 3D."""
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_dir = Path(temp_dir) / "input"
            temp_input_dir.mkdir()

            # Copy images
            for i, src_path in enumerate(frame_paths):
                dst_path = temp_input_dir / f"{i:06d}.jpg"
                shutil.copy2(src_path, dst_path)

            # Run Matrix3D unposed few-shot pipeline
            cmd = [
                "python",
                str(self.matrix3d_path / "pipeline_unposed_few_shot_to_3d.py"),
                "--data_path",
                str(temp_input_dir),
                "--exp_name",
                "unposed-fewshot-to-3d",
                "--gpu",
                "0" if self.device == "cuda" else "cpu",
            ]

            logger.info(f"Running Matrix3D unposed few-shot: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.matrix3d_path, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Matrix3D failed: {result.stderr}")
                return []

            # Find the output PLY file
            results_dir = self.matrix3d_path / "results" / "unposed-fewshot-to-3d" / "input"
            ply_files = list(results_dir.glob("*.ply"))
            if not ply_files:
                logger.error("No PLY file found in Matrix3D output")
                return []

            ply_path = ply_files[0]
            logger.info(f"Loading PLY from {ply_path}")

            # Load the PLY as GaussianFrame
            frame = load_static_gaussian_ply(ply_path)
            frame.timestamp_ms = timestamps_ms[len(timestamps_ms) // 2]  # Use middle timestamp
            return [frame]
