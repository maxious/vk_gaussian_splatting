"""FastGS integration for high-quality Gaussian Splatting training."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..types import GaussianFrame

if TYPE_CHECKING:
    from ..ply_io import load_static_gaussian_ply

logger = logging.getLogger(__name__)


class FastGSProcessor:
    """FastGS processor for high-quality Gaussian Splatting training.

    FastGS is a training-based approach that produces SOTA-quality
    3D Gaussian Splats from multi-view images.

    Requirements:
    - COLMAP-formatted dataset (images/ + sparse/0/)
    - FastGS installed and in PYTHONPATH
    - CUDA-capable GPU

    Paper: https://arxiv.org/abs/2511.04283
    Repo: https://github.com/fastgs/FastGS
    """

    def __init__(
        self,
        device: str = "cuda",
        iterations: int = 30_000,
        save_iterations: list[int] | None = None,
        optimizer_type: str = "sparse_adam",
        source_path: str | None = None,
        fastgs_path: str | None = None,
        white_background: bool = False,
        eval: bool = False,
        sh_degree: int = 3,
    ):
        """Initialize FastGS processor.

        Args:
            device: PyTorch device (default: cuda)
            iterations: Number of training iterations (default: 30_000)
            save_iterations: Iterations to save checkpoints (default: [30_000])
            optimizer_type: 'default' or 'sparse_adam' (default: sparse_adam)
            source_path: Path to COLMAP dataset (default: auto-detect)
            fastgs_path: Path to FastGS installation (default: auto-detect)
            white_background: Use white background (default: False)
            eval: Use train/test split (default: False)
            sh_degree: Spherical harmonics degree (default: 3)
        """
        self.device = device
        self.iterations = iterations
        self.save_iterations = save_iterations or [iterations]
        self.optimizer_type = optimizer_type
        self.white_background = white_background
        self.eval = eval
        self.sh_degree = sh_degree

        self.fastgs_path = self._find_fastgs_path(fastgs_path)
        self.source_path = source_path

        logger.info(f"FastGS initialized: {self.fastgs_path}")
        logger.info(f"Iterations: {self.iterations}, Optimizer: {self.optimizer_type}")

    def _find_fastgs_path(self, provided_path: str | None) -> str:
        """Find FastGS installation path."""
        if provided_path is not None:
            return provided_path

        candidates = [
            Path(__file__).parent.parent.parent.parent.parent.parent / "FastGS",
            Path.home() / "FastGS",
            Path.home() / "src" / "FastGS",
        ]

        for cand in candidates:
            if cand.exists() and (cand / "train.py").exists():
                logger.info(f"Found FastGS at: {cand}")
                return str(cand)

        try:
            for path in sys.path:
                if (
                    Path(path).exists()
                    and (Path(path) / "train.py").exists()
                    and (Path(path) / "scene").exists()
                ):
                    logger.info(f"Found FastGS in sys.path: {path}")
                    return path
        except Exception:
            pass

        raise ValueError("FastGS not found")

    def _check_colmap_dataset(self, dataset_path: Path) -> bool:
        """Check if directory is a valid COLMAP dataset."""
        required = [
            dataset_path / "sparse",
            dataset_path / "sparse" / "0",
        ]

        for req in required:
            if not req.exists():
                return False

        # Check for cameras.bin or similar
        sparse_dir = dataset_path / "sparse" / "0"
        colmap_files = list(sparse_dir.glob("*.bin")) + list(sparse_dir.glob("*.txt"))
        return len(colmap_files) > 0

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames with FastGS training.

        Args:
            frame_paths: List of frame image paths
            timestamps_ms: Corresponding timestamps in milliseconds
            per_frame: If True, process each frame individually.
                      For FastGS, this is ignored - always trains single model.

        Returns:
            List of GaussianFrame objects (single frame with trained Gaussians)
        """
        if not frame_paths:
            return []

        # Determine dataset path
        if self.source_path is None:
            dataset_path = frame_paths[0].parent
        else:
            dataset_path = Path(self.source_path)

        logger.info(f"FastGS processing dataset: {dataset_path}")

        # Check if dataset is COLMAP-formatted
        if not self._check_colmap_dataset(dataset_path):
            raise RuntimeError(
                f"Dataset is not COLMAP-formatted: {dataset_path}\n"
                f"Expected structure: images/ + sparse/0/\n"
                f"Please run COLMAP on your images first, or use depth-based "
                f"processors (DA3/MoGe/SHARP) for quick conversion."
            )

        # Run FastGS training
        with tempfile.TemporaryDirectory() as temp_output:
            output_path = Path(temp_output)

            # Build command
            cmd = self._build_training_command(dataset_path, output_path)

            logger.info(f"Running FastGS training...")
            logger.info(f"Command: {' '.join(cmd[:3])} ...")

            try:
                # Run FastGS training
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )

                # Log output
                for line in result.stdout.split("\n"):
                    if line.strip():
                        logger.info(f"FastGS: {line}")

                if result.returncode != 0:
                    logger.error(f"FastGS training failed:")
                    logger.error(result.stderr)
                    raise RuntimeError(f"FastGS training failed with code {result.returncode}")

            except subprocess.TimeoutExpired:
                logger.error("FastGS training timed out after 1 hour")
                raise RuntimeError("FastGS training timeout")
            except subprocess.CalledProcessError as e:
                logger.error(f"FastGS training failed: {e}")
                raise

            # Load the final trained PLY
            final_iteration = max(self.save_iterations)
            ply_path = (
                output_path / "point_cloud" / f"iteration_{final_iteration}" / "point_cloud.ply"
            )

            if not ply_path.exists():
                # Try to find any saved PLY
                ply_files = list((output_path / "point_cloud").glob("**/point_cloud.ply"))
                if not ply_files:
                    raise RuntimeError(f"FastGS did not produce output PLY. Check training logs.")
                ply_path = ply_files[-1]
                logger.info(f"Using latest PLY: {ply_path}")

            logger.info(f"Loading trained Gaussians from: {ply_path}")

            # Load PLY and return as GaussianFrame
            from ..ply_io import load_static_gaussian_ply

            frame = load_static_gaussian_ply(ply_path)
            frame.frame_idx = 0
            frame.timestamp_ms = timestamps_ms[0] if timestamps_ms else 0.0

            return [frame]

    def _build_training_command(self, dataset_path: Path, output_path: Path) -> list[str]:
        """Build FastGS training command."""
        cmd = [
            sys.executable,
            os.path.join(self.fastgs_path, "train.py"),
            "--source_path",
            str(dataset_path),
            "--model_path",
            str(output_path),
            "--iterations",
            str(self.iterations),
            "--optimizer_type",
            self.optimizer_type,
            "--sh_degree",
            str(self.sh_degree),
        ]

        for save_iter in self.save_iterations:
            cmd.append(str(save_iter))

        cmd.append("--quiet")

        if self.white_background:
            cmd.append("--white_background")

        if self.eval:
            cmd.append("--eval")

        return cmd

    def __repr__(self) -> str:
        return (
            f"FastGSProcessor(device={self.device}, "
            f"iterations={self.iterations}, optimizer={self.optimizer_type})"
        )
