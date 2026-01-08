"""
TRELLIS.2 processor for XPU-only inference.
Saves intermediate MeshWithVoxel to disk for later CUDA processing.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add vendored TRELLIS.2 package to path
_VKGS_TRELLIS2_PARENT = Path(__file__).parent.parent
if str(_VKGS_TRELLIS2_PARENT) not in sys.path:
    sys.path.insert(0, str(_VKGS_TRELLIS2_PARENT))

import torch
import numpy as np
from PIL import Image

from .base import GaussianProcessor

logger = logging.getLogger(__name__)

try:
    from vkgs_trellis2.pipelines import Trellis2ImageTo3DPipeline

    TRELLIS2_AVAILABLE = True
except ImportError as e:
    import traceback

    logger.warning(f"Failed to import TRELLIS.2: {e}")
    traceback.print_exc()
    TRELLIS2_AVAILABLE = False


class Trellis2XPUProcessor(GaussianProcessor):
    """TRELLIS.2 processor that runs on XPU and saves intermediate mesh data."""

    def __init__(
        self,
        model_id: str = "microsoft/TRELLIS.2-4B",
        xpu_device: str = "xpu:0",
        low_vram: bool = True,
    ):
        if not TRELLIS2_AVAILABLE:
            raise ImportError(
                "TRELLIS.2 not found. The vendored package should be in offline/vkgs_trellis2/"
            )

        # Check XPU availability
        if not torch.xpu.is_available():
            raise RuntimeError(
                "XPU not available. Install XPU-enabled PyTorch: "
                "pip install torch --index-url https://download.pytorch.org/whl/xpu"
            )

        self.xpu_device = xpu_device
        self.model_id = model_id
        self.low_vram = low_vram

        # Environment setup
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        logger.info(f"Loading TRELLIS.2 pipeline: {model_id}")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
        self.pipeline.to(xpu_device)
        logger.info(f"TRELLIS.2 pipeline moved to {xpu_device}")

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list:
        """Not applicable - TRELLIS.2 produces MeshWithVoxel, not GaussianFrame."""
        logger.warning("TRELLIS.2 produces Mesh/GLB outputs, not GaussianFrame.")
        return []

    def infer_and_save(
        self,
        image_path: Path,
        output_path: Path,
        pipeline_type: str = "1024_cascade",
        num_samples: int = 1,
        seed: int = 42,
        max_num_tokens: int = 49152,
    ) -> Path:
        """
        Run TRELLIS.2 inference on XPU and save MeshWithVoxel to disk.

        Args:
            image_path: Input image path
            output_path: Output .pt file path (will be created)
            pipeline_type: '512', '1024', '1024_cascade', '1536_cascade'
            num_samples: Number of samples to generate
            seed: Random seed
            max_num_tokens: Maximum tokens for cascade pipeline

        Returns:
            Path to saved .pt file
        """
        logger.info(f"Running TRELLIS.2 inference on {image_path}")

        # Load and preprocess image
        image = Image.open(image_path)

        # Run pipeline on XPU - use return_latent=True to skip CUDA decode
        _, (shape_slat, tex_slat, res) = self.pipeline.run(
            image,
            num_samples=num_samples,
            seed=seed,
            pipeline_type=pipeline_type,
            max_num_tokens=max_num_tokens,
            preprocess_image=True,
            return_latent=True,
        )

        # Move latent tensors to CPU before saving
        logger.info("Transferring latents to CPU for saving")
        latent_data = {
            'shape_slat': self._sparse_tensor_to_cpu(shape_slat),
            'tex_slat': self._sparse_tensor_to_cpu(tex_slat),
            'res': res,
        }

        # Save to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving latent data to {output_path}")
        torch.save(latent_data, output_path)

        logger.info(
            f"Mesh data saved successfully ({output_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )
        return output_path

    def _sparse_tensor_to_cpu(self, slat):
        """Transfer SparseTensor latent from XPU to CPU."""
        from vkgs_trellis2.modules.sparse import SparseTensor
        
        # Extract tensor data and move to CPU
        return {
            "feats": slat.feats.cpu() if slat.feats is not None else None,
            "coords": slat.coords.cpu() if slat.coords is not None else None,
            "shape": slat.shape,
            "layout": slat.layout,
            "spatial_shape": slat.spatial_shape if hasattr(slat, 'spatial_shape') else None,
            "format_version": 2,  # Version 2 for latent format
        }

    def _mesh_to_cpu(self, mesh_xpu):
        """Transfer MeshWithVoxel from XPU to CPU."""
        from vkgs_trellis2.representations import MeshWithVoxel

        # Extract all attributes and move to CPU
        return {
            "vertices": mesh_xpu.vertices.cpu(),
            "faces": mesh_xpu.faces.cpu(),
            "origin": mesh_xpu.origin
            if isinstance(mesh_xpu.origin, (list, tuple))
            else mesh_xpu.origin.cpu(),
            "voxel_size": mesh_xpu.voxel_size,
            "coords": mesh_xpu.coords.cpu() if mesh_xpu.coords is not None else None,
            "attrs": mesh_xpu.attrs.cpu() if mesh_xpu.attrs is not None else None,
            "voxel_shape": mesh_xpu.voxel_shape,
            "layout": mesh_xpu.layout,
            "format_version": 1,  # Version for future compatibility
        }

    @staticmethod
    def load_from_disk(mesh_path: Path) -> dict:
        """Load MeshWithVoxel data from disk.

        Returns:
            Dictionary with mesh data (vertices, faces, coords, attrs, etc.)
        """
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh data file not found: {mesh_path}")

        logger.info(f"Loading mesh data from {mesh_path}")
        data = torch.load(mesh_path, weights_only=False)

        # Check version
        version = data.get("format_version", 0)
        if version != 1:
            logger.warning(f"Unknown format version {version}, may have compatibility issues")

        logger.info(f"Loaded mesh data: {mesh_path.stat().st_size / 1024 / 1024:.2f} MB")
        return data
