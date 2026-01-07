import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)

# Add TRELLIS.2 to path if not installed
TRELLIS2_PATH = Path("C:/Users/maxious/trel/TRELLIS.2")
if TRELLIS2_PATH.exists() and str(TRELLIS2_PATH) not in sys.path:
    sys.path.append(str(TRELLIS2_PATH))

try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    import o_voxel

    TRELLIS2_AVAILABLE = True
except ImportError:
    TRELLIS2_AVAILABLE = False


class Trellis2Processor(GaussianProcessor):
    """Processor using Microsoft TRELLIS.2 model."""

    def __init__(
        self,
        model_id: str = "microsoft/TRELLIS.2-4B",
        device: str = "cuda",
    ):
        if not TRELLIS2_AVAILABLE:
            raise ImportError(
                "TRELLIS.2 not found. Please ensure it is in PYTHONPATH or C:/Users/maxious/trel/TRELLIS.2"
            )

        self.device = device
        self.model_id = model_id

        # Environment setup
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        logger.info(f"Loading TRELLIS.2 pipeline: {model_id}")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
        self.pipeline.cuda()

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """
        TRELLIS.2 produces Meshes/GLB, not Gaussians.
        This method will return an empty list, but the GLB export will be handled via `export_glb`.
        """
        logger.warning(
            "TRELLIS.2 produces GLB/Mesh outputs, not Gaussians. GaussianFrame generation skipped."
        )
        return []

    def export_glb(self, image_path: Path, output_path: Path):
        """Export GLB using TRELLIS.2"""
        logger.info(f"Running TRELLIS.2 on {image_path}")
        image = Image.open(image_path)

        # Run pipeline
        mesh = self.pipeline.run(image)[0]

        # Simplify (optional, to fit in viewer limits if needed)
        # Using 1M faces target or similar
        mesh.simplify(1000000)

        logger.info(f"Exporting GLB to {output_path}")
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1000000,
            texture_size=4096,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb.export(str(output_path), extension_webp=True)
        logger.info("GLB export complete")
