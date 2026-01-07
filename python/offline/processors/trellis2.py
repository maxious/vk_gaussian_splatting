import os
import sys
import logging
from pathlib import Path

# Add vendored TRELLIS.2 package to path
_VKGS_TRELLIS2_PARENT = Path(__file__).parent.parent
if str(_VKGS_TRELLIS2_PARENT) not in sys.path:
    sys.path.insert(0, str(_VKGS_TRELLIS2_PARENT))

import numpy as np
import torch
from PIL import Image

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)

try:
    from vkgs_trellis2.pipelines import Trellis2ImageTo3DPipeline
    from vkgs_trellis2.renderers import EnvMap

    # o-voxel is a separate package with C++ extensions that needs to be installed
    # See: https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel
    try:
        import o_voxel

        O_VOXEL_AVAILABLE = True
    except ImportError:
        O_VOXEL_AVAILABLE = False
        logger.warning(
            "o-voxel package not found. GLB export will be disabled. "
            "Install with: pip install o-voxel (requires CUDA compilation)"
        )

    TRELLIS2_AVAILABLE = True
except ImportError as e:
    import traceback

    logger.warning(f"Failed to import TRELLIS.2: {e}")
    traceback.print_exc()
    TRELLIS2_AVAILABLE = False
    O_VOXEL_AVAILABLE = False


class Trellis2Processor(GaussianProcessor):
    """Processor using Microsoft TRELLIS.2 model."""

    def __init__(
        self,
        model_id: str = "microsoft/TRELLIS.2-4B",
        device: str = "cuda",
    ):
        if not TRELLIS2_AVAILABLE:
            raise ImportError(
                "TRELLIS.2 not found. The vendored package should be in offline/vkgs_trellis2/"
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
        if not O_VOXEL_AVAILABLE:
            raise ImportError(
                "o-voxel package is required for GLB export. "
                "Install from https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel"
            )

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

    def export_vxz(self, image_path: Path, output_path: Path):
        """Export VXZ o-voxel file using TRELLIS.2"""
        if not O_VOXEL_AVAILABLE:
            raise ImportError(
                "o-voxel package is required for VXZ export. "
                "Install from https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel"
            )

        logger.info(f"Running TRELLIS.2 on {image_path}")
        image = Image.open(image_path)

        # Run pipeline
        mesh = self.pipeline.run(image)[0]

        logger.info(f"Exporting VXZ o-voxel to {output_path}")

        # Convert mesh to o-voxel format and export
        from o_voxel.convert import mesh_to_voxel

        voxel = mesh_to_voxel(mesh)

        # Export to VXZ
        o_voxel.io.write_vxz(
            str(output_path),
            coord=voxel.coord,
            attr=voxel.attr,
            chunk_size=256,
            filter="none",
            compression="lzma",
            compression_level=9,
            attr_interleave="as_is",
        )
        logger.info("VXZ export complete")
