"""
TRELLIS.2 processor for CUDA o-voxel export.
Loads intermediate MeshWithVoxel from disk and exports to GLB/VXZ.
"""

import logging
from pathlib import Path

import torch
from PIL import Image

from .base import GaussianProcessor

logger = logging.getLogger(__name__)

try:
    # o-voxel is a separate package with C++ extensions that needs to be installed
    # See: https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel
    import o_voxel

    O_VOXEL_AVAILABLE = True
except ImportError as e:
    import traceback

    logger.warning(f"Failed to import o-voxel: {e}")
    traceback.print_exc()
    O_VOXEL_AVAILABLE = False


class Trellis2CUDAProcessor(GaussianProcessor):
    """TRELLIS.2 processor that loads mesh data and exports via o-voxel on CUDA."""

    def __init__(self, cuda_device: str = "cuda:0"):
        if not O_VOXEL_AVAILABLE:
            raise ImportError(
                "o-voxel package not found. Install from https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel"
            )

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. Install CUDA-enabled PyTorch: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )

        self.cuda_device = cuda_device
        logger.info(f"CUDA processor initialized with device: {cuda_device}")

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list:
        """Not applicable - this processor only handles export from saved mesh data."""
        logger.warning("TRELLIS2CUDAProcessor only handles export from saved mesh data.")
        return []

    def load_and_export_glb(
        self,
        mesh_path: Path,
        output_path: Path,
        decimation_target: int = 1000000,
        texture_size: int = 4096,
    ) -> Path:
        """
        Load MeshWithVoxel from disk and export to GLB using o-voxel.

        Args:
            mesh_path: Path to saved .pt file (from Trellis2XPUProcessor)
            output_path: Output GLB file path
            decimation_target: Target face count for simplification
            texture_size: Texture resolution for baking

        Returns:
            Path to exported GLB file
        """
        logger.info(f"Loading mesh data from {mesh_path}")

        # Load mesh data from disk
        from .trellis2_xpu import Trellis2XPUProcessor

        data = Trellis2XPUProcessor.load_from_disk(mesh_path)

        # Transfer to CUDA
        logger.info(f"Transferring mesh to {self.cuda_device}")
        mesh_cuda = self._dict_to_cuda(data)

        # Reconstruct MeshWithVoxel object
        mesh = self._reconstruct_mesh(mesh_cuda)

        # Simplify (optional, to fit in viewer limits if needed)
        logger.info(f"Simplifying mesh to {decimation_target} faces")
        mesh.simplify(decimation_target)

        # Export to GLB using o-voxel
        logger.info(f"Exporting GLB to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb.export(str(output_path), extension_webp=True)

        logger.info(f"GLB export complete: {output_path}")
        logger.info(f"Output size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return output_path

    def load_and_export_vxz(
        self,
        mesh_path: Path,
        output_path: Path,
        chunk_size: int = 256,
        compression: str = "lzma",
        compression_level: int = 9,
    ) -> Path:
        """
        Load MeshWithVoxel from disk and export to VXZ using o-voxel.

        Args:
            mesh_path: Path to saved .pt file (from Trellis2XPUProcessor)
            output_path: Output VXZ file path
            chunk_size: Chunk size for VXZ format
            compression: Compression algorithm ('lzma', 'zlib', 'none')
            compression_level: Compression level (0-9)

        Returns:
            Path to exported VXZ file
        """
        logger.info(f"Loading mesh data from {mesh_path}")

        # Load mesh data from disk
        from .trellis2_xpu import Trellis2XPUProcessor

        data = Trellis2XPUProcessor.load_from_disk(mesh_path)

        # Transfer to CUDA
        logger.info(f"Transferring mesh to {self.cuda_device}")
        mesh_cuda = self._dict_to_cuda(data)

        # Reconstruct MeshWithVoxel object
        mesh = self._reconstruct_mesh(mesh_cuda)

        # Export to VXZ using o-voxel
        logger.info(f"Exporting VXZ to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert mesh to o-voxel format and export
        from o_voxel.convert import mesh_to_voxel

        voxel = mesh_to_voxel(mesh)

        # Export to VXZ
        o_voxel.io.write_vxz(
            str(output_path),
            coord=voxel.coord,
            attr=voxel.attr,
            chunk_size=chunk_size,
            filter="none",
            compression=compression,
            compression_level=compression_level,
            attr_interleave="as_is",
        )

        logger.info(f"VXZ export complete: {output_path}")
        logger.info(f"Output size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return output_path

    def _dict_to_cuda(self, data: dict) -> dict:
        """Transfer all tensors in the mesh data dictionary to CUDA."""
        cuda_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                cuda_data[key] = value.to(self.cuda_device)
            else:
                cuda_data[key] = value
        return cuda_data

    def _reconstruct_mesh(self, data: dict):
        """Reconstruct MeshWithVoxel object from dictionary data."""
        from vkgs_trellis2.representations import MeshWithVoxel

        return MeshWithVoxel(
            vertices=data["vertices"],
            faces=data["faces"],
            origin=data["origin"],
            voxel_size=data["voxel_size"],
            coords=data["coords"],
            attrs=data["attrs"],
            voxel_shape=data["voxel_shape"],
            layout=data["layout"],
        )
