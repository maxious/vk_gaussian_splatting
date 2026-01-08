"""
TRELLIS.2 processor for CUDA o-voxel export.
Loads intermediate latent or mesh data from disk and exports to GLB/VXZ.

Supports two data formats:
- format_version=2: Latent data (shape_slat, tex_slat) - requires decode
- format_version=1: Mesh data (vertices, faces, etc.) - direct export
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .base import GaussianProcessor

logger = logging.getLogger(__name__)

try:
    import o_voxel

    O_VOXEL_AVAILABLE = True
except ImportError as e:
    import traceback

    logger.warning(f"Failed to import o-voxel: {e}")
    traceback.print_exc()
    O_VOXEL_AVAILABLE = False


class Trellis2CUDAProcessor(GaussianProcessor):
    """TRELLIS.2 processor that decodes latents and exports via o-voxel on CUDA."""

    def __init__(
        self,
        cuda_device: str = "cuda:0",
        model_id: str = "microsoft/TRELLIS.2-4B",
    ):
        if not O_VOXEL_AVAILABLE:
            raise ImportError(
                "o-voxel package not found. Install from https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. Install CUDA-enabled PyTorch: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )

        self.cuda_device = cuda_device
        self.model_id = model_id
        self._pipeline = None
        logger.info(f"CUDA processor initialized with device: {cuda_device}")

    def _get_pipeline(self):
        """Lazy-load the TRELLIS.2 pipeline for decoding latents."""
        if self._pipeline is None:
            logger.info(f"Loading TRELLIS.2 pipeline for decoding: {self.model_id}")
            from ..vkgs_trellis2.pipelines import Trellis2ImageTo3DPipeline

            self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_id)
            self._pipeline.to(self.cuda_device)
            logger.info(f"Pipeline loaded on {self.cuda_device}")
        return self._pipeline

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

    def _load_and_decode(self, data_path: Path):
        """Load data from disk and decode if latent format."""
        logger.info(f"Loading data from {data_path}")
        data = torch.load(data_path, weights_only=False)

        if "shape_slat" in data and "tex_slat" in data:
            logger.info("Detected latent format, decoding to mesh...")
            return self._decode_latents(data)
        elif "vertices" in data and "faces" in data:
            logger.info("Detected mesh format, reconstructing mesh...")
            cuda_data = self._dict_to_cuda(data)
            return self._reconstruct_mesh(cuda_data)
        else:
            raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")

    def _decode_latents(self, data: dict):
        """Decode latent data to MeshWithVoxel using the pipeline."""
        from ..vkgs_trellis2.modules.sparse import SparseTensor

        pipeline = self._get_pipeline()

        shape_slat_data = data["shape_slat"]
        tex_slat_data = data["tex_slat"]
        res = data["res"]

        logger.info(f"Reconstructing SparseTensors for resolution {res}")

        shape_slat = SparseTensor(
            feats=shape_slat_data["feats"].to(self.cuda_device),
            coords=shape_slat_data["coords"].to(self.cuda_device),
            shape=shape_slat_data["shape"],
        )
        if shape_slat_data.get("layout"):
            shape_slat._layout = shape_slat_data["layout"]
        if shape_slat_data.get("spatial_shape"):
            shape_slat._spatial_shape = shape_slat_data["spatial_shape"]

        tex_slat = SparseTensor(
            feats=tex_slat_data["feats"].to(self.cuda_device),
            coords=tex_slat_data["coords"].to(self.cuda_device),
            shape=tex_slat_data["shape"],
        )
        if tex_slat_data.get("layout"):
            tex_slat._layout = tex_slat_data["layout"]
        if tex_slat_data.get("spatial_shape"):
            tex_slat._spatial_shape = tex_slat_data["spatial_shape"]

        logger.info("Decoding latents to mesh (this may take a moment)...")
        meshes = pipeline.decode_latent(shape_slat, tex_slat, res)

        if not meshes:
            raise RuntimeError("Decode returned no meshes")

        logger.info(f"Decoded {len(meshes)} mesh(es)")
        return meshes[0]

    def load_and_export_glb(
        self,
        mesh_path: Path,
        output_path: Path,
        decimation_target: int = 1000000,
        texture_size: int = 4096,
    ) -> Path:
        """
        Load latent/mesh data from disk and export to GLB using o-voxel.

        Args:
            mesh_path: Path to saved .pt file (from Trellis2XPUProcessor)
            output_path: Output GLB file path
            decimation_target: Target face count for simplification
            texture_size: Texture resolution for baking

        Returns:
            Path to exported GLB file
        """
        mesh = self._load_and_decode(mesh_path)

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
        Load latent/mesh data from disk and export to VXZ using o-voxel.

        Args:
            mesh_path: Path to saved .pt file (from Trellis2XPUProcessor)
            output_path: Output VXZ file path
            chunk_size: Chunk size for VXZ format
            compression: Compression algorithm ('lzma', 'zlib', 'none')
            compression_level: Compression level (0-9)

        Returns:
            Path to exported VXZ file
        """
        mesh = self._load_and_decode(mesh_path)

        logger.info(f"Exporting VXZ to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        o_voxel.io.write_vxz(
            str(output_path),
            coord=mesh.coords,
            attr=mesh.attrs,
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
        from ..vkgs_trellis2.representations import MeshWithVoxel

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
