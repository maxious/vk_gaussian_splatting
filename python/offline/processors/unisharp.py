"""UniSHARP Gaussian Processor.

Integrates Insta360 Research Team's UniSHARP model for single-image-to-3D
Gaussian Splatting with support for perspective, fisheye, and panoramic cameras.

Setup:
    cd python/
    git clone https://github.com/Insta360-Research-Team/UniSHARP.git unisharp
    git clone https://github.com/lpiccinelli-eth/UniK3D.git UniK3D

Model weights are auto-downloaded from Hugging Face Hub:
    Insta360-Research/Unisharp (4.73 GB checkpoint)
    lpiccinelli/unik3d-vitl (UniK3D backbone, ~1.5 GB)

Note: gsplat is NOT required — it's only used by UniSHARP's optional
rendering utilities, not the model forward pass that this processor uses.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)


@dataclass
class UniSHARPProfilingStats:
    """Profiling statistics for UniSHARP processing."""

    io_time: float = 0.0
    preprocess_time: float = 0.0
    inference_time: float = 0.0
    postprocess_time: float = 0.0
    total_frames: int = 0
    frame_times: list[float] = field(default_factory=list)

    def log_summary(self):
        if self.total_frames == 0:
            return
        logger.info("=== UniSHARP Processing Profile ===")
        logger.info(f"Total frames: {self.total_frames}")
        logger.info(
            f"I/O (load+mask):     {self.io_time:.2f}s ({self.io_time / self.total_frames * 1000:.1f}ms/frame)"
        )
        logger.info(
            f"Preprocess (resize): {self.preprocess_time:.2f}s ({self.preprocess_time / self.total_frames * 1000:.1f}ms/frame)"
        )
        logger.info(
            f"Inference (GPU):     {self.inference_time:.2f}s ({self.inference_time / self.total_frames * 1000:.1f}ms/frame)"
        )
        logger.info(
            f"Postprocess (CPU):   {self.postprocess_time:.2f}s ({self.postprocess_time / self.total_frames * 1000:.1f}ms/frame)"
        )
        total = self.io_time + self.preprocess_time + self.inference_time + self.postprocess_time
        logger.info(
            f"Total measured:      {total:.2f}s ({total / self.total_frames * 1000:.1f}ms/frame)"
        )
        if self.frame_times:
            avg = sum(self.frame_times) / len(self.frame_times)
            logger.info(f"Avg wall-clock:      {avg * 1000:.1f}ms/frame")


@dataclass
class PreloadedFrame:
    """Pre-loaded frame with optional mask."""

    index: int
    path: Path
    timestamp_ms: float
    image: np.ndarray
    height: int
    width: int
    mask_applied: bool = False
    mask: np.ndarray | None = None


class UniSHARPGaussianProcessor(GaussianProcessor):
    """Process images to Gaussian splats using Insta360's UniSHARP model.

    UniSHARP extends SHARP with universal camera support (perspective,
    fisheye, panoramic) via UniK3D's ray-based geometry representation.

    Args:
        checkpoint_path: Path to pretrained model checkpoint (.pt file).
            If not provided, downloads from HF Hub (Insta360-Research/Unisharp).
        device: PyTorch device string (e.g., 'cuda', 'cuda:0', 'cpu').
        camera_model: Camera model type ('pinhole', 'fisheye624', 'spherical').
            Defaults to 'pinhole' for perspective images.
        distance_init_cap_m: Soft cap on initial depth in meters (default: model's
            internal cap of 100.0). Lower values compress the scene depth range.
            For indoor scenes try 5-10, for outdoor try 20-50.
        internal_size: Processing resolution as (height, width).
            Uses UniSHARP's default (768 max edge) if not specified.
        enable_profiling: Whether to collect and log timing statistics.
    """

    # UniSHARP default for perspective: 768px max edge
    DEFAULT_MAX_EDGE = 768
    SH_C0 = 0.28209479177387814

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        camera_model: str = "pinhole",
        distance_init_cap_m: float | None = None,
        internal_size: tuple[int, int] | None = None,
        enable_profiling: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.camera_model = camera_model
        self.distance_init_cap_m = distance_init_cap_m
        self.internal_size = internal_size
        self.enable_profiling = enable_profiling
        self.model = None
        self._torch = None
        self._F = None

    @staticmethod
    def _discover_devices(device_spec: str = "auto") -> list[str]:
        """Discover all available GPU devices based on device_spec.

        Args:
            device_spec: Device specification like 'auto', 'cuda', 'xpu', 'cpu'

        Returns:
            List of device strings (e.g., ['cuda:0', 'cuda:1'] or ['xpu:0'])
        """
        if ":" in device_spec:
            device_type, device_indices = device_spec.split(":", 1)
            indices = [int(i.strip()) for i in device_indices.split(",")]
            return [f"{device_type}:{i}" for i in indices]

        if device_spec == "auto":
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                devices = [f"cuda:{i}" for i in range(num_devices)]
                logger.info(f"Auto-detected {num_devices} CUDA device(s): {devices}")
                return devices

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                num_devices = torch.xpu.device_count()
                devices = [f"xpu:{i}" for i in range(num_devices)]
                logger.info(f"Auto-detected {num_devices} XPU device(s): {devices}")
                return devices

            if torch.backends.mps.is_available():
                logger.info("Auto-detected MPS device (Apple Silicon)")
                return ["mps"]

            logger.info("No GPU detected, using CPU")
            return ["cpu"]

        if device_spec == "cuda":
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                return [f"cuda:{i}" for i in range(num_devices)]
            logger.warning("CUDA requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "xpu":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                num_devices = torch.xpu.device_count()
                return [f"xpu:{i}" for i in range(num_devices)]
            logger.warning("XPU requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "mps":
            if torch.backends.mps.is_available():
                return ["mps"]
            logger.warning("MPS requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "cpu":
            return ["cpu"]

        logger.warning(f"Unknown device spec '{device_spec}', falling back to CPU")
        return ["cpu"]

    def _resolve_internal_size(self, H: int, W: int) -> tuple[int, int]:
        """Resolve processing resolution preserving aspect ratio.

        UniSHARP auto-resizes: max 768px for perspective, max 1536px for panorama.
        """
        if self.internal_size is not None:
            return self.internal_size

        max_edge = self.DEFAULT_MAX_EDGE
        if self.camera_model == "spherical":
            max_edge = 1536

        if H >= W:
            new_h = min(H, max_edge)
            new_w = int(W * (new_h / H))
        else:
            new_w = min(W, max_edge)
            new_h = int(H * (new_w / W))

        # Ensure even dimensions (common for vision models)
        new_h = (new_h // 2) * 2
        new_w = (new_w // 2) * 2

        return (new_h, new_w)

    def _load_model(self):
        """Load UniSHARP model with UniK3D backbone.

        Downloads checkpoint from HuggingFace Hub if not provided locally.
        Requires UniSHARP repo cloned to python/unisharp/ and
        UniK3D repo cloned to python/UniK3D/.
        """
        if self.model is not None:
            return

        import ssl

        import torch.nn.functional as F

        self._torch = torch
        self._F = F

        torch.set_float32_matmul_precision("high")
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]

        # Ensure UniSHARP and UniK3D are on sys.path
        import sys

        unisharp_dir = Path(__file__).resolve().parents[2] / "unisharp"
        unik3d_dir = Path(__file__).resolve().parents[2] / "UniK3D"

        if not unisharp_dir.exists():
            raise FileNotFoundError(
                f"UniSHARP not found at {unisharp_dir}. "
                "Clone it: git clone https://github.com/Insta360-Research-Team/UniSHARP.git python/unisharp"
            )
        if not unik3d_dir.exists():
            raise FileNotFoundError(
                f"UniK3D not found at {unik3d_dir}. "
                "Clone it: git clone https://github.com/lpiccinelli-eth/UniK3D.git python/UniK3D"
            )

        if str(unisharp_dir) not in sys.path:
            sys.path.insert(0, str(unisharp_dir))
        if str(unik3d_dir) not in sys.path:
            sys.path.insert(0, str(unik3d_dir))

        from unisharp.models.unisharp_feature import UnisharpFeatureConfig, UnisharpFeatureModel

        logger.info(f"Loading UniSHARP model on {self.device}...")

        # Build config — defaults match UniSHARP inference defaults
        cfg = UnisharpFeatureConfig()

        model = UnisharpFeatureModel(cfg).to(self.device)

        # Load checkpoint
        if self.checkpoint_path and self.checkpoint_path.exists():
            logger.info(f"Loading UniSHARP checkpoint from: {self.checkpoint_path}")
            model.load_from_checkpoint(str(self.checkpoint_path), strict=False)
        else:
            # Auto-download from HF Hub
            from huggingface_hub import hf_hub_download

            logger.info("Downloading UniSHARP checkpoint from Insta360-Research/Unisharp...")
            ckpt_path = hf_hub_download(
                repo_id="Insta360-Research/Unisharp",
                filename="pretained_model.pt",
                cache_dir=str(Path.home() / ".cache" / "huggingface" / "hub"),
            )
            logger.info(f"Loading checkpoint from HF cache: {ckpt_path}")
            model.load_from_checkpoint(ckpt_path, strict=False)

        model.eval()
        self.model = model

        # Clear GPU cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        logger.info(f"UniSHARP model ready on {self.device}")

    def _build_camera_intrinsics(self, H: int, W: int, f_px: float) -> torch.Tensor:
        """Build 3x3 camera intrinsics matrix for pinhole camera model."""
        return (
            torch.tensor(
                [
                    [f_px, 0.0, W / 2.0],
                    [0.0, f_px, H / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            .float()
            .to(self.device)
        )

    def _preprocess_image(self, img: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess image: normalize to [0,1] float32 and keep uint8 version.

        UniSHARP requires both float32 image (normalized [0,1]) and uint8 image
        (for the UniK3D encoder which uses DINOv2).

        Returns:
            (image_float, image_u8) — both [1, 3, H, W] tensors on device
        """
        img_pt = torch.from_numpy(img).to(self.device, non_blocking=True).float()
        img_pt = img_pt.permute(2, 0, 1).unsqueeze(0) / 255.0

        # Resize to internal processing size
        _, _, H, W = img_pt.shape
        target_h, target_w = self._resolve_internal_size(H, W)
        img_resized = self._F.interpolate(
            img_pt,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )

        # UniSHARP also needs uint8 version
        img_u8 = (img_resized * 255.0).clamp(0, 255).to(torch.uint8)

        return img_resized, img_u8

    def _gaussians_to_frame(
        self,
        gaussians,
        H: int,
        W: int,
        mask: np.ndarray | None,
        f_px: float,
    ) -> GaussianFrame:
        """Convert UniSHARP Gaussians3D output to GaussianFrame.

        UniSHARP outputs Gaussians3D NamedTuple with the same fields as SHARP:
        mean_vectors, singular_values, quaternions, colors, opacities.

        Conversion: singular_values → log-scales, opacities → logits,
        colors → SH DC coefficients.
        """
        means_tensor = gaussians.mean_vectors.squeeze(0)
        scales_linear = gaussians.singular_values.squeeze(0)
        rotations_tensor = gaussians.quaternions.squeeze(0)
        opacities_prob = gaussians.opacities.squeeze(0)
        colors_linear = gaussians.colors.squeeze(0)

        # Pass through raw linear values — write_static_gaussian_ply() handles
        # log (scales) and logit (opacities) conversions downstream
        scales_tensor = self._torch.clamp(scales_linear, min=1e-8)
        opacities_tensor = self._torch.clamp(opacities_prob, 1e-6, 1.0 - 1e-6)
        colors_sh = (colors_linear - 0.5) / self.SH_C0

        # Build valid mask (all Gaussians valid by default)
        valid_mask = self._torch.ones(
            means_tensor.shape[0], dtype=self._torch.bool, device=self.device
        )

        # Apply geometric mask if provided
        # Note: UniSHARP outputs Gaussians in metric camera space, not NDC.
        # The pinhole projection (x*f/z + W/2) may not align with the image
        # mask for all scenes. Image-level masking (zeroing background pixels
        # before inference) is the primary filter.
        if mask is not None:
            mask_np = mask.astype(np.float32)
            mask_tensor = (
                self._torch.from_numpy(mask_np).to(self.device).unsqueeze(0).unsqueeze(0)
            )

            x = means_tensor[:, 0]
            y = means_tensor[:, 1]
            z = means_tensor[:, 2]

            valid_z = z > 1e-3
            valid_mask = valid_mask & valid_z

            z_safe = self._torch.where(valid_z, z, self._torch.ones_like(z))

            u_px = (x * f_px / z_safe) + W / 2.0
            v_px = (y * f_px / z_safe) + H / 2.0

            u_norm = 2.0 * (u_px / W) - 1.0
            v_norm = 2.0 * (v_px / H) - 1.0

            grid_coords = (
                self._torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
            )

            mask_sampled = self._torch.nn.functional.grid_sample(
                mask_tensor,
                grid_coords,
                mode="nearest",
                align_corners=False,
                padding_mode="zeros",
            )

            geometric_mask = mask_sampled.reshape(-1) > 0.5
            in_bounds = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)
            logger.debug(
                "Mask projection: %d/%d in fg, %d in image bounds, z>0: %d",
                geometric_mask.sum().item(), geometric_mask.numel(),
                in_bounds.sum().item(), valid_z.sum().item(),
            )

            # Only apply geometric mask if it actually filters (non-zero in fg).
            # If zero Gaussians hit the mask, skip geometric filtering and rely
            # on image-level masking alone.
            if geometric_mask.sum() > 0:
                valid_mask = valid_mask & geometric_mask
            else:
                logger.debug(
                    "Skipping geometric mask filter (no Gaussians in foreground). "
                    "Relying on image-level masking (background pixels zeroed)."
                )

        # Filter by valid mask
        means_tensor = means_tensor[valid_mask]
        scales_tensor = scales_tensor[valid_mask]
        rotations_tensor = rotations_tensor[valid_mask]
        opacities_tensor = opacities_tensor[valid_mask]
        colors_sh = colors_sh[valid_mask]

        return GaussianFrame(
            frame_idx=0,
            timestamp_ms=0.0,
            means=means_tensor.cpu().numpy().astype(np.float32),
            scales=scales_tensor.cpu().numpy().astype(np.float32),
            rotations=rotations_tensor.cpu().numpy().astype(np.float32),
            colors=colors_sh.cpu().numpy().astype(np.float32),
            opacities=opacities_tensor.cpu().numpy().astype(np.float32),
        )

    def _process_frame(
        self,
        img: np.ndarray,
        H: int,
        W: int,
        mask: np.ndarray | None = None,
    ) -> GaussianFrame:
        """Process a single image frame through UniSHARP.

        Args:
            img: RGB image as numpy array (H, W, 3), uint8 [0,255]
            H: Original image height
            W: Original image width
            mask: Optional boolean mask for background removal

        Returns:
            GaussianFrame with extracted Gaussians
        """
        # Preprocess: normalize and resize
        img_resized, img_u8 = self._preprocess_image(img)

        # Build camera intrinsics
        f_px = max(H, W) * 0.8
        intrinsics = self._build_camera_intrinsics(H, W, f_px)

        # Run UniSHARP inference
        with self._torch.no_grad():
            output = self.model(
                image=img_resized,
                image_u8=img_u8,
                camera_intrinsics=intrinsics.unsqueeze(0),
                camera_model=self.camera_model,
                distance_init_cap_m=self.distance_init_cap_m,
                return_aux=False,
            )

        # When return_aux=False, output is Gaussians3D directly (not a dict)
        gaussians = output

        # Convert to GaussianFrame
        return self._gaussians_to_frame(gaussians, H, W, mask, f_px)

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = True,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process frames using UniSHARP.

        Args:
            frame_paths: List of image file paths
            timestamps_ms: Corresponding timestamps in milliseconds
            per_frame: If True, process each frame individually for separate PLYs
            masks_dir: Optional directory containing mask images
            mask_first_frame: If False, skip masking the first frame

        Returns:
            List of GaussianFrame objects
        """
        import cv2
        from tqdm import tqdm

        # Silence UniSHARP logging noise
        logging.getLogger("unisharp").setLevel(logging.WARNING)
        logging.getLogger("unik3d").setLevel(logging.WARNING)

        self._load_model()

        stats = UniSHARPProfilingStats() if self.enable_profiling else None
        results: list[GaussianFrame] = []

        pbar = tqdm(total=len(frame_paths), desc="UniSHARP processing", unit="frame")

        try:
            for i, frame_path in enumerate(frame_paths):
                frame_start = time.perf_counter()

                # I/O: Load image
                io_start = time.perf_counter()
                img = cv2.imread(str(frame_path))
                if img is None:
                    logger.warning(f"Failed to load image: {frame_path}")
                    pbar.update(1)
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H, W = img.shape[:2]

                # Load mask if applicable
                mask = None
                mask_applied = False
                if masks_dir is not None and (mask_first_frame or i > 0):
                    mask_path = None
                    for ext in [frame_path.suffix, ".png", ".jpg", ".jpeg"]:
                        candidate = masks_dir / f"{frame_path.stem}{ext}"
                        if candidate.exists():
                            mask_path = candidate
                            break
                    if mask_path is not None:
                        mask_img = cv2.imread(str(mask_path))
                        if mask_img is not None:
                            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                            black_pixels = np.all(mask_img == 0, axis=2)
                            mask = ~black_pixels
                            mask_applied = True
                            img[black_pixels] = 0
                            logger.debug("Mask loaded: %s (%d/%d foreground pixels)",
                                mask_path.name, mask.sum(), mask.size)
                        else:
                            logger.warning("Failed to load mask: %s", mask_path)

                if stats:
                    stats.io_time += time.perf_counter() - io_start

                # Inference + postprocess
                infer_start = time.perf_counter()
                frame = self._process_frame(img, H, W, mask if mask_applied else None)
                frame.frame_idx = i
                frame.timestamp_ms = timestamps_ms[i]

                if stats:
                    stats.inference_time += time.perf_counter() - infer_start

                results.append(frame)

                frame_time = time.perf_counter() - frame_start
                if stats:
                    stats.frame_times.append(frame_time)
                    stats.total_frames += 1

                pbar.set_postfix(file=frame_path.name)
                pbar.update(1)

        finally:
            pbar.close()

        if stats:
            stats.log_summary()

        return results

