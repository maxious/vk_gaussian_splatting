"""InfiniSplat Gaussian Processor - single image to 3D Gaussians.

Uses the InfiniSplat model (zju3dv/InfiniSplat, SIGGRAPH Asia 2026) for
monocular or depth-sensor-guided 3D Gaussian reconstruction.

The InfiniSplat source is expected at ``python/third_party/infinisplat/``
(added as a git submodule from https://github.com/zju3dv/InfiniSplat.git).
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)

INFINISPLAT_REPO = "PLUS-WAVE/InfiniSplat"
INFINISPLAT_RGB_CKPT = "checkpoints/infinisplat_rgb.ckpt"
INFINISPLAT_LIDAR_CKPT = "checkpoints/infinisplat_lidar.ckpt"
INFERENCE_HEIGHT = 1152
INFERENCE_WIDTH = 1536
DEFAULT_FOCAL_35MM_MM = 30.0
SH_C0 = 0.28209479177387814


def _convert_rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def _linear_rgb_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.clamp(min=0.0).pow(1.0 / 2.4) - 0.055,
    )


def _canonicalize_quaternions(q: torch.Tensor) -> torch.Tensor:
    idx = q.abs().argmax(dim=-1, keepdim=True)
    sign = torch.gather(q, -1, idx).sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign


def _decompose_covariance_to_quat_scale(
    cov: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose 3x3 cov matrix to quaternions (wxyz) and singular values.

    Uses scipy on CPU (non-differentiable, matching InfiniSplat's own approach).
    """
    from scipy.spatial.transform import Rotation

    device, dtype = cov.device, cov.dtype
    cov_np = cov.detach().cpu().to(torch.float64).reshape(-1, 3, 3)
    cov_np = 0.5 * (cov_np + cov_np.transpose(-1, -2))

    eigvals, eigvecs = torch.linalg.eigh(cov_np)
    sort_idx = torch.argsort(eigvals, dim=-1, descending=True)
    eigvals = torch.gather(eigvals, -1, sort_idx)
    eigvecs = torch.gather(eigvecs, -1, sort_idx.unsqueeze(-2).expand(-1, 3, 3))
    eigvals = eigvals.clamp_min(1e-12)
    singular_values = eigvals.sqrt().to(dtype=dtype, device=device)

    R_np = eigvecs.numpy().reshape(-1, 3, 3)
    det_r = np.linalg.det(R_np)
    R_np[det_r < 0, :, -1] *= -1

    quat_np = Rotation.from_matrix(R_np).as_quat()
    quat_np = quat_np[:, [3, 0, 1, 2]]

    q = torch.from_numpy(quat_np).to(dtype=dtype, device=device).reshape(*cov.shape[:-2], 4)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = _canonicalize_quaternions(q)
    return q, singular_values


def _convert_focal_mm_to_px(width: float, height: float, focal_mm: float) -> float:
    return focal_mm * math.sqrt(width**2 + height**2) / math.sqrt(36.0**2 + 24.0**2)


def _extract_exif_focal_px(image: Image.Image) -> float | None:
    try:
        exif = image.getexif()
        focal_35mm = None
        ifdg = exif.get_ifd(0x8769)
        if ifdg:
            from PIL.ExifTags import TAGS

            tag_map = {v: k for k, v in TAGS.items()}
            if "FocalLengthIn35mmFilm" in tag_map:
                fid = tag_map["FocalLengthIn35mmFilm"]
                if fid in ifdg:
                    val = ifdg[fid]
                    if isinstance(val, tuple) and len(val) == 2 and val[1] != 0:
                        focal_35mm = float(val[0]) / float(val[1])
                    else:
                        try:
                            focal_35mm = float(val)
                        except (TypeError, ValueError):
                            pass

        if focal_35mm is None or focal_35mm < 1.0:
            if "FocalLength" in exif:
                val = exif["FocalLength"]
                if isinstance(val, tuple) and len(val) == 2 and val[1] != 0:
                    focal_mm = float(val[0]) / float(val[1])
                else:
                    try:
                        focal_mm = float(val)
                    except (TypeError, ValueError):
                        return None
                focal_35mm = focal_mm * 8.4 if focal_mm < 10.0 else focal_mm

        if focal_35mm is not None and focal_35mm >= 1.0:
            return _convert_focal_mm_to_px(image.width, image.height, focal_35mm)
    except Exception:
        pass
    return None


def _resolve_intrinsics(
    image: Image.Image,
    focal_length_px: float | None = None,
) -> torch.Tensor:
    w, h = image.size
    if focal_length_px is not None:
        f_px = focal_length_px
    else:
        exif = _extract_exif_focal_px(image)
        f_px = exif if exif is not None else _convert_focal_mm_to_px(w, h, DEFAULT_FOCAL_35MM_MM)
    K = torch.tensor(
        [[f_px, 0.0, (w - 1) / 2.0], [0.0, f_px, (h - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return K


def _scale_intrinsics(
    K: torch.Tensor,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> torch.Tensor:
    s = K.clone()
    s[0] *= float(dst_w) / float(src_w)
    s[1] *= float(dst_h) / float(src_h)
    return s


def _download_checkpoint(ckpt_name: str, cache_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "hub"
    local = hf_hub_download(
        repo_id=INFINISPLAT_REPO,
        filename=ckpt_name,
        cache_dir=cache_dir,
    )
    return Path(local)


def _find_infinisplat_src() -> Path | None:
    candidates = [
        Path(__file__).parent.parent.parent / "third_party" / "infinisplat" / "src",
        Path(__file__).parent.parent.parent / "Infinisplat" / "src",
        Path(__file__).parent.parent.parent / "infinisplat" / "src",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return None


def _import_infinisplat():
    src_path = _find_infinisplat_src()
    if src_path:
        parent = str(src_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from src.model.encoder import get_encoder
        from src.demo.config import RootCfg, load_typed_root_config

        return get_encoder, RootCfg, load_typed_root_config, src_path
    try:
        from infinisplat.model.encoder import get_encoder
        from infinisplat.demo.config import RootCfg, load_typed_root_config

        return get_encoder, RootCfg, load_typed_root_config, None
    except ImportError:
        raise ImportError(
            "InfiniSplat source not found. Clone it to python/third_party/infinisplat/:\n"
            "  git clone https://github.com/zju3dv/InfiniSplat.git python/third_party/infinisplat"
        )


class InfiniSplatGaussianProcessor(GaussianProcessor):
    """Generate 3D Gaussians from single images using InfiniSplat.

    Supports two modes:
      - rgb: RGB-only monocular reconstruction (default)
      - lidar: Depth-sensor-guided reconstruction with prompt depth

    Model checkpoints are auto-downloaded from HuggingFace on first use.
    The InfiniSplat source must be available at ``python/third_party/infinisplat/``
    (refer to AGENTS.md for submodule setup).
    """

    MODE_EXPERIMENTS = {
        "rgb": "infinisplat_hypersim_rgb",
        "lidar": "infinisplat_hypersim_lidar",
    }
    MODE_CHECKPOINTS = {
        "rgb": INFINISPLAT_RGB_CKPT,
        "lidar": INFINISPLAT_LIDAR_CKPT,
    }

    def __init__(
        self,
        *,
        device: str = "cuda",
        mode: str = "rgb",
        focal_length_px: float | None = None,
        disable_floater_filter: bool = False,
        checkpoint_path: str | Path | None = None,
        weights_cache_dir: Path | None = None,
    ):
        if mode not in self.MODE_EXPERIMENTS:
            raise ValueError(
                f"InfiniSplat mode must be one of {list(self.MODE_EXPERIMENTS.keys())}, got {mode!r}"
            )
        self._device = device
        self._mode = mode
        self._focal_length_px = focal_length_px
        self._disable_floater_filter = disable_floater_filter
        self._checkpoint_path = checkpoint_path
        self._weights_cache_dir = weights_cache_dir
        self._encoder = None
        self._device_torch = torch.device(device)

    def _ensure_model(self):
        if self._encoder is not None:
            return

        get_encoder, RootCfg, load_typed_root_config, src_path = _import_infinisplat()

        config_dir = src_path.parent / "config" if src_path else None

        experiment = self.MODE_EXPERIMENTS[self._mode]
        ckpt_name = self.MODE_CHECKPOINTS[self._mode]

        checkpoint_path: Path
        if self._checkpoint_path:
            checkpoint_path = Path(self._checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"InfiniSplat checkpoint not found: {checkpoint_path}")
        else:
            ckpt_file = _download_checkpoint(ckpt_name, cache_dir=self._weights_cache_dir)
            checkpoint_path = ckpt_file

        if config_dir and config_dir.is_dir():
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                cfg_dict = compose(config_name="inference", overrides=[f"+experiment={experiment}"])
            cfg = load_typed_root_config(cfg_dict)
        else:
            raise RuntimeError(
                "InfiniSplat config directory not found. "
                "Ensure the InfiniSplat repo is cloned to python/third_party/infinisplat/"
            )

        logger.info("Loading InfiniSplat %s encoder from %s", self._mode, checkpoint_path)
        encoder = get_encoder(cfg.model.encoder)

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        encoder_state = {
            k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        if not encoder_state:
            raise KeyError("No encoder weights found in checkpoint (expected 'encoder.' prefix).")
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
        if missing or unexpected:
            logger.warning(
                "Encoder state dict mismatch: missing=%d, unexpected=%d",
                len(missing),
                len(unexpected),
            )

        encoder = encoder.to(self._device_torch)
        encoder.eval()
        self._encoder = encoder
        logger.info("InfiniSplat encoder loaded successfully")

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = True,
        **kwargs,
    ) -> list[GaussianFrame]:
        self._ensure_model()
        frames: list[GaussianFrame] = []
        for idx, (path, ts) in enumerate(zip(frame_paths, timestamps_ms)):
            logger.info(
                "InfiniSplat processing frame %d/%d: %s", idx + 1, len(frame_paths), path.name
            )
            frame = self._process_single(path, idx, ts)
            frames.append(frame)
        return frames

    def _process_single(
        self,
        image_path: Path,
        frame_idx: int,
        timestamp_ms: float,
    ) -> GaussianFrame:
        assert self._encoder is not None

        img_pil = Image.open(image_path).convert("RGB")
        img_pil = ImageOps.exif_transpose(img_pil)
        orig_w, orig_h = img_pil.size

        orig_K = _resolve_intrinsics(img_pil, focal_length_px=self._focal_length_px)

        inf_h, inf_w = INFERENCE_HEIGHT, INFERENCE_WIDTH
        if (orig_h, orig_w) != (inf_h, inf_w):
            img_inf = img_pil.resize((inf_w, inf_h), Image.Resampling.BILINEAR)
            inf_K = _scale_intrinsics(orig_K, orig_w, orig_h, inf_w, inf_h)
        else:
            img_inf = img_pil
            inf_K = orig_K.clone()

        img_tensor = torch.from_numpy(np.array(img_inf, dtype=np.float32) / 255.0).permute(2, 0, 1)

        intrinsics_norm = inf_K.clone()
        intrinsics_norm[0] /= inf_w
        intrinsics_norm[1] /= inf_h

        device = self._device_torch
        context = {
            "image": img_tensor.unsqueeze(0).unsqueeze(0).to(device),
            "intrinsics": intrinsics_norm.unsqueeze(0).unsqueeze(0).to(device),
            "extrinsics": torch.eye(4, device=device).unsqueeze(0).unsqueeze(0),
        }

        with torch.inference_mode():
            output = self._encoder(context)

        gaussians = output["gaussians"]

        if not self._disable_floater_filter:
            gaussians = self._filter_floaters(gaussians)

        means = gaussians.mean_vectors[0]
        opacities = gaussians.opacities[0]
        colors_linear = gaussians.colors[0].clamp(0.0, 1.0)

        if gaussians.covariances is not None:
            quats, scales = _decompose_covariance_to_quat_scale(gaussians.covariances[0])
        else:
            quats = gaussians.quaternions[0]
            quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            quats = _canonicalize_quaternions(quats)
            scales = gaussians.singular_values[0].clamp_min(1e-8)

        colors_sh = _convert_rgb_to_sh(_linear_rgb_to_srgb(colors_linear))

        return GaussianFrame(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            means=means.detach().cpu().numpy().astype(np.float32),
            scales=scales.detach().cpu().numpy().astype(np.float32),
            rotations=quats.detach().cpu().numpy().astype(np.float32),
            colors=colors_sh.detach().cpu().numpy().astype(np.float32),
            opacities=opacities.detach().cpu().numpy().astype(np.float32),
            intrinsic=orig_K.detach().cpu().numpy().astype(np.float32),
            extrinsic=np.eye(4, dtype=np.float32),
            image_size=(orig_h, orig_w),
        )

    @torch.inference_mode()
    def _filter_floaters(self, gaussians: Any) -> Any:
        """Remove spatial outlier Gaussians using kNN distance threshold."""
        from scipy.spatial import cKDTree

        points = gaussians.mean_vectors[0]
        candidate = torch.isfinite(points).all(dim=-1)
        idx = torch.nonzero(candidate).flatten()
        if idx.numel() < 2:
            return gaussians

        pts_np = points[idx].detach().float().cpu().numpy()
        k = min(16, len(pts_np) - 1)
        tree = cKDTree(pts_np)
        dists, _ = tree.query(pts_np, k=k + 1, workers=-1)
        mean_d = dists[:, 1:].mean(axis=1)
        thresh = float(mean_d.mean() + 2.5 * mean_d.std())
        keep_np = mean_d <= thresh

        keep = torch.zeros(points.shape[0], device=points.device, dtype=torch.bool)
        keep[idx[torch.from_numpy(keep_np).to(device=points.device)]] = True
        if not keep.any():
            return gaussians

        from types import SimpleNamespace

        g = SimpleNamespace()
        g.mean_vectors = gaussians.mean_vectors[:, keep]
        g.singular_values = gaussians.singular_values[:, keep]
        g.quaternions = gaussians.quaternions[:, keep]
        g.colors = gaussians.colors[:, keep]
        g.opacities = gaussians.opacities[:, keep]
        if gaussians.covariances is not None:
            g.covariances = gaussians.covariances[:, keep]
        else:
            g.covariances = None
        return g


def parse_infinisplat_model_id(model_id: str) -> dict:
    """Parse InfiniSplat model_id string into configuration parameters.

    Formats:
        infinisplat                 → rgb mode, default focal
        infinisplat:lidar           → lidar (depth-sensor) mode
        infinisplat:rgb:focal=1200  → rgb mode, focal length 1200px
        infinisplat:lidar:no_filter → lidar mode, skip floater filter
    """
    config = {
        "mode": "rgb",
        "focal_length_px": None,
        "disable_floater_filter": False,
        "checkpoint_path": None,
    }

    if ":" not in model_id:
        return config

    parts = model_id.split(":")[1:]
    # First positional part is the mode
    if parts and parts[0] in ("rgb", "lidar"):
        config["mode"] = parts[0]
        parts = parts[1:]

    for part in parts:
        if part == "no_filter":
            config["disable_floater_filter"] = True
        elif part.startswith("focal="):
            config["focal_length_px"] = float(part.split("=", 1)[1])
        elif part.startswith("ckpt="):
            config["checkpoint_path"] = part.split("=", 1)[1]
        else:
            logger.warning("Unknown InfiniSplat config part: %s", part)

    return config
