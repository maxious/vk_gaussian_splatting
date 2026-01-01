"""Export video frames to Gaussian Splatting PLY files using DA3.

Supports two modes:
1. Per-frame static PLY files (one PLY per frame or chunk)
2. FreeTimeGS PLY with temporal parameters (motion vectors, time center, time scale)

Usage:
    # Export per-frame static PLYs:
    python -m offline.export_gaussian_ply --input video.mp4 --output ./gaussians/ --mode frames

    # Export single FreeTimeGS PLY with motion:
    python -m offline.export_gaussian_ply --input video.mp4 --output scene.ply --mode freetimegs

Requires DA3-GIANT model with infer_gs=True for Gaussian output.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


@dataclass
class GaussianFrame:
    """Gaussian splat data for a single frame."""

    frame_idx: int
    timestamp_ms: float
    means: np.ndarray  # (N, 3) positions
    scales: np.ndarray  # (N, 3) log-scale
    rotations: np.ndarray  # (N, 4) quaternion wxyz
    colors: np.ndarray  # (N, 3) SH DC term (f_dc)
    opacities: np.ndarray  # (N,) logit opacity


def write_static_gaussian_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    sh_rest: np.ndarray | None = None,
    flip_y: bool = False,
) -> None:
    """Write a static 3DGS PLY file using plyfile.

    Args:
        path: Output PLY file path
        means: (N, 3) float32 positions
        scales: (N, 3) float32 log-scales
        rotations: (N, 4) float32 quaternions (wxyz order)
        colors: (N, 3) float32 SH DC coefficients
        opacities: (N,) float32 logit opacities
        sh_rest: Optional (N, 45) float32 higher-order SH coefficients
        flip_y: If True, negate Y coordinates to flip the coordinate system
    """
    n_points = len(means)

    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        n_sh = sh_rest.shape[1]
        for i in range(n_sh):
            dtype_list.append((f"f_rest_{i}", "f4"))

    dtype_list.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )

    elements = np.empty(n_points, dtype=dtype_list)

    elements["x"] = means[:, 0]
    elements["y"] = -means[:, 1] if flip_y else means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["f_dc_0"] = colors[:, 0]
    elements["f_dc_1"] = colors[:, 1]
    elements["f_dc_2"] = colors[:, 2]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        for i in range(sh_rest.shape[1]):
            elements[f"f_rest_{i}"] = sh_rest[:, i]

    elements["opacity"] = opacities
    elements["scale_0"] = scales[:, 0]
    elements["scale_1"] = scales[:, 1]
    elements["scale_2"] = scales[:, 2]
    elements["rot_0"] = rotations[:, 0]
    elements["rot_1"] = rotations[:, 1]
    elements["rot_2"] = rotations[:, 2]
    elements["rot_3"] = rotations[:, 3]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=False).write(str(path))

    logger.info(f"Wrote {n_points} Gaussians to {path}")


def write_freetimegs_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    motion: np.ndarray,
    time_center: np.ndarray,
    time_scale: np.ndarray,
    sh_rest: np.ndarray | None = None,
    flip_y: bool = False,
) -> None:
    """Write a FreeTimeGS PLY file with temporal parameters using plyfile.

    The temporal model is:
        position(t) = mean + motion * (t - time_center)
        opacity(t) = opacity * exp(-0.5 * ((t - time_center) / time_scale)^2)

    Args:
        flip_y: If True, negate Y coordinates to flip the coordinate system
        path: Output PLY file path
        means: (N, 3) float32 positions at time_center
        scales: (N, 3) float32 log-scales
        rotations: (N, 4) float32 quaternions (wxyz order)
        colors: (N, 3) float32 SH DC coefficients
        opacities: (N,) float32 logit opacities
        motion: (N, 3) float32 velocity vectors
        time_center: (N,) float32 temporal center (0-1 normalized)
        time_scale: (N,) float32 temporal width (log-space, will be exp'd by viewer)
        sh_rest: Optional (N, 45) float32 higher-order SH coefficients
    """
    n_points = len(means)

    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        for i in range(sh_rest.shape[1]):
            dtype_list.append((f"f_rest_{i}", "f4"))

    dtype_list.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("motion_0", "f4"),
            ("motion_1", "f4"),
            ("motion_2", "f4"),
            ("t", "f4"),
            ("t_scale", "f4"),
        ]
    )

    elements = np.empty(n_points, dtype=dtype_list)

    elements["x"] = means[:, 0]
    elements["y"] = -means[:, 1] if flip_y else means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["f_dc_0"] = colors[:, 0]
    elements["f_dc_1"] = colors[:, 1]
    elements["f_dc_2"] = colors[:, 2]

    if sh_rest is not None and sh_rest.shape[1] > 0:
        for i in range(sh_rest.shape[1]):
            elements[f"f_rest_{i}"] = sh_rest[:, i]

    elements["opacity"] = opacities
    elements["scale_0"] = scales[:, 0]
    elements["scale_1"] = scales[:, 1]
    elements["scale_2"] = scales[:, 2]
    elements["rot_0"] = rotations[:, 0]
    elements["rot_1"] = rotations[:, 1]
    elements["rot_2"] = rotations[:, 2]
    elements["rot_3"] = rotations[:, 3]
    elements["motion_0"] = motion[:, 0]
    elements["motion_1"] = motion[:, 1]
    elements["motion_2"] = motion[:, 2]
    elements["t"] = time_center
    elements["t_scale"] = time_scale

    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=False).write(str(path))

    logger.info(f"Wrote {n_points} FreeTimeGS Gaussians to {path}")


class DA3GaussianProcessor:
    """Process video frames to Gaussian splats using DA3."""

    def __init__(
        self,
        model_id: str = "depth-anything/DA3-GIANT",
        device: str = "cuda",
        process_res: int = 518,
    ):
        self.model_id = model_id
        self.device = device
        self.process_res = process_res
        self.model = None

    def _load_model(self):
        """Lazy load the DA3 model."""
        if self.model is not None:
            return

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for DA3 inference")

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            raise RuntimeError(
                "depth-anything-3 not installed. Run: uv pip install depth-anything-3"
            )

        logger.info(f"Loading model: {self.model_id} (this may take a while to download...)")
        logger.info("Model files are cached in ~/.cache/huggingface/hub/")

        import huggingface_hub

        huggingface_hub.logging.set_verbosity_info()

        self.model = DepthAnything3.from_pretrained(self.model_id)
        logger.info(f"Model loaded, moving to {self.device}...")
        self.model = self.model.to(self.device).eval()
        self.dtype = torch.float16
        logger.info("Model ready for inference")

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
            List of GaussianFrame objects (one per frame if per_frame=True,
            otherwise one merged frame)
        """
        import torch

        self._load_model()

        if per_frame:
            return self._process_frames_individually(frame_paths, timestamps_ms)
        else:
            return self._process_frames_merged(frame_paths, timestamps_ms)

    def _process_frames_merged(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
    ) -> list[GaussianFrame]:
        """Process all frames together, returning merged Gaussians."""
        import torch

        logger.info(f"Processing {len(frame_paths)} frames merged with infer_gs=True")

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                images = [str(p) for p in frame_paths]

                predictions = self.model.inference(
                    images,
                    process_res=self.process_res,
                    ref_view_strategy="saddle_balanced",
                    infer_gs=True,
                )

        if predictions.gaussians is None:
            raise RuntimeError(
                "Model did not return Gaussians. "
                "Make sure you're using DA3-GIANT with infer_gs=True"
            )

            gaussians = predictions.gaussians

            means = gaussians.means[0].cpu().numpy()
            scales = gaussians.scales[0].cpu().numpy()

            # Necessary comment: Convert linear scales (0+) to log scales (-inf, +inf) for PLY format compatibility
            if np.all(scales > 0):
                scales = np.log(np.maximum(scales, 1e-10))

            rotations = gaussians.rotations[0].cpu().numpy()
            opacities = gaussians.opacities[0].cpu().numpy()

        harmonics = gaussians.harmonics[0].cpu().numpy()
        colors = harmonics[:, :, 0] if harmonics.ndim == 3 else harmonics

        if opacities.ndim == 2:
            opacities = opacities[:, 0]

        mid_ts = timestamps_ms[len(timestamps_ms) // 2] if timestamps_ms else 0

        return [
            GaussianFrame(
                frame_idx=0,
                timestamp_ms=mid_ts,
                means=means.astype(np.float32),
                scales=scales.astype(np.float32),
                rotations=rotations.astype(np.float32),
                colors=colors.astype(np.float32),
                opacities=opacities.astype(np.float32),
            )
        ]


class SharpGaussianProcessor:
    """Process images to Gaussian splats using Apple's SHARP model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        vit_preset: str = "dinov2l16_384",
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.vit_preset = vit_preset
        self.predictor = None

    def _load_model(self):
        if self.predictor is not None:
            return

        import torch
        import ssl
        from sharp.models import create_predictor, PredictorParams

        # Disable SSL verification for model download if needed
        ssl._create_default_https_context = ssl._create_unverified_context

        logger.info(f"Initializing SHARP model with preset: {self.vit_preset}...")
        params = PredictorParams()

        # Configure model backbone preset
        params.monodepth.patch_encoder_preset = self.vit_preset
        params.monodepth.image_encoder_preset = self.vit_preset
        params.gaussian_decoder.patch_encoder_preset = self.vit_preset
        params.gaussian_decoder.image_encoder_preset = self.vit_preset

        self.predictor = create_predictor(params)

        if self.model_path and self.model_path.exists():
            logger.info(f"Loading SHARP weights from local file: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location="cpu")
        else:
            # Check default local resource location
            # Assuming script is run from project root or python dir, try relative paths
            local_resource_path = Path("_downloaded_resources/sharp/sharp_2572gikvuh.pt")
            if not local_resource_path.exists():
                # Try relative to python/ directory if running from there
                local_resource_path = Path("../_downloaded_resources/sharp/sharp_2572gikvuh.pt")

            if local_resource_path.exists():
                logger.info(
                    f"Loading SHARP weights from default local resource: {local_resource_path}"
                )
                state_dict = torch.load(local_resource_path, map_location="cpu")
            else:
                url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
                logger.info(f"Downloading SHARP weights from {url}...")
                state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

        self.predictor.load_state_dict(state_dict)
        self.predictor.eval().to(self.device)
        logger.info("SHARP model ready")

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = True,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """Process frames using SHARP."""
        import torch
        import cv2
        from sharp.cli.predict import predict_image
        from sharp.utils import color_space as cs_utils
        from tqdm import tqdm

        self._load_model()
        results = []

        # Silence SHARP CLI logs
        logging.getLogger("sharp.cli.predict").setLevel(logging.WARNING)

        # Constants from SHARP utils
        SH_C0 = 0.28209479177387814

        pbar = tqdm(
            zip(frame_paths, timestamps_ms),
            total=len(frame_paths),
            desc="SHARP processing",
            unit="frame",
        )
        for i, (path, ts) in enumerate(pbar):
            pbar.set_postfix(file=path.name)

            # Load image
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Failed to load image: {path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]

            # Apply mask if provided (skip first frame if requested)
            if masks_dir is not None and (mask_first_frame or i > 0):
                mask_path = None
                for ext in [path.suffix, ".png", ".jpg", ".jpeg"]:
                    candidate = masks_dir / f"{path.stem}{ext}"
                    if candidate.exists():
                        mask_path = candidate
                        break
                if mask_path is not None:
                    mask = cv2.imread(str(mask_path))
                    if mask is not None:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                        # Where mask is black (background), set image to black
                        black_pixels = np.all(mask == 0, axis=2)
                        img[black_pixels] = 0
                        n_masked = black_pixels.sum()
                        logger.info(f"Masked {n_masked} background pixels in {path.name}")
                    else:
                        logger.warning(f"Failed to load mask: {mask_path}")
                else:
                    logger.warning(f"Mask not found for {path.stem} in {masks_dir}")

            # SHARP expects focal length in pixels.
            # Default to FOV ~60 degrees if unknown (f ~= W) or use 500 like demo
            # Ideally we should estimate from EXIF or use a reasonable default.
            # Using max(H, W) is a safe bet for "normal" lenses.
            f_px = max(H, W) * 0.8

            with torch.no_grad():
                gaussians = predict_image(self.predictor, img, f_px, device=self.device)

            # --- Convert to standard 3DGS format ---

            # 1. Means (already in world space relative to camera)
            means = gaussians.mean_vectors.squeeze(0).cpu().numpy()

            # 2. Scales: Linear -> Log
            # SHARP outputs linear scales (singular values)
            scales_linear = gaussians.singular_values.squeeze(0)
            scales = torch.log(torch.clamp(scales_linear, min=1e-8)).cpu().numpy()

            # 3. Rotations: Quaternions (already normalized usually)
            rotations = gaussians.quaternions.squeeze(0).cpu().numpy()

            # 4. Opacities: Prob -> Logit
            # SHARP outputs probabilities (0-1)
            opacities_prob = gaussians.opacities.squeeze(0)
            # Inverse sigmoid: log(p / (1-p))
            opacities_prob = torch.clamp(opacities_prob, 1e-6, 1.0 - 1e-6)
            opacities = torch.log(opacities_prob / (1.0 - opacities_prob)).cpu().numpy()

            # 5. Colors: Linear RGB -> sRGB -> SH DC
            # SHARP predicts Linear RGB.
            # Apple's save_ply converts Linear->sRGB before saving.
            # We follow this to match the "ML-SHARP" color profile in our viewer.
            colors_linear = gaussians.colors.squeeze(0)
            colors_srgb = cs_utils.linearRGB2sRGB(colors_linear)

            # Convert RGB to SH DC (degree 0)
            # sh = (rgb - 0.5) / C0
            colors_sh = (colors_srgb - 0.5) / SH_C0
            colors = colors_sh.cpu().numpy()

            # Filter out black/background splats (SHARP prefers black background)
            if remove_black_splats:
                valid_mask = np.any(colors_linear.cpu().numpy() > 0.01, axis=1)
                n_removed = len(means) - valid_mask.sum()
                if n_removed > 0:
                    logger.info(f"Removed {n_removed} black/background splats from frame {i}")
                    means = means[valid_mask]
                    scales = scales[valid_mask]
                    rotations = rotations[valid_mask]
                    colors = colors[valid_mask]
                    opacities = opacities[valid_mask]

            results.append(
                GaussianFrame(
                    frame_idx=i,
                    timestamp_ms=ts,
                    means=means.astype(np.float32),
                    scales=scales.astype(np.float32),
                    rotations=rotations.astype(np.float32),
                    colors=colors.astype(np.float32),
                    opacities=opacities.astype(np.float32),
                )
            )

        return results

    def _process_frames_individually(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
    ) -> list[GaussianFrame]:
        """Process each frame individually for per-frame PLY output."""
        import torch

        results = []

        for i, (frame_path, ts) in enumerate(zip(frame_paths, timestamps_ms)):
            logger.info(f"Processing frame {i + 1}/{len(frame_paths)}: {frame_path.name}")

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    predictions = self.model.inference(
                        [str(frame_path)],
                        process_res=self.process_res,
                        ref_view_strategy="first",
                        infer_gs=True,
                    )

            if predictions.gaussians is None:
                logger.warning(f"Frame {i} did not return Gaussians, skipping")
                continue

            gaussians = predictions.gaussians

            means = gaussians.means[0].cpu().numpy()
            scales = gaussians.scales[0].cpu().numpy()

            # Necessary comment: Convert linear scales (0+) to log scales (-inf, +inf) for PLY format compatibility
            if np.all(scales > 0):
                scales = np.log(np.maximum(scales, 1e-10))

            rotations = gaussians.rotations[0].cpu().numpy()
            opacities = gaussians.opacities[0].cpu().numpy()

            harmonics = gaussians.harmonics[0].cpu().numpy()
            colors = harmonics[:, :, 0] if harmonics.ndim == 3 else harmonics

            if opacities.ndim == 2:
                opacities = opacities[:, 0]

            results.append(
                GaussianFrame(
                    frame_idx=i,
                    timestamp_ms=ts,
                    means=means.astype(np.float32),
                    scales=scales.astype(np.float32),
                    rotations=rotations.astype(np.float32),
                    colors=colors.astype(np.float32),
                    opacities=opacities.astype(np.float32),
                )
            )

        return results


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    frame_skip: int = 1,
    max_frames: int | None = None,
) -> tuple[list[Path], list[float]]:
    """Extract frames from video.

    Returns:
        (frame_paths, timestamps_ms)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_paths = []
    timestamps_ms = []
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            if max_frames is not None and saved >= max_frames:
                break

            frame_path = output_dir / f"frame_{saved:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            timestamps_ms.append(idx * 1000.0 / fps)
            saved += 1

            if saved % 50 == 0:
                logger.info(f"Extracted {saved} frames")

        idx += 1

    cap.release()
    logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
    return frame_paths, timestamps_ms





def prune_gaussian_frame(frame: GaussianFrame, opacity_threshold: float = 0.05) -> GaussianFrame:
    """Prune Gaussians with low opacity.

    Args:
        frame: GaussianFrame to prune
        opacity_threshold: Minimum sigmoid(opacity) to keep

    Returns:
        New GaussianFrame with pruned data
    """
    # Opacities are logits (inverse sigmoid)
    # sigmoid(x) = 1 / (1 + exp(-x))
    # we want sigmoid(op) > threshold
    # 1 / (1 + exp(-op)) > threshold
    # 1 + exp(-op) < 1/threshold
    # exp(-op) < 1/threshold - 1
    # -op < ln(1/threshold - 1)
    # op > -ln(1/threshold - 1) = ln(threshold / (1-threshold)) if threshold < 1

    # Or just use scipy/numpy sigmoid if available, but doing it on logits is fine
    # Let's compute actual probability to be safe and clear
    probs = 1.0 / (1.0 + np.exp(-frame.opacities))
    mask = probs > opacity_threshold

    n_before = len(frame.means)
    n_after = np.sum(mask)

    if n_after < n_before:
        logger.info(
            f"Pruned frame {frame.frame_idx}: {n_before} -> {n_after} splats (threshold {opacity_threshold})"
        )

    return GaussianFrame(
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        means=frame.means[mask],
        scales=frame.scales[mask],
        rotations=frame.rotations[mask],
        colors=frame.colors[mask],
        opacities=frame.opacities[mask],
    )


def export_video_to_gaussian_plys(
    video_path: Path,
    output_path: Path,
    mode: str = "frames",
    model_id: str = "depth-anything/DA3-GIANT",
    frame_skip: int = 5,
    chunk_size: int = 10,
    max_frames: int | None = None,
    device: str = "cuda",
    process_res: int = 518,
    opacity_threshold: float = 0.0,
    flip_y: bool = False,
) -> None:
    """Convert video to Gaussian PLY files.

    Args:
        video_path: Input video file
        output_path: Output path (directory for 'frames' mode, file for 'freetimegs')
        mode: 'frames' for per-frame PLYs, 'freetimegs' for single temporal PLY
        model_id: DA3 model ID (must support infer_gs=True)
        frame_skip: Process every Nth frame
        chunk_size: Number of frames to process together
        max_frames: Maximum frames to process (None for all)
        device: PyTorch device
        process_res: Processing resolution for DA3
    """
    temp_dir = (
        output_path.parent / "temp_frames" if mode == "freetimegs" else output_path / "temp_frames"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, timestamps_ms = extract_video_frames(
        video_path, temp_dir, frame_skip=frame_skip, max_frames=max_frames
    )

    if not frame_paths:
        raise ValueError("No frames extracted from video")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if "sharp" in model_id.lower():
        # Parse SHARP model configuration
        model_path = None
        vit_preset = "dinov2l16_384"  # Default

        if ":" in model_id:
            # Format: sharp:dinov3l16_384 or sharp:/path/to/model.pt
            parts = model_id.split(":", 1)
            config_part = parts[1]

            if Path(config_part).exists() or "\\" in config_part or "/" in config_part:
                model_path = config_part
            else:
                vit_preset = config_part

        processor = SharpGaussianProcessor(
            model_path=model_path, device=device, vit_preset=vit_preset
        )
    else:
        processor = DA3GaussianProcessor(
            model_id=model_id,
            device=device,
            process_res=process_res,
        )

    all_frames: list[GaussianFrame] = []

    for chunk_start in range(0, len(frame_paths), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(frame_paths))
        chunk_paths = frame_paths[chunk_start:chunk_end]
        chunk_timestamps = timestamps_ms[chunk_start:chunk_end]

        logger.info(
            f"Processing chunk {chunk_start // chunk_size + 1}: frames {chunk_start}-{chunk_end - 1}"
        )

        try:
            # For frames mode, process individually to get per-frame PLYs
            # For freetimegs mode, process merged for unified scene
            per_frame = mode == "frames"
            chunk_frames = processor.process_frames(
                chunk_paths, chunk_timestamps, per_frame=per_frame
            )

            for i, frame in enumerate(chunk_frames):
                if per_frame:
                    frame.frame_idx = chunk_start + i

                # Apply pruning if requested
                if opacity_threshold > 0:
                    frame = prune_gaussian_frame(frame, opacity_threshold)
                    # Update the list with pruned frame if needed, but we used chunk_frames in all_frames extend
                    # So we should update chunk_frames list
                    chunk_frames[i] = frame

            all_frames.extend(chunk_frames)

            if mode == "frames":
                output_path.mkdir(parents=True, exist_ok=True)
                for frame in chunk_frames:
                    ply_path = output_path / f"frame_{frame.frame_idx:06d}.ply"
                    write_static_gaussian_ply(
                        ply_path,
                        frame.means,
                        frame.scales,
                        frame.rotations,
                        frame.colors,
                        frame.opacities,
                    )
        except Exception as e:
            logger.error(f"Failed to process chunk: {e}")
            raise

    if mode == "freetimegs":
        from offline.motion_tracking_cpu import compute_motion_vectors

        logger.info("Computing motion vectors (CPU-accelerated with FAISS)...")
        (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
            compute_motion_vectors(all_frames, fps)
        )

    # Zero out motion for static splats (motion magnitude <= 0.001)
    motion_magnitude = np.linalg.norm(motion, axis=1)
    static_mask = motion_magnitude <= 0.001
    n_static = static_mask.sum()
    if n_static > 0:
        logger.info(f"Zeroing motion for {n_static} static splats (motion <= 0.001)")
        motion[static_mask] = 0.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
    write_freetimegs_ply(
        output_path,
        means,
        scales,
        rotations,
        colors,
        opacities,
        motion,
        time_center,
        time_scale,
        flip_y=flip_y,
    )

    logger.info(f"Export complete: {output_path}")


def load_static_gaussian_ply(path: Path) -> GaussianFrame:
    """Load a static 3DGS PLY file into a GaussianFrame.

    Args:
        path: Path to the PLY file

    Returns:
        GaussianFrame with loaded data (frame_idx/timestamp from filename if possible)
    """
    import re

    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        vertex_count = 0
        properties: list[str] = []
        in_vertex = False

        for line in header_lines:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element "):
                in_vertex = False
            elif in_vertex and line.startswith("property float "):
                properties.append(line.split()[-1])

        prop_to_idx = {p: i for i, p in enumerate(properties)}

        stride = len(properties)
        data = np.frombuffer(f.read(vertex_count * stride * 4), dtype=np.float32)
        data = data.reshape(vertex_count, stride)

    means = np.column_stack(
        [
            data[:, prop_to_idx["x"]],
            data[:, prop_to_idx["y"]],
            data[:, prop_to_idx["z"]],
        ]
    )

    colors = np.column_stack(
        [
            data[:, prop_to_idx["f_dc_0"]],
            data[:, prop_to_idx["f_dc_1"]],
            data[:, prop_to_idx["f_dc_2"]],
        ]
    )

    scales = np.column_stack(
        [
            data[:, prop_to_idx["scale_0"]],
            data[:, prop_to_idx["scale_1"]],
            data[:, prop_to_idx["scale_2"]],
        ]
    )

    rotations = np.column_stack(
        [
            data[:, prop_to_idx["rot_0"]],
            data[:, prop_to_idx["rot_1"]],
            data[:, prop_to_idx["rot_2"]],
            data[:, prop_to_idx["rot_3"]],
        ]
    )

    opacities = data[:, prop_to_idx["opacity"]]

    frame_idx = 0
    timestamp_ms = 0.0
    match = re.search(r"frame_(\d+)", path.stem)
    if match:
        frame_idx = int(match.group(1))
        timestamp_ms = frame_idx * 33.33

    return GaussianFrame(
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        rotations=rotations.astype(np.float32),
        colors=colors.astype(np.float32),
        opacities=opacities.astype(np.float32),
    )


def export_images_to_gaussian_plys(
    input_dir: Path,
    output_path: Path,
    mode: str = "frames",
    fps: float = 30.0,
    model_id: str = "depth-anything/DA3-GIANT",
    image_pattern: str = "*.jpg",
    max_frames: int | None = None,
    device: str = "cuda",
    process_res: int = 518,
    opacity_threshold: float = 0.0,
    masks_dir: Path | None = None,
    mask_first_frame: bool = True,
    remove_black_splats: bool = True,
    save_frequency: int = 5,
    flip_y: bool = False,
) -> None:
    """Process images with DA3 and export to Gaussian PLY files.

    Two-step pipeline:
    1. Run DA3 on each image to get per-frame Gaussians
    2. (Optional) Track Gaussians across frames and compute motion vectors for FreeTimeGS

    Args:
        input_dir: Directory containing input images
        output_path: Output path (directory for 'frames', file for 'freetimegs')
        mode: 'frames' for per-frame PLYs, 'freetimegs' for single temporal PLY
        fps: Assumed frame rate for temporal normalization
        model_id: DA3 model ID
        image_pattern: Glob pattern for images
        max_frames: Maximum frames to process
        device: PyTorch device
        process_res: Processing resolution for DA3
        opacity_threshold: Prune Gaussians with opacity below this threshold
        flip_y: If True, negate Y coordinates to flip the coordinate system
    """
    image_paths = sorted(input_dir.glob(image_pattern))
    if max_frames:
        image_paths = image_paths[:max_frames]

    if not image_paths:
        raise ValueError(f"No images found in {input_dir} matching '{image_pattern}'")

    logger.info(f"Found {len(image_paths)} images to process")

    timestamps_ms = [i * (1000.0 / fps) for i in range(len(image_paths))]

    if "sharp" in model_id.lower():
        # Heuristic: if model_id contains "sharp", use Sharp processor
        model_path = (
            model_id if Path(model_id).exists() or "\\" in model_id or "/" in model_id else None
        )
        if model_id.lower() == "sharp":
            model_path = None

        processor = SharpGaussianProcessor(model_path=model_path, device=device)
    else:
        processor = DA3GaussianProcessor(
            model_id=model_id,
            device=device,
            process_res=process_res,
        )

    # Process frames individually first
    if isinstance(processor, SharpGaussianProcessor):
        frames = processor.process_frames(
            image_paths,
            timestamps_ms,
            per_frame=True,
            masks_dir=masks_dir,
            mask_first_frame=mask_first_frame,
            remove_black_splats=remove_black_splats,
        )
    else:
        frames = processor.process_frames(image_paths, timestamps_ms, per_frame=True)

    if not frames:
        raise RuntimeError("No frames processed successfully")

    # Apply pruning if requested
    if opacity_threshold > 0:
        frames = [prune_gaussian_frame(f, opacity_threshold) for f in frames]

    logger.info(
        f"Processed {len(frames)} frames, total {sum(len(f.means) for f in frames)} Gaussians"
    )

    if mode == "frames":
        output_path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            ply_path = output_path / f"frame_{frame.frame_idx:06d}.ply"
            write_static_gaussian_ply(
                ply_path,
                frame.means,
                frame.scales,
                frame.rotations,
                frame.colors,
                frame.opacities,
                flip_y=flip_y,
            )
            if (i + 1) % save_frequency == 0:
                logger.info(f"Saved {i + 1}/{len(frames)} PLY files")
        logger.info(f"Exported {len(frames)} PLY files to {output_path}")
        return

    # FreeTimeGS mode
    from offline.motion_tracking_cpu import compute_motion_vectors

    logger.info("Computing motion vectors (CPU-accelerated with FAISS)...")
    (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
        compute_motion_vectors(frames, fps)
    )

    # Zero out motion for static splats (motion magnitude <= 0.001)
        motion_magnitude = np.linalg.norm(motion, axis=1)
        static_mask = motion_magnitude <= 0.001
        n_static = static_mask.sum()
        if n_static > 0:
            logger.info(f"Zeroing motion for {n_static} static splats (motion <= 0.001)")
            motion[static_mask] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_freetimegs_ply(
        output_path,
        means,
        scales,
        rotations,
        colors,
        opacities,
        motion,
        time_center,
        time_scale,
    )

    logger.info(f"Wrote FreeTimeGS PLY with {len(means)} Gaussians to {output_path}")


def postprocess_plys_to_freetimegs(
    input_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    max_match_distance: float = 0.05,
    ply_pattern: str = "frame_*.ply",
    flip_y: bool = False,
) -> None:
    """Postprocess existing per-frame PLY files to a single FreeTimeGS PLY.

    Loads all PLY files matching the pattern, tracks Gaussians across frames,
    computes motion vectors and temporal parameters, then outputs a single
    FreeTimeGS PLY file.

    Args:
        input_dir: Directory containing per-frame PLY files
        output_path: Output FreeTimeGS PLY file path
        fps: Assumed frame rate if not derivable from filenames
        max_match_distance: Maximum distance for matching Gaussians across frames
        ply_pattern: Glob pattern for PLY files
        flip_y: If True, negate Y coordinates to flip the coordinate system
    """
    import glob as glob_module

    ply_files = sorted(input_dir.glob(ply_pattern))

    if not ply_files:
        raise ValueError(f"No PLY files found in {input_dir} matching '{ply_pattern}'")

    logger.info(f"Found {len(ply_files)} PLY files to postprocess")

    frames: list[GaussianFrame] = []
    for i, ply_path in enumerate(ply_files):
        logger.info(f"Loading {i + 1}/{len(ply_files)}: {ply_path.name}")
        frame = load_static_gaussian_ply(ply_path)
        frame.frame_idx = i
        frame.timestamp_ms = i * (1000.0 / fps)
        frames.append(frame)

    logger.info(
        f"Loaded {len(frames)} frames, total {sum(len(f.means) for f in frames)} Gaussian observations"
    )

    from offline.motion_tracking_cpu import compute_motion_vectors

    logger.info("Computing motion vectors (CPU-accelerated with FAISS)...")
    (means, scales, rotations, colors, opacities, motion, time_center, time_scale) = (
        compute_motion_vectors(frames, fps, max_match_distance=max_match_distance)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_freetimegs_ply(
        output_path,
        means,
        scales,
        rotations,
        colors,
        opacities,
        motion,
        time_center,
        time_scale,
    )

    logger.info(f"Wrote FreeTimeGS PLY with {len(means)} Gaussians to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export video to Gaussian Splatting PLY files using DA3"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    export_parser = subparsers.add_parser("export", help="Export video to Gaussian PLYs")
    export_parser.add_argument("--input", "-i", type=Path, required=True, help="Input video file")
    export_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path (directory for frames mode, file for freetimegs)",
    )
    export_parser.add_argument(
        "--mode",
        choices=["frames", "freetimegs"],
        default="frames",
        help="Export mode: 'frames' for per-frame PLYs, 'freetimegs' for temporal PLY",
    )
    export_parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3-GIANT",
        help="DA3 model ID (must support infer_gs)",
    )
    export_parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame")
    export_parser.add_argument(
        "--chunk-size", type=int, default=10, help="Frames per processing chunk"
    )
    export_parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process"
    )
    export_parser.add_argument(
        "--process-res", type=int, default=518, help="Processing resolution for DA3"
    )
    export_parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.0,
        help="Prune Gaussians with opacity below this threshold (e.g. 0.05)",
    )
    export_parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Negate Y coordinates to flip the coordinate system (useful for SHARP models)",
    )
    export_parser.add_argument("--device", type=str, default="cuda")
    export_parser.add_argument("-v", "--verbose", action="store_true")

    postprocess_parser = subparsers.add_parser(
        "postprocess", help="Postprocess per-frame PLYs to FreeTimeGS PLY with motion vectors"
    )
    postprocess_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory containing per-frame PLY files",
    )
    postprocess_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output FreeTimeGS PLY file"
    )
    postprocess_parser.add_argument(
        "--fps", type=float, default=30.0, help="Frame rate for temporal normalization"
    )
    postprocess_parser.add_argument(
        "--max-match-distance",
        type=float,
        default=0.05,
        help="Maximum distance for matching Gaussians across frames",
    )
    postprocess_parser.add_argument(
        "--pattern", type=str, default="frame_*.ply", help="Glob pattern for PLY files"
    )
    postprocess_parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Negate Y coordinates to flip the coordinate system (useful for SHARP models). Only use if input PLYs weren't already flipped.",
    )
    postprocess_parser.add_argument("-v", "--verbose", action="store_true")

    images_parser = subparsers.add_parser(
        "images", help="Process images with DA3 and export to Gaussian PLY files"
    )
    images_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input directory containing images"
    )
    images_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path (directory for frames, file for freetimegs)",
    )
    images_parser.add_argument(
        "--mode",
        choices=["frames", "freetimegs"],
        default="frames",
        help="Export mode: 'frames' for per-frame PLYs, 'freetimegs' for temporal PLY",
    )
    images_parser.add_argument(
        "--fps", type=float, default=30.0, help="Frame rate for temporal normalization"
    )
    images_parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3-GIANT",
        help="Model ID. For SHARP: 'sharp' (DINOv2) or 'sharp:dinov3l16_384' (DINOv3)",
    )
    images_parser.add_argument(
        "--pattern", type=str, default="*.jpg", help="Glob pattern for image files"
    )
    images_parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process"
    )
    images_parser.add_argument(
        "--process-res", type=int, default=518, help="Processing resolution for DA3"
    )
    images_parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Directory containing mask images for background removal",
    )
    images_parser.add_argument(
        "--no-mask-first-frame",
        action="store_true",
        help="Apply mask to the first frame (default: skip first frame)",
    )
    images_parser.add_argument(
        "--no-remove-black-splats",
        action="store_true",
        help="Keep black splats instead of removing them (default: remove)",
    )
    images_parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Negate Y coordinates to flip the coordinate system (useful for SHARP models). If using postprocess afterward, don't flip there too.",
    )
    images_parser.add_argument("--device", type=str, default="cuda")
    images_parser.add_argument("-v", "--verbose", action="store_true")

    legacy_parser = subparsers.add_parser("legacy", help="Legacy CLI (deprecated)")
    legacy_parser.add_argument("--input", "-i", type=Path, required=True)
    legacy_parser.add_argument("--output", "-o", type=Path, required=True)
    legacy_parser.add_argument("--mode", choices=["frames", "freetimegs"], default="frames")
    legacy_parser.add_argument("--model", type=str, default="depth-anything/DA3-GIANT")
    legacy_parser.add_argument("--frame-skip", type=int, default=5)
    legacy_parser.add_argument("--chunk-size", type=int, default=10)
    legacy_parser.add_argument("--max-frames", type=int, default=None)
    legacy_parser.add_argument("--process-res", type=int, default=518)
    legacy_parser.add_argument("--device", type=str, default="cuda")
    legacy_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        if hasattr(args, "input"):
            args.command = "legacy"
        else:
            parser.print_help()
            return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.verbose:
        # Silence noisy libraries even in verbose mode
        for lib in ["matplotlib", "PIL", "urllib3", "timm", "huggingface_hub", "torch"]:
            logging.getLogger(lib).setLevel(logging.INFO)

    if args.command == "export" or args.command == "legacy":
        export_video_to_gaussian_plys(
            args.input,
            args.output,
            mode=args.mode,
            model_id=args.model,
            frame_skip=args.frame_skip,
            chunk_size=args.chunk_size,
            max_frames=args.max_frames,
            device=args.device,
            process_res=args.process_res,
            opacity_threshold=args.opacity_threshold if hasattr(args, "opacity_threshold") else 0.0,
            flip_y=getattr(args, "flip_y", False),
        )
    elif args.command == "postprocess":
        postprocess_plys_to_freetimegs(
            args.input,
            args.output,
            fps=args.fps,
            max_match_distance=args.max_match_distance,
            ply_pattern=args.pattern,
            flip_y=getattr(args, "flip_y", False),
        )
    elif args.command == "images":
        export_images_to_gaussian_plys(
            args.input,
            args.output,
            mode=args.mode,
            fps=args.fps,
            model_id=args.model,
            image_pattern=args.pattern,
            max_frames=args.max_frames,
            device=args.device,
            process_res=args.process_res,
            opacity_threshold=args.opacity_threshold if hasattr(args, "opacity_threshold") else 0.0,
            masks_dir=args.masks_dir if hasattr(args, "masks_dir") else None,
            mask_first_frame=not getattr(args, "no_mask_first_frame", False),
            remove_black_splats=not getattr(args, "no_remove_black_splats", False),
            flip_y=getattr(args, "flip_y", False),
        )


if __name__ == "__main__":
    main()
