"""MotionCrafter Processor for 4D Geometry & Motion."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)


def compute_normals_from_point_map(points: np.ndarray) -> np.ndarray:
    """Compute normals from point map using cross product of partial derivatives."""
    H, W, _ = points.shape
    padded = np.pad(points, ((1, 1), (1, 1), (0, 0)), mode="edge")
    du = (padded[1:-1, 2:, :] - padded[1:-1, :-2, :]) / 2.0
    dv = (padded[2:, 1:-1, :] - padded[:-2, 1:-1, :]) / 2.0
    normals = np.cross(du, dv)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-8)
    avg_z = np.mean(points[:, :, 2])
    if avg_z > 0:
        flip = normals[:, :, 2] > 0
        normals[flip] = -normals[flip]
    else:
        flip = normals[:, :, 2] < 0
        normals[flip] = -normals[flip]
    return normals


class MotionCrafterProcessor(GaussianProcessor):
    """Process video to 4D Gaussians (geometry + motion) using MotionCrafter."""

    def __init__(
        self,
        model_path: str | None = None,
        config_path: str | None = None,
        device: str = "cuda",
        process_res: int = 512,
        max_frames: int = 16,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.process_res = process_res
        self.max_frames = max_frames
        self.model = None

    def _find_and_import_motioncrafter(self):
        """Attempt to find and import MotionCrafter modules."""
        try:
            import motioncrafter
            return
        except ImportError:
            pass

        candidates = [
            Path("python/motioncrafter"),
            Path("3rdparty/MotionCrafter"),
            Path("../MotionCrafter"),
            Path("./MotionCrafter"),
        ]

        found = False
        for p in candidates:
            if p.exists() and (p / "run.py").exists():
                logger.info(f"Found MotionCrafter at {p}")
                if str(p.resolve()) not in sys.path:
                    sys.path.insert(0, str(p.resolve()))
                found = True
                break

        if not found and self.model_path:
            p = Path(self.model_path)
            if p.exists() and (p / "run.py").exists():
                logger.info(f"Using model path as repo path: {p}")
                if str(p.resolve()) not in sys.path:
                    sys.path.insert(0, str(p.resolve()))
                found = True

        if not found:
            logger.warning("MotionCrafter repository not found. Please clone it.")

    def _load_model(self):
        """Lazy load MotionCrafter model."""
        if self.model is not None:
            return

        self._find_and_import_motioncrafter()

        try:
            from omegaconf import OmegaConf
            from scripts.evaluation.funcs import load_model_checkpoint
            from utils.utils import instantiate_from_config
        except ImportError as e:
            logger.error(f"Failed to import MotionCrafter modules: {e}")
            raise

        logger.info("Loading MotionCrafter pipeline...")

        if self.config_path:
            config_file = self.config_path
        else:
            config_file = "configs/inference/run.yaml"
            for p in sys.path:
                candidate = Path(p) / "configs/inference/run.yaml"
                if candidate.exists():
                    config_file = str(candidate)
                    break

        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        config = OmegaConf.load(config_file)

        try:
            self.model = load_model_checkpoint(self.model, config_file, self.model_path, self.device)
        except Exception:
            self.model = instantiate_from_config(config.model)
            if self.model_path:
                logger.info(f"Loading checkpoint from {self.model_path}")
                sd = torch.load(self.model_path, map_location="cpu")
                if "state_dict" in sd:
                    sd = sd["state_dict"]
                m, u = self.model.load_state_dict(sd, strict=False)
                if len(m) > 0:
                    logger.warning(f"Missing keys: {m}")
                if len(u) > 0:
                    logger.warning(f"Unexpected keys: {u}")
            self.model.to(self.device)
            self.model.eval()

        logger.info("MotionCrafter model loaded")

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = True,
    ) -> list[GaussianFrame]:
        """Process frames using MotionCrafter."""
        self._load_model()

        images = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        if not images:
            return []

        num_frames = len(images)
        H_orig, W_orig = images[0].shape[:2]
        scale = self.process_res / max(H_orig, W_orig)
        H_target = int(round(H_orig * scale / 64) * 64)
        W_target = int(round(W_orig * scale / 64) * 64)

        logger.info(f"Resizing images to {W_target}x{H_target} for inference")

        images_resized = [cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_AREA) for img in images]

        pixel_values = torch.from_numpy(np.stack(images_resized)).float() / 127.5 - 1.0
        pixel_values = pixel_values.permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

        try:
            batch = {"video": pixel_values}
            if hasattr(self.model, "inference"):
                outputs = self.model.inference(batch)
            else:
                outputs = self.model(pixel_values)
        except Exception as e:
            logger.error(f"MotionCrafter inference failed: {e}")
            raise

        def to_np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if x.ndim == 5:
                x = x[0].transpose(1, 2, 3, 0)
            elif x.ndim == 4:
                x = x.transpose(1, 2, 3, 0)
            return x

        keys_map = {
            "point_map": ["point_map", "points", "rec_point_map"],
            "scene_flow": ["scene_flow", "flow", "rec_deform_map", "deform_map"],
            "valid_mask": ["valid_mask", "mask", "rec_valid_mask"],
        }

        point_map_np = None
        scene_flow_np = None
        valid_mask_np = None

        if isinstance(outputs, dict):
            for key, candidates in keys_map.items():
                for c in candidates:
                    if c in outputs:
                        val = to_np(outputs[c])
                        if key == "point_map":
                            point_map_np = val
                        elif key == "scene_flow":
                            scene_flow_np = val
                        elif key == "valid_mask":
                            valid_mask_np = val
                        break

        if point_map_np is None:
            logger.error(f"Could not find point_map in outputs")
            return []

        T_out = point_map_np.shape[0]
        gaussians = []

        for t in range(T_out):
            if t >= num_frames:
                break

            ts = timestamps_ms[t]
            points = point_map_np[t]
            flow = scene_flow_np[t] if scene_flow_np is not None else None
            mask = valid_mask_np[t] if valid_mask_np is not None else None

            if mask is not None and mask.ndim == 3:
                mask = mask[..., 0]

            points_flat = points.reshape(-1, 3)
            flow_flat = flow.reshape(-1, 3) if flow is not None else None
            mask_flat = mask.reshape(-1) > 0.5 if mask is not None else np.ones(len(points_flat), dtype=bool)

            color_img = images_resized[t]
            colors_flat = color_img.reshape(-1, 3) / 255.0

            points_valid = points_flat[mask_flat]
            colors_valid = colors_flat[mask_flat]
            flow_valid = flow_flat[mask_flat] if flow_flat is not None else None

            if len(points_valid) == 0:
                continue

            normals_grid = compute_normals_from_point_map(points)
            normals_flat = normals_grid.reshape(-1, 3)
            normals_valid = normals_flat[mask_flat]

            nz = normals_valid[:, 2]
            nx = normals_valid[:, 0]
            ny = normals_valid[:, 1]

            qw = 1.0 + nz
            qx = -ny
            qy = nx
            qz = np.zeros_like(nx)

            antiparallel = qw < 1e-3
            qw[antiparallel] = 0.0
            qx[antiparallel] = 1.0

            quats = np.stack([qw, qx, qy, qz], axis=1)
            quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)

            scale_val = 1.0 / max(H_target, W_target) * 2.0
            scales_valid = np.full((len(points_valid), 3), np.log(scale_val), dtype=np.float32)
            opacities_valid = np.full(len(points_valid), 10.0, dtype=np.float32)

            SH_C0 = 0.28209479177387814
            colors_sh = (colors_valid - 0.5) / SH_C0

            gaussians.append(
                GaussianFrame(
                    frame_idx=t,
                    timestamp_ms=ts,
                    means=points_valid.astype(np.float32),
                    scales=scales_valid.astype(np.float32),
                    rotations=quats.astype(np.float32),
                    colors=colors_sh.astype(np.float32),
                    opacities=opacities_valid.astype(np.float32),
                    flow=flow_valid.astype(np.float32) if flow_valid is not None else None,
                )
            )

        return gaussians
