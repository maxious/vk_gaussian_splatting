"""Gaussian processor using tttLRM (Test-Time Training LRM).

Produces Gaussian splats from multi-view posed images using the tttLRM model.
Each "frame" is a JSON manifest describing multiple camera views of a scene
at one timestep, in the format expected by tttLRM's dataset_scene.py.

Supports multi-device (multi-XPU/CUDA) processing via DeviceWorkerPool.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from common.device_worker_pool import DeviceWorker, DeviceWorkerPool
from ..types import GaussianFrame
from .base import GaussianProcessor
logger = logging.getLogger(__name__)
# Add file handler for debugging
_tttlrm_log_file = os.environ.get("TTTLRM_LOG_FILE", "/tmp/tttlrm_debug.log")
_handler = logging.FileHandler(_tttlrm_log_file, mode="a")
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)

_TTTLRM_ROOT = str(Path(__file__).parent.parent.parent / "third_party" / "tttlrm")


def _ensure_tttlrm_on_path() -> None:
    """Add tttlrm vendored directory to sys.path, ensuring it takes priority.

    Evicts any previously-cached ``utils`` / ``model`` / ``data`` packages
    so that imports resolve to the tttLRM vendored versions.
    """
    import importlib

    # Remove then re-insert at position 0
    if _TTTLRM_ROOT in sys.path:
        sys.path.remove(_TTTLRM_ROOT)
    sys.path.insert(0, _TTTLRM_ROOT)
    # Evict cached top-level modules that tttLRM shadows
    for prefix in ("utils", "model", "data"):
        for mod_name in list(sys.modules.keys()):
            if mod_name == prefix or mod_name.startswith(prefix + "."):
                del sys.modules[mod_name]
    # Invalidate caches so the import machinery re-scans sys.path
    importlib.invalidate_caches()


def _import_tttlrm_module(dotted_name: str, filename: str):
    """Directly load a tttLRM module by file path, bypassing sys.path conflicts."""
    import importlib.util
    import types

    parts = filename.split("/")
    filepath = Path(_TTTLRM_ROOT) / filename
    spec = importlib.util.spec_from_file_location(dotted_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _setup_tttlrm_utils() -> None:
    """Pre-register the tttLRM 'utils' package so sub-imports work."""
    import types

    if "utils" not in sys.modules or "sp_support" not in dir(sys.modules.get("utils", types.ModuleType("x"))):
        # Create a virtual 'utils' package pointing to tttlrm's utils
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(Path(_TTTLRM_ROOT) / "utils")]
        utils_pkg.__package__ = "utils"
        sys.modules["utils"] = utils_pkg

    for submod, filename in [
        ("utils.sp_support", "utils/sp_support.py"),
        ("utils.ddp_utils", "utils/ddp_utils.py"),
        ("utils.metrics", "utils/metrics.py"),
        ("utils.camera_utils", "utils/camera_utils.py"),
    ]:
        filepath = Path(_TTTLRM_ROOT) / filename
        if filepath.exists() and submod not in sys.modules:
            _import_tttlrm_module(submod, filename)


def _patch_sp_support_single_gpu() -> None:
    """Make sp_support work without torch.distributed initialisation."""
    _ensure_tttlrm_on_path()
    _setup_tttlrm_utils()
    sp_support = sys.modules["utils.sp_support"]

    sp_support._SP_GROUP = None
    sp_support.get_sp_rank = lambda: 0
    sp_support.get_sp_world_size = lambda: 1
    sp_support.get_sp_replicas = lambda: 1
    sp_support.get_sp_replica_id = lambda: 0
    sp_support.sp_broadcast = lambda x: x
    sp_support.sp_all_reduce = lambda tensor, op=None: tensor

    def _sp_input_broadcast_scatter(
        x: torch.Tensor,
        scatter_dim: int = 1,
        different_size: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[int, ...]]:
        if different_size:
            return x, tuple(x.shape)
        return x

    sp_support.sp_input_broadcast_scatter = _sp_input_broadcast_scatter

    def _sp_all_gather(
        x: torch.Tensor,
        gather_dim: int = 2,
        length: int = -1,
    ) -> torch.Tensor:
        if length != -1:
            slices = [slice(None)] * x.dim()
            slices[gather_dim] = slice(0, length)
            return x[tuple(slices)]
        return x

    sp_support.sp_all_gather = _sp_all_gather


def _resize_and_crop(
    image: Image.Image,
    target_size: tuple[int, int],
    fxfycxcy: list[float],
) -> tuple[Image.Image, list[float]]:
    """Resize and centre-crop image, adjusting intrinsics.

    Mirrors ``data.dataset_scene.resize_and_crop``.
    """
    original_width, original_height = image.size
    target_height, target_width = target_size

    fx, fy, cx, cy = fxfycxcy

    scale_x = target_width / original_width
    scale_y = target_height / original_height

    new_width = int(round(original_width * scale_x))
    new_height = int(round(original_height * scale_y))
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    cropped_image = resized_image.crop((left, top, left + target_width, top + target_height))

    new_fx = fx * scale_x
    new_fy = fy * scale_y
    new_cx = cx * scale_x - left
    new_cy = cy * scale_y - top

    return cropped_image, [new_fx, new_fy, new_cx, new_cy]


def _normalize_with_mean_pose(
    c2ws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre and scale poses using the mean camera.

    Mirrors ``data.dataset_scene.normalize_with_mean_pose`` with
    ``frame_method='mean_cam'``.
    """

    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x)

    centre = c2ws[:, :3, 3].mean(0)
    vec2 = _normalize(c2ws[:, :3, 2].sum(0))
    up = c2ws[:, :3, 1].sum(0)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))

    apos = np.eye(4)
    apos[:3] = np.stack([vec0, vec1, vec2, centre], 1)

    c2ws = np.linalg.inv(apos) @ c2ws

    scene_scale = np.max(np.abs(c2ws[:, :3, 3]))
    c2ws[:, :3, 3] /= scene_scale
    return c2ws, np.linalg.inv(apos), 1.0 / scene_scale


class TttLRMDeviceWorker(DeviceWorker[dict, dict]):
    """DeviceWorker that runs tttLRM inference on a single device.

    Each worker loads its own copy of the model and processes manifests
    independently, enabling multi-XPU/CUDA parallelism via DeviceWorkerPool.
    """

    def __init__(
        self,
        device: str,
        worker_id: int,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        image_size: int = 536,
        image_size_x: int = 960,
        num_input_views: int = 64,
        opacity_threshold: float = 0.001,
        autoregressive: bool = False,
        ttt_update_views: int = 0,
    ) -> None:
        super().__init__(device, worker_id)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.image_size = image_size
        self.image_size_x = image_size_x
        self.num_input_views = num_input_views
        self.opacity_threshold = opacity_threshold
        self.autoregressive = autoregressive
        self.ttt_update_views = ttt_update_views

    def load_model(self) -> None:
        # Disable torch.compile on XPU (inductor backend not supported)
        if "xpu" in self.device:
            os.environ["TTTLRM_NO_COMPILE"] = "1"

        _ensure_tttlrm_on_path()
        _patch_sp_support_single_gpu()

        import omegaconf
        from easydict import EasyDict as edict  # type: ignore[import-untyped]
        from model.model import tttLRM  # type: ignore[import-untyped]

        # Build config
        if self.config_path is not None:
            cfg = omegaconf.OmegaConf.load(self.config_path)
        else:
            config_name = "dl3dv_ar.yaml" if self.autoregressive else "dl3dv_full.yaml"
            default_yaml = Path(_TTTLRM_ROOT) / "configs" / config_name
            cfg = omegaconf.OmegaConf.load(str(default_yaml))

        cfg.sp_size = 1
        cfg.inference = False
        cfg.evaluation = True
        cfg.model.act_ckpt = False
        cfg.model.image_size = self.image_size
        cfg.model.image_size_x = self.image_size_x
        cfg.training.num_input_views = self.num_input_views
        cfg.training.num_virtual_views = self.num_input_views
        cfg.training.num_target_views = min(8, self.num_input_views)
        cfg.training.target_has_input = True
        cfg.training.num_views = self.num_input_views
        cfg.training.sample_ar = False
        cfg.training.sample_mixed_length = False
        cfg.training.depth_loss_weight = 0.0
        cfg.training.perceptual_loss_weight = 0.0
        if self.ttt_update_views > 0:
            cfg.model.ttt_update_views = self.ttt_update_views
        self._config = edict(cfg)

        # Build model and move to device (autocast handles dtype)
        model = tttLRM(self._config)
        model = model.to(self.device)

        # Load checkpoint
        ckpt_path = self.checkpoint_path
        if ckpt_path is None:
            from huggingface_hub import hf_hub_download

            ckpt_name = "dl3dv_ar.pt" if self.autoregressive else "dl3dv_full.pt"
            ckpt_path = hf_hub_download(repo_id="chenwang/tttLRM", filename=ckpt_name)

        logger.info("Worker %d: Loading tttLRM checkpoint from %s", self.worker_id, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        model = model.to(torch.bfloat16)
        model.eval()
        self.model = model
        logger.info("Worker %d: tttLRM ready on %s", self.worker_id, self.device)

    def process_item(self, item: dict) -> dict:
        """Process a single manifest (or a view-subset of one).

        Args:
            item: dict with keys 'manifest_path', 'frame_idx', 'timestamp_ms',
                and optionally 'view_indices' (list[int]) for split-view mode.

        Returns:
            dict with Gaussian attributes (numpy arrays)
        """
        assert self.model is not None
        manifest_path = Path(item["manifest_path"])
        frame_idx = item["frame_idx"]
        timestamp_ms = item["timestamp_ms"]
        view_indices = item.get("view_indices")

        with open(manifest_path) as f:
            manifest = json.load(f)

        batch = _prepare_batch_static(
            manifest, manifest_path,
            self.num_input_views, self.image_size, self.image_size_x,
            view_indices=view_indices,
        )

        device = torch.device(self.device)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Dynamically adjust config for actual view count
        actual_views = batch["num_input_views"][0].item()
        self._config.training.num_input_views = actual_views
        self._config.training.num_virtual_views = actual_views
        self._config.training.num_target_views = min(
            self._config.training.num_target_views, actual_views,
        )
        self._config.training.num_views = actual_views

        dtype = torch.bfloat16 if "xpu" in self.device else torch.bfloat16
        with torch.no_grad(), torch.autocast(
            enabled=True, device_type=device.type, dtype=dtype,
        ):
            result = self.model(batch, gaussians_only=True)
        gaussians = result.gaussians
        xyz = gaussians["xyz"][0].cpu().numpy()
        feature = gaussians["feature"][0].cpu().numpy()
        f_dc = feature[:, 0, :]
        scale = gaussians["scale"][0].cpu().numpy()
        rotation = gaussians["rotation"][0].cpu().numpy()
        opacity = gaussians["opacity"][0].squeeze(-1).cpu().numpy()

        logger.debug(
            "Worker %d: Raw gaussians from model: xyz=%s, feature=%s, scale=%s, rotation=%s, opacity=%s",
            self.worker_id,
            xyz.shape,
            feature.shape,
            scale.shape,
            rotation.shape,
            opacity.shape,
        )

        # Log opacity stats
        sigmoid_opacity_vals = 1.0 / (1.0 + np.exp(-opacity))
        logger.debug(
            "Worker %d: Opacity stats - min=%.6f, max=%.6f, mean=%.6f, median=%.6f",
            self.worker_id,
            sigmoid_opacity_vals.min(),
            sigmoid_opacity_vals.max(),
            sigmoid_opacity_vals.mean(),
            np.median(sigmoid_opacity_vals),
        )

        if self.opacity_threshold > 0:
            sigmoid_opacity = 1.0 / (1.0 + np.exp(-opacity))
            mask = sigmoid_opacity > self.opacity_threshold
            xyz, f_dc, scale, rotation, opacity = (
                xyz[mask], f_dc[mask], scale[mask], rotation[mask], opacity[mask],
            )

        if "xpu" in self.device:
            torch.xpu.empty_cache()

        logger.info(
            "Worker %d: Processed manifest %d: %d Gaussians",
            self.worker_id, frame_idx, len(xyz),
        )

        result = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "means": xyz.astype(np.float32),
            "scales": scale.astype(np.float32),
            "rotations": rotation.astype(np.float32),
            "colors": f_dc.astype(np.float32),
            "opacities": opacity.astype(np.float32),
        }
        if "split_part" in item:
            result["split_part"] = item["split_part"]
        return result


def _prepare_batch_static(
    manifest: dict,
    manifest_path: Path,
    num_input_views: int,
    image_size: int,
    image_size_x: int,
    view_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Build the batch dict expected by tttLRM.forward() (standalone function for workers).

    Args:
        view_indices: If set, only include these view indices in the batch.
            Pose normalisation always uses ALL available views so that
            split-view batches share the same coordinate frame.
    """
    frames = manifest["frames"]
    image_base_dir = manifest.get("dataset_root", str(manifest_path.parent))

    n_total = min(len(frames), num_input_views)

    # Always normalise using ALL available poses for a consistent frame
    all_c2ws = np.array([np.linalg.inv(np.array(f["w2c"])) for f in frames[:n_total]])
    all_c2ws, _apos_inv, scene_scale = _normalize_with_mean_pose(all_c2ws)

    selected = view_indices if view_indices is not None else list(range(n_total))
    n_views = len(selected)

    target_size = (image_size, image_size_x)
    to_tensor = transforms.ToTensor()

    images: list[torch.Tensor] = []
    fxfycxcy_list: list[list[float]] = []
    c2ws_list: list[np.ndarray] = []

    for idx in selected:
        frame = frames[idx]
        img_path = Path(image_base_dir) / frame["file_path"]
        image = Image.open(img_path)
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1])
            image = bg
        elif image.mode != "RGB":
            image = image.convert("RGB")

        intrinsics = [frame["fx"], frame["fy"], frame["cx"], frame["cy"]]
        image, intrinsics = _resize_and_crop(image, target_size, intrinsics)

        images.append(to_tensor(image))
        fxfycxcy_list.append(intrinsics)
        c2ws_list.append(all_c2ws[idx])

    c2ws = torch.from_numpy(np.array(c2ws_list)).float()
    fxfycxcy = torch.from_numpy(np.array(fxfycxcy_list)).float()
    image_stack = torch.stack(images)

    virtual_c2ws = c2ws.clone()
    virtual_fxfycxcy = fxfycxcy.clone()
    virtual_input_indices = torch.arange(n_views).long()

    image_choices = torch.arange(n_views).long().unsqueeze(-1)
    scene_indices = torch.zeros_like(image_choices)
    indices = torch.cat([image_choices, scene_indices], dim=-1)

    return {
        "image": image_stack.unsqueeze(0),
        "c2w": c2ws.unsqueeze(0),
        "fxfycxcy": fxfycxcy.unsqueeze(0),
        "index": indices.unsqueeze(0),
        "scene_name": manifest.get("scene_name", "unknown"),
        "virtual_c2w": virtual_c2ws.unsqueeze(0),
        "virtual_fxfycxcy": virtual_fxfycxcy.unsqueeze(0),
        "virtual_input_indices": virtual_input_indices.unsqueeze(0),
        "num_input_views": torch.tensor([n_views]),
        "scene_scale": scene_scale,
        "apos": _apos_inv,
    }


class TttLRMGaussianProcessor(GaussianProcessor):
    """Gaussian processor using tttLRM (Test-Time Training LRM).

    Produces Gaussian splats from multi-view posed images.  Each "frame"
    passed to :meth:`process_frames` should be a path to a JSON manifest
    that describes one timestep with all of its camera views, using the
    same format as tttLRM's ``data/dataset_scene.py``.

    Manifest JSON format::

        {
            "frames": [
                {
                    "file_path": "relative/path/to/image.jpg",
                    "w2c": [[...], ...],   // 4x4 world-to-camera matrix
                    "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0
                },
                ...
            ]
        }
    """

    def __init__(
        self,
        device: str = "cuda",
        device_spec: str | None = None,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        image_size: int = 536,
        image_size_x: int = 960,
        num_input_views: int = 64,
        opacity_threshold: float = 0.001,
        autoregressive: bool = False,
        split_views: bool = False,
        ttt_update_views: int = 0,
    ) -> None:
        self.opacity_threshold = opacity_threshold
        self.image_size = image_size
        self.image_size_x = image_size_x
        self.num_input_views = num_input_views
        self.autoregressive = autoregressive
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.split_views = split_views

        # Determine multi-device config
        self.device_spec = device_spec or device
        devices = DeviceWorkerPool._discover_devices(self.device_spec)
        self.use_multi_device = len(devices) > 1 or split_views

        # Auto-select ttt_update_views for XPU when split_views is active
        if ttt_update_views == 0 and split_views and "xpu" in devices[0]:
            ttt_update_views = 4
            logger.info("Auto-selected ttt_update_views=%d for XPU split-view mode", ttt_update_views)

        if self.use_multi_device:
            logger.info("Multi-device tttLRM: using %d devices: %s", len(devices), devices)
            self._pool: DeviceWorkerPool | None = DeviceWorkerPool(
                worker_class=TttLRMDeviceWorker,
                devices=devices,
                worker_kwargs={
                    "checkpoint_path": checkpoint_path,
                    "config_path": config_path,
                    "image_size": image_size,
                    "image_size_x": image_size_x,
                    "num_input_views": num_input_views,
                    "opacity_threshold": opacity_threshold,
                    "autoregressive": autoregressive,
                    "ttt_update_views": ttt_update_views,
                },
            )
            self.device = torch.device(devices[0])
            self._model = None
        else:
            self._pool = None
            self.device = torch.device(devices[0])
            if self.device.type == "xpu":
                os.environ["TTTLRM_NO_COMPILE"] = "1"
            _ensure_tttlrm_on_path()
            _patch_sp_support_single_gpu()
            self._config = self._build_config(config_path)
            self._model = self._build_model(checkpoint_path)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_config(self, config_path: str | None) -> Any:
        import omegaconf
        from easydict import EasyDict as edict  # type: ignore[import-untyped]

        if config_path is not None:
            cfg = omegaconf.OmegaConf.load(config_path)
        else:
            config_name = "dl3dv_ar.yaml" if self.autoregressive else "dl3dv_full.yaml"
            default_yaml = Path(_TTTLRM_ROOT) / "configs" / config_name
            cfg = omegaconf.OmegaConf.load(str(default_yaml))

        # Override for single-GPU inference
        cfg.sp_size = 1
        cfg.inference = False
        cfg.evaluation = True

        cfg.model.act_ckpt = False
        cfg.model.image_size = self.image_size
        cfg.model.image_size_x = self.image_size_x

        cfg.training.num_input_views = self.num_input_views
        cfg.training.num_virtual_views = self.num_input_views
        # Use some input views as targets to allow loss computation
        cfg.training.num_target_views = min(8, self.num_input_views)
        cfg.training.target_has_input = True
        cfg.training.num_views = self.num_input_views
        cfg.training.sample_ar = False
        cfg.training.sample_mixed_length = False
        cfg.training.depth_loss_weight = 0.0
        cfg.training.perceptual_loss_weight = 0.0

        return edict(cfg)

    @staticmethod
    def _download_checkpoint(filename: str = "dl3dv_full.pt") -> str:
        """Download a tttLRM checkpoint from HuggingFace Hub (cached)."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="chenwang/tttLRM",
            filename=filename,
        )
        logger.info("tttLRM checkpoint cached at %s", path)
        return path

    def _build_model(self, checkpoint_path: str | None) -> torch.nn.Module:
        from model.model import tttLRM  # type: ignore[import-untyped]

        model = tttLRM(self._config).to(self.device)

        if checkpoint_path is None:
            ckpt_name = "dl3dv_ar.pt" if self.autoregressive else "dl3dv_full.pt"
            checkpoint_path = self._download_checkpoint(ckpt_name)

        logger.info("Loading tttLRM checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model

    # ------------------------------------------------------------------
    # Manifest / batch preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _load_manifest(manifest_path: Path) -> dict:
        with open(manifest_path) as f:
            return json.load(f)

    def _prepare_batch(self, manifest: dict, manifest_path: Path) -> dict[str, Any]:
        """Build the batch dict expected by ``tttLRM.forward()``.

        This mirrors the logic in ``data.dataset_scene.Dataset.__getitem__``
        but deterministically uses the first ``num_input_views`` frames
        (no random sampling).
        """
        frames = manifest["frames"]
        # Use dataset_root if present (from SIGA converter), else fall back to manifest dir
        image_base_dir = manifest.get("dataset_root", str(manifest_path.parent))

        n_views = min(len(frames), self.num_input_views)
        if n_views < self.num_input_views:
            logger.warning(
                "Manifest %s has only %d views (requested %d)",
                manifest_path,
                len(frames),
                self.num_input_views,
            )

        target_size = (self.image_size, self.image_size_x)
        to_tensor = transforms.ToTensor()

        images: list[torch.Tensor] = []
        fxfycxcy_list: list[list[float]] = []
        c2ws_list: list[np.ndarray] = []

        for frame in frames[:n_views]:
            img_path = Path(image_base_dir) / frame["file_path"]
            image = Image.open(img_path)
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[-1])
                image = bg
            elif image.mode != "RGB":
                image = image.convert("RGB")

            intrinsics = [frame["fx"], frame["fy"], frame["cx"], frame["cy"]]
            image, intrinsics = _resize_and_crop(image, target_size, intrinsics)

            images.append(to_tensor(image))
            fxfycxcy_list.append(intrinsics)
            c2ws_list.append(np.linalg.inv(np.array(frame["w2c"])))

        c2ws_np = np.array(c2ws_list)
        c2ws_np, _apos_inv, scene_scale = _normalize_with_mean_pose(c2ws_np)

        c2ws = torch.from_numpy(c2ws_np).float()
        fxfycxcy = torch.from_numpy(np.array(fxfycxcy_list)).float()
        image_stack = torch.stack(images)  # (V, 3, H, W)

        # Virtual views = input views (same cameras used for Gaussian prediction)
        virtual_c2ws = c2ws[:n_views].clone()
        virtual_fxfycxcy = fxfycxcy[:n_views].clone()
        virtual_input_indices = torch.arange(n_views).long()

        # Dummy index / metadata expected by prepare_input_target
        image_choices = torch.arange(n_views).long().unsqueeze(-1)
        scene_indices = torch.zeros_like(image_choices)
        indices = torch.cat([image_choices, scene_indices], dim=-1)

        batch: dict[str, Any] = {
            "image": image_stack.unsqueeze(0),          # (1, V, 3, H, W)
            "c2w": c2ws.unsqueeze(0),                   # (1, V, 4, 4)
            "fxfycxcy": fxfycxcy.unsqueeze(0),          # (1, V, 4)
            "index": indices.unsqueeze(0),              # (1, V, 2)
            "scene_name": manifest.get("scene_name", "unknown"),
            "virtual_c2w": virtual_c2ws.unsqueeze(0),
            "virtual_fxfycxcy": virtual_fxfycxcy.unsqueeze(0),
            "virtual_input_indices": virtual_input_indices.unsqueeze(0),
            "num_input_views": torch.tensor([n_views]),
            "scene_scale": scene_scale,
            "apos": _apos_inv,
        }
        return batch

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = False,
    ) -> list[GaussianFrame]:
        """Process multi-view manifests to extract Gaussians.

        Args:
            frame_paths: Paths to JSON manifest files.  Each manifest
                describes one timestep with all camera views.
            timestamps_ms: Corresponding timestamps in milliseconds.
            per_frame: Ignored (always produces one GaussianFrame per
                manifest).

        Returns:
            List of :class:`GaussianFrame` objects.
        """
        if self.use_multi_device and self._pool is not None:
            return self._process_frames_multi_device(frame_paths, timestamps_ms)
        return self._process_frames_single_device(frame_paths, timestamps_ms)

    def _process_frames_multi_device(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
    ) -> list[GaussianFrame]:
        """Distribute manifests across multiple devices via DeviceWorkerPool."""
        if self.split_views:
            return self._process_frames_split_views(frame_paths, timestamps_ms)

        items = [
            {
                "manifest_path": str(p),
                "frame_idx": i,
                "timestamp_ms": ts,
            }
            for i, (p, ts) in enumerate(zip(frame_paths, timestamps_ms))
        ]

        logger.info(
            "Processing %d manifests across %d devices",
            len(items),
            len(self._pool.devices),
        )

        results = self._pool.map(items)

        frames = [
            GaussianFrame(
                frame_idx=r["frame_idx"],
                timestamp_ms=r["timestamp_ms"],
                means=r["means"],
                scales=r["scales"],
                rotations=r["rotations"],
                colors=r["colors"],
                opacities=r["opacities"],
            )
            for r in results
        ]
        frames.sort(key=lambda f: f.frame_idx)

        logger.info(
            "Multi-device processing complete: %d frames, %d total Gaussians",
            len(frames),
            sum(len(f.means) for f in frames),
        )
        return frames

    def _process_frames_split_views(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
    ) -> list[GaussianFrame]:
        """Split each manifest's views across devices, then merge Gaussians.

        Each device gets a disjoint subset of views, processes them
        independently with chunked TTT, and the resulting Gaussians
        are concatenated.  Pose normalisation uses ALL views so that
        the coordinate frame is consistent across devices.
        """
        n_devices = len(self._pool.devices)
        items: list[dict] = []

        for manifest_idx, (p, ts) in enumerate(zip(frame_paths, timestamps_ms)):
            with open(p) as f:
                manifest = json.load(f)
            n_total = min(len(manifest["frames"]), self.num_input_views)

            # Distribute views evenly; last device gets the remainder
            base, remainder = divmod(n_total, n_devices)
            offset = 0
            for dev_idx in range(n_devices):
                count = base + (1 if dev_idx < remainder else 0)
                view_indices = list(range(offset, offset + count))
                offset += count
                items.append({
                    "manifest_path": str(p),
                    "frame_idx": manifest_idx,
                    "timestamp_ms": ts,
                    "view_indices": view_indices,
                    "split_part": dev_idx,
                })

        logger.info(
            "Split-view processing: %d manifest(s) × %d devices = %d items",
            len(frame_paths),
            n_devices,
            len(items),
        )

        results = self._pool.map(items)

        # Group by frame_idx and merge
        from collections import defaultdict

        by_frame: dict[int, list[dict]] = defaultdict(list)
        for r in results:
            by_frame[r["frame_idx"]].append(r)

        frames: list[GaussianFrame] = []
        for frame_idx in sorted(by_frame.keys()):
            parts = by_frame[frame_idx]
            parts.sort(key=lambda r: r.get("split_part", 0))
            frame = GaussianFrame(
                frame_idx=frame_idx,
                timestamp_ms=parts[0]["timestamp_ms"],
                means=np.concatenate([p["means"] for p in parts]),
                scales=np.concatenate([p["scales"] for p in parts]),
                rotations=np.concatenate([p["rotations"] for p in parts]),
                colors=np.concatenate([p["colors"] for p in parts]),
                opacities=np.concatenate([p["opacities"] for p in parts]),
            )
            frames.append(frame)

        frames.sort(key=lambda f: f.frame_idx)
        logger.info(
            "Split-view processing complete: %d frames, %d total Gaussians",
            len(frames),
            sum(len(f.means) for f in frames),
        )
        return frames

    def _process_frames_single_device(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
    ) -> list[GaussianFrame]:
        """Process all manifests on a single device (original path)."""
        frames: list[GaussianFrame] = []
        self._model.eval()

        with (
            torch.no_grad(),
            torch.autocast(
                enabled=True,
                device_type=self.device.type,
                dtype=torch.bfloat16,
            ),
        ):
            for i, (manifest_path, ts) in enumerate(zip(frame_paths, timestamps_ms)):
                manifest = self._load_manifest(Path(manifest_path))
                batch = self._prepare_batch(manifest, Path(manifest_path))

                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Dynamically adjust config for actual view count
                actual_views = batch["num_input_views"][0].item()
                self._config.training.num_input_views = actual_views
                self._config.training.num_virtual_views = actual_views
                self._config.training.num_target_views = min(
                    self._config.training.num_target_views, actual_views,
                )
                self._config.training.num_views = actual_views

                result = self._model(batch, gaussians_only=True)
                gaussians = result.gaussians

                xyz = gaussians["xyz"][0].cpu().numpy()
                feature = gaussians["feature"][0].cpu().numpy()
                f_dc = feature[:, 0, :]
                scale = gaussians["scale"][0].cpu().numpy()
                rotation = gaussians["rotation"][0].cpu().numpy()
                opacity = gaussians["opacity"][0].squeeze(-1).cpu().numpy()

                if self.opacity_threshold > 0:
                    sigmoid_opacity = 1.0 / (1.0 + np.exp(-opacity))
                    mask = sigmoid_opacity > self.opacity_threshold
                    xyz = xyz[mask]
                    f_dc = f_dc[mask]
                    scale = scale[mask]
                    rotation = rotation[mask]
                    opacity = opacity[mask]

                frame = GaussianFrame(
                    frame_idx=i,
                    timestamp_ms=ts,
                    means=xyz.astype(np.float32),
                    scales=scale.astype(np.float32),
                    rotations=rotation.astype(np.float32),
                    colors=f_dc.astype(np.float32),
                    opacities=opacity.astype(np.float32),
                )
                frames.append(frame)
                logger.info(
                    "Processed manifest %d/%d: %d Gaussians",
                    i + 1,
                    len(frame_paths),
                    len(frame),
                )

                if self.device.type == "xpu":
                    torch.xpu.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return frames
