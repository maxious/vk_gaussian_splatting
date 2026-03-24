from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..types import GaussianFrame
from .base import GaussianProcessor

logger = logging.getLogger(__name__)

INFINIDEPTH_REPO = "ritianyu/InfiniDepth"
INFINIDEPTH_MODELS = {
    "depth": "infinidepth.ckpt",
    "depthsensor": "infinidepth_depthsensor.ckpt",
    "gs": "infinidepth_gs.ckpt",
    "depthsensor_gs": "infinidepth_depthsensor_gs.ckpt",
    "moge2": "moge2.pt",
    "skyseg": "skyseg.onnx",
}


def _download_infinidepth_checkpoint(filename: str, cache_dir: Path | None = None) -> Path:
    """Download InfiniDepth checkpoint from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "hub"
    local_path = hf_hub_download(
        repo_id=INFINIDEPTH_REPO,
        filename=filename,
        cache_dir=cache_dir,
    )
    return Path(local_path)


class InfiniDepthGaussianProcessor(GaussianProcessor):
    def __init__(
        self,
        *,
        model_type: str = "InfiniDepth",
        device: str = "cuda",
        input_size: tuple[int, int] = (768, 1024),
        sample_point_num: int = 2_000_000,
        coord_deterministic_sampling: bool = True,
        enable_skyseg_model: bool = False,
        sample_sky_mask_dilate_px: int = 0,
        fx_org: float | None = None,
        fy_org: float | None = None,
        cx_org: float | None = None,
        cy_org: float | None = None,
        depth_model_path: str | Path | None = None,
        gs_model_path: str | Path | None = None,
        moge2_pretrained: str | Path | None = None,
        sky_model_ckpt_path: str | Path | None = None,
        infinidepth_root: str | Path | None = None,
        debug_export_ply_dir: str | Path | None = None,
        masks_dir: str | Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = False,
    ):
        self.model_type = model_type
        self.device_spec = device
        self.input_size = input_size
        self.sample_point_num = int(sample_point_num)
        self.coord_deterministic_sampling = bool(coord_deterministic_sampling)
        self.enable_skyseg_model = bool(enable_skyseg_model)
        self.sample_sky_mask_dilate_px = int(sample_sky_mask_dilate_px)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.mask_first_frame = bool(mask_first_frame)
        self.remove_black_splats = bool(remove_black_splats)

        self.fx_org = fx_org
        self.fy_org = fy_org
        self.cx_org = cx_org
        self.cy_org = cy_org

        self.infinidepth_root = self._resolve_infinidepth_root(infinidepth_root)
        self.depth_model_path = self._resolve_checkpoint_path(
            depth_model_path
            or os.environ.get("INFINIDEPTH_DEPTH_CKPT")
            or "checkpoints/depth/infinidepth.ckpt"
        )
        self.gs_model_path = self._resolve_checkpoint_path(
            gs_model_path
            or os.environ.get("INFINIDEPTH_GS_CKPT")
            or "checkpoints/gs/infinidepth_gs.ckpt"
        )
        self.moge2_pretrained = self._resolve_checkpoint_path(
            moge2_pretrained
            or os.environ.get("INFINIDEPTH_MOGE2_CKPT")
            or "checkpoints/moge-2-vitl-normal/model.pt"
        )
        self.sky_model_ckpt_path = self._resolve_checkpoint_path(
            sky_model_ckpt_path
            or os.environ.get("INFINIDEPTH_SKYSEG_CKPT")
            or "checkpoints/sky/skyseg.onnx"
        )

        self.debug_export_ply_dir = (
            Path(debug_export_ply_dir).expanduser().resolve()
            if debug_export_ply_dir is not None
            else None
        )

        self._torch = None
        self._depth_model = None
        self._gs_predictor = None
        self._device = None

        self._build_model = None
        self._prepare_metric_depth_inputs = None
        self._resolve_camera_intrinsics_for_inference = None
        self._build_camera_matrices = None
        self._filter_gaussians_by_statistical_outlier = None
        self._unpack_gaussians_for_export = None
        self._build_sparse_uniform_gaussians = None
        self._run_optional_sampling_sky_mask = None
        self._depth_to_disparity = None
        self._load_image = None
        self._GSPixelAlignPredictor = None
        self._export_ply = None

    @staticmethod
    def _resolve_infinidepth_root(infinidepth_root: str | Path | None) -> Path | None:
        candidate = infinidepth_root or os.environ.get("INFINIDEPTH_ROOT")
        if candidate is None:
            return None
        return Path(candidate).expanduser().resolve()

    def _resolve_checkpoint_path(self, checkpoint: str | Path) -> Path:
        p = Path(checkpoint).expanduser()
        if p.is_absolute():
            return p

        cwd_candidate = Path.cwd() / p
        if cwd_candidate.exists():
            return cwd_candidate.resolve()

        if self.infinidepth_root is not None:
            root_candidate = self.infinidepth_root / p
            if root_candidate.exists():
                return root_candidate.resolve()

        if self.infinidepth_root is not None:
            return (self.infinidepth_root / p).resolve()
        return cwd_candidate.resolve()

    def _ensure_infinidepth_imports(self) -> None:
        if self._build_model is not None:
            return

        if self.infinidepth_root is not None:
            root_str = str(self.infinidepth_root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)

        try:
            import torch

            from InfiniDepth.gs import GSPixelAlignPredictor, export_ply
            from InfiniDepth.utils import gs_utils
            from InfiniDepth.utils.inference_utils import (
                build_camera_matrices,
                filter_gaussians_by_statistical_outlier,
                prepare_metric_depth_inputs,
                resolve_camera_intrinsics_for_inference,
                run_optional_sampling_sky_mask,
                unpack_gaussians_for_export as unpack_gaussians_for_export_inference,
            )
            from InfiniDepth.utils.io_utils import depth_to_disparity, load_image
            from InfiniDepth.utils.model_utils import build_model
        except ImportError as exc:
            root_hint = (
                f" (INFINIDEPTH_ROOT={self.infinidepth_root})"
                if self.infinidepth_root is not None
                else ""
            )
            raise ImportError(
                "Failed to import InfiniDepth modules. "
                "Install InfiniDepth or set INFINIDEPTH_ROOT to a valid checkout"
                f"{root_hint}."
            ) from exc

        self._torch = torch
        self._build_model = build_model
        self._prepare_metric_depth_inputs = prepare_metric_depth_inputs
        self._resolve_camera_intrinsics_for_inference = resolve_camera_intrinsics_for_inference
        self._build_camera_matrices = build_camera_matrices
        self._filter_gaussians_by_statistical_outlier = filter_gaussians_by_statistical_outlier
        self._unpack_gaussians_for_export = getattr(
            gs_utils,
            "unpack_gaussians_for_export",
            unpack_gaussians_for_export_inference,
        )
        self._build_sparse_uniform_gaussians = gs_utils._build_sparse_uniform_gaussians
        self._run_optional_sampling_sky_mask = run_optional_sampling_sky_mask
        self._depth_to_disparity = depth_to_disparity
        self._load_image = load_image
        self._GSPixelAlignPredictor = GSPixelAlignPredictor
        self._export_ply = export_ply

    def _resolve_device(self) -> str:
        assert self._torch is not None

        device = self.device_spec.lower()
        if device == "auto":
            if self._torch.cuda.is_available():
                return "cuda"
            if hasattr(self._torch, "xpu") and self._torch.xpu.is_available():
                return "xpu"
            if hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if device.startswith("cuda") and not self._torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            return "cpu"

        if device.startswith("xpu") and (
            not hasattr(self._torch, "xpu") or not self._torch.xpu.is_available()
        ):
            logger.warning("XPU requested but unavailable, falling back to CPU")
            return "cpu"

        if device == "mps" and (
            not hasattr(self._torch.backends, "mps") or not self._torch.backends.mps.is_available()
        ):
            logger.warning("MPS requested but unavailable, falling back to CPU")
            return "cpu"

        return self.device_spec

    def _load_model(self) -> None:
        if self._depth_model is not None and self._gs_predictor is not None:
            return

        self._ensure_infinidepth_imports()
        assert self._torch is not None

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        if not self.depth_model_path.exists():
            logger.info("Downloading InfiniDepth depth model from HuggingFace...")
            self.depth_model_path = _download_infinidepth_checkpoint(
                INFINIDEPTH_MODELS["depth"], cache_dir
            )
        if not self.gs_model_path.exists():
            logger.info("Downloading InfiniDepth GS model from HuggingFace...")
            self.gs_model_path = _download_infinidepth_checkpoint(
                INFINIDEPTH_MODELS["gs"], cache_dir
            )
        if not self.moge2_pretrained.exists():
            logger.info("Downloading MoGe-2 model from HuggingFace...")
            self.moge2_pretrained = _download_infinidepth_checkpoint(
                INFINIDEPTH_MODELS["moge2"], cache_dir
            )

        resolved_device = self._resolve_device()
        self._device = self._torch.device(resolved_device)

        logger.info(
            "Loading InfiniDepth depth model=%s gs=%s device=%s",
            self.depth_model_path,
            self.gs_model_path,
            self._device,
        )

        self._depth_model = self._build_model(  # type: ignore[operator]
            self.model_type,
            model_path=str(self.depth_model_path),
        ).to(self._device)
        self._depth_model.eval()

        self._gs_predictor = self._GSPixelAlignPredictor(dino_feature_dim=1024).to(self._device)  # type: ignore[operator]
        self._gs_predictor.load_from_infinidepth_gs_checkpoint(str(self.gs_model_path))
        self._gs_predictor.eval()

    def process_frames(
        self,
        frame_paths: list[Path],
        timestamps_ms: list[float],
        per_frame: bool = True,
        masks_dir: str | Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = False,
    ) -> list[GaussianFrame]:
        if not frame_paths:
            return []

        if len(frame_paths) != len(timestamps_ms):
            raise ValueError(
                f"frame_paths/timestamps_ms length mismatch: {len(frame_paths)} != {len(timestamps_ms)}"
            )

        masks_dir_path = Path(masks_dir) if masks_dir else self.masks_dir
        mask_first = mask_first_frame or self.mask_first_frame
        remove_black = remove_black_splats or self.remove_black_splats

        if not per_frame and len(frame_paths) > 1:
            logger.info(
                "InfiniDepth currently runs per-frame inference; processing %d frames independently",
                len(frame_paths),
            )

        self._load_model()
        assert self._torch is not None

        results: list[GaussianFrame] = []
        with self._torch.no_grad():
            for frame_idx, (frame_path, timestamp_ms) in enumerate(
                zip(frame_paths, timestamps_ms, strict=True)
            ):
                frame = self._process_single_frame(
                    frame_idx=frame_idx,
                    frame_path=frame_path,
                    timestamp_ms=timestamp_ms,
                    masks_dir=masks_dir_path,
                    mask_first_frame=mask_first,
                    remove_black_splats=remove_black,
                )
                results.append(frame)
        return results

    def _process_single_frame(
        self,
        *,
        frame_idx: int,
        frame_path: Path,
        timestamp_ms: float,
        masks_dir: Path | None = None,
        mask_first_frame: bool = False,
        remove_black_splats: bool = False,
    ) -> GaussianFrame:
        assert self._torch is not None
        assert self._depth_model is not None
        assert self._gs_predictor is not None
        assert self._device is not None

        if not frame_path.exists():
            raise FileNotFoundError(f"Input frame not found: {frame_path}")

        _, image, (org_h_raw, org_w_raw) = self._load_image(  # type: ignore[operator]
            str(frame_path), self.input_size
        )
        org_h = int(org_h_raw)
        org_w = int(org_w_raw)
        image = image.to(self._device)
        b, _, h, w = image.shape

        custom_mask_foreground = None
        custom_mask_background_for_filtering = None
        if masks_dir is not None:
            mask_frame_idx = 0 if mask_first_frame else frame_idx
            mask_path = masks_dir / f"frame_{mask_frame_idx:06d}.png"
            if mask_path.exists():
                from PIL import Image as PILImage

                custom_mask = (
                    self._torch.from_numpy(
                        np.array(PILImage.open(mask_path).convert("L")).astype(np.float32) / 255.0
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self._device)
                )
                custom_mask = self._torch.nn.functional.interpolate(
                    custom_mask, size=(h, w), mode="bilinear", align_corners=False
                )
                custom_mask_foreground = custom_mask[0, 0] > 0.5
                custom_mask_background_for_filtering = custom_mask[0, 0] <= 0.5
                image = image * custom_mask_foreground.float().unsqueeze(0).unsqueeze(0)
                logger.debug(f"Applied mask to image before processing: {mask_path}")
            else:
                logger.warning(f"Custom mask not found: {mask_path}, skipping")

        gt_depth, prompt_depth, gt_depth_mask, _, moge2_intrinsics = (
            self._prepare_metric_depth_inputs(  # type: ignore[operator]
                input_depth_path=None,
                input_size=self.input_size,
                image=image,
                device=self._device,
                moge2_pretrained=str(self.moge2_pretrained),
            )
        )

        fx_org, fy_org, cx_org, cy_org, _ = self._resolve_camera_intrinsics_for_inference(  # type: ignore[operator]
            fx_org=self.fx_org,
            fy_org=self.fy_org,
            cx_org=self.cx_org,
            cy_org=self.cy_org,
            org_h=org_h,
            org_w=org_w,
            image=image,
            moge2_pretrained=str(self.moge2_pretrained),
            moge2_intrinsics=moge2_intrinsics,
        )

        gt = self._depth_to_disparity(gt_depth)  # type: ignore[operator]
        prompt = self._depth_to_disparity(prompt_depth)  # type: ignore[operator]

        _, _, _, _, intrinsics, extrinsics = self._build_camera_matrices(  # type: ignore[operator]
            fx_org=fx_org,
            fy_org=fy_org,
            cx_org=cx_org,
            cy_org=cy_org,
            org_h=org_h,
            org_w=org_w,
            h=h,
            w=w,
            batch=b,
            device=self._device,
        )

        sky_mask = self._run_optional_sampling_sky_mask(  # type: ignore[operator]
            image=image,
            enable_skyseg_model=self.enable_skyseg_model,
            sky_model_ckpt_path=str(self.sky_model_ckpt_path),
            dilate_px=self.sample_sky_mask_dilate_px,
        )

        combined_mask = sky_mask
        if custom_mask_background_for_filtering is not None:
            if combined_mask is not None:
                combined_mask = combined_mask | custom_mask_background_for_filtering
            else:
                combined_mask = custom_mask_background_for_filtering

        depthmap, dino_tokens, query_3d_uniform_coord, pred_depth_3d = (
            self._depth_model.inference_for_gs(
                image=image,
                intrinsics=intrinsics,
                gt_depth=gt,
                gt_depth_mask=gt_depth_mask,
                prompt_depth=prompt,
                prompt_mask=prompt > 0,
                sky_mask=combined_mask,
                sample_point_num=self.sample_point_num,
                coord_deterministic_sampling=self.coord_deterministic_sampling,
            )
        )

        if query_3d_uniform_coord is None or pred_depth_3d is None:
            raise RuntimeError(
                "InfiniDepth inference_for_gs did not return sparse 3D-uniform queries"
            )

        dense_gaussians = self._gs_predictor(
            image=image,
            depthmap=depthmap,
            dino_tokens=dino_tokens,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

        pixel_gaussians = self._build_sparse_uniform_gaussians(  # type: ignore[operator]
            dense_gaussians=dense_gaussians,
            query_3d_uniform_coord=query_3d_uniform_coord,
            pred_depth_3d=pred_depth_3d,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            h=h,
            w=w,
        )

        filtered = self._filter_gaussians_by_statistical_outlier(pixel_gaussians)  # type: ignore[operator]
        if hasattr(filtered, "means"):
            pixel_gaussians = filtered
        elif isinstance(filtered, tuple) and filtered:
            first = filtered[0]
            if hasattr(first, "means"):
                pixel_gaussians = first

        gaussians_for_export: Any = pixel_gaussians
        means, harmonics, opacities, scales, rotations = self._unpack_gaussians_for_export(  # type: ignore[operator]
            gaussians_for_export
        )

        if harmonics.ndim == 3:
            colors = harmonics[:, :, 0]
        elif harmonics.ndim == 2:
            colors = harmonics
        else:
            raise RuntimeError(
                f"Unexpected harmonics shape from InfiniDepth: {tuple(harmonics.shape)}"
            )

        means_np = self._to_numpy_f32(means)
        scales_np = self._to_numpy_f32(scales)
        rotations_np = self._to_numpy_f32(rotations)
        colors_np = self._to_numpy_f32(colors)
        opacities_np = self._to_numpy_f32(opacities).reshape(-1)

        self._maybe_export_debug_ply(
            frame_idx=frame_idx,
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            fx_org=float(fx_org),
            fy_org=float(fy_org),
            cx_org=float(cx_org),
            cy_org=float(cy_org),
            org_h=int(org_h),
            org_w=int(org_w),
            extrinsics=extrinsics,
        )

        return GaussianFrame(
            frame_idx=frame_idx,
            timestamp_ms=float(timestamp_ms),
            means=means_np,
            scales=scales_np,
            rotations=rotations_np,
            colors=colors_np,
            opacities=opacities_np,
            intrinsic=intrinsics[0].cpu().numpy() if hasattr(intrinsics, "__getitem__") else None,
            extrinsic=extrinsics[0].cpu().numpy()
            if hasattr(extrinsics, "__getitem__")
            else extrinsics.cpu().numpy()
            if hasattr(extrinsics, "cpu")
            else extrinsics,
            image_size=(int(org_h), int(org_w)),
            color_space_index=1,
        )

    @staticmethod
    def _to_numpy_f32(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        arr = np.asarray(value, dtype=np.float32)
        return np.ascontiguousarray(arr)

    def _maybe_export_debug_ply(
        self,
        *,
        frame_idx: int,
        means,
        harmonics,
        opacities,
        scales,
        rotations,
        fx_org: float,
        fy_org: float,
        cx_org: float,
        cy_org: float,
        org_h: int,
        org_w: int,
        extrinsics,
    ) -> None:
        if self.debug_export_ply_dir is None:
            return

        self.debug_export_ply_dir.mkdir(parents=True, exist_ok=True)
        ply_path = self.debug_export_ply_dir / f"frame_{frame_idx:06d}.ply"

        self._export_ply(  # type: ignore[operator]
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            path=str(ply_path),
            scales=scales,
            rotations=rotations,
            focal_length_px=(fx_org, fy_org),
            principal_point_px=(cx_org, cy_org),
            image_shape=(org_h, org_w),
            extrinsic_matrix=extrinsics[0],
        )
