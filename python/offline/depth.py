"""Depth extraction command for offline preprocessing."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Result from depth model inference."""

    depth: np.ndarray  # (H, W) metric depth
    z_min: float
    z_max: float
    normals: Optional[np.ndarray] = None  # (H, W, 3) in [-1, 1]


def _is_moge_model(model_id: str) -> bool:
    """Check if model_id refers to a MoGe model."""
    return "moge" in model_id.lower()


def _infer_moge_frame(moge_model, frame_rgb: np.ndarray, device: str, dtype) -> DepthResult:
    """Run MoGe inference on a single frame and return DepthResult."""
    img_tensor = torch.tensor(frame_rgb / 255.0, dtype=dtype, device=device).permute(2, 0, 1)

    with torch.no_grad():
        output = moge_model.infer(img_tensor, resolution_level=9)

    points = output["points"].cpu().numpy()  # (H, W, 3)
    depth = points[..., 2]  # Z component is depth

    normals = None
    if "normal" in output and output["normal"] is not None:
        normals = output["normal"].cpu().numpy()  # (H, W, 3) in [-1, 1]

    mask = output.get("mask", None)
    if mask is not None:
        mask_np = mask.cpu().numpy() > 0.5
        valid_depth = depth[mask_np]
        if len(valid_depth) > 0:
            z_min = float(valid_depth.min())
            z_max = float(valid_depth.max())
        else:
            z_min, z_max = 0.0, 1.0
    else:
        z_min = float(depth.min())
        z_max = float(depth.max())

    return DepthResult(depth=depth, z_min=z_min, z_max=z_max, normals=normals)


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    """Convert normals from [-1, 1] range to [0, 255] uint8 for visualization."""
    normalized = ((normals + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return normalized


def _is_image_file(path: str) -> bool:
    """Check if path is an image file based on extension."""
    ext = str(path).lower()
    return ext.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))


def run_depth(args) -> None:
    """Run depth extraction on video or image and output H.265 video."""

    is_image = _is_image_file(args.input)

    if is_image:
        # Load single image
        img = cv2.imread(str(args.input))
        if img is None:
            raise RuntimeError(f"Cannot open image: {args.input}")
        frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
        height, width = img.shape[:2]
        fps = 30.0  # Default FPS for single image
        frame_count = 1
        logger.info(f"Loaded image: {args.input}")
    else:
        # Load video
        cap = cv2.VideoCapture(str(args.input))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.input}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        for idx in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    logger.info(f"Read {len(frames)} frames from {args.input}")
    logger.info(f"  Resolution: {width}x{height}, FPS: {fps:.2f}")

    from backend.models.depth_model import MultiDeviceDepthModel

    logger.info(f"Using InfiniDepth model: {args.model} with device_spec={args.device_spec}")
    model = MultiDeviceDepthModel(
        model_id=args.model,
        device_spec=args.device_spec,
    )
    model.process_res = args.process_res

    args.output.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    depth_frames = []
    z_mins = []
    z_maxs = []
    has_normals = False

    for batch_start in range(0, len(frames), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(frames))
        batch_frames = frames[batch_start:batch_end]

        results = model.infer_depth_batch(
            batch_frames,
            target_sizes=[(fr.shape[1], fr.shape[0]) for fr in batch_frames],
            batch_size=len(batch_frames),
        )

        for depth_pred in results:
            depth = depth_pred.depth
            z_min, z_max = depth_pred.z_min, depth_pred.z_max
            normals = depth_pred.normals
            z_mins.append(z_min)
            z_maxs.append(z_max)

            if normals is not None:
                has_normals = True

            if z_max > z_min:
                normalized = ((depth - z_min) / (z_max - z_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(depth, dtype=np.uint8)

            if has_normals and normals is not None:
                normal_viz = _normalize_normals(normals)
                depth_expanded = np.stack([normalized] * 3, axis=2)
                combined = np.concatenate([depth_expanded, normal_viz], axis=1)
                # depth_frames stores (H, W, 3) uint8
                depth_frames.append(combined)
            else:
                # Convert grayscale to 3-channel RGB
                rgb_frame = np.stack([normalized] * 3, axis=2)
                depth_frames.append(rgb_frame)

        elapsed = time.perf_counter() - start_time
        fps_rate = batch_end / elapsed if elapsed > 0 else 0
        logger.info(f"  Processed {batch_end}/{len(frames)} frames ({fps_rate:.1f} fps)")

    # Encode depth video using imageio (uses ffmpeg internally)
    import imageio

    depth_video_path = args.output / "depth_sequence.mp4"
    logger.info(f"Encoding {len(depth_frames)} depth frames to H.265 lossless...")
    imageio.mimwrite(
        depth_video_path,
        depth_frames,
        fps=fps,
        codec="hevc",
        quality=None,
        pixelformat="yuv444p" if has_normals else "yuv420p",
        output_params=["-crf", "0"],
    )

    # Encode RGB video
    rgb_video_path = args.output / "rgb_sequence.mp4"
    logger.info(f"Encoding {len(frames)} RGB frames to H.265...")
    imageio.mimwrite(
        rgb_video_path,
        frames,
        fps=fps,
        codec="hevc",
        quality=None,
        pixelformat="yuv420p",
        output_params=["-crf", "18"],
    )

    logger.info(f"Wrote depth video to {depth_video_path}")
    logger.info(f"Wrote RGB video to {rgb_video_path}")
    processed = len(frames)
    total_time = time.perf_counter() - start_time
    avg_fps = processed / total_time if total_time > 0 else 0

    metadata = {
        "video_path": "rgb_sequence.mp4",
        "depth_video_path": "depth_sequence.mp4",
        "frame_count": processed,
        "fps": fps,
        "source_width": width,
        "source_height": height,
        "source_resolution": [width, height],
        "model_id": args.model,
        "device_spec": args.device_spec,
        "process_res": args.process_res,
        "processing_time_s": total_time,
        "avg_fps": avg_fps,
        "format": "hevc_lossless",
        "z_min": min(z_mins),
        "z_max": max(z_maxs),
        "has_normals": has_normals,
        "side_by_side": has_normals,
    }
    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    model.shutdown()
