"""Depth extraction command for offline preprocessing."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.models.depth_model import MultiDeviceDepthModel
from offline.formats import VdzFrame, write_vdz_frame

logger = logging.getLogger(__name__)


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    """Convert normals from [-1, 1] range to [0, 255] uint8 for visualization.

    Args:
        normals: (H, W, 3) array in [-1, 1] range

    Returns:
        (H, W, 3) uint8 array
    """
    # Convert from [-1, 1] to [0, 255]
    normalized = ((normals + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return normalized


def run_depth(args) -> None:
    """Run depth extraction on video."""
    # Suppress verbose DA3 logs
    os.environ["DA3_LOG_LEVEL"] = "WARN"

    # Read video frames
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    timestamps_ms = []
    for idx in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()

    logger.info(f"Read {len(frames)} frames from {args.input}")
    logger.info(f"  Resolution: {width}x{height}, FPS: {fps:.2f}")

    # Initialize multi-device model
    logger.info(f"Using device_spec={args.device_spec}")
    model = MultiDeviceDepthModel(
        model_id=args.model,
        device_spec=args.device_spec,
    )
    model.process_res = args.process_res

    # Process frames
    args.output.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    if args.format == "video":
        # Use torchcodec to encode H.265 lossless video
        from torchcodec.encoders import VideoEncoder

        depth_tensors = []
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

                # Normalize to 0-255
                if z_max > z_min:
                    normalized = ((depth - z_min) / (z_max - z_min) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(depth, dtype=np.uint8)

                if has_normals and normals is not None:
                    # Create side-by-side: depth on left, normals on right
                    normal_viz = _normalize_normals(normals)
                    # Expand depth to have channel dimension for concatenation
                    depth_expanded = np.stack([normalized] * 3, axis=2)  # (H, W) -> (H, W, 3)
                    # Concatenate horizontally
                    combined = np.concatenate([depth_expanded, normal_viz], axis=1)
                    # Convert to tensor (H, W, 3) -> (1, 3, H, W) for video encoding
                    tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)
                else:
                    # Just depth
                    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
                    tensor = tensor.repeat(1, 3, 1, 1)
                depth_tensors.append(tensor)

        # Stack all frames
        all_frames = torch.cat(depth_tensors, dim=0)

        # Encode to H.265 lossless
        video_path = args.output / "depth_sequence.mp4"
        logger.info(f"Encoding {all_frames.shape[0]} frames to H.265 lossless...")
        encoder = VideoEncoder(frames=all_frames, frame_rate=fps)
        encoder.to_file(video_path, codec="hevc", pixel_format="yuv444p", crf=0)

        logger.info(f"Wrote depth video to {video_path}")
        processed = len(frames)
        total_time = time.perf_counter() - start_time
        avg_fps = processed / total_time if total_time > 0 else 0

        metadata = {
            "video_path": str(args.input),
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

    else:
        # Original VDZ format
        vdz_path = args.output / "depth_sequence.vdz"
        logger.info(f"Processing {len(frames)} frames (batch_size={args.batch_size})")

        processed = 0
        z_mins = []
        z_maxs = []

        with open(vdz_path, "wb") as f:
            for batch_start in range(0, len(frames), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(frames))
                batch_frames = frames[batch_start:batch_end]
                batch_timestamps = timestamps_ms[batch_start:batch_end]

                results = model.infer_depth_batch(
                    batch_frames,
                    target_sizes=[(fr.shape[1], fr.shape[0]) for fr in batch_frames],
                    batch_size=len(batch_frames),
                )

                for depth_pred, ts in zip(results, batch_timestamps):
                    z_mins.append(depth_pred.z_min)
                    z_maxs.append(depth_pred.z_max)

                    vdz_frame = VdzFrame(
                        timestamp_ms=ts,
                        width=depth_pred.depth.shape[1],
                        height=depth_pred.depth.shape[0],
                        depth=depth_pred.depth,
                        z_min=depth_pred.z_min,
                        z_max=depth_pred.z_max,
                    )
                    write_vdz_frame(f, vdz_frame, compress=True)
                    processed += 1

                elapsed = time.perf_counter() - start_time
                fps_rate = processed / elapsed if elapsed > 0 else 0
                logger.info(f"  Processed {processed}/{len(frames)} frames ({fps_rate:.1f} fps)")

        total_time = time.perf_counter() - start_time
        avg_fps = len(frames) / total_time if total_time > 0 else 0

        logger.info(f"Wrote {processed} depth frames to {vdz_path}")
        logger.info(f"Total time: {total_time:.1f}s, Average: {avg_fps:.1f} fps")

        metadata = {
            "video_path": str(args.input),
            "depth_video_path": "depth_sequence.vdz",
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
            "format": "vdz",
            "z_min": min(z_mins),
            "z_max": max(z_maxs),
        }
        with open(args.output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    model.shutdown()
