"""Utilities for inspecting PyAV and torchcodec frames."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameInfo:
    time_ms: float
    index: int
    pts: int | None
    key_frame: bool


def frame_info_from_av(frame) -> FrameInfo:
    time_ms = float(frame.time) * 1000 if frame.time is not None else -1.0
    return FrameInfo(
        time_ms=time_ms,
        index=getattr(frame, "index", -1),
        pts=frame.pts,
        key_frame=bool(frame.key_frame),
    )


def frame_info_from_torchcodec(frame_batch, frame_idx: int = 0) -> FrameInfo:
    """Create FrameInfo from torchcodec FrameBatch.

    Args:
        frame_batch: torchcodec FrameBatch object
        frame_idx: Index of the specific frame within the batch
    """
    # Get PTS from the frame batch - pts_seconds can be a tensor or list
    pts_seconds = frame_batch.pts_seconds

    if hasattr(pts_seconds, "item"):  # It's a tensor
        pts_sec = float(pts_seconds[frame_idx].item()) if frame_idx < len(pts_seconds) else 0.0
    elif isinstance(pts_seconds, (list, tuple)):
        pts_sec = float(pts_seconds[frame_idx]) if frame_idx < len(pts_seconds) else 0.0
    else:
        # Scalar tensor
        pts_sec = float(pts_seconds) if frame_idx == 0 else 0.0

    time_ms = pts_sec * 1000.0

    # Get keyframe info if available
    key_frame = (
        getattr(frame_batch, "key_frames", [False])[frame_idx]
        if hasattr(frame_batch, "key_frames")
        else False
    )

    return FrameInfo(
        time_ms=time_ms,
        index=frame_idx,
        pts=int(pts_sec * 1000) if pts_sec else None,
        key_frame=key_frame,
    )

    return FrameInfo(
        time_ms=time_ms,
        index=frame_idx,
        pts=int(pts_sec * 1000) if pts_sec else None,
        key_frame=key_frame,
    )
