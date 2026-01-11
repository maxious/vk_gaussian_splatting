"""HLS streaming generator for RGB + RGB-packed depth side-by-side video.

This module generates a standards-compliant HLS stream where each frame contains:
- Left half: Original RGB video
- Right half: RGB-packed 16-bit depth data

The RGB packing scheme:
- Red channel: High byte of depth value (depth >> 8) & 0xFF
- Green channel: Low byte of depth value (depth >> 0) & 0xFF
- Blue channel: 0x80 (mid-point for visual contrast)

On the C++ client, the shader unpacks with:
    float depth = (color.r * 256.0 + color.g) * scale + z_min;
    where scale = (z_max - z_min) / 65535.0
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Optional, cast

import cv2
import numpy as np

from backend.config import get_settings
from backend.models.depth_model import (
    DepthModel,
    DepthPrediction,
    MultiDeviceDepthModel,
    get_depth_model,
)
from backend.video.io import FrameDecoder, VideoMetadata


@dataclass
class HlsMetadata:
    """Metadata returned to client for stream configuration."""

    stream_url: str  # URL to the master playlist
    width: int  # Total width (RGB width * 2)
    height: int  # Height of the video
    fps: float  # Frames per second
    duration_s: float  # Total duration in seconds
    z_min: float  # Minimum depth value for unpacking
    z_max: float  # Maximum depth value for maxking
    scale: float  # Scale factor: (z_max - z_min) / 65535.0
    rgb_width: int  # Width of RGB portion
    depth_width: int  # Width of depth portion


@dataclass
class HlsGeneratorState:
    """Tracks the state of HLS generation."""

    status: str = "pending"  # pending, processing, ready, error
    progress: float = 0.0  # 0.0 to 1.0
    frame_count: int = 0
    total_frames: int = 0
    error_message: str | None = None
    start_time: float = 0.0  # Unix timestamp when generation started
    eta_seconds: float | None = None  # Estimated time remaining in seconds
    frames_per_second: float = 0.0  # Processing speed
    sampling_frames: int = 10  # Number of frames used for depth range sampling
    sampling_start_time: float = 0.0  # When sampling phase started
    production_start_time: float = 0.0  # When actual HLS encoding started


class HlsGenerator:
    """Generates HLS stream with side-by-side RGB and RGB-packed depth."""

    def __init__(
        self,
        session_id: str,
        source_path: Path,
        output_dir: Path,
        fps: float = 30.0,
        segment_duration: float = 2.0,
        hls_time: float = 2.0,
        process_res: int = 640,
    ) -> None:
        """Initialize HLS generator.

        Args:
            session_id: Unique session identifier
            source_path: Path to input video file
            output_dir: Directory for output HLS files
            fps: Target FPS for output stream
            segment_duration: Duration of each HLS segment in seconds
            hls_time: FFmpeg hls_time parameter
            process_res: Depth processing resolution
        """
        self.session_id = session_id
        self.source_path = source_path
        self.output_dir = output_dir
        self.fps = fps
        self.segment_duration = segment_duration
        self.hls_time = hls_time
        self.process_res = process_res

        self.state = HlsGeneratorState()
        self._lock = threading.Lock()
        self._depth_model: Any = None
        self._ffmpeg_process: Optional[subprocess.Popen] = None

    def _get_depth_model(self) -> DepthModel | MultiDeviceDepthModel:
        """Get the appropriate depth model based on settings."""
        settings = get_settings()
        if self._depth_model is None:
            self._depth_model = get_depth_model(use_multi_device=settings.use_multi_device)
        return self._depth_model

    def pack_depth_rgb(self, depth: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
        """Pack 16-bit depth into RGB channels.

        Args:
            depth: 2D float32 depth array in meters
            z_min: Minimum depth value for normalization
            z_max: Maximum depth value for normalization

        Returns:
            3-channel uint8 RGB image where R=high byte, G=low byte, B=128
        """
        # Normalize to 0-65535 range
        scale = (z_max - z_min) / 65535.0
        if scale <= 0:
            scale = 1.0

        normalized = np.clip((depth - z_min) / scale, 0, 65535)
        uint16_val = np.rint(normalized).astype(np.uint16)

        # Pack into RGB
        rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (uint16_val >> 8) & 0xFF  # Red = high byte
        rgb[..., 1] = uint16_val & 0xFF  # Green = low byte
        rgb[..., 2] = 128  # Blue = mid-point for visual contrast

        return rgb

    def create_side_by_side(self, rgb: np.ndarray, depth_rgb: np.ndarray) -> np.ndarray:
        """Create side-by-side composition.

        Args:
            rgb: Original RGB frame (H, W, 3) uint8
            depth_rgb: RGB-packed depth (H, W, 3) uint8

        Returns:
            Side-by-side frame (H, W*2, 3) uint8
        """
        # Ensure depth has same height as RGB (resize if needed)
        if depth_rgb.shape[0] != rgb.shape[0]:
            depth_rgb = cv2.resize(
                depth_rgb,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return np.concatenate([rgb, depth_rgb], axis=1)

    async def generate(self) -> HlsMetadata:
        """Generate HLS stream. This is a blocking operation.

        Returns:
            HlsMetadata with stream URL and configuration
        """
        settings = get_settings()

        # Record start time for ETA calculation
        self.state.start_time = time.time()
        self.state.status = "processing"

        # Clean output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        # Get video metadata
        decoder = FrameDecoder(self.source_path)
        video_meta = decoder.metadata()
        decoder.close()

        # Use actual frame count from video metadata for accurate progress tracking
        if video_meta.frames:
            self.state.total_frames = int(video_meta.frames)
        else:
            duration_ms_val = video_meta.duration_ms
            fps_val = self.fps
            if duration_ms_val is not None and fps_val is not None:
                self.state.total_frames = int(duration_ms_val / 1000.0 * fps_val)
            else:
                self.state.total_frames = 0

        # Calculate dimensions
        rgb_width = video_meta.width
        rgb_height = video_meta.height
        depth_width = rgb_width  # Same width for depth
        depth_height = rgb_height
        total_width = rgb_width * 2

        # Calculate global z_min/z_max from a sample of frames
        z_min, z_max = await self._calculate_depth_range(decoder, video_meta)

        # Mark when actual HLS encoding starts (after sampling)
        self.state.production_start_time = time.time()

        # Store video duration for accurate frame timing
        video_duration_ms = (
            video_meta.duration_ms or (video_meta.frames / self.fps * 1000)
            if video_meta.frames
            else None
        )

        # Create FFmpeg pipeline for HLS generation
        hls_dir = self.output_dir / "hls"
        hls_dir.mkdir(parents=True)

        playlist_path = hls_dir / "playlist.m3u8"
        segment_pattern = hls_dir / "segment_%03d.ts"

        # FFmpeg command for HLS encoding
        # Using setpts=PTS-STARTPTS to ensure timestamps start at 0.0
        # This prevents "Invalid pts in seconds" errors from FFmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{total_width}x{rgb_height}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",  # Read from stdin
            "-fflags",
            "+genpts",  # Generate PTS if missing
            "-filter:v",
            "setpts=PTS-STARTPTS",  # Reset timestamps to start at 0
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(int(self.fps)),  # GOP size = 1 second
            "-hls_time",
            str(self.hls_time),
            "-hls_list_size",
            "0",  # Include all segments in playlist
            "-hls_segment_filename",
            str(segment_pattern),
            "-start_number",
            "0",
            str(playlist_path),
        ]

        # Start FFmpeg process
        self._ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            # Process frames and feed to FFmpeg
            decoder = FrameDecoder(self.source_path)
            frame_interval_ms = 1000.0 / self.fps
            frame_idx = 0
            max_frame_idx = None

            while True:
                # Check if process is still running
                if self._ffmpeg_process.poll() is not None:
                    raise RuntimeError("FFmpeg process died unexpectedly")

                # Calculate target timestamp for this frame
                target_time_ms = frame_idx * frame_interval_ms

                # Stop if we exceed video duration (with small tolerance)
                if video_duration_ms is not None and target_time_ms >= video_duration_ms - 1:
                    break

                try:
                    frame, frame_info = decoder.decode_at(target_time_ms)
                except StopIteration:
                    break

                progress = (frame_idx + 1) / max(self.state.total_frames, 1)

                production_elapsed_s = time.time() - self.state.production_start_time

                if frame_idx < 20:
                    current_fps = (frame_idx + 1) / max(production_elapsed_s, 0.001)
                    smoothing_alpha = 0.3
                else:
                    current_fps = (frame_idx + 1) / max(production_elapsed_s, 0.001)
                    smoothing_alpha = 0.1

                if self.state.frames_per_second == 0.0:
                    self.state.frames_per_second = current_fps
                else:
                    self.state.frames_per_second = (
                        smoothing_alpha * current_fps
                        + (1 - smoothing_alpha) * self.state.frames_per_second
                    )

                remaining_frames = max(self.state.total_frames - (frame_idx + 1), 0)
                eta_seconds = (
                    remaining_frames / self.state.frames_per_second
                    if self.state.frames_per_second > 0 and frame_idx >= 10
                    else None
                )

                with self._lock:
                    self.state.progress = progress
                    self.state.frame_count = frame_idx + 1
                    self.state.eta_seconds = eta_seconds

                # Ensure RGB is uint8
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                elif frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)

                # Run depth inference
                prediction = await self._get_depth_model().infer_depth_async(
                    frame,
                    process_res=self.process_res,
                    target_size=(depth_width, depth_height),
                )

                # Pack depth into RGB
                depth_rgb = self.pack_depth_rgb(prediction.depth, z_min, z_max)

                # Create side-by-side frame
                composite = self.create_side_by_side(frame, depth_rgb)

                # Write to FFmpeg stdin
                composite_bytes = composite.tobytes()
                stdin = self._ffmpeg_process.stdin
                if stdin is not None:
                    try:
                        stdin.write(composite_bytes)  # type: ignore[arg-type]
                    except Exception:
                        # FFmpeg may reject frames with invalid timestamps (e.g., last frame slightly out of bounds)
                        break

                frame_idx += 1

            decoder.close()

        except Exception as e:
            with self._lock:
                self.state.status = "error"
                self.state.error_message = str(e)
            raise

        finally:
            # Close FFmpeg stdin and wait for completion
            if self._ffmpeg_process.stdin:
                self._ffmpeg_process.stdin.close()
            self._ffmpeg_process.wait()

        # Write metadata JSON
        scale = (z_max - z_min) / 65535.0
        if scale <= 0:
            scale = 1.0

        metadata = HlsMetadata(
            stream_url=str(playlist_path.absolute()),
            width=total_width,
            height=rgb_height,
            fps=self.fps,
            duration_s=frame_idx / self.fps,
            z_min=z_min,
            z_max=z_max,
            scale=scale,
            rgb_width=rgb_width,
            depth_width=depth_width,
        )

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "stream_url": metadata.stream_url,
                    "width": metadata.width,
                    "height": metadata.height,
                    "fps": metadata.fps,
                    "duration_s": metadata.duration_s,
                    "z_min": metadata.z_min,
                    "z_max": metadata.z_max,
                    "scale": metadata.scale,
                    "rgb_width": metadata.rgb_width,
                    "depth_width": metadata.depth_width,
                },
                f,
                indent=2,
            )

        with self._lock:
            self.state.status = "ready"
            self.state.progress = 1.0

        return metadata

    async def _calculate_depth_range(
        self, decoder: FrameDecoder, video_meta: VideoMetadata
    ) -> tuple[float, float]:
        """Calculate global z_min/z_max from a sample of frames.

        Uses percentile-based estimation for robustness.
        """
        settings = get_settings()
        sample_count = 10
        frame_interval_ms = (
            (video_meta.duration_ms or 0) / sample_count if video_meta.duration_ms else 1000
        )

        z_min_samples = []
        z_max_samples = []

        for i in range(sample_count):
            try:
                frame, _ = decoder.decode_at(i * frame_interval_ms)
            except StopIteration:
                break

            # Run depth inference on sample
            prediction = await self._get_depth_model().infer_depth_async(
                frame,
                process_res=min(self.process_res, 384),  # Lower res for sampling
            )

            z_min_samples.append(prediction.z_min)
            z_max_samples.append(prediction.z_max)

        if z_min_samples and z_max_samples:
            z_min = float(np.percentile(z_min_samples, 5))  # 5th percentile
            z_max = float(np.percentile(z_max_samples, 95))  # 95th percentile
        else:
            z_min = 0.5
            z_max = 10.0

        return z_min, z_max

    def get_state(self) -> HlsGeneratorState:
        """Get current generation state."""
        with self._lock:
            return HlsGeneratorState(
                status=self.state.status,
                progress=self.state.progress,
                frame_count=self.state.frame_count,
                total_frames=self.state.total_frames,
                error_message=self.state.error_message,
                start_time=self.state.start_time,
                eta_seconds=self.state.eta_seconds,
                frames_per_second=self.state.frames_per_second,
            )

    def cancel(self) -> None:
        """Cancel ongoing generation."""
        if self._ffmpeg_process:
            self._ffmpeg_process.terminate()
            self._ffmpeg_process.wait()
        with self._lock:
            self.state.status = "cancelled"


# Global generator registry (session_id -> generator)
_hls_generators: dict[str, HlsGenerator] = {}
_hls_generators_lock = threading.Lock()


def get_hls_generator(session_id: str) -> Optional[HlsGenerator]:
    """Get an existing HLS generator."""
    with _hls_generators_lock:
        return _hls_generators.get(session_id)


def register_hls_generator(session_id: str, generator: HlsGenerator) -> None:
    """Register a new HLS generator."""
    with _hls_generators_lock:
        _hls_generators[session_id] = generator


def unregister_hls_generator(session_id: str) -> None:
    """Unregister an HLS generator (cleanup)."""
    with _hls_generators_lock:
        _hls_generators.pop(session_id, None)
