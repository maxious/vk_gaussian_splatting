"""Video decoding helpers built on top of PyAV and torchcodec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import threading

# Optional PyAV import (only needed for FrameDecoder)
try:
    import av  # type: ignore[import-untyped]

    _PYAV_AVAILABLE = True
except ImportError:
    av = None  # type: ignore[assignment]
    _PYAV_AVAILABLE = False

from backend.utils.frame_info import FrameInfo, frame_info_from_av, frame_info_from_torchcodec


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    frames: Optional[int]
    fps: float
    duration_ms: Optional[float]

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0


class FrameDecoder:
    """Thin wrapper around PyAV for timestamp-based decoding."""

    SEEK_PAD_MS = 0.0
    STREAM_WINDOW_MS = 1000.0
    MAX_SCAN_FRAMES = 360

    def __init__(self, source: Path) -> None:
        self.source = source
        self._container = av.open(str(source))
        self._stream = self._container.streams.video[0]
        self._frame_iter: Optional[Iterator] = None
        self._last_frame_time_ms: Optional[float] = None

    def metadata(self) -> VideoMetadata:
        stream = self._stream
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        duration_ms = float(stream.duration * stream.time_base * 1000) if stream.duration else None
        return VideoMetadata(
            width=stream.width,
            height=stream.height,
            frames=stream.frames or None,
            fps=fps,
            duration_ms=duration_ms,
        )

    def decode_at(self, time_ms: float) -> tuple[np.ndarray, FrameInfo]:
        """Decode the frame nearest to the requested timestamp (ms).

        The decoder defaults to forward streaming: if the caller requests a
        timestamp slightly ahead of the last frame we returned, we simply advance
        the existing decode iterator. When the caller jumps far ahead or backwards
        we fall back to a guarded seek toward the preceding keyframe, then resume
        forward decoding until we reach (or slightly surpass) the target time.
        """

        time_ms = max(time_ms, 0.0)
        if not self._should_stream_forward(time_ms):
            self._seek_near(time_ms)
        frame, info = self._advance_to(time_ms)
        return frame, info

    def iter_frames(self) -> Iterator[np.ndarray]:
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                yield frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        self._container.close()

    def __del__(self) -> None:  # pragma: no cover - destructor best-effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _reset_iterator(self) -> None:
        self._frame_iter = self._container.decode(self._stream)

    def should_stream_forward(self, time_ms: float) -> bool:
        if self._last_frame_time_ms is None:
            return False
        delta = time_ms - self._last_frame_time_ms
        return 0.0 <= delta <= self.STREAM_WINDOW_MS

    def _should_stream_forward(self, time_ms: float) -> bool:
        return self.should_stream_forward(time_ms)

    def _seek_near(self, time_ms: float) -> None:
        # Seek exactly to the target time with backward=True.
        # This finds the closest keyframe <= time_ms.
        seek_ts = time_ms / 1000.0
        time_base = float(self._stream.time_base)
        target_pts = int(seek_ts / time_base)
        self._container.seek(target_pts, stream=self._stream, any_frame=False, backward=True)
        self._reset_iterator()
        self._last_frame_time_ms = None

    def _advance_to(self, time_ms: float) -> tuple[np.ndarray, FrameInfo]:
        if self._frame_iter is None:
            self._reset_iterator()
        assert self._frame_iter is not None
        frames_examined = 0
        while True:
            try:
                frame = next(self._frame_iter)
            except StopIteration as exc:  # pragma: no cover - EOF
                self._frame_iter = None
                raise StopIteration from exc
            info = frame_info_from_av(frame)
            frames_examined += 1
            actual_time = info.time_ms if info.time_ms >= 0 else time_ms
            self._last_frame_time_ms = actual_time
            if (
                info.time_ms < 0
                or actual_time >= time_ms
                or frames_examined >= self.MAX_SCAN_FRAMES
            ):
                return frame.to_ndarray(format="rgb24"), info


class TorchcodecFrameDecoder:
    """Video decoder using torchcodec for fast, accurate timestamp decoding.

    Advantages over PyAV:
    - ~2x faster decoding with num_ffmpeg_threads=0
    - Accurate PTS timestamps from video stream
    - Supports GPU decoding (CUDA backend)

    Note: torchcodec VideoDecoder has state issues when reusing - create fresh
    decoder instances for each decode_at() call or accept sequential access.
    """

    SEEK_PAD_MS = 0.0
    STREAM_WINDOW_MS = 1000.0

    def __init__(self, source: Path, num_ffmpeg_threads: int = 0) -> None:
        """Initialize torchcodec decoder.

        Args:
            source: Path to video file
            num_ffmpeg_threads: FFmpeg thread count. 0 = auto, 1 = single-threaded.
                               Use 0 for single decoder, 1 for multiple parallel decoders.
        """
        self.source = source
        self._num_ffmpeg_threads = num_ffmpeg_threads
        self._container = self._create_decoder()
        self._stream = self._container.metadata
        self._total_frames: Optional[int] = None
        self._fps: Optional[float] = None
        self._frame_indices: list[int] = []

        # Initialize frame index mapping
        self._init_frame_mapping()

    def _create_decoder(self):
        """Create a fresh VideoDecoder instance."""
        try:
            from torchcodec.decoders import VideoDecoder
        except ImportError as exc:
            raise ImportError(
                "torchcodec is required for TorchcodecFrameDecoder. "
                "Install with: pip install torchcodec"
            ) from exc
        return VideoDecoder(str(self.source), num_ffmpeg_threads=self._num_ffmpeg_threads)

    def _init_frame_mapping(self) -> None:
        """Initialize frame index to timestamp mapping."""
        total = self._stream.num_frames
        fps = self._stream.average_fps

        if total is None or fps is None or fps == 0:
            self._total_frames = 0
            self._fps = 30.0
            self._frame_indices = []
            return

        self._total_frames = total
        self._fps = float(fps)
        self._frame_indices = list(range(total))

    def metadata(self) -> VideoMetadata:
        fps = self._fps or 30.0
        duration_ms = (self._total_frames / fps * 1000) if self._total_frames else None
        # torchcodec metadata may have Optional width/height
        width = self._stream.width or 640
        height = self._stream.height or 480
        return VideoMetadata(
            width=width,
            height=height,
            frames=self._total_frames,
            fps=fps,
            duration_ms=duration_ms,
        )

    def decode_at(self, time_ms: float) -> tuple[np.ndarray, FrameInfo]:
        """Decode the frame nearest to the requested timestamp (ms).

        Uses get_frame_played_at for accurate timestamp-based seeking.
        torchcodec's get_frame_played_at(seconds) returns the frame at that exact timestamp.
        """
        if self._total_frames is None or self._total_frames == 0:
            raise StopIteration("No frames in video")

        time_ms = max(time_ms, 0.0)
        time_sec = time_ms / 1000.0

        # Create fresh decoder for this decode operation (torchcodec state issues)
        decoder = self._create_decoder()

        try:
            # Use get_frame_played_at for accurate timestamp-based seeking
            frame = decoder.get_frame_played_at(time_sec)

            # Convert to numpy (RGB)
            frame_np = frame.data.permute(1, 2, 0).cpu().numpy()

            # Get accurate timestamp from PTS
            pts_ms = float(frame.pts_seconds) * 1000

            info = FrameInfo(
                time_ms=pts_ms,
                index=-1,  # Frame object doesn't expose index
                pts=int(pts_ms) if pts_ms else None,
                key_frame=False,  # Frame object doesn't expose key_frame
            )

            return frame_np, info

        finally:
            del decoder

    def should_stream_forward(self, time_ms: float) -> bool:
        """Check if we can stream forward from current position."""
        # Torchcodec doesn't maintain forward streaming state
        # Always return False to force fresh decode
        return False

    def close(self) -> None:
        if hasattr(self, "_container") and self._container is not None:
            try:
                del self._container
                self._container = None
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DecoderPool:
    """Manages a pool of FrameDecoders for parallel random access with locality awareness.

    Supports both PyAV and torchcodec decoders. Torchcodec is preferred for:
    - ~2x faster decoding (with num_ffmpeg_threads=0)
    - Accurate PTS timestamps from video stream
    """

    def __init__(
        self,
        source: Path,
        count: int = 4,
        use_torchcodec: bool = True,
        num_ffmpeg_threads: int = 0,
    ) -> None:
        """Initialize decoder pool.

        Args:
            source: Path to video file
            count: Number of decoders in pool
            use_torchcodec: Use torchcodec instead of PyAV (default: True)
            num_ffmpeg_threads: FFmpeg thread count for torchcodec (0 = auto)
        """
        self.source = source
        self.count = count
        self.use_torchcodec = use_torchcodec

        if use_torchcodec:
            # torchcodec: Create decoders that can be reused (each creates fresh VideoDecoder per decode)
            self._decoders = [
                TorchcodecFrameDecoder(source, num_ffmpeg_threads=num_ffmpeg_threads)
                for _ in range(count)
            ]
        else:
            # PyAV: Traditional decoder pool with stateful decoders
            self._decoders = [FrameDecoder(source) for _ in range(count)]

        self._free_decoders = list(self._decoders)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def metadata(self) -> VideoMetadata:
        # Just peek at one decoder
        with self._cond:
            while not self._free_decoders:
                self._cond.wait()
            decoder = self._free_decoders.pop()

        try:
            return decoder.metadata()
        finally:
            with self._cond:
                self._free_decoders.append(decoder)
                self._cond.notify()

    def decode_at(self, time_ms: float) -> tuple[np.ndarray, FrameInfo]:
        # Block until a decoder is available
        with self._cond:
            while not self._free_decoders:
                self._cond.wait()

            # Smart Scheduling: Find a decoder that is close to the target time
            # to avoid expensive seeking.
            best_decoder = None
            for d in self._free_decoders:
                if d.should_stream_forward(time_ms):
                    best_decoder = d
                    break

            # If no suitable decoder found, pick the most recently used one (LIFO)
            # to keep "hot" decoders active, or just any.
            if best_decoder is None:
                best_decoder = self._free_decoders.pop()  # Pop from end (LIFO)
            else:
                self._free_decoders.remove(best_decoder)

        try:
            return best_decoder.decode_at(time_ms)
        finally:
            with self._cond:
                self._free_decoders.append(best_decoder)
                self._cond.notify()

    def close(self) -> None:
        with self._lock:
            for decoder in self._decoders:
                try:
                    decoder.close()
                except Exception:
                    pass
            self._decoders.clear()
            self._free_decoders.clear()
