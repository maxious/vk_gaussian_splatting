"""Offline video depth extraction using Depth-Anything-3.

Simplified depth-only extraction with batched inference.
Uses chunk-based processing similar to DA3-streaming for memory efficiency.

Usage:
    python -m offline.video_depth --input video.mp4 --output ./depth_output/

    # Or with installed package:
    vkgs-depth --input video.mp4 --output ./depth_output/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

import cv2
import numpy as np

from .formats import VdzFrame, write_vdz_frame

if TYPE_CHECKING:
    import torch
    from depth_anything_3.api import DepthAnything3

logger = logging.getLogger(__name__)


@dataclass
class DepthConfig:
    """Configuration for depth extraction."""

    # Chunk settings (from DA3-streaming research)
    # Larger chunks = better temporal coherence but more VRAM
    chunk_size: int = 60  # frames per chunk
    overlap: int = 30  # 50% overlap for smooth transitions

    # Model settings
    model_id: str = "depth-anything/DA3MONO-LARGE"  # Fastest monocular depth
    process_res: int = 518  # Processing resolution (divisible by 14)
    device: str = "cuda"

    # Performance settings
    num_workers: int = 8  # Parallel image loading threads
    use_amp: bool = True  # Automatic mixed precision

    # Output settings
    compress_vdz: bool = True
    save_rgb: bool = False  # Also save RGB frames


@dataclass
class ChunkDepthResult:
    """Result from processing a single chunk."""

    chunk_idx: int
    frame_indices: list[int]
    depths: np.ndarray  # (N, H, W) float32 metric depth
    timestamps_ms: list[float]


class VideoReader:
    """Memory-efficient video frame reader."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_s = self.frame_count / self.fps if self.fps > 0 else 0

    def read_frame(self, idx: int) -> tuple[np.ndarray, float]:
        """Read single frame. Returns (RGB array, timestamp_ms)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError(f"Frame {idx} not available")

        timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms

    def read_frames_parallel(
        self,
        indices: list[int],
        num_workers: int = 8,
    ) -> list[tuple[np.ndarray, float]]:
        """Read multiple frames in parallel using thread pool."""
        results: list[tuple[np.ndarray, float]] = []

        def read_single(args: tuple[int, int]) -> tuple[int, np.ndarray | None, float]:
            pos, frame_idx = args  # enumerate gives (position, value)
            cap = cv2.VideoCapture(str(self.video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.release()
            if ret:
                return pos, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms
            return pos, None, 0.0

        # Collect results into ordered list
        ordered: dict[int, tuple[np.ndarray, float]] = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for pos, frame, ts in executor.map(read_single, enumerate(indices)):
                if frame is not None:
                    ordered[pos] = (frame, ts)

        for i in range(len(indices)):
            if i in ordered:
                results.append(ordered[i])

        return results

    def close(self):
        self.cap.release()


def check_cuda_available() -> None:
    """Assert that CUDA-enabled PyTorch is available."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Depth inference requires a CUDA-enabled PyTorch.\n"
            "Install with:\n"
            "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
        )

    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


class DepthExtractor:
    """Depth extraction using Depth-Anything-3."""

    def __init__(self, config: DepthConfig):
        self.config = config
        self.model: "DepthAnything3 | None" = None
        self.dtype: "torch.dtype | None" = None

    def _ensure_model_loaded(self):
        """Lazy load the model on first use."""
        if self.model is not None:
            return

        import torch

        check_cuda_available()

        from depth_anything_3.api import DepthAnything3

        logger.info(f"Loading model: {self.config.model_id}")
        self.model = DepthAnything3.from_pretrained(self.config.model_id)
        self.model = self.model.to(self.config.device).eval()

        # Use bfloat16 for newer GPUs (compute >= 8.0), else float16
        if torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

        logger.info(f"Model loaded, using {self.dtype}")

    def process_chunk(
        self,
        frames: list[np.ndarray],
        chunk_idx: int,
        frame_indices: list[int],
        timestamps_ms: list[float],
    ) -> ChunkDepthResult:
        """Process a batch of frames through the depth model."""
        import torch
        from PIL import Image

        self._ensure_model_loaded()
        assert self.model is not None

        logger.info(f"Processing chunk {chunk_idx}: {len(frames)} frames")
        start_time = time.perf_counter()

        # Clear GPU memory before processing
        torch.cuda.empty_cache()

        # Convert numpy arrays to PIL Images for the API
        images: list[np.ndarray | Image.Image | str] = [Image.fromarray(f) for f in frames]

        with torch.no_grad():
            if self.config.use_amp:
                with torch.autocast("cuda", dtype=self.dtype):
                    predictions = self.model.inference(
                        images,
                        process_res=self.config.process_res,
                    )
            else:
                predictions = self.model.inference(
                    images,
                    process_res=self.config.process_res,
                )

        # Extract depth - shape is (N, H, W)
        depths = predictions.depth
        if depths.ndim == 4:  # Sometimes (N, 1, H, W)
            depths = np.squeeze(depths, axis=1)

        elapsed = time.perf_counter() - start_time
        fps = len(frames) / elapsed
        logger.info(f"Chunk {chunk_idx} complete: {elapsed:.1f}s ({fps:.1f} FPS)")

        # Clear GPU memory after processing
        torch.cuda.empty_cache()

        return ChunkDepthResult(
            chunk_idx=chunk_idx,
            frame_indices=frame_indices,
            depths=depths.astype(np.float32),
            timestamps_ms=timestamps_ms,
        )


class VideoDepthProcessor:
    """Main processor for extracting depth from video."""

    def __init__(self, config: DepthConfig):
        self.config = config
        self.extractor = DepthExtractor(config)

    def _get_chunk_indices(
        self,
        total_frames: int,
    ) -> list[tuple[int, int]]:
        """Calculate chunk start/end indices with overlap."""
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap

        if total_frames <= chunk_size:
            return [(0, total_frames)]

        chunks = []
        step = chunk_size - overlap

        start = 0
        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunks.append((start, end))

            if end >= total_frames:
                break

            start += step

        return chunks

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
    ) -> None:
        """Process entire video and write VDZ depth sequence."""
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing video: {video_path}")
        video = VideoReader(video_path)

        logger.info(
            f"Video info: {video.frame_count} frames, "
            f"{video.fps:.2f} FPS, {video.width}x{video.height}, "
            f"{video.duration_s:.1f}s"
        )

        # Calculate chunks
        chunks = self._get_chunk_indices(video.frame_count)
        logger.info(
            f"Processing in {len(chunks)} chunks "
            f"(size={self.config.chunk_size}, overlap={self.config.overlap})"
        )

        # Process and write incrementally
        vdz_path = output_dir / "depth_sequence.vdz"
        total_start = time.perf_counter()

        with open(vdz_path, "wb") as f:
            frame_idx = 0
            overlap = self.config.overlap

            for chunk_idx, (start, end) in enumerate(chunks):
                frame_indices = list(range(start, end))

                # Read frames (parallel for better throughput)
                frames_data = video.read_frames_parallel(
                    frame_indices,
                    num_workers=self.config.num_workers,
                )

                frames = [fd[0] for fd in frames_data]
                timestamps = [fd[1] for fd in frames_data]

                # Process through depth model
                result = self.extractor.process_chunk(
                    frames,
                    chunk_idx,
                    frame_indices,
                    timestamps,
                )

                # Determine which frames to write (handle overlap)
                if chunk_idx == 0:
                    # First chunk: write all except overlap/2 at end
                    if chunk_idx < len(chunks) - 1:
                        write_start, write_end = 0, len(result.depths) - overlap // 2
                    else:
                        write_start, write_end = 0, len(result.depths)
                elif chunk_idx == len(chunks) - 1:
                    # Last chunk: skip overlap/2 at start
                    write_start, write_end = overlap // 2, len(result.depths)
                else:
                    # Middle chunks: skip overlap/2 on both ends
                    write_start = overlap // 2
                    write_end = len(result.depths) - overlap // 2

                # Write frames to VDZ
                for local_idx in range(write_start, write_end):
                    depth = result.depths[local_idx]
                    timestamp_ms = result.timestamps_ms[local_idx]

                    z_min = float(np.percentile(depth, 1))
                    z_max = float(np.percentile(depth, 99))

                    vdz_frame = VdzFrame(
                        timestamp_ms=timestamp_ms,
                        width=depth.shape[1],
                        height=depth.shape[0],
                        depth=depth,
                        z_min=z_min,
                        z_max=z_max,
                    )
                    write_vdz_frame(f, vdz_frame, compress=self.config.compress_vdz)
                    frame_idx += 1

        total_elapsed = time.perf_counter() - total_start
        avg_fps = frame_idx / total_elapsed

        logger.info(
            f"Wrote {frame_idx} depth frames to {vdz_path} "
            f"({total_elapsed:.1f}s, avg {avg_fps:.1f} FPS)"
        )

        # Write metadata
        meta = {
            "video_path": str(video_path),
            "frame_count": frame_idx,
            "fps": video.fps,
            "duration_s": video.duration_s,
            "source_resolution": [video.width, video.height],
            "config": {
                "chunk_size": self.config.chunk_size,
                "overlap": self.config.overlap,
                "model_id": self.config.model_id,
                "process_res": self.config.process_res,
            },
            "processing_time_s": total_elapsed,
            "avg_fps": avg_fps,
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        video.close()
        logger.info("Processing complete")


def main():
    parser = argparse.ArgumentParser(
        description="Extract depth maps from video using Depth-Anything-3"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=60, help="Frames per chunk (default: 60)")
    parser.add_argument(
        "--overlap", type=int, default=30, help="Overlap between chunks (default: 30)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3MONO-LARGE",
        help="Model ID (default: depth-anything/DA3MONO-LARGE)",
    )
    parser.add_argument(
        "--process-res", type=int, default=518, help="Processing resolution (default: 518)"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel frame loading workers (default: 8)"
    )
    parser.add_argument("--no-compress", action="store_true", help="Disable VDZ compression")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = DepthConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model_id=args.model,
        process_res=args.process_res,
        num_workers=args.workers,
        compress_vdz=not args.no_compress,
        use_amp=not args.no_amp,
    )

    processor = VideoDepthProcessor(config)
    processor.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
