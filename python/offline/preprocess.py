"""Offline video preprocessing using DA3-streaming for temporally consistent depth.

This module provides chunk-based depth estimation with Sim3 alignment between
chunks, producing per-frame VDZ files with globally consistent scale and
optional camera pose estimation.

Usage:
    python -m offline.preprocess --input video.mp4 --output ./preprocessed/

    # Or with installed package:
    vkgs-preprocess --input video.mp4 --output ./preprocessed/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch

from common.datasets import VideoDataset
from numba import jit


logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for offline preprocessing."""

    # Chunk settings (from DA3-streaming)
    chunk_size: int = 60
    overlap: int = 30

    # Model settings
    # DA3-LARGE has camera decoder for pose estimation
    # DA3METRIC-LARGE is depth-only (no camera params)
    model_id: str = "depth-anything/DA3-LARGE"
    process_res: int = 504
    device: str = "cuda"

    # Output settings
    output_depth_format: str = "vdz"  # "vdz" or "npz"
    compress_vdz: bool = True
    save_camera_poses: bool = True
    downsample_factor: int = 1

    # Alignment settings
    align_method: str = "sim3"  # "sim3" or "scale+se3"
    conf_threshold_coef: float = 0.5

    # Loop closure (optional, expensive)
    loop_closure: bool = False
    loop_similarity_threshold: float = 0.85


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""

    chunk_idx: int
    frame_indices: list[int]
    depths: np.ndarray  # (N, H, W)
    confidences: np.ndarray  # (N, H, W)
    intrinsics: np.ndarray  # (N, 3, 3)
    extrinsics: np.ndarray  # (N, 3, 4) w2c
    timestamps_ms: list[float]


@dataclass
class AlignmentTransform:
    """Sim3 transform between chunks."""

    scale: float
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3


class VideoFrameExtractor:
    """Extract frames from video file."""

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

    def extract_all(self, output_dir: Path) -> list[Path]:
        """Extract all frames to directory. Returns list of frame paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.frame_count):
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_path = output_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)

            if i % 100 == 0:
                logger.info(f"Extracted frame {i}/{self.frame_count}")

        return frame_paths

    def get_frame(self, idx: int) -> tuple[np.ndarray, float]:
        """Get single frame and timestamp."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError(f"Frame {idx} not available")

        timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms

    def close(self):
        self.cap.release()


def check_cuda_available() -> None:
    """Assert that CUDA-enabled PyTorch is available."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Depth inference requires a CUDA-enabled PyTorch installation.\n"
            "Install CUDA-enabled PyTorch with:\n"
            "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "Or for cu130:\n"
            "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130\n"
            f"\nCurrent PyTorch: {torch.__version__}\n"
            f"CUDA available: {torch.cuda.is_available()}\n"
            f"CUDA version (compile): {torch.version.cuda}"
        )

    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


class DA3StreamingProcessor:
    """Wrapper around DA3-streaming for chunk-based inference."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy load the DA3 model."""
        import torch

        # Verify CUDA is available
        check_cuda_available()

        from depth_anything_3.api import DepthAnything3

        logger.info(f"Loading model: {self.config.model_id}")
        self.model = DepthAnything3.from_pretrained(self.config.model_id)
        self.model = self.model.to(self.config.device).eval()
        self.dtype = torch.float16

    def process_chunk_from_paths(
        self,
        frame_paths: list[Path],
        chunk_idx: int,
        timestamps_ms: list[float],
    ) -> ChunkResult:
        """Process a chunk of frames from file paths (original method)."""
        import torch
        from PIL import Image

        logger.info(f"Processing chunk {chunk_idx}: {len(frame_paths)} frames")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.dtype):
                # DA3 accepts list of paths or PIL images
                images = [str(p) for p in frame_paths]

                # Use ref_view_strategy for temporal consistency
                # Options: "first", "middle", "saddle_balanced", "saddle_sim_range"
                predictions = self.model.inference(  # type: ignore[attr-defined]  # type: ignore[attr-defined]
                    images,
                    process_res=self.config.process_res,
                    ref_view_strategy="saddle_balanced",
                )

        return self._extract_predictions(predictions, chunk_idx, timestamps_ms, len(frame_paths))

    def process_chunk_from_tensors(
        self,
        frame_tensors,
        chunk_idx: int,
        timestamps_ms: list[float],
    ) -> ChunkResult:
        """Process a chunk of frames from torch tensors (torchcodec optimization).

        This method avoids the inefficient PNG extraction -> re-decoding pipeline
        by accepting torch tensors directly from torchcodec.FrameBatch.

        Args:
            frame_tensors: (B, C, H, W) uint8 tensor from torchcodec
            chunk_idx: Chunk index for logging
            timestamps_ms: List of timestamps for each frame

        Returns:
            ChunkResult with depth, confidence, intrinsics, extrinsics
        """
        logger.info(f"Processing chunk {chunk_idx}: {frame_tensors.shape[0]} frames (torchcodec)")

        # Convert torch tensors to numpy arrays for DA3
        # FrameBatch.data is (B, C, H, W) uint8, need (B, H, W, C) for DA3
        frame_np_list = [
            frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            for frame in frame_tensors
        ]

        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.dtype):
                # Use ref_view_strategy for temporal consistency
                predictions = self.model.inference(  # type: ignore[attr-defined]
                    frame_np_list,
                    process_res=self.config.process_res,
                    ref_view_strategy="saddle_balanced",
                )

        return self._extract_predictions(predictions, chunk_idx, timestamps_ms, len(frame_np_list))

    def _extract_predictions(
        self,
        predictions,
        chunk_idx: int,
        timestamps_ms: list[float],
        num_frames: int,
    ) -> ChunkResult:
        """Extract and format predictions from DA3 model output."""
        import numpy as np

        # Extract results - depth is (N, H, W)
        depths = predictions.depth
        if depths.ndim == 4:  # Sometimes (N, 1, H, W)
            depths = np.squeeze(depths, axis=1)

        # Confidence may be None
        if predictions.conf is not None:
            confs = predictions.conf
            if confs.ndim == 4:
                confs = np.squeeze(confs, axis=1)
        else:
            confs = np.ones_like(depths)

        # Intrinsics (N, 3, 3)
        intrinsics = predictions.intrinsics
        if intrinsics is None:
            # Fallback: estimate intrinsics from image size
            H, W = depths.shape[1], depths.shape[2]
            fx = fy = max(H, W)  # Approximate
            cx, cy = W / 2, H / 2
            intrinsics = np.array(
                [[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]] * len(depths), dtype=np.float32
            )

        # Extrinsics from DA3-LARGE are (N, 3, 4) - this is w2c format already
        extrinsics = predictions.extrinsics
        if extrinsics is None:
            # Fallback: identity poses
            extrinsics = np.zeros((len(depths), 3, 4), dtype=np.float32)
            for i in range(len(depths)):
                extrinsics[i] = np.eye(4, dtype=np.float32)[:3, :]

        return ChunkResult(
            chunk_idx=chunk_idx,
            frame_indices=list(range(num_frames)),
            depths=depths,
            confidences=confs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            timestamps_ms=timestamps_ms,
        )

    # Backward compatibility alias
    process_chunk = process_chunk_from_paths


@jit(nopython=True)
def estimate_sim3(source_points: np.ndarray, target_points: np.ndarray) -> AlignmentTransform:
    """Estimate Sim3 transform from source to target point clouds.

    Adapted from DA3-streaming/loop_utils/sim3utils.py
    """
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt((src_centered**2).sum(axis=1).mean())
    scale_tgt = np.sqrt((tgt_centered**2).sum(axis=1).mean())
    s = scale_tgt / scale_src

    src_scaled = src_centered * s
    H = src_scaled.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src

    return AlignmentTransform(scale=s, rotation=R, translation=t)


@jit(nopython=True)
def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """Convert depth map to world-space point cloud.

    Args:
        depth: (H, W) depth map
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (3, 4) world-to-camera transform

    Returns:
        (H, W, 3) point cloud in world space
    """
    H, W = depth.shape

    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera space
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    points_cam = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    # Transform to world space
    R = np.ascontiguousarray(extrinsics[:3, :3])
    t = np.ascontiguousarray(extrinsics[:3, 3])

    # w2c -> c2w
    R_inv = R.T
    t_inv = -R_inv @ t

    points_world = points_cam @ R_inv.T + t_inv

    return points_world


def align_chunks(
    chunk1: ChunkResult,
    chunk2: ChunkResult,
    overlap: int,
    conf_threshold: float,
) -> AlignmentTransform:
    """Align chunk2 to chunk1 using overlapping frames."""

    # Get overlapping depth maps
    depths1 = chunk1.depths[-overlap:]
    depths2 = chunk2.depths[:overlap]
    confs1 = chunk1.confidences[-overlap:]
    confs2 = chunk2.confidences[:overlap]
    intrinsics1 = chunk1.intrinsics[-overlap:]
    intrinsics2 = chunk2.intrinsics[:overlap]
    extrinsics1 = chunk1.extrinsics[-overlap:]
    extrinsics2 = chunk2.extrinsics[:overlap]

    # Convert to point clouds
    all_pts1 = []
    all_pts2 = []

    for i in range(overlap):
        pts1 = depth_to_point_cloud(depths1[i], intrinsics1[i], extrinsics1[i])
        pts2 = depth_to_point_cloud(depths2[i], intrinsics2[i], extrinsics2[i])

        # Filter by confidence
        mask = (confs1[i] > conf_threshold) & (confs2[i] > conf_threshold)

        all_pts1.append(pts1[mask])
        all_pts2.append(pts2[mask])

    pts1_flat = np.concatenate(all_pts1, axis=0)
    pts2_flat = np.concatenate(all_pts2, axis=0)

    logger.info(f"Aligning with {len(pts1_flat)} corresponding points")

    # Estimate Sim3
    transform = estimate_sim3(pts2_flat, pts1_flat)

    return transform


def apply_sim3(points: np.ndarray, transform: AlignmentTransform) -> np.ndarray:
    """Apply Sim3 transform to points."""
    s, R, t = transform.scale, transform.rotation, transform.translation
    return (s * (R @ points.T)).T + t


class OfflinePreprocessor:
    """Main offline preprocessing pipeline."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.processor = DA3StreamingProcessor(config)

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        temp_dir: Optional[Path] = None,
    ) -> None:
        """Process entire video with chunk-based alignment (original PNG extraction method)."""
        from .formats import (
            VdzFrame,
            write_vdz_frame,
            write_camera_poses,
            CameraPose,
            VdsHeader,
            write_vds_header,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_dir or output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames
        logger.info(f"Extracting frames from {video_path}")
        extractor = VideoFrameExtractor(video_path)
        frame_dir = temp_dir / "frames"
        frame_paths = extractor.extract_all(frame_dir)

        # Get timestamps
        timestamps = [i * 1000.0 / extractor.fps for i in range(len(frame_paths))]

        # Process chunks
        chunk_results, transforms = self._process_chunks_from_paths(
            frame_paths, timestamps, extractor.fps
        )

        # Write output
        logger.info("Writing output files")
        self._write_output(
            output_dir,
            extractor,
            chunk_results,
            transforms,
        )

        extractor.close()
        logger.info(f"Preprocessing complete: {output_dir}")

    def process_video_torchcodec(
        self,
        video_path: Path,
        output_dir: Path,
        temp_dir: Optional[Path] = None,
    ) -> None:
        """Process entire video with torchcodec (direct video decoding, no PNG intermediate).

        This method is more efficient than process_video() because it:
        - Avoids PNG encoding/decoding overhead
        - Uses direct memory-to-memory video decoding
        - Eliminates disk I/O for intermediate files

        Requires torchcodec to be installed: pip install torchcodec
        """
        from common.datasets import VideoDataset

        from .formats import (
            VdzFrame,
            write_vdz_frame,
            write_camera_poses,
            CameraPose,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_dir or output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create VideoDataset for direct video decoding
        logger.info(f"Opening video with torchcodec: {video_path}")
        dataset = VideoDataset(video_path)

        # Get video metadata
        fps = dataset._metadata.average_fps
        num_frames = len(dataset)

        # Get timestamps from VideoDataset
        timestamps = dataset._timestamps_ms

        # Process chunks using torch tensors
        chunk_results, transforms = self._process_chunks_from_tensors(dataset, timestamps, fps)

        # Write output
        logger.info("Writing output files")
        self._write_output(
            output_dir,
            None,  # No VideoFrameExtractor in torchcodec mode
            chunk_results,
            transforms,
        )

        dataset.close()
        logger.info(f"Preprocessing complete (torchcodec): {output_dir}")

    def _process_chunks_from_paths(
        self,
        frame_paths: list[Path],
        timestamps: list[float],
        fps: float,
    ) -> tuple[list[ChunkResult], list[AlignmentTransform]]:
        """Process video frames from file paths (original method)."""
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        chunks = self._get_chunk_indices(len(frame_paths), chunk_size, overlap)

        logger.info(f"Processing {len(frame_paths)} frames in {len(chunks)} chunks")

        chunk_results: list[ChunkResult] = []
        transforms: list[AlignmentTransform] = []

        for i, (start, end) in enumerate(chunks):
            chunk_frames = frame_paths[start:end]
            chunk_timestamps = timestamps[start:end]

            result = self.processor.process_chunk_from_paths(chunk_frames, i, chunk_timestamps)
            chunk_results.append(result)

            # Align with previous chunk
            if i > 0:
                conf_threshold = np.mean(result.confidences) * self.config.conf_threshold_coef
                transform = align_chunks(
                    chunk_results[i - 1],
                    result,
                    overlap,
                    conf_threshold,
                )
                transforms.append(transform)
                logger.info(f"Chunk {i} aligned: scale={transform.scale:.4f}")

        # Accumulate transforms
        cumulative_transforms = self._accumulate_transforms(transforms)

        return chunk_results, cumulative_transforms

    def _process_chunks_from_tensors(
        self,
        dataset: VideoDataset,
        timestamps: list[float],
        fps: float,
    ) -> tuple[list[ChunkResult], list[AlignmentTransform]]:
        """Process video frames using torchcodec tensors (optimized method)."""
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        num_frames = len(dataset)
        chunks = self._get_chunk_indices(num_frames, chunk_size, overlap)

        logger.info(f"Processing {num_frames} frames in {len(chunks)} chunks (torchcodec)")

        chunk_results: list[ChunkResult] = []
        transforms: list[AlignmentTransform] = []

        for i, (start, end) in enumerate(chunks):
            # Get frame indices for this chunk
            chunk_indices = list(range(start, end))
            chunk_timestamps = timestamps[start:end]

            # Decode batch of frames using torchcodec
            frame_batch = dataset._decoder.get_frames_at(chunk_indices)
            frame_tensors = frame_batch.data  # (B, C, H, W) uint8

            result = self.processor.process_chunk_from_tensors(frame_tensors, i, chunk_timestamps)
            chunk_results.append(result)

            # Align with previous chunk
            if i > 0:
                conf_threshold = (
                    float(np.mean(result.confidences)) * self.config.conf_threshold_coef
                )
                transform = align_chunks(
                    chunk_results[i - 1],
                    result,
                    overlap,
                    conf_threshold,
                )
                transforms.append(transform)
                logger.info(f"Chunk {i} aligned: scale={transform.scale:.4f}")

        # Accumulate transforms
        cumulative_transforms = self._accumulate_transforms(transforms)

        return chunk_results, cumulative_transforms

    def _get_chunk_indices(
        self,
        total_frames: int,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """Calculate chunk start/end indices."""
        chunks = []
        start = 0

        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunks.append((start, end))

            if end >= total_frames:
                break

            start = end - overlap

        return chunks

    def _accumulate_transforms(
        self,
        transforms: list[AlignmentTransform],
    ) -> list[AlignmentTransform]:
        """Accumulate sequential transforms to get global alignment."""
        if not transforms:
            return []

        cumulative = [transforms[0]]

        for i in range(1, len(transforms)):
            prev = cumulative[-1]
            curr = transforms[i]

            # Compose: T_cumulative = T_prev * T_curr
            R_new = prev.rotation @ curr.rotation
            s_new = prev.scale * curr.scale
            t_new = prev.scale * (prev.rotation @ curr.translation) + prev.translation

            cumulative.append(
                AlignmentTransform(
                    scale=s_new,
                    rotation=R_new,
                    translation=t_new,
                )
            )

        return cumulative

    def _write_output(
        self,
        output_dir: Path,
        extractor: Optional[VideoFrameExtractor],
        chunk_results: list[ChunkResult],
        transforms: list[AlignmentTransform],
    ) -> None:
        """Write VDZ sequence and camera poses."""
        from .formats import VdzFrame, write_vdz_frame, CameraPose, write_camera_poses

        overlap = self.config.overlap
        all_poses: list[CameraPose] = []

        # Get video metadata from extractor if available
        video_path = str(extractor.video_path) if extractor else "torchcodec"
        fps = extractor.fps if extractor else 0.0
        duration_s = extractor.duration_s if extractor else 0.0
        width = extractor.width if extractor else 0
        height = extractor.height if extractor else 0

        # Write VDZ sequence
        vdz_path = output_dir / "depth_sequence.vdz"
        with open(vdz_path, "wb") as f:
            frame_idx = 0

            for chunk_idx, result in enumerate(chunk_results):
                # Determine which frames to write (skip overlap except for last chunk)
                if chunk_idx == 0:
                    start_local = 0
                    end_local = (
                        len(result.depths) - (overlap // 2)
                        if chunk_idx < len(chunk_results) - 1
                        else len(result.depths)
                    )
                elif chunk_idx == len(chunk_results) - 1:
                    start_local = overlap // 2
                    end_local = len(result.depths)
                else:
                    start_local = overlap // 2
                    end_local = len(result.depths) - (overlap // 2)

                # Get transform for this chunk
                if chunk_idx > 0 and transforms:
                    transform = transforms[chunk_idx - 1]
                else:
                    transform = AlignmentTransform(1.0, np.eye(3), np.zeros(3))

                for local_idx in range(start_local, end_local):
                    depth = result.depths[local_idx]
                    timestamp_ms = result.timestamps_ms[local_idx]

                    # Apply scale correction from alignment
                    depth_corrected = depth * transform.scale

                    z_min = float(np.percentile(depth_corrected, 1))
                    z_max = float(np.percentile(depth_corrected, 99))

                    vdz_frame = VdzFrame(
                        timestamp_ms=timestamp_ms,
                        width=depth.shape[1],
                        height=depth.shape[0],
                        depth=depth_corrected.astype(np.float32),
                        z_min=z_min,
                        z_max=z_max,
                    )
                    write_vdz_frame(f, vdz_frame, compress=self.config.compress_vdz)

                    # Build camera pose
                    if self.config.save_camera_poses:
                        w2c = np.eye(4)
                        w2c[:3, :] = result.extrinsics[local_idx]
                        c2w = np.linalg.inv(w2c)

                        # Apply Sim3 to camera pose
                        if chunk_idx > 0:
                            S = np.eye(4)
                            S[:3, :3] = transform.scale * transform.rotation
                            S[:3, 3] = transform.translation
                            c2w = S @ c2w

                        all_poses.append(
                            CameraPose(
                                frame_idx=frame_idx,
                                timestamp_ms=timestamp_ms,
                                intrinsics=result.intrinsics[local_idx],
                                extrinsics=c2w,
                            )
                        )

                    frame_idx += 1

        logger.info(f"Wrote {frame_idx} depth frames to {vdz_path}")

        # Write camera poses
        if self.config.save_camera_poses and all_poses:
            write_camera_poses(output_dir, all_poses)
            logger.info(f"Wrote {len(all_poses)} camera poses")

        # Write metadata
        meta = {
            "video_path": video_path,
            "frame_count": frame_idx,
            "fps": fps,
            "duration_s": duration_s,
            "width": width,
            "height": height,
            "config": {
                "chunk_size": self.config.chunk_size,
                "overlap": self.config.overlap,
                "model_id": self.config.model_id,
                "process_res": self.config.process_res,
                "decoder": "torchcodec" if not extractor else "cv2",
            },
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            import json

            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Offline video-to-depth preprocessing with DA3-streaming"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=60, help="Frames per chunk")
    parser.add_argument("--overlap", type=int, default=30, help="Overlap between chunks")
    parser.add_argument("--model", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--process-res", type=int, default=518, help="Processing resolution")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-compress", action="store_true", help="Disable VDZ compression")
    parser.add_argument("--no-poses", action="store_true", help="Skip camera pose estimation")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = PreprocessConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model_id=args.model,
        process_res=args.process_res,
        device=args.device,
        compress_vdz=not args.no_compress,
        save_camera_poses=not args.no_poses,
    )

    preprocessor = OfflinePreprocessor(config)
    preprocessor.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
