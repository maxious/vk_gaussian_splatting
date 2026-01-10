"""Generic frame dataset for loading images from disk with optimized I/O.

This module provides datasets for efficient image/video loading for ML pipelines.

Features:
- FrameDataset: Image sequence loading from disk with background workers
- VideoDataset: Direct video decoding using torchcodec (no PNG intermediate)
- Background loading with torch DataLoader workers
- Pin-memory support for fast CPU->GPU/XPU transfers
- C-contiguous array conversion for zero-copy tensor creation
- Optional mask loading and resizing
- Timestamp tracking
- Optional parallel preload for small datasets (joblib-based)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset

FrameDatasetItem = Tuple[int, torch.Tensor, float, int, int, Optional[torch.Tensor]]
VideoDatasetItem = Tuple[int, torch.Tensor, float, int, int]


class FrameDataset(Dataset[FrameDatasetItem]):
    """Generic dataset for loading image sequences from disk.

    Supports various image formats (PNG, JPG, etc.) and optional mask loading.
    Designed for use with DataLoader for background loading and pinned memory.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = FrameDataset(image_paths, timestamps_ms)
        >>> dataloader = DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True)
        >>> for batch in dataloader:
        ...     indices, images, timestamps = batch
        ...     # images is (B, C, H, W) tensor, ready for GPU

    Parallel Preload:
        For small datasets, use parallel_preload to load all frames in parallel
        at initialization. This is useful when the full dataset fits in memory
        and you want to avoid I/O bottlenecks during training.
    """

    def __init__(
        self,
        frame_paths: List[Path],
        timestamps_ms: List[float],
        masks_dir: Optional[Path] = None,
        mask_first_frame: bool = False,
        transform: "Optional[Callable]" = None,
        image_mode: str = "RGB",
        parallel_preload: int = 0,
    ):
        """Initialize FrameDataset.

        Args:
            frame_paths: List of paths to image files
            timestamps_ms: List of timestamps for each frame
            masks_dir: Optional directory containing mask files
            mask_first_frame: Whether to apply mask to first frame
            transform: Optional transform function (img, mask) -> (img, mask)
            image_mode: OpenCV color mode ("RGB", "BGR", "GRAYSCALE")
            parallel_preload: Number of parallel workers for preloading (0 = disabled).
                             Use for small datasets that fit in memory.
        """
        self.frame_paths = frame_paths
        self.timestamps_ms = timestamps_ms
        self.masks_dir = masks_dir
        self.mask_first_frame = mask_first_frame
        self.transform = transform
        self.image_mode = image_mode
        self.parallel_preload = parallel_preload

        # Map image_mode to cv2 constants
        mode_map = {
            "RGB": cv2.COLOR_BGR2RGB,
            "BGR": None,  # Keep as BGR
            "GRAYSCALE": cv2.COLOR_BGR2GRAY,
        }
        self.color_convert = mode_map.get(image_mode)

        # Parallel preload if requested
        self._cached_frames: Optional[List[np.ndarray]] = None
        if parallel_preload > 0:
            self._preload_all()

    def _preload_all(self) -> None:
        """Preload all frames in parallel using joblib."""
        if len(self.frame_paths) == 0:
            self._cached_frames = []
            return

        # Use threading backend since cv2 releases GIL during I/O
        n_jobs = min(self.parallel_preload, len(self.frame_paths))
        loaded = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._load_image)(path) for path in self.frame_paths
        )
        self._cached_frames = cast(List[np.ndarray], list(loaded))

    def __len__(self) -> int:
        return len(self.frame_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load a single image from disk."""
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")

        if self.color_convert is not None:
            img = cv2.cvtColor(img, self.color_convert)

        return img

    def _load_mask(self, image_path: Path, H: int, W: int) -> Optional[np.ndarray]:
        """Load and resize mask for an image."""
        if self.masks_dir is None:
            return None

        if not self.mask_first_frame and image_path == self.frame_paths[0]:
            return None

        mask_path = None
        for ext in [image_path.suffix, ".png", ".jpg", ".jpeg"]:
            candidate = self.masks_dir / f"{image_path.stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            return None

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return mask

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, float, int, int, Optional[torch.Tensor]]:
        """Get a single frame.

        Returns:
            Tuple of (index, image_tensor, timestamp_ms, height, width, mask_tensor or None)
            - image_tensor: (C, H, W) float tensor in [0, 1] range
            - mask_tensor: (1, H, W) float tensor or None
        """
        if self._cached_frames is not None:
            img = self._cached_frames[index]
        else:
            path = self.frame_paths[index]
            img = self._load_image(path)

        ts = self.timestamps_ms[index]
        H, W = img.shape[:2]

        mask = self._load_mask(self.frame_paths[index], H, W)

        # Apply transforms
        if self.transform:
            img, mask = self.transform(img, mask)

        # Convert to tensor with C-contiguity for zero-copy
        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).permute(2, 0, 1)

        if mask is not None:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros(1, H, W, dtype=torch.float32)

        return index, img_tensor, ts, H, W, mask_tensor


def frame_dataset_factory(
    frame_paths: List[Path],
    timestamps_ms: List[float],
    masks_dir: Optional[Path] = None,
    mask_first_frame: bool = False,
    transform: Optional[Callable] = None,
    image_mode: str = "RGB",
    parallel_preload: int = 0,
) -> FrameDataset:
    """Factory function for creating FrameDataset with optimal defaults.

    This is the recommended way to create a FrameDataset as it provides
    a consistent interface across the codebase.

    Args:
        parallel_preload: Number of parallel workers for preloading (0 = disabled).
                         Use for small datasets that fit in memory.
    """
    return FrameDataset(
        frame_paths=frame_paths,
        timestamps_ms=timestamps_ms,
        masks_dir=masks_dir,
        mask_first_frame=mask_first_frame,
        transform=transform,
        image_mode=image_mode,
        parallel_preload=parallel_preload,
    )


class VideoDataset(Dataset[VideoDatasetItem]):
    """Dataset for direct video decoding using torchcodec.

    This dataset avoids the inefficient PNG extraction -> re-decoding pipeline
    by decoding video frames directly to tensors using torchcodec.FrameBatch.

    Advantages over PNG extraction:
    - No disk I/O for intermediate PNG files
    - Direct memory-to-memory video decoding
    - FrameBatch provides PTS timestamps automatically
    - Supports GPU decoding with CUDA backend

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = VideoDataset(video_path)
        >>> dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
        >>> for batch in dataloader:
        ...     indices, images, timestamps = batch
        ...     # images is (B, C, H, W) uint8 tensor
        ...     # timestamps is (B,) float tensor in milliseconds

    Note:
        Requires torchcodec to be installed: `pip install torchcodec`
    """

    def __init__(
        self,
        video_path: Path,
        frame_indices: Optional[List[int]] = None,
        transform: "Optional[Callable]" = None,
    ):
        """Initialize VideoDataset.

        Args:
            video_path: Path to video file
            frame_indices: Optional list of frame indices to load (None = all frames)
            transform: Optional transform function (tensor, idx) -> transformed_tensor
        """
        self.video_path = Path(video_path)
        self.transform = transform

        # Import torchcodec here to make it optional
        try:
            from torchcodec.decoders import VideoDecoder
        except ImportError as exc:
            raise ImportError(
                "torchcodec is required for VideoDataset. Install with: pip install torchcodec"
            ) from exc

        self._decoder = VideoDecoder(str(video_path))
        self._metadata = self._decoder.metadata

        total_frames = self._metadata.num_frames
        if total_frames is None:
            raise ValueError(f"Cannot determine frame count for video: {video_path}")
        average_fps = self._metadata.average_fps
        if average_fps is None:
            raise ValueError(f"Cannot determine FPS for video: {video_path}")

        # Build frame index mapping
        if frame_indices is not None:
            self._frame_indices = list(frame_indices)
        else:
            self._frame_indices = list(range(total_frames))

        # Pre-compute timestamps in milliseconds
        self._timestamps_ms: list[float] = []
        for idx in self._frame_indices:
            # Get frame info for this index
            if idx < total_frames:
                # Approximate: use frame index / FPS
                pts_sec = idx / average_fps
                self._timestamps_ms.append(pts_sec * 1000.0)
            else:
                self._timestamps_ms.append(0.0)

    def __len__(self) -> int:
        return len(self._frame_indices)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, float, int, int]:
        """Get a single frame.

        Returns:
            Tuple of (index, image_tensor, timestamp_ms, height, width)
            - index: Frame index in original video
            - image_tensor: (C, H, W) uint8 tensor
            - timestamp_ms: Timestamp in milliseconds
            - height: Frame height
            - width: Frame width
        """
        frame_idx = self._frame_indices[index]
        timestamp_ms = self._timestamps_ms[index]

        frame_batch = self._decoder.get_frames_at([frame_idx])
        frame_tensor = frame_batch.data[0]

        H, W = frame_tensor.shape[1], frame_tensor.shape[2]

        if self.transform:
            frame_tensor = self.transform(frame_tensor, index)

        return frame_idx, frame_tensor, timestamp_ms, H, W

    def iterate_batches(
        self, batch_size: int
    ) -> "Iterator[Tuple[torch.Tensor, List[float], List[int]]]":
        """Efficiently iterate over frames in batches using FrameBatch.

        This method is more efficient than using DataLoader because it
        uses torchcodec's batch decoding capabilities.

        Args:
            batch_size: Number of frames per batch

        Yields:
            Tuple of (frame_timestamps, timestamp_ms, frame_indices)
            - frames: (B, C, H, W) uint8 tensor
            - timestamps_ms: (B,) list of timestamps in milliseconds
            - frame_indices: (B,) list of frame indices
        """
        num_frames = len(self._frame_indices)

        for start_idx in range(0, num_frames, batch_size):
            end_idx = min(start_idx + batch_size, num_frames)
            batch_frame_indices = self._frame_indices[start_idx:end_idx]

            # Decode batch of frames
            frame_batch = self._decoder.get_frames_at(batch_frame_indices)

            frames = frame_batch.data  # (B, C, H, W) uint8

            # Capture actual timestamps from FrameBatch (precise PTS, not approximated)
            timestamps_ms = [ts * 1000.0 for ts in frame_batch.pts_seconds.tolist()]

            yield frames, timestamps_ms, batch_frame_indices

    def iterate_batches_streaming(
        self, batch_size: int
    ) -> "Iterator[Tuple[torch.Tensor, List[float], List[int]]]":
        """Stream through video forward-only using get_frames_in_range().

        This is the optimal method for real-time streaming because:
        - Uses get_frames_in_range() for sequential forward decoding
        - Captures precise timestamps from FrameBatch.pts_seconds
        - Decoder maintains minimal state (no seeking required)

        Args:
            batch_size: Number of frames per batch

        Yields:
            Tuple of (frames, timestamps_ms, frame_indices)
            - frames: (B, C, H, W) uint8 tensor
            - timestamps_ms: (B,) list of timestamps in milliseconds (precise PTS)
            - frame_indices: (B,) list of frame indices in original video
        """
        num_frames = len(self._frame_indices)
        start_frame = 0

        while start_frame < num_frames:
            end_frame = min(start_frame + batch_size, num_frames)
            batch_frame_indices = self._frame_indices[start_frame:end_frame]

            # Get actual frame numbers from indices
            actual_start = batch_frame_indices[0]
            actual_end = batch_frame_indices[-1]

            # Decode range of frames - this is efficient for forward streaming
            # API: get_frames_in_range(start, stop, step=1) - stop is exclusive
            frame_batch = self._decoder.get_frames_in_range(
                actual_start, actual_start + len(batch_frame_indices)
            )

            frames = frame_batch.data  # (B, C, H, W) uint8

            # Capture actual timestamps from FrameBatch (precise PTS, not approximated)
            timestamps_ms = [ts * 1000.0 for ts in frame_batch.pts_seconds.tolist()]

            yield frames, timestamps_ms, batch_frame_indices

            start_frame = end_frame

    def close(self) -> None:
        """Close the video decoder and release resources.

        Note: torchcodec VideoDecoder doesn't have an explicit close method.
        Resources are released when the object is garbage collected.
        """
        pass

    def __del__(self) -> None:
        """Destructor to ensure decoder is closed."""
        try:
            self.close()
        except Exception:
            pass


def video_dataset_factory(
    video_path: Path,
    frame_indices: Optional[List[int]] = None,
    transform: Optional[Callable] = None,
) -> VideoDataset:
    """Factory function for creating VideoDataset with optimal defaults.

    This is the recommended way to create a VideoDataset as it provides
    a consistent interface across the codebase.

    Args:
        video_path: Path to video file
        frame_indices: Optional list of frame indices to load
        transform: Optional transform function
    """
    return VideoDataset(
        video_path=video_path,
        frame_indices=frame_indices,
        transform=transform,
    )
