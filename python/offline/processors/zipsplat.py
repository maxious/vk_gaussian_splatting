"""ZipSplat adapter for 4DAnyone multi-view video output."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def _video_info(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open generated view video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        raise ValueError(f"Generated view video has no frames: {path}")
    return fps, frame_count


def _read_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from generated view video")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _load_zipsplat():
    try:
        from zipsplat import ZipSplat, load_image
    except ImportError as exc:
        raise RuntimeError(
            "ZipSplat is not installed. Install it from https://github.com/cvg/ZipSplat "
            "and run this command with a CUDA-enabled PyTorch environment."
        ) from exc
    return ZipSplat, load_image


def export_4danyone_with_zipsplat(
    result_dir: Path,
    output_dir: Path,
    *,
    weights: str = "zipsplat",
    max_frames: int | None = None,
    frame_skip: int = 1,
    compression_ratio: float | None = None,
) -> int:
    """Predict one static ZipSplat scene for each 4DAnyone timestamp.

    ``result_dir`` is the 4DAnyone ``fdanyone/<clip>`` directory. The generated
    view videos must be in ``videos/dense``. Frames are written to a temporary
    directory because ZipSplat's public API loads images by path.
    """
    result_dir = result_dir.expanduser().resolve()
    dense_dir = result_dir / "videos" / "dense"
    view_videos = sorted(dense_dir.glob("*.mp4"))
    if not view_videos:
        raise ValueError(f"No generated view videos found in {dense_dir}")

    metadata_path = result_dir / "metadata.json"
    if metadata_path.exists():
        try:
            json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid 4DAnyone metadata: {metadata_path}") from exc

    fps, frame_count = _video_info(view_videos[0])
    for path in view_videos[1:]:
        other_fps, other_count = _video_info(path)
        if other_count != frame_count:
            raise ValueError(
                f"Generated views have different frame counts: {view_videos[0].name} "
                f"has {frame_count}, {path.name} has {other_count}"
            )
        if fps and other_fps and abs(fps - other_fps) > 0.01:
            raise ValueError(f"Generated views have different frame rates: {path}")

    selected = list(range(0, frame_count, max(1, frame_skip)))
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("No timestamps selected")
    if compression_ratio is not None and not 0.0 < compression_ratio <= 1.0:
        raise ValueError("--compression-ratio must be in the interval (0, 1]")

    ZipSplat, load_image = _load_zipsplat()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ZipSplat requires a CUDA-enabled PyTorch runtime")

    model = ZipSplat(weights=weights).cuda().eval()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    captures = [cv2.VideoCapture(str(path)) for path in view_videos]
    try:
        with tempfile.TemporaryDirectory(prefix="zipsplat_4danyone_") as temp_dir:
            temp_path = Path(temp_dir)
            for output_index, source_index in enumerate(selected):
                image_paths = []
                for view_index, cap in enumerate(captures):
                    image = _read_frame(cap, source_index)
                    image_path = temp_path / f"{view_index:04d}_{output_index:06d}.png"
                    if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Could not write temporary image: {image_path}")
                    image_paths.append(image_path)

                images = [load_image(path) for path in image_paths]
                with torch.inference_mode():
                    gaussians = model(
                        images,
                        compression=1.0 if compression_ratio is None else compression_ratio,
                    )[0]
                output_path = output_dir / f"frame_{output_index:06d}.ply"
                gaussians.save_ply(output_path)
                logger.info(
                    "Wrote timestamp %d/%d from %d views to %s",
                    output_index + 1,
                    len(selected),
                    len(view_videos),
                    output_path,
                )
    finally:
        for cap in captures:
            cap.release()

    (output_dir / "zipsplat_manifest.json").write_text(
        json.dumps(
            {
                "source": str(result_dir),
                "views": len(view_videos),
                "source_frames": selected,
                "fps": fps,
                "frame_skip": frame_skip,
                "weights": weights,
            },
            indent=2,
        )
        + "\n"
    )
    return len(selected)
