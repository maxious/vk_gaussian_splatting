"""Convert SIGA2025VVC-Dataset camera calibrations to tttLRM JSON manifests.

SIGA2025VVC format:
  - extri.yml: OpenCV YAML with camera names, rotation matrices (Rot_{id}),
    and translation vectors (T_{id}) as world-to-camera transforms.
  - intri.yml: OpenCV YAML with intrinsic matrices (K_{id}) and distortion (D_{id}).
  - images/{cam_id}/{frame:06d}.jpg

tttLRM format (per timestep):
  {
    "scene_name": "...",
    "frames": [
      {"file_path": "images/00/000000.jpg", "fx": ..., "fy": ..., "cx": ..., "cy": ...,
       "w2c": [[4x4 world-to-camera matrix]]}
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenCV YAML parsing
# ---------------------------------------------------------------------------


def _parse_opencv_yaml(path: Path) -> dict:
    """Parse an OpenCV YAML file (FileStorage format).

    Uses ``cv2.FileStorage`` when available, otherwise falls back to a regex-
    based parser that handles ``!!opencv-matrix`` nodes and plain sequences.
    """
    try:
        import cv2

        return _parse_with_cv2(path, cv2)
    except ImportError:
        logger.debug("cv2 not available, using fallback YAML parser")
        return _parse_opencv_yaml_fallback(path)


def _parse_with_cv2(path: Path, cv2) -> dict:  # noqa: ANN001
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    result: dict = {}
    root = fs.root()
    for key in root.keys():
        node = fs.getNode(key)
        if node.isSeq():
            result[key] = [node.at(i).string() for i in range(int(node.size()))]
        elif node.isMap():
            # opencv-matrix: read as mat
            mat = node.mat()
            if mat is not None:
                result[key] = mat
            else:
                result[key] = node.string() or node.real()
        elif node.isString():
            result[key] = node.string()
        elif node.isReal():
            result[key] = node.real()
        elif node.isInt():
            result[key] = int(node.real())
        else:
            mat = node.mat()
            if mat is not None:
                result[key] = mat
    fs.release()
    return result


def _parse_opencv_yaml_fallback(path: Path) -> dict:
    """Fallback parser for OpenCV YAML ``!!opencv-matrix`` files."""
    text = path.read_text()
    # Strip the %YAML directive and opencv-storage line
    lines = text.splitlines()
    result: dict = {}

    # Parse the "names:" sequence
    names_match = re.search(
        r"^names:\s*\n((?:\s+-\s+.*\n?)+)", text, re.MULTILINE
    )
    if names_match:
        entries = re.findall(r'-\s+"?([^"\n]+)"?', names_match.group(1))
        result["names"] = [e.strip() for e in entries]

    # Parse !!opencv-matrix blocks
    matrix_pattern = re.compile(
        r"^(\w+):\s+!!opencv-matrix\s*\n"
        r"\s+rows:\s+(\d+)\s*\n"
        r"\s+cols:\s+(\d+)\s*\n"
        r"\s+dt:\s+(\w)\s*\n"
        r"\s+data:\s+\[([^\]]+)\]",
        re.MULTILINE,
    )
    for m in matrix_pattern.finditer(text):
        name = m.group(1)
        rows = int(m.group(2))
        cols = int(m.group(3))
        data = [float(x.strip()) for x in m.group(5).split(",")]
        result[name] = np.array(data, dtype=np.float64).reshape(rows, cols)

    return result


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------


def convert_siga_to_lrm_manifests(
    dataset_root: Path,
    output_dir: Path,
    frame_indices: list[int] | None = None,
    camera_indices: list[str] | None = None,
    max_cameras: int | None = None,
) -> list[Path]:
    """Convert SIGA2025VVC dataset to tttLRM-compatible JSON manifests.

    Parameters
    ----------
    dataset_root:
        Root of the SIGA2025VVC dataset (contains ``extri.yml``, ``intri.yml``,
        and ``images/``).
    output_dir:
        Directory to write the JSON manifests.
    frame_indices:
        Which temporal frame indices to convert.  ``None`` means the first
        frame only (index 0).
    camera_indices:
        Which camera IDs to include (e.g. ``["00", "03"]``).  ``None`` means
        all cameras listed in ``extri.yml``.
    max_cameras:
        Limit the number of cameras included in each manifest.

    Returns
    -------
    list[Path]
        Paths of all created manifest JSON files.
    """
    extri_path = dataset_root / "extri.yml"
    intri_path = dataset_root / "intri.yml"
    if not extri_path.exists():
        raise FileNotFoundError(f"Missing extrinsics file: {extri_path}")
    if not intri_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {intri_path}")

    extri = _parse_opencv_yaml(extri_path)
    intri = _parse_opencv_yaml(intri_path)

    # Determine camera list
    cam_ids: list[str] = extri.get("names", [])
    if not cam_ids:
        raise ValueError("No camera names found in extri.yml")

    if camera_indices is not None:
        cam_ids = [c for c in cam_ids if c in camera_indices]
    if max_cameras is not None:
        cam_ids = cam_ids[:max_cameras]

    if not cam_ids:
        raise ValueError("No cameras selected after filtering")

    logger.info("Using %d cameras: %s", len(cam_ids), cam_ids)

    # Determine frame indices
    if frame_indices is None:
        frame_indices = [0]

    # Derive a scene name from the dataset folder
    scene_base = dataset_root.name

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for frame_idx in frame_indices:
        frame_str = f"{frame_idx:06d}"
        scene_name = f"{scene_base}_frame{frame_str}"
        frames = []

        for cam_id in cam_ids:
            # -- Extrinsics (world-to-camera) --
            rot_key = f"Rot_{cam_id}"
            t_key = f"T_{cam_id}"

            if rot_key not in extri:
                logger.warning("Missing %s in extri.yml, skipping camera %s", rot_key, cam_id)
                continue
            if t_key not in extri:
                logger.warning("Missing %s in extri.yml, skipping camera %s", t_key, cam_id)
                continue

            rot = np.asarray(extri[rot_key], dtype=np.float64).reshape(3, 3)
            tvec = np.asarray(extri[t_key], dtype=np.float64).reshape(3, 1)

            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = rot
            w2c[:3, 3] = tvec.ravel()

            # -- Intrinsics --
            k_key = f"K_{cam_id}"
            if k_key not in intri:
                logger.warning("Missing %s in intri.yml, skipping camera %s", k_key, cam_id)
                continue

            K = np.asarray(intri[k_key], dtype=np.float64).reshape(3, 3)
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

            file_path = f"images/{cam_id}/{frame_str}.jpg"
            full_path = dataset_root / file_path
            if not full_path.exists():
                logger.debug("Image not found: %s, skipping camera %s", full_path, cam_id)
                continue

            frames.append(
                {
                    "file_path": file_path,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "w2c": w2c.tolist(),
                }
            )

        if not frames:
            logger.warning("No valid frames for timestep %d, skipping", frame_idx)
            continue

        manifest = {
            "scene_name": scene_name,
            "dataset_root": str(dataset_root.resolve()),
            "frames": frames,
        }
        out_path = output_dir / f"{scene_name}.json"
        out_path.write_text(json.dumps(manifest, indent=2))
        logger.info("Wrote %s (%d cameras)", out_path.name, len(frames))
        created.append(out_path)

    return created


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SIGA2025VVC dataset to tttLRM JSON manifests"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory of the SIGA2025VVC dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for JSON manifests",
    )
    parser.add_argument(
        "--frame-indices",
        type=int,
        nargs="*",
        default=None,
        help="Temporal frame indices to convert (default: first frame only)",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="Maximum number of cameras to include",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=None,
        help="Specific camera IDs to include (e.g. 00 01 03)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    paths = convert_siga_to_lrm_manifests(
        dataset_root=args.root,
        output_dir=args.output,
        frame_indices=args.frame_indices,
        camera_indices=args.cameras,
        max_cameras=args.max_cameras,
    )
    logger.info("Created %d manifest(s)", len(paths))


if __name__ == "__main__":
    main()
