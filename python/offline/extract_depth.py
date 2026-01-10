"""Extract depth frames from VDZ file as images."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from matplotlib import cm

from offline.formats import read_all_vdz_frames

logger = logging.getLogger(__name__)

# Valid colormaps
COLORMAPS = {
    "viridis": cm.viridis,
    "magma": cm.magma,
    "plasma": cm.plasma,
    "inferno": cm.inferno,
    "gray": cm.gray,
}


def run_extract_depth(args) -> None:
    """Extract depth frames from VDZ file as images."""
    frames = read_all_vdz_frames(args.input)
    logger.info(f"Read {len(frames)} frames from {args.input}")

    # Parse frame range
    frame_indices = None
    if args.frame_range:
        frame_indices = set()
        if "-" in args.frame_range:
            start, end = args.frame_range.split("-")
            frame_indices.update(range(int(start), int(end) + 1))
        else:
            for part in args.frame_range.split(","):
                frame_indices.add(int(part))
        logger.info(f"Extracting frames: {sorted(frame_indices)}")

    # Get colormap
    colormap = None if args.colormap == "raw" else COLORMAPS.get(args.colormap, cm.viridis)

    args.output.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        if frame_indices is not None and i not in frame_indices:
            continue

        depth = frame.depth

        # Apply colormap or keep raw
        if colormap is None:
            # Normalize to 0-255 for raw metric depth
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                vis = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                vis = np.zeros_like(depth, dtype=np.uint8)
        else:
            # Normalize to 0-1 for colormap
            d_min, d_max = frame.z_min, frame.z_max
            depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
            colored = colormap(depth_norm)
            vis = (colored[:, :, :3] * 255).astype(np.uint8)

        # Resize if needed
        if args.max_width and vis.shape[1] > args.max_width:
            scale = args.max_width / vis.shape[1]
            new_size = (args.max_width, int(vis.shape[0] * scale))
            vis = cv2.resize(vis, new_size, interpolation=cv2.INTER_AREA)

        # Save image
        output_path = args.output / f"depth_{i:06d}.{args.format}"
        cv2.imwrite(str(output_path), vis)

    logger.info(f"Extracted {len(frames)} depth images to {args.output}")
