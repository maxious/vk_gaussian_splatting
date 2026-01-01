"""Export video frames to Gaussian Splatting PLY files.

This module has been refactored into smaller submodules in the `offline` package.
This file is kept for backwards compatibility.
"""

from .cli import main
from .exporters import (
    export_images_to_gaussian_plys,
    export_video_to_gaussian_plys,
    postprocess_plys_to_freetimegs,
)
from .ply_io import load_static_gaussian_ply, write_freetimegs_ply, write_static_gaussian_ply
from .processors.da3 import DA3GaussianProcessor
from .processors.sharp import SharpGaussianProcessor
from .types import GaussianFrame
from .video_utils import extract_video_frames, prune_gaussian_frame

# Deprecated: These were internal but exposed
match_gaussians_bidirectional = None  # Never implemented in original file

if __name__ == "__main__":
    main()
