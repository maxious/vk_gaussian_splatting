"""Offline preprocessing tools for video-to-depth conversion."""

import sys
from pathlib import Path

# Add vendored TRELLIS package to path
_VKGS_TRELLIS_PATH = Path(__file__).parent / "vkgs_trellis"
if str(_VKGS_TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(_VKGS_TRELLIS_PATH))
