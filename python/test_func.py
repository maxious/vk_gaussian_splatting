import sys

sys.path.insert(0, ".")

from pathlib import Path
import numpy as np
from offline.motion_tracking_cpu import match_gaussians_sliding_window_faiss

# Load data
data_dir = Path("tests/data/sharp_sequence")
ply_files = sorted(data_dir.glob("*.ply"))[:5]

all_means = []
for p in ply_files:
    # Simple PLY load
    with open(p, "rb") as f:
        # Skip header
        line = b""
        while not line.startswith(b"end_header"):
            line = f.readline()

        # Read data
        data = np.frombuffer(f.read(), dtype=np.float32)
        # Assuming standard PLY format: x,y,z,nx,ny,nz,f_dc_0,f_dc_1,f_dc_2,opacity,scale_0,scale_1,scale_2,rot_0,rot_1,rot_2,rot_3
        # We only need x,y,z
        n_points = len(data) // 16  # 16 floats per vertex
        data = data.reshape(n_points, 16)
        means = data[:, :3]
        all_means.append(means.astype(np.float32))

print(f"Loaded {len(all_means)} frames")

# Test function
result = match_gaussians_sliding_window_faiss(all_means, max_distance=0.05, window_size=3)
print(f"Found {len(result)} matches")
