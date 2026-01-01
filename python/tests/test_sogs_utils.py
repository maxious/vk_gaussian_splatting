"""Synthetic data generator for SOG compression testing."""

import numpy as np
import torch
from pathlib import Path
from typing import Dict


def generate_synthetic_gaussians(n_gaussians: int = 100, seed: int = 42) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic Gaussian splat data with known properties for testing.

    Args:
        n_gaussians: Number of Gaussians to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary with synthetic Gaussian data
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate means (positions) in a unit cube
    means = torch.randn(n_gaussians, 3, dtype=torch.float32) * 0.5 + 0.5

    # Generate opacities (sigmoid of random values)
    opacities_raw = torch.randn(n_gaussians, dtype=torch.float32)
    opacities = torch.sigmoid(opacities_raw)

    # Generate scales (positive values, log-normal distribution)
    scales_raw = torch.randn(n_gaussians, 3, dtype=torch.float32)
    scales = torch.exp(scales_raw * 0.5 + 0.5)  # Mean around 1.5

    # Generate quaternions (random unit quaternions)
    quats_raw = torch.randn(n_gaussians, 4, dtype=torch.float32)
    quats = quats_raw / torch.norm(quats_raw, dim=1, keepdim=True)

    # Generate SH0 colors (in sRGB-like range)
    sh0_raw = torch.randn(n_gaussians, 3, dtype=torch.float32)
    sh0 = torch.sigmoid(sh0_raw)  # Values between 0 and 1

    return {
        "means": means.cuda(),
        "opacities": opacities.cuda(),
        "scales": scales.cuda(),
        "quats": quats.cuda(),
        "sh0": sh0.unsqueeze(-1).cuda(),  # (N, 1, 3)
    }


def create_test_ply_file(data: Dict[str, torch.Tensor], filepath: Path) -> None:
    """
    Create a PLY file from synthetic Gaussian data for testing read_ply.

    Args:
        data: Gaussian data dictionary
        filepath: Output PLY file path
    """
    import plyfile

    # Convert to numpy and CPU
    means = data["means"].cpu().numpy()
    opacities = data["opacities"].cpu().numpy()
    scales = data["scales"].cpu().numpy()
    quats = data["quats"].cpu().numpy()
    sh0 = data["sh0"].cpu().numpy().squeeze(-2)  # (N, 3)

    # Create PLY data structure
    vertex_data = []
    for i in range(len(means)):
        vertex = (
            means[i, 0],
            means[i, 1],
            means[i, 2],  # x, y, z
            opacities[i],  # opacity
            scales[i, 0],
            scales[i, 1],
            scales[i, 2],  # scale_x, scale_y, scale_z
            quats[i, 0],
            quats[i, 1],
            quats[i, 2],
            quats[i, 3],  # rot_0 to rot_3
            sh0[i, 0],
            sh0[i, 1],
            sh0[i, 2],  # f_dc_0 to f_dc_2
        )
        vertex_data.append(vertex)

    # Define PLY element
    from plyfile import PlyElement, PlyData
    import numpy as np

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    vertex_array = np.array(vertex_data, dtype=dtype)
    el = PlyElement.describe(vertex_array, "vertex")
    PlyData([el]).write(str(filepath))
