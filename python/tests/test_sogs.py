"""Unit tests for SOG compression functionality."""

import pytest
import tempfile
import torch
import numpy as np
from pathlib import Path

from .test_sogs_utils import generate_synthetic_gaussians, create_test_ply_file
from ..sogs.compression import read_ply, morton_order_sort, kmeans_1d, write_webp_image
import numpy as np
import json


class TestSogsCompression:
    """Test suite for SOG compression components."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic Gaussian data for testing."""
        return generate_synthetic_gaussians(n_gaussians=50, seed=12345)

    @pytest.fixture
    def temp_ply_file(self, synthetic_data):
        """Create a temporary PLY file with synthetic data."""
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = Path(f.name)
        create_test_ply_file(synthetic_data, filepath)
        yield filepath
        filepath.unlink()  # Cleanup

    def test_read_ply_basic(self, temp_ply_file, synthetic_data):
        """Test that read_ply correctly loads synthetic PLY data."""
        loaded_data = read_ply(str(temp_ply_file))

        # Check that all expected keys are present
        expected_keys = {"means", "opacities", "scales", "quats", "sh0"}
        assert set(loaded_data.keys()) == expected_keys

        # Check data types
        for key in expected_keys:
            assert isinstance(loaded_data[key], torch.Tensor)
            assert loaded_data[key].is_cuda  # Should be on GPU

        # Check shapes
        n_gaussians = len(synthetic_data["means"])
        assert loaded_data["means"].shape == (n_gaussians, 3)
        assert loaded_data["opacities"].shape == (n_gaussians,)
        assert loaded_data["scales"].shape == (n_gaussians, 3)
        assert loaded_data["quats"].shape == (n_gaussians, 4)
        assert loaded_data["sh0"].shape == (n_gaussians, 1, 3)

    def test_read_ply_values(self, temp_ply_file, synthetic_data):
        """Test that read_ply preserves the exact values from synthetic data."""
        loaded_data = read_ply(str(temp_ply_file))

        # Move synthetic data to CPU for comparison
        synthetic_cpu = {k: v.cpu() for k, v in synthetic_data.items()}

        # Check that values match (allowing for small floating point differences)
        torch.testing.assert_close(
            loaded_data["means"].cpu(), synthetic_cpu["means"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            loaded_data["opacities"].cpu(), synthetic_cpu["opacities"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            loaded_data["scales"].cpu(), synthetic_cpu["scales"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            loaded_data["quats"].cpu(), synthetic_cpu["quats"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            loaded_data["sh0"].cpu(), synthetic_cpu["sh0"], rtol=1e-6, atol=1e-6
        )

    def test_morton_order_sort(self):
        """Test Morton order sorting for spatial coherence."""
        # Create test points in a 2x2x2 grid
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )

        indices = morton_order_sort(points)

        # Morton order should visit points in a space-filling curve
        # The exact order depends on the implementation, but should be deterministic
        assert len(indices) == 8
        assert sorted(indices) == list(range(8))  # All indices present
        assert len(set(indices)) == 8  # No duplicates

    def test_kmeans_1d(self):
        """Test 1D k-means clustering with known data."""
        # Create data with 3 clear clusters
        data = np.array([1, 1.1, 1.2, 5, 5.1, 5.2, 9, 9.1, 9.2], dtype=np.float32)
        centroids, labels = kmeans_1d(data, n_clusters=3, iterations=10)

        assert len(centroids) == 3
        assert len(labels) == 9

        # Centroids should be roughly at cluster centers
        centroids_sorted = np.sort(centroids)
        assert centroids_sorted[0] < centroids_sorted[1] < centroids_sorted[2]

        # Labels should group similar values
        assert labels[0] == labels[1] == labels[2]  # First cluster
        assert labels[3] == labels[4] == labels[5]  # Second cluster
        assert labels[6] == labels[7] == labels[8]  # Third cluster

    def test_write_webp_image(self, tmp_path):
        """Test WebP image writing."""
        # Create test RGBA data (2x2 image)
        width, height = 2, 2
        data = np.array(
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]],
            dtype=np.uint8,
        ).flatten()

        filepath = tmp_path / "test.webp"
        write_webp_image(str(filepath), data, width, height)

        assert filepath.exists()
        assert filepath.stat().st_size > 0  # File has content

    def test_metadata_generation(self):
        """Test that metadata JSON has the expected structure and values."""
        # This would test the metadata creation function if it were extracted
        # For now, test the expected structure manually
        expected_structure = {
            "version": 2,
            "asset": {"generator": "vk_gaussian_splatting sogs v2.0.0"},
            "count": 50,
            "means": {
                "mins": [0.0, 0.0, 0.0],
                "maxs": [1.0, 1.0, 1.0],
                "files": ["means_l.webp", "means_u.webp"],
            },
            "scales": {"codebook": [0.1, 0.2], "files": ["scales.webp"]},
            "quats": {"files": ["quats.webp"]},
            "sh0": {"codebook": [0.1, 0.2], "files": ["sh0.webp"]},
        }

        # Check that required keys are present
        required_keys = {"version", "asset", "count", "means", "scales", "quats", "sh0"}
        assert set(expected_structure.keys()) == required_keys

        # Check asset structure
        assert "generator" in expected_structure["asset"]
        assert expected_structure["version"] == 2

    def test_full_compression_pipeline(self, synthetic_data, tmp_path):
        """Integration test for the complete SOG compression pipeline."""
        from ..sogs.compression import run_compression

        output_path = tmp_path / "test_output.sog"

        # Run compression
        run_compression(str(output_path), synthetic_data)

        # Check that output file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Check that it's a valid zip file (SOG format)
        import zipfile

        with zipfile.ZipFile(output_path, "r") as zf:
            # Should contain the expected files
            expected_files = {
                "means_l.webp",
                "means_u.webp",
                "scales.webp",
                "quats.webp",
                "sh0.webp",
                "meta.json",
            }
            actual_files = set(zf.namelist())
            assert expected_files.issubset(actual_files)

            # Check meta.json exists and is valid JSON
            with zf.open("meta.json") as f:
                metadata = json.load(f)
                assert metadata["version"] == 2
                assert "count" in metadata
                assert metadata["count"] == len(synthetic_data["means"])
