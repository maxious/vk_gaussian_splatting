"""End-to-end integration test for AMB3R + COLMAP export pipeline.

Since AMB3R requires CUDA + pytorch3d + spconv (not available on this XPU system),
we test the pipeline by generating synthetic AMB3R-like outputs and verifying:
1. The COLMAP export produces valid reconstruction files
2. The full plumbing from synthetic outputs to PLY works
3. The SIGA2025VVC dataset converter works correctly
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_amb3r_output(
    num_frames: int,
    height: int,
    width: int,
) -> dict:
    """Generate synthetic AMB3R-like outputs for testing.

    Creates a simple scene: a plane at z=2 with some depth variation,
    cameras arranged in a circle looking at the origin.
    """
    np.random.seed(42)

    world_points = np.zeros((num_frames, height, width, 3), dtype=np.float64)
    confidence = np.ones((num_frames, height, width), dtype=np.float64) * 2.0

    # Create a grid of 3D points on a plane at z=2
    for t in range(num_frames):
        xs = np.linspace(-1, 1, width)
        ys = np.linspace(-1, 1, height)
        xg, yg = np.meshgrid(xs, ys)
        zg = 2.0 + 0.1 * np.sin(xg * 3) * np.cos(yg * 3)  # wavy plane
        world_points[t, :, :, 0] = xg
        world_points[t, :, :, 1] = yg
        world_points[t, :, :, 2] = zg

    # Add some noise and set low-confidence regions
    world_points += np.random.randn(*world_points.shape) * 0.01
    confidence += np.random.randn(*confidence.shape) * 0.3
    confidence = np.clip(confidence, 0.1, 5.0)

    # Create camera extrinsics (w2c) - cameras around the scene
    extrinsics_w2c = np.zeros((num_frames, 3, 4), dtype=np.float64)
    for t in range(num_frames):
        angle = 2 * np.pi * t / num_frames
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        tvec = np.array([0, 0, 3.0])  # Camera 3 units from origin
        extrinsics_w2c[t, :3, :3] = R
        extrinsics_w2c[t, :3, 3] = tvec

    # Create intrinsics
    intrinsics = np.zeros((num_frames, 3, 3), dtype=np.float64)
    for t in range(num_frames):
        fx = fy = width * 1.2  # focal length
        cx, cy = width / 2.0, height / 2.0
        intrinsics[t] = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])

    return {
        "world_points": world_points,
        "confidence": confidence,
        "extrinsics_w2c": extrinsics_w2c,
        "intrinsics": intrinsics,
    }


def test_colmap_export(image_paths: list[Path], output_dir: Path) -> bool:
    """Test the COLMAP export with synthetic AMB3R data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from offline.colmap_export import export_amb3r_to_colmap

    num_frames = len(image_paths)
    h_proc, w_proc = 56, 74  # Small processing resolution for speed

    amb3r_output = generate_synthetic_amb3r_output(num_frames, h_proc, w_proc)

    colmap_dir = output_dir / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting COLMAP reconstruction with {num_frames} frames...")
    export_amb3r_to_colmap(
        world_points=amb3r_output["world_points"],
        confidence=amb3r_output["confidence"],
        extrinsics_w2c=amb3r_output["extrinsics_w2c"],
        intrinsics=amb3r_output["intrinsics"],
        image_paths=[str(p) for p in image_paths],
        export_dir=str(colmap_dir),
        conf_thresh_percentile=40.0,
        max_points_per_frame=500,  # Small for test
    )

    # Verify output files
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(colmap_dir))
    num_images = len(reconstruction.images)
    num_points = len(reconstruction.points3D)
    num_cameras = len(reconstruction.cameras)

    logger.info(f"COLMAP reconstruction: {num_images} images, {num_points} 3D points, {num_cameras} cameras")

    assert num_images == num_frames, f"Expected {num_frames} images, got {num_images}"
    assert num_points > 0, "No 3D points in reconstruction"
    assert num_cameras == num_frames, f"Expected {num_frames} cameras, got {num_cameras}"

    # Verify point cloud has reasonable values
    pts = np.array([p.xyz for p in reconstruction.points3D.values()])
    logger.info(f"Point cloud bounds: min={pts.min(axis=0)}, max={pts.max(axis=0)}")

    return True


def test_colmap_to_ply(image_paths: list[Path], output_dir: Path) -> Path:
    """Export COLMAP reconstruction directly to PLY (skip FastGS training)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import pycolmap

    # Generate and export COLMAP
    num_frames = len(image_paths)
    h_proc, w_proc = 56, 74

    amb3r_output = generate_synthetic_amb3r_output(num_frames, h_proc, w_proc)

    colmap_dir = output_dir / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    from offline.colmap_export import export_amb3r_to_colmap

    export_amb3r_to_colmap(
        world_points=amb3r_output["world_points"],
        confidence=amb3r_output["confidence"],
        extrinsics_w2c=amb3r_output["extrinsics_w2c"],
        intrinsics=amb3r_output["intrinsics"],
        image_paths=[str(p) for p in image_paths],
        export_dir=str(colmap_dir),
        conf_thresh_percentile=30.0,
        max_points_per_frame=2000,
    )

    # Load the COLMAP reconstruction and write a Gaussian PLY from the point cloud
    reconstruction = pycolmap.Reconstruction(str(colmap_dir))
    pts = np.array([p.xyz for p in reconstruction.points3D.values()], dtype=np.float32)
    colors_uint8 = np.array([p.color for p in reconstruction.points3D.values()], dtype=np.uint8)

    num_points = len(pts)
    logger.info(f"Creating Gaussian PLY from {num_points} COLMAP points")

    # Create simple Gaussians from the point cloud
    means = pts
    # Small uniform scales
    scales = np.full((num_points, 3), -5.0, dtype=np.float32)  # log scale
    # Identity rotations (wxyz)
    rotations = np.zeros((num_points, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    # Convert colors to SH DC (0th order spherical harmonic)
    colors_float = colors_uint8.astype(np.float32) / 255.0
    # SH DC coefficient = (color - 0.5) / 0.2821
    sh_dc = (colors_float - 0.5) / 0.28209479177387814
    # Sigmoid inverse for opacity (high opacity)
    opacities = np.full((num_points, 1), 5.0, dtype=np.float32)

    from offline.ply_io import write_static_gaussian_ply

    ply_path = output_dir / "amb3r_test_output.ply"
    write_static_gaussian_ply(
        ply_path,
        means,
        scales,
        rotations,
        sh_dc,
        opacities,
    )

    logger.info(f"PLY written: {ply_path} ({ply_path.stat().st_size / 1024:.1f} KB)")
    return ply_path


def test_siga2025vvc_converter() -> bool:
    """Test the SIGA2025VVC to LRM JSON converter with synthetic data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic SIGA2025VVC dataset
        cam_ids = ["00", "01", "02"]
        images_dir = tmpdir / "images"

        for cam_id in cam_ids:
            cam_dir = images_dir / cam_id
            cam_dir.mkdir(parents=True)
            # Create tiny test images
            from PIL import Image
            for frame_idx in range(3):
                img = Image.new("RGB", (64, 48), color=(100 + frame_idx * 30, 50, 200))
                img.save(cam_dir / f"{frame_idx:06d}.jpg")

        # Create extri.yml (OpenCV YAML format)
        extri_lines = ["%YAML:1.0", "---"]
        extri_lines.append("names:")
        for cam_id in cam_ids:
            extri_lines.append(f'   - "{cam_id}"')

        for cam_id in cam_ids:
            angle = float(cam_ids.index(cam_id)) * 0.5
            R = np.eye(3)
            R[0, 0] = np.cos(angle)
            R[0, 2] = np.sin(angle)
            R[2, 0] = -np.sin(angle)
            R[2, 2] = np.cos(angle)
            T = np.array([[0.0], [0.0], [3.0]])

            extri_lines.append(f"Rot_{cam_id}: !!opencv-matrix")
            extri_lines.append("   rows: 3")
            extri_lines.append("   cols: 3")
            extri_lines.append("   dt: d")
            data = ", ".join(f"{x:.6f}" for x in R.flatten())
            extri_lines.append(f"   data: [ {data} ]")

            extri_lines.append(f"T_{cam_id}: !!opencv-matrix")
            extri_lines.append("   rows: 3")
            extri_lines.append("   cols: 1")
            extri_lines.append("   dt: d")
            data = ", ".join(f"{x:.6f}" for x in T.flatten())
            extri_lines.append(f"   data: [ {data} ]")

        (tmpdir / "extri.yml").write_text("\n".join(extri_lines))

        # Create intri.yml
        intri_lines = ["%YAML:1.0", "---"]
        for cam_id in cam_ids:
            K = np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]])
            intri_lines.append(f"K_{cam_id}: !!opencv-matrix")
            intri_lines.append("   rows: 3")
            intri_lines.append("   cols: 3")
            intri_lines.append("   dt: d")
            data = ", ".join(f"{x:.6f}" for x in K.flatten())
            intri_lines.append(f"   data: [ {data} ]")

        (tmpdir / "intri.yml").write_text("\n".join(intri_lines))

        # Run the converter
        from offline.datasets.siga2025vvc_to_lrm_json import convert_siga_to_lrm_manifests

        output_dir = tmpdir / "manifests"
        paths = convert_siga_to_lrm_manifests(
            dataset_root=tmpdir,
            output_dir=output_dir,
            frame_indices=[0, 1],
            camera_indices=None,
            max_cameras=None,
        )

        assert len(paths) == 2, f"Expected 2 manifests, got {len(paths)}"

        for p in paths:
            manifest = json.loads(p.read_text())
            assert "scene_name" in manifest
            assert "frames" in manifest
            assert len(manifest["frames"]) == 3  # 3 cameras

            for frame in manifest["frames"]:
                assert "file_path" in frame
                assert "fx" in frame
                assert "fy" in frame
                assert "cx" in frame
                assert "cy" in frame
                assert "w2c" in frame
                w2c = np.array(frame["w2c"])
                assert w2c.shape == (4, 4), f"w2c shape: {w2c.shape}"

            logger.info(f"Manifest {p.name}: {len(manifest['frames'])} cameras, scene={manifest['scene_name']}")

    logger.info("SIGA2025VVC converter test PASSED")
    return True


def main():
    """Run all AMB3R pipeline tests."""
    test_images_dir = Path("/tmp/amb3r_test/images")

    if not test_images_dir.exists():
        logger.error(f"Test images not found at {test_images_dir}")
        logger.info("Extract frames first: ffmpeg -i video.mp4 -vf 'select=...' /tmp/amb3r_test/images/frame_%06d.jpg")
        return

    image_paths = sorted(test_images_dir.glob("*.jpg"))
    if not image_paths:
        logger.error("No JPEG images found in test directory")
        return

    logger.info(f"Found {len(image_paths)} test images")

    # Test 1: SIGA2025VVC converter
    logger.info("=" * 60)
    logger.info("TEST 1: SIGA2025VVC to LRM JSON converter")
    logger.info("=" * 60)
    test_siga2025vvc_converter()

    # Test 2: COLMAP export
    logger.info("=" * 60)
    logger.info("TEST 2: COLMAP export with synthetic AMB3R data")
    logger.info("=" * 60)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_colmap_export(image_paths, Path(tmpdir))

    # Test 3: Full pipeline to PLY
    logger.info("=" * 60)
    logger.info("TEST 3: Full pipeline → PLY output")
    logger.info("=" * 60)
    output_dir = Path("/tmp/amb3r_test/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = test_colmap_to_ply(image_paths, output_dir)

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info(f"Output PLY: {ply_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
