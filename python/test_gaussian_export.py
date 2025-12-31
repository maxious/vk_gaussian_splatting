"""Test script for Gaussian PLY export pipeline.

This tests the export_gaussian_ply module with a small video sample.
Requires DA3-GIANT model and CUDA-enabled PyTorch.

Usage:
    cd python
    uv run python test_gaussian_export.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_static_ply_export():
    """Test exporting video frames to static PLY files."""
    import numpy as np
    from offline.export_gaussian_ply import (
        write_static_gaussian_ply,
        GaussianFrame,
    )
    
    output_dir = Path("../_downloaded_resources/test_gaussian_export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_points = 100
    means = np.random.randn(n_points, 3).astype(np.float32) * 0.5
    scales = np.random.randn(n_points, 3).astype(np.float32) * 0.1
    rotations = np.random.randn(n_points, 4).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    colors = np.random.rand(n_points, 3).astype(np.float32)
    opacities = np.random.randn(n_points).astype(np.float32)
    
    ply_path = output_dir / "test_static.ply"
    write_static_gaussian_ply(
        ply_path,
        means,
        scales,
        rotations,
        colors,
        opacities,
    )
    
    assert ply_path.exists(), "PLY file not created"
    assert ply_path.stat().st_size > 0, "PLY file is empty"
    
    logger.info(f"Static PLY test passed: {ply_path}")
    return True


def test_freetimegs_ply_export():
    """Test exporting to FreeTimeGS PLY format."""
    import numpy as np
    from offline.export_gaussian_ply import write_freetimegs_ply
    
    output_dir = Path("../_downloaded_resources/test_gaussian_export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_points = 100
    means = np.random.randn(n_points, 3).astype(np.float32) * 0.5
    scales = np.random.randn(n_points, 3).astype(np.float32) * 0.1
    rotations = np.random.randn(n_points, 4).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    colors = np.random.rand(n_points, 3).astype(np.float32)
    opacities = np.random.randn(n_points).astype(np.float32)
    motion = np.random.randn(n_points, 3).astype(np.float32) * 0.1
    time_center = np.random.rand(n_points).astype(np.float32)
    time_scale = np.random.randn(n_points).astype(np.float32) * 0.5
    
    ply_path = output_dir / "test_freetimegs.ply"
    write_freetimegs_ply(
        ply_path,
        means,
        scales,
        rotations,
        colors,
        opacities,
        motion,
        time_center,
        time_scale,
    )
    
    assert ply_path.exists(), "FreeTimeGS PLY file not created"
    assert ply_path.stat().st_size > 0, "FreeTimeGS PLY file is empty"
    
    with open(ply_path, "rb") as f:
        header = f.read(1024).decode("utf-8", errors="ignore")
        assert "property float motion_0" in header, "Missing motion_0 property"
        assert "property float t" in header, "Missing t property"
        assert "property float t_scale" in header, "Missing t_scale property"
    
    logger.info(f"FreeTimeGS PLY test passed: {ply_path}")
    return True


def test_motion_vector_computation():
    """Test motion vector computation from frame sequence."""
    import numpy as np
    from offline.export_gaussian_ply import GaussianFrame, compute_motion_vectors
    
    frames = []
    n_points = 50
    for i in range(5):
        base_pos = np.random.randn(n_points, 3).astype(np.float32)
        motion_offset = np.array([0.1, 0, 0], dtype=np.float32) * i
        means = base_pos + motion_offset
        
        frames.append(GaussianFrame(
            frame_idx=i,
            timestamp_ms=i * 100.0,
            means=means,
            scales=np.zeros((n_points, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (n_points, 1)).astype(np.float32),
            colors=np.ones((n_points, 3), dtype=np.float32) * 0.5,
            opacities=np.zeros(n_points, dtype=np.float32),
        ))
    
    (means, scales, rotations, colors, opacities,
     motion, time_center, time_scale) = compute_motion_vectors(frames, fps=10.0)
    
    assert len(means) > 0, "No Gaussians in output"
    assert motion.shape == (len(means), 3), "Motion shape mismatch"
    assert time_center.shape == (len(means),), "Time center shape mismatch"
    assert time_scale.shape == (len(means),), "Time scale shape mismatch"
    
    logger.info(f"Motion vector computation test passed: {len(means)} Gaussians")
    return True


def test_da3_gaussian_inference():
    """Test DA3 Gaussian inference (requires CUDA and model download).
    
    This test is skipped if CUDA is not available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping DA3 inference test")
            return True
    except ImportError:
        logger.warning("PyTorch not installed, skipping DA3 inference test")
        return True
    
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        logger.warning("depth-anything-3 not installed, skipping DA3 inference test")
        return True
    
    video_path = Path("../_downloaded_resources/BigBuckBunny_320x180.mp4")
    if not video_path.exists():
        logger.warning(f"Test video not found: {video_path}, skipping DA3 test")
        return True
    
    from offline.export_gaussian_ply import (
        extract_video_frames,
        DA3GaussianProcessor,
    )
    
    output_dir = Path("../_downloaded_resources/test_gaussian_export")
    temp_dir = output_dir / "temp_frames"
    
    frame_paths, timestamps = extract_video_frames(
        video_path,
        temp_dir,
        frame_skip=100,
        max_frames=3,
    )
    
    if not frame_paths:
        logger.error("No frames extracted")
        return False
    
    logger.info(f"Extracted {len(frame_paths)} test frames")
    
    processor = DA3GaussianProcessor(
        model_id="depth-anything/DA3-GIANT",
        device="cuda",
        process_res=518,
    )
    
    try:
        gaussian_frames = processor.process_frames(frame_paths, timestamps)
        logger.info(f"Processed {len(gaussian_frames)} frames to Gaussians")
        
        for i, frame in enumerate(gaussian_frames):
            logger.info(
                f"Frame {i}: {len(frame.means)} Gaussians, "
                f"means shape={frame.means.shape}, "
                f"scales shape={frame.scales.shape}"
            )
        
        return True
    except Exception as e:
        logger.error(f"DA3 inference failed: {e}")
        return False


def main():
    tests = [
        ("Static PLY export", test_static_ply_export),
        ("FreeTimeGS PLY export", test_freetimegs_ply_export),
        ("Motion vector computation", test_motion_vector_computation),
        ("DA3 Gaussian inference", test_da3_gaussian_inference),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        logger.info(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        try:
            if test_fn():
                passed += 1
                logger.info(f"✓ {name} PASSED")
            else:
                failed += 1
                logger.error(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            logger.exception(f"✗ {name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
