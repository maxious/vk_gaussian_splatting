"""Export AMB3R predictions to COLMAP format using pycolmap."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_amb3r_to_colmap(
    world_points: np.ndarray,
    confidence: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: list[str | Path],
    export_dir: str | Path,
    conf_thresh_percentile: float = 40.0,
    max_points_per_frame: int = 5000,
) -> None:
    """Export AMB3R predictions to COLMAP binary format.

    Args:
        world_points: (T, H, W, 3) world-space 3D points
        confidence: (T, H, W) or (T, H, W, 1) per-pixel confidence
        extrinsics_w2c: (T, 3, 4) world-to-camera extrinsics
        intrinsics: (T, 3, 3) camera intrinsic matrices
        image_paths: List of T image file paths
        export_dir: Output directory for COLMAP reconstruction
        conf_thresh_percentile: Percentile threshold for confidence filtering
        max_points_per_frame: Max points to keep per frame after filtering
    """
    import pycolmap
    from PIL import Image

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    num_frames = len(image_paths)

    # Squeeze confidence if needed
    if confidence.ndim == 4:
        confidence = confidence[..., 0]

    # Confidence threshold
    conf_thresh = np.percentile(confidence, conf_thresh_percentile)

    # Collect all valid points with their frame index and pixel coords
    all_points = []
    all_colors = []
    all_frame_indices = []
    all_pixel_coords = []

    for fidx in range(num_frames):
        frame_conf = confidence[fidx]  # (H, W)
        frame_pts = world_points[fidx]  # (H, W, 3)
        h, w = frame_conf.shape

        # Valid mask: confident + finite
        valid = (frame_conf >= conf_thresh) & np.all(np.isfinite(frame_pts), axis=-1)

        if valid.sum() == 0:
            continue

        # Get valid indices
        valid_y, valid_x = np.where(valid)

        # Subsample if too many points
        if len(valid_y) > max_points_per_frame:
            indices = np.random.choice(len(valid_y), max_points_per_frame, replace=False)
            valid_y = valid_y[indices]
            valid_x = valid_x[indices]

        pts = frame_pts[valid_y, valid_x]  # (N, 3)

        # Get colors from original image
        img = np.array(Image.open(image_paths[fidx]).convert("RGB"))
        orig_h, orig_w = img.shape[:2]
        # Scale pixel coords to original image size
        scale_x = orig_w / w
        scale_y = orig_h / h
        img_y = np.clip((valid_y * scale_y).astype(int), 0, orig_h - 1)
        img_x = np.clip((valid_x * scale_x).astype(int), 0, orig_w - 1)
        colors = img[img_y, img_x]  # (N, 3)

        all_points.append(pts)
        all_colors.append(colors)
        all_frame_indices.append(np.full(len(pts), fidx, dtype=np.int32))
        # Store pixel coords in processed resolution (will be scaled later)
        all_pixel_coords.append(np.stack([valid_x, valid_y], axis=-1).astype(np.float64))

    if not all_points:
        raise RuntimeError("No valid points found after confidence filtering")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    frame_indices = np.concatenate(all_frame_indices, axis=0)
    pixel_coords = np.concatenate(all_pixel_coords, axis=0)

    num_points = len(points)
    logger.info(f"Exporting to COLMAP with {num_points} points from {num_frames} frames")

    # Build pycolmap Reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add all 3D points (with empty tracks initially)
    point3d_ids = []
    for vidx in range(num_points):
        point3d_id = reconstruction.add_point3D(
            points[vidx], pycolmap.Track(), colors[vidx]
        )
        point3d_ids.append(point3d_id)

    h_proc = world_points.shape[1]
    w_proc = world_points.shape[2]

    for fidx in range(num_frames):
        img = Image.open(image_paths[fidx])
        orig_w, orig_h = img.size

        # Scale intrinsics from processing resolution to original resolution
        K = intrinsics[fidx].copy()
        K[0, :] *= orig_w / w_proc
        K[1, :] *= orig_h / h_proc

        pycolmap_intri = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])

        # w2c extrinsic
        extrinsic = extrinsics_w2c[fidx]  # (3, 4)
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3]
        )

        # Camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # Rig
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

        # Image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = camera.camera_id

        # Frame
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = camera.camera_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # Build 2D points for this frame and update tracks
        point2d_list = []
        points_in_frame = frame_indices == fidx
        for vidx in np.where(points_in_frame)[0]:
            # Scale pixel coords to original image size
            px = pixel_coords[vidx].copy()
            px[0] *= orig_w / w_proc
            px[1] *= orig_h / h_proc
            point3d_id = point3d_ids[vidx]
            point2d_list.append(pycolmap.Point2D(px, point3d_id))
            reconstruction.point3D(point3d_id).track.add_element(
                image.image_id, len(point2d_list) - 1
            )

        image.frame_id = image.image_id
        image.name = os.path.basename(str(image_paths[fidx]))
        image.points2D = pycolmap.Point2DList(point2d_list)
        reconstruction.add_image(image)

    # Write reconstruction
    reconstruction.write(str(export_dir))
    logger.info(f"COLMAP reconstruction written to {export_dir}")
