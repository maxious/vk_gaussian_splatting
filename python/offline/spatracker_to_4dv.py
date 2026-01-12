"""Convert SpaTrackerV2 results to 4DV format."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from offline.export_4dv import export_4dv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert SpaTrackerV2 npz to 4DV")
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input result.npz from SpaTracker"
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .4dv file")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate")
    parser.add_argument(
        "--static-threshold",
        type=float,
        default=0.01,
        help="Variance threshold for static tracks (default: 0.01)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    logger.info(f"Loading {args.input}...")
    data = np.load(args.input, allow_pickle=True)

    # SpaTracker output keys: 'coords', 'extrinsics', 'intrinsics', 'depths', 'video', 'visibs', 'unc_metric'
    # coords: (T, N, 3) - World coordinates? Or Camera? Inference says: "coords" = c2w * track3d + t
    # So 'coords' are World coordinates.

    coords = data["coords"]  # (T, N, 3)
    visibs = data["visibs"]  # (T, N)
    video = data[
        "video"
    ]  # (T, H, W, 3) or (T, 3, H, W)? Inference says: video_tensor is (T, C, H, W), saved as (T, H, W, 3) likely if permuted back?
    # Inference: data_npz_load["video"] = (video_tensor).cpu().numpy()/255
    # Check inference.py: video_tensor was permuted to (N, C, H, W).
    # So saved video is likely (T, 3, H, W) or (T, H, W, 3). Let's assume standard image layout.

    T, N, _ = coords.shape
    logger.info(f"Loaded {N} tracks over {T} frames")

    # Compute motion variance to identify static vs dynamic tracks
    # This is key for temporal compression: static tracks can be represented more efficiently
    variances = np.var(coords, axis=0).sum(axis=-1)  # (N,)
    is_static = variances < args.static_threshold
    logger.info(
        f"Identified {np.sum(is_static)} static tracks and {np.sum(~is_static)} dynamic tracks"
    )

    # Filter invalid tracks (low visibility)
    # If a track is invisible for most frames, maybe skip?
    # For now, keep all.

    # We need to format this for export_4dv.
    # export_4dv takes: means, scales, rotations, colors, opacities, motion, time_center, time_scale
    # These arrays are "per splat".
    # But 4DV assumes a "canonical" set of splats with motion coefficients?
    # Actually, FreeTimeGS / 4DV assumes:
    # x(t) = x_0 + \sum motion_i * basis_i(t)
    # If we have arbitrary trajectories, we might need to fit the motion basis.

    # Or, we can treat this as "One gaussian per track"
    # means = coords at t=0 (or t_center)
    # motion = polynomial coefficients?
    # export_4dv expects 'motion' array. In 'motion_tracking_cpu.py', fit_trajectories returns 'velocity'.
    # FreeTimeGS typically models linear motion per segment or polynomial.
    # The current `export_4dv` implementation takes `motion` which seems to be velocity?
    # Let's check `export_4dv.py`:
    # packed_motions[idx] = pack_11_10_11(mx, my, mz)
    # It just quantizes a vector.
    # The C++ loader interprets this based on `time_scale` and `time_center`.
    # x(t) = x0 + v * (t - t0)
    # So it supports Linear Motion.

    # SpaTracker tracks are non-linear.
    # We can break tracks into chunks (which 4DV supports via "chunks"? No, 4DV chunks are spatial/batching).
    # Wait, 4DV format in `export_4dv.py` is one set of splats.
    # If we want to represent non-linear motion, we might need multiple splats per track (one per segment) fading in/out?
    # Or just fit a single linear motion and accept error?
    # SpaTracker is useful for *complex* motion. Linear approximation destroys value.

    # However, for "compression", maybe we just want the *points*.
    # If `export_4dv` only supports linear motion, then we are limited.
    # But `export_4dv` seems to support `time_scale`.

    # Let's look at `motion_tracking_cpu.py`: `fit_trajectories` computes `velocity`.
    # It seems the current 4DV format is indeed Linear Motion Gaussian Splatting (FreeTimeGS is often linear per short interval).

    # Workaround: Fit piecewise linear?
    # Or just use the "frame 0" + velocity for the whole video? That would be bad.

    # Actually, FreeTimeGS paper (Wang2025) might support more.
    # But our `export_4dv.py` implementation:
    # packed_motions ...
    # This implies a single motion vector per splat.

    # So to use SpaTracker effectively, we should probably output a *sequence* of PLYs or SOGs (one per frame)
    # where the Gaussians are consistent (same ID).
    # But `export_4dv` is for a single file.

    # If the goal is "compress gaussian splat frames", then `4dv` is the target format.
    # If `4dv` only supports linear motion, we can only compress linear segments.
    # Maybe we can split the video into short clips where motion is approx linear?

    # Let's assume we fit a single linear motion for now, or just use the center.
    # But to fully utilize SpaTracker, we'd need a format that supports arbitrary trajectories (like `co4d` or similar).
    # Since `vk_gaussian_splatting` supports `4dv`, let's stick to it.

    # Let's compute average velocity.

    # 1. Compute means (average position or position at t_center)
    # 2. Compute velocity (linear fit)

    # Time normalized 0..1
    times = np.linspace(0, 1, T)

    # Fit X = X0 + V*t
    # V = covariance(X, t) / var(t)

    means_center = np.mean(coords, axis=0)  # (N, 3)

    # Vectorized fit
    t_mean = np.mean(times)
    t_centered = times - t_mean

    # X_centered: (T, N, 3) - (1, N, 3)
    coords_centered = coords - means_center[None, :, :]

    # num: sum(t * x) over T
    numerator = np.sum(t_centered[:, None, None] * coords_centered, axis=0)  # (N, 3)
    denominator = np.sum(t_centered**2)

    velocity = numerator / denominator  # (N, 3)

    # Force static tracks to have zero velocity for compression efficiency
    velocity[is_static] = 0.0

    # Scales: heuristic 0.02
    scales = np.ones((N, 3), dtype=np.float32) * 0.02
    scales = np.log(scales)

    # Rotations: Identity
    rotations = np.zeros((N, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w=1

    # Colors: Sample from video at t_center?
    # Or average color?
    # We need 2D projections to sample color.
    # SpaTracker gives `video` but aligning 3D points to it requires reprojection.
    # `data['intrinsics']` and `data['extrinsics']` are available.
    # Let's pick the middle frame.
    mid_idx = T // 2
    img = video[mid_idx]  # (3, H, W) or (H, W, 3)?
    # data_npz_load["video"] = (video_tensor).cpu().numpy()/255
    # video_tensor shape is (T, C, H, W).
    # So img is (C, H, W).
    C, H, W = img.shape

    # Reproject means_center to mid_idx camera
    # P_cam = E * P_world
    # p_pix = K * P_cam

    # Extrinsics in npz: data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
    # So it's w2c.
    w2c = data["extrinsics"][mid_idx]  # (4, 4)
    K = data["intrinsics"][mid_idx]  # (3, 3)

    # Homogeneous
    ones = np.ones((N, 1))
    P_world = np.hstack([means_center, ones])  # (N, 4)
    P_cam = (w2c @ P_world.T).T  # (N, 4)
    P_cam_3 = P_cam[:, :3]

    uv_hom = (K @ P_cam_3.T).T  # (N, 3)
    u = uv_hom[:, 0] / (uv_hom[:, 2] + 1e-6)
    v = uv_hom[:, 1] / (uv_hom[:, 2] + 1e-6)

    # Sample colors
    colors = np.zeros((N, 3), dtype=np.float32)
    u_int = np.clip(np.round(u), 0, W - 1).astype(int)
    v_int = np.clip(np.round(v), 0, H - 1).astype(int)

    # img is (C, H, W) -> (3, H, W)
    # numpy indexing: img[:, v, u]
    cols = img[:, v_int, u_int].T  # (N, 3)

    # Convert to SH DC
    SH_C0 = 0.28209479177387814
    colors = (cols - 0.5) / SH_C0

    # Opacities
    opacities = np.ones(N, dtype=np.float32) * 10.0

    # Time params
    time_center = np.ones(N, dtype=np.float32) * 0.5
    time_scale = np.ones(N, dtype=np.float32) * 1.0  # spanning 0..1

    logger.info(f"Exporting to {args.output}")
    export_4dv(
        args.output,
        means_center.astype(np.float32),
        scales.astype(np.float32),
        rotations.astype(np.float32),
        colors.astype(np.float32),
        opacities.astype(np.float32),
        velocity.astype(np.float32),
        time_center.astype(np.float32),
        time_scale.astype(np.float32),
    )


if __name__ == "__main__":
    main()
