import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from offline.export_4dv import export_4dv
from offline.any4d_integration import (
    load_any4d_model,
    ANY4D_AVAILABLE,
    load_images,
    load_moge_model,
    loss_of_one_batch_multi_view,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_any4d_export(
    input_path, output_path, model_path=None, fps=30.0, num_splats=100_000, device="cuda"
):
    """
    Run Any4D export to 4DV.
    """
    if not ANY4D_AVAILABLE:
        logger.error("Any4D dependencies not found. Please install them first.")
        return

    # 1. Load images
    if input_path.is_file():
        # TODO: Handle video file extraction
        logger.error(
            "Direct video file support not yet implemented in this script. Please extract frames first."
        )
        return

    image_paths = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    if not image_paths:
        logger.error(f"No images found in {input_path}")
        return

    # 2. Load Model
    logger.info("Loading Any4D model...")
    try:
        model = load_any4d_model(model_path, device=device)
    except Exception as e:
        logger.error(f"Failed to load Any4D model: {e}")
        return

    # 3. Run Inference
    # We run pairwise inference: Ref (Frame 0) -> Target (Frame t)
    # This gives us P(0) and Flow(t)

    # First, run for t=0 to get canonical points
    logger.info("Running Any4D inference...")

    # We process in batches/windows if needed, but Any4D demo does pairwise relative to a reference.
    # Let's treat Frame 0 as reference for the whole sequence (simple approach).
    # Any4D `loss_of_one_batch_multi_view` handles a list of views.
    # It computes everything relative to the first view in the list?
    # Checking demo_inference.py:
    # It constructs `views = [image_paths[ref_idx], image_paths[idx]]` and runs inference.
    # This confirms pairwise processing is standard.

    # Step 3a: Get Canonical Points from Frame 0
    # We can just run inference on [Frame 0, Frame 1] to get Frame 0 points.

    ref_idx = 0

    # Run first pair to get points
    # We need to construct the list of all images to load them properly via load_images (which handles resizing etc)
    # But Any4D load_images loads ALL of them. This might be heavy for VRAM if N is large.
    # Demo script loads subsets.

    # Load Frame 0 and Frame 1
    # We rely on `run_inference` from integration script which calls `load_images` on the list.
    # Let's modify usage to be pairwise to save memory if needed, but `run_inference` takes list.

    # Let's try to load all views first (metadata) but not full tensors if possible?
    # load_images returns list of dicts with tensors. Memory intensive for long videos.
    # We should iterate.

    moge_model = load_moge_model(device=device)

    # We store trajectories: list of (N, 3) positions for each frame
    trajectories = []

    # Get Canonical Points (P0)
    # We use Frame 0 as reference.
    # We need to find valid points (mask).

    p0 = np.zeros((num_splats, 3), dtype=np.float32)
    colors0 = np.zeros((num_splats, 3), dtype=np.float32)
    scales0 = np.log(np.ones((num_splats, 3)) * 0.01)

    # We will accumulate positions for subsampled points

    num_frames = len(image_paths)

    # List to store flow/positions for selected points
    selected_indices = None

    # Iterate frames
    for t in tqdm(range(num_frames)):
        if t == 0:
            # For t=0, we just need the points.
            # We pair with t=1 to trigger the model (it expects multiview?)
            # Actually, `loss_of_one_batch_multi_view` works with >=1 views?
            # Config says `task=images_only`.
            # Let's pass [Frame 0, Frame 1]
            target_idx = 1 if num_frames > 1 else 0
        else:
            target_idx = t

        current_pair_paths = [image_paths[0], image_paths[target_idx]]

        # Load pair
        views = load_images(
            current_pair_paths,
            size=(518, 336),  # Standard Any4D resolution
            verbose=False,
            norm_type="dinov2",
            patch_size=14,
            compute_moge_mask=True,
            moge_model=moge_model,
            binary_mask_path=None,
        )

        # Inference
        with torch.no_grad():
            output = loss_of_one_batch_multi_view(
                views,
                model,
                None,
                device,
                use_amp=True,
            )

        # Extract data
        # output keys: pred1, pred2...
        # pred1 corresponds to Frame 0
        # pred2 corresponds to Frame t

        pred_ref = output["pred1"]
        pred_target = output["pred2"]

        # P0 from pred1
        pts3d_0 = pred_ref["pts3d"][0].cpu().numpy()  # (H*W, 3) or similar

        if t == 0:
            # Initialize Canonical Points
            # Get mask for valid points
            # Demo uses `non_ambiguous_mask` and depth checks

            mask = views[0]["non_ambiguous_mask"].cpu().numpy()

            # Additional masking (depth < 40)
            depth_z = pred_ref["pts3d_cam"][..., 2:3][0].squeeze(-1).cpu().numpy()
            mask = mask & (depth_z < 40.0)

            # Subsample
            valid_indices = np.where(mask.flatten())[0]

            if len(valid_indices) > num_splats:
                selected_indices = np.random.choice(valid_indices, num_splats, replace=False)
            else:
                selected_indices = valid_indices

            p0 = pts3d_0[selected_indices]

            # Get Colors (from image)
            img0 = views[0]["img"][0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) normalized?
            # Need to denormalize? `load_images` normalizes.
            # We can reload raw image or use `rgb` utility from Any4D
            # For 4DV we want SH DC.
            # Let's treat normalized RGB as "color" for now, or revert normalization.
            # DINOv2 norm: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img0_raw = (img0 * std + mean).clip(0, 1)

            colors0 = img0_raw.reshape(-1, 3)[selected_indices]

            # Scales
            # Heuristic or from local density
            # Any4D doesn't output splat scales directly.
            # We use a default small scale.
            scales0 = np.ones((len(p0), 3)) * 0.01
            scales0 = np.log(scales0)  # Log scale

            trajectories.append(p0)

        else:
            # t > 0
            # We want P(t).
            # output["pred2"] contains info for View 2.
            # BUT Any4D predicts SCENE FLOW from View 1 to View 2 in `scene_flow` key of `pred2`?
            # Demo: `cur_scene_flow = cur_pred_result["pred2"]["scene_flow"]`
            # `pts3d_after_motion = cur_pts3d + cur_scene_flow`
            # where `cur_pts3d` is pred1 points.

            if "scene_flow" in pred_target:
                flow = pred_target["scene_flow"][0].cpu().numpy()  # (N, 3)
                # Apply flow to P0
                # Flow is dense, aligned with P0
                flow_selected = flow[selected_indices]
                pt = p0 + flow_selected
                trajectories.append(pt)
            else:
                logger.warning(f"No scene flow found for frame {t}")
                trajectories.append(p0)  # Assume static if no flow

    # 4. Fit Linear Motion
    # We have trajectories (T, N, 3)
    traj_array = np.stack(trajectories, axis=0)  # (T, N, 3)
    T, N, _ = traj_array.shape

    logger.info(f"Fitting motion for {N} splats over {T} frames...")

    # Fit X(t) = X0 + V*t
    # times 0..1
    times = np.linspace(0, 1, T)

    means_center = np.mean(traj_array, axis=0)  # (N, 3)

    t_mean = np.mean(times)
    t_centered = times - t_mean

    coords_centered = traj_array - means_center[None, :, :]

    numerator = np.sum(t_centered[:, None, None] * coords_centered, axis=0)
    denominator = np.sum(t_centered**2)

    velocity = numerator / denominator  # (N, 3)

    # 5. Export
    logger.info(f"Exporting to {output_path}")

    # Prepare data for export_4dv
    rotations = np.zeros((N, 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    opacities = np.ones(N, dtype=np.float32) * 10.0  # Logit for ~1.0

    # Colors to SH DC
    SH_C0 = 0.28209479177387814
    colors_sh = (colors0 - 0.5) / SH_C0

    time_center = np.ones(N, dtype=np.float32) * 0.5
    time_scale = np.ones(N, dtype=np.float32) * 1.0

    export_4dv(
        output_path,
        means_center.astype(np.float32),
        scales0.astype(np.float32),
        rotations.astype(np.float32),
        colors_sh.astype(np.float32),
        opacities.astype(np.float32),
        velocity.astype(np.float32),
        time_center.astype(np.float32),
        time_scale.astype(np.float32),
    )


def main():
    if not ANY4D_AVAILABLE:
        logger.error("Any4D dependencies not found. Please install them first.")
        return

    parser = argparse.ArgumentParser(description="Convert Any4D results to 4DV format")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input folder containing images or video file",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .4dv file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to Any4D checkpoint")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate")
    parser.add_argument(
        "--num-splats", type=int, default=100_000, help="Number of splats to subsample"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    run_any4d_export(
        args.input,
        args.output,
        model_path=args.model_path,
        fps=args.fps,
        num_splats=args.num_splats,
        device=args.device,
    )
