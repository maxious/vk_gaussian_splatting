"""Analyze Gaussian stability across frames.

Checks whether DA3 Gaussians have stable correspondence across frames by:
1. Comparing same-index Gaussians (pixel-aligned assumption)
2. Comparing matched Gaussians (position-based matching)
3. Analyzing attribute correlation (color, scale, opacity)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from offline.export_gaussian_ply import (
    GaussianFrame,
    load_static_gaussian_ply,
    match_gaussians_bidirectional,
)

logger = logging.getLogger(__name__)


def analyze_same_index_stability(frames: list[GaussianFrame]) -> dict:
    """Check if same-index Gaussians are stable across frames.
    
    If DA3 produces consistent pixel-to-Gaussian mapping, same indices
    should represent the same world points (for static scenes).
    """
    if len(frames) < 2:
        return {}
    
    n_gaussians = min(len(f.means) for f in frames)
    
    position_diffs = []
    color_diffs = []
    opacity_diffs = []
    
    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        
        pos_diff = np.linalg.norm(f1.means[:n_gaussians] - f2.means[:n_gaussians], axis=1)
        color_diff = np.linalg.norm(f1.colors[:n_gaussians] - f2.colors[:n_gaussians], axis=1)
        opacity_diff = np.abs(f1.opacities[:n_gaussians] - f2.opacities[:n_gaussians])
        
        position_diffs.append(pos_diff)
        color_diffs.append(color_diff)
        opacity_diffs.append(opacity_diff)
    
    position_diffs = np.concatenate(position_diffs)
    color_diffs = np.concatenate(color_diffs)
    opacity_diffs = np.concatenate(opacity_diffs)
    
    return {
        "position": {
            "mean": float(np.mean(position_diffs)),
            "median": float(np.median(position_diffs)),
            "std": float(np.std(position_diffs)),
            "p95": float(np.percentile(position_diffs, 95)),
            "p99": float(np.percentile(position_diffs, 99)),
        },
        "color": {
            "mean": float(np.mean(color_diffs)),
            "median": float(np.median(color_diffs)),
            "std": float(np.std(color_diffs)),
        },
        "opacity": {
            "mean": float(np.mean(opacity_diffs)),
            "median": float(np.median(opacity_diffs)),
            "std": float(np.std(opacity_diffs)),
        },
    }


def analyze_matched_stability(
    frames: list[GaussianFrame],
    max_distance: float = 0.05,
) -> dict:
    """Analyze attribute stability for position-matched Gaussians."""
    if len(frames) < 2:
        return {}
    
    all_color_diffs = []
    all_opacity_diffs = []
    all_scale_diffs = []
    match_rates = []
    
    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        
        matches = match_gaussians_bidirectional(f1.means, f2.means, max_distance)
        match_rate = len(matches) / len(f1.means)
        match_rates.append(match_rate)
        
        if len(matches) == 0:
            continue
        
        idx1 = [m[0] for m in matches]
        idx2 = [m[1] for m in matches]
        
        color_diff = np.linalg.norm(f1.colors[idx1] - f2.colors[idx2], axis=1)
        opacity_diff = np.abs(f1.opacities[idx1] - f2.opacities[idx2])
        scale_diff = np.linalg.norm(f1.scales[idx1] - f2.scales[idx2], axis=1)
        
        all_color_diffs.extend(color_diff)
        all_opacity_diffs.extend(opacity_diff)
        all_scale_diffs.extend(scale_diff)
    
    if not all_color_diffs:
        return {"match_rate": {"mean": np.mean(match_rates)}}
    
    return {
        "match_rate": {
            "mean": float(np.mean(match_rates)),
            "min": float(np.min(match_rates)),
            "max": float(np.max(match_rates)),
        },
        "color": {
            "mean": float(np.mean(all_color_diffs)),
            "median": float(np.median(all_color_diffs)),
            "std": float(np.std(all_color_diffs)),
        },
        "opacity": {
            "mean": float(np.mean(all_opacity_diffs)),
            "median": float(np.median(all_opacity_diffs)),
            "std": float(np.std(all_opacity_diffs)),
        },
        "scale": {
            "mean": float(np.mean(all_scale_diffs)),
            "median": float(np.median(all_scale_diffs)),
            "std": float(np.std(all_scale_diffs)),
        },
    }


def analyze_index_vs_matched_correspondence(
    frames: list[GaussianFrame],
    max_distance: float = 0.05,
) -> dict:
    """Compare same-index vs position-matched correspondences.
    
    If same-index works well, it means pixel ordering is stable.
    """
    if len(frames) < 2:
        return {}
    
    same_index_match_count = 0
    position_match_count = 0
    overlap_count = 0
    total_pairs = 0
    
    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        n = min(len(f1.means), len(f2.means))
        
        same_idx_dists = np.linalg.norm(f1.means[:n] - f2.means[:n], axis=1)
        same_idx_matches = set(np.where(same_idx_dists < max_distance)[0])
        
        pos_matches = match_gaussians_bidirectional(f1.means, f2.means, max_distance)
        pos_match_set = set(m[0] for m in pos_matches if m[0] < n)
        
        overlap = same_idx_matches & pos_match_set
        
        same_index_match_count += len(same_idx_matches)
        position_match_count += len(pos_match_set)
        overlap_count += len(overlap)
        total_pairs += n
    
    return {
        "same_index_match_rate": same_index_match_count / total_pairs if total_pairs > 0 else 0,
        "position_match_rate": position_match_count / total_pairs if total_pairs > 0 else 0,
        "overlap_rate": overlap_count / max(same_index_match_count, 1),
        "total_gaussians_per_frame": total_pairs // max(len(frames) - 1, 1),
    }


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Analyze Gaussian stability across frames")
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Directory containing per-frame PLY files")
    parser.add_argument("--pattern", type=str, default="frame_*.ply",
                        help="Glob pattern for PLY files")
    parser.add_argument("--max-distance", type=float, default=0.05,
                        help="Max distance for position matching")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to analyze")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    ply_files = sorted(args.input.glob(args.pattern))
    if args.max_frames:
        ply_files = ply_files[:args.max_frames]
    
    logger.info(f"Loading {len(ply_files)} PLY files...")
    frames = []
    for i, path in enumerate(ply_files):
        frame = load_static_gaussian_ply(path)
        frame.frame_idx = i
        frame.timestamp_ms = i * 33.33
        frames.append(frame)
        logger.debug(f"Loaded {path.name}: {len(frame.means)} Gaussians")
    
    logger.info(f"Analyzing {len(frames)} frames...")
    
    print("\n=== Same-Index Stability ===")
    print("(Assumes Gaussian N in frame 1 = Gaussian N in frame 2)")
    same_idx = analyze_same_index_stability(frames)
    print(json.dumps(same_idx, indent=2))
    
    print("\n=== Position-Matched Stability ===")
    print("(Matches Gaussians by nearest position)")
    matched = analyze_matched_stability(frames, args.max_distance)
    print(json.dumps(matched, indent=2))
    
    print("\n=== Index vs Matched Comparison ===")
    comparison = analyze_index_vs_matched_correspondence(frames, args.max_distance)
    print(json.dumps(comparison, indent=2))
    
    print("\n=== Summary ===")
    if same_idx.get("position", {}).get("median", float("inf")) < args.max_distance:
        print("[OK] Same-index Gaussians are spatially stable - pixel ordering is consistent!")
        print("  Consider using index-based tracking instead of position matching.")
    else:
        print("[!!] Same-index Gaussians move significantly - camera motion or scene change.")
        print("  Position-based matching is required.")


if __name__ == "__main__":
    main()
