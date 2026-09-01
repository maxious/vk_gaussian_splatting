"""
SOG-XT (KISS-GS) container encoder.

Encodes a Gaussian splat scene into the SOG-XT container format: a set of
ordinary WebP images plus a small ``meta.json`` (v3).  This is the container
format used by KISS-GS (arXiv:2608.26948, https://fraunhoferhhi.github.io/KISS-GS/),
extending Self-Organizing Gaussians (SOG) with:

  * A view-dependent color codebook that is itself stored as a 2D image
    (``f_rest_centroids.webp``) indexed by a UV label plane
    (``f_rest_labels.webp``).
  * Positions stored as a coarse/high byte and a detail/low byte plane
    (``means_bytes_1.webp`` / ``means_bytes_0.webp``) after a signed log remap.

The container produced here round-trips through the reference decoder
(``decode_sogxt.py``) and through the C++ ``SogXtLoader`` in the viewer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

# ============================================================================
# Helpers
# ============================================================================


def morton_order_sort(points: np.ndarray) -> np.ndarray:
    """Sort points in Morton (Z-order) curve order for spatial coherence."""
    if len(points) == 0:
        return np.arange(0)
    min_val = points.min(axis=0)
    max_val = points.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    normalized = (points - min_val) / range_val
    quantized = (normalized * 1023).astype(np.uint32)

    x = quantized[:, 0]
    y = quantized[:, 1]
    z = quantized[:, 2]

    def spread(v: np.ndarray) -> np.ndarray:
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    x = spread(x)
    y = spread(y)
    z = spread(z)
    codes = (z << 2) | (y << 1) | x
    return np.argsort(codes)


def _as_numpy(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "detach"):  # torch.Tensor
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


def _quantize_u8(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map ``values`` in [vmin, vmax] to uint8 0..255."""
    span = float(vmax - vmin)
    if span <= 0.0:
        span = 1.0
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    return np.rint(norm * 255.0).astype(np.uint8)


def _quantize_u16_low_high(
    values: np.ndarray, vmin: float, vmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map ``values`` in [vmin, vmax] to a low/high byte pair (16-bit split)."""
    span = float(vmax - vmin)
    if span <= 0.0:
        span = 1.0
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    u16 = np.rint(norm * 65535.0).astype(np.uint16)
    low = (u16 & 0xFF).astype(np.uint8)
    high = ((u16 >> 8) & 0xFF).astype(np.uint8)
    return low, high


def _signed_log(values: np.ndarray) -> np.ndarray:
    """sign(x) * log(|x| + 1) used by SOG-XT for positions."""
    return np.sign(values) * np.log1p(np.abs(values))


def _quantize_u8_per_channel(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel min/max quantization.  Returns (u8, mins, maxs)."""
    mins = values.reshape(-1, values.shape[-1]).min(axis=0)
    maxs = values.reshape(-1, values.shape[-1]).max(axis=0)
    span = maxs - mins
    span[span == 0.0] = 1.0
    norm = np.clip((values - mins) / span, 0.0, 1.0)
    u8 = np.rint(norm * 255.0).astype(np.uint8)
    return u8, mins, maxs


def _quantize_u8_scaled(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Quantize with pre-computed per-channel mins/maxs."""
    span = maxs - mins
    span[span == 0.0] = 1.0
    norm = np.clip((values - mins) / span, 0.0, 1.0)
    return np.rint(norm * 255.0).astype(np.uint8)


def _save_webp(path: Path, arr: np.ndarray) -> None:
    """Write an HxW or HxWx{1,3,4} uint8 array as lossless WebP via Pillow."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    elif arr.shape[2] == 3:
        im = Image.fromarray(arr, mode="RGB")
    elif arr.shape[2] == 4:
        im = Image.fromarray(arr, mode="RGBA")
    elif arr.shape[2] == 1:
        im = Image.fromarray(arr[:, :, 0], mode="L")
    else:
        raise ValueError(f"Unsupported WebP channel count: {arr.shape[2]}")
    im.save(path, format="WEBP", lossless=True, quality=100, method=4, exact=True)


def _tile_image(centroid_grid: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    """
    Inverse of ``untile_feature_grid_3x5``.

    Args:
        centroid_grid: (side, side, 45) uint8 centroid values.
        tile_rows / tile_cols: e.g. 3 / 5 for the 15 SH coefficients (3 RGB
            channels each -> 45 channels).

    Returns:
        (side*tile_rows, side*tile_cols, 3) tiled RGB image.
    """
    side, _, _ = centroid_grid.shape
    tiled = np.zeros((side * tile_rows, side * tile_cols, 3), dtype=np.uint8)
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            coeff = tr * tile_cols + tc
            for ch in range(3):
                plane = centroid_grid[:, :, ch * (tile_rows * tile_cols) + coeff]
                tiled[tr * side : (tr + 1) * side, tc * side : (tc + 1) * side, ch] = plane
    return tiled


def _build_codebook(f_rest: np.ndarray, k: int, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means (FAISS) on the 45-dim SH residual vectors -> (centroids, labels)."""
    try:
        import faiss
    except ImportError:  # pragma: no cover - numpy fallback for tiny scenes
        return _build_codebook_numpy(f_rest, k, iterations)

    data = np.ascontiguousarray(f_rest, dtype=np.float32)
    kmeans = faiss.Kmeans(d=data.shape[1], k=k, niter=iterations, verbose=False, gpu=False, seed=42)
    kmeans.train(data)
    centroids = kmeans.centroids.astype(np.float32)
    _, labels = kmeans.index.search(data, 1)
    return centroids, labels.reshape(-1).astype(np.int64)


def _build_codebook_numpy(
    f_rest: np.ndarray, k: int, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Small-scale k-means fallback (no FAISS)."""
    data = np.ascontiguousarray(f_rest, dtype=np.float32)
    n = data.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(42)
    centroids = data[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iterations):
        dists = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        labels = dists.argmin(axis=1)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = data[mask].mean(axis=0)
    return centroids, labels


# ============================================================================
# Encoder
# ============================================================================


def run_sogxt_compression(
    output_dir: Path,
    splats: dict[str, object],
    iterations: int = 10,
    sh_codebook_side: int | None = None,
    grid_side: int | None = None,
    verbose: bool = True,
) -> None:
    """
    Encode a Gaussian splat scene into a SOG-XT container directory.

    Args:
        output_dir: Directory to write the container into (meta.json + *.webp).
        splats: Dictionary with keys:
            'means'      (N, 3)      positions
            'opacities'  (N,)        opacity in logit space
            'scales'     (N, 3)      scales in log space
            'quats'      (N, 4)      rotation quaternions (w, x, y, z)
            'f_dc'       (N, 3)      base color SH DC coefficients
            'f_rest'     (N, 45)     higher-order SH coefficients, INRIA layout
                                     [R_0..14, G_0..14, B_0..14]
        iterations: K-means iterations for the color codebook.
        sh_codebook_side: Side length of the square color codebook grid
            (clamped to [16, 256]); derived from the primitive count when unset.
        grid_side: Side length of the attribute image grid; derived from the
            primitive count (aligned to a multiple of 4) when unset.
        verbose: Print progress.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    means = _as_numpy(splats["means"]).astype(np.float64)  # (N, 3)
    opacities = _as_numpy(splats["opacities"]).astype(np.float64)  # (N,) logit
    scales = _as_numpy(splats["scales"]).astype(np.float64)  # (N, 3) log
    quats = _as_numpy(splats["quats"]).astype(np.float64)  # (N, 4)
    f_dc = _as_numpy(splats["f_dc"]).astype(np.float64)  # (N, 3)
    f_rest = _as_numpy(splats["f_rest"]).astype(np.float64)  # (N, 45)

    n = means.shape[0]
    if n == 0:
        raise ValueError("Cannot encode an empty scene.")
    if verbose:
        print(f"Encoding {n} Gaussians to SOG-XT container: {output_dir}")

    # Spatial sort (Morton order; KISS-GS uses PLAS, but the container decoder
    # is order-agnostic, so any coherent permutation is valid).
    indices = morton_order_sort(means.astype(np.float32))
    means = means[indices]
    opacities = opacities[indices]
    scales = scales[indices]
    quats = quats[indices]
    f_dc = f_dc[indices]
    f_rest = f_rest[indices]

    # --- Grid layout -------------------------------------------------------
    if grid_side is None:
        grid_side = math.ceil(math.sqrt(n))
        grid_side = ((grid_side + 3) // 4) * 4
    grid_side = int(grid_side)
    cells = grid_side * grid_side
    if cells < n:
        raise ValueError(f"grid_side {grid_side} too small for {n} Gaussians")
    if verbose:
        print(f"Grid: {grid_side}x{grid_side} ({cells} cells, {cells - n} inactive)")

    # Pad each attribute into the full grid (active splats first, rest zeros).
    def pad_grid(values: np.ndarray, per_splat: int) -> np.ndarray:
        out = np.zeros((cells, per_splat), dtype=values.dtype)
        out[:n] = values.reshape(n, per_splat)
        return out

    means_grid = pad_grid(means, 3)
    opacities_grid = pad_grid(opacities, 1).reshape(-1)
    scales_grid = pad_grid(scales, 3)
    quats_grid = pad_grid(quats, 4)
    f_dc_grid = pad_grid(f_dc, 3)

    # --- Color codebook side ----------------------------------------------
    if sh_codebook_side is None:
        k = min(65536, max(256, n // 8))
        sh_codebook_side = math.ceil(math.sqrt(k))
    sh_codebook_side = int(np.clip(sh_codebook_side, 16, 256))
    if verbose:
        print(f"Color codebook: {sh_codebook_side}x{sh_codebook_side}")

    # --- Active mask -------------------------------------------------------
    active = np.zeros(cells, dtype=np.uint8)
    active[:n] = 255
    active_mask_img = active.reshape(grid_side, grid_side)
    _save_webp(output_dir / "active_mask.webp", active_mask_img)

    # --- Means: signed log + 16-bit low/high split -------------------------
    means_sl_full = _signed_log(means_grid)
    means_min = float(means_sl_full[:n].min())
    means_max = float(means_sl_full[:n].max())
    low, high = _quantize_u16_low_high(means_sl_full, means_min, means_max)
    low_img = low.reshape(grid_side, grid_side, 3)
    high_img = high.reshape(grid_side, grid_side, 3)
    _save_webp(output_dir / "means_bytes_0.webp", low_img)
    _save_webp(output_dir / "means_bytes_1.webp", high_img)

    # --- Opacities (logit), scales (log), quats, base color ----------------
    op_min = float(opacities_grid[:n].min())
    op_max = float(opacities_grid[:n].max())
    op_u8 = _quantize_u8(opacities_grid, op_min, op_max).reshape(grid_side, grid_side)
    _save_webp(output_dir / "opacities.webp", op_u8)

    _, scales_mins, scales_maxs = _quantize_u8_per_channel(scales_grid[:n])
    scales_u8_full = _quantize_u8_scaled(scales_grid, scales_mins, scales_maxs)
    _save_webp(output_dir / "scales.webp", scales_u8_full.reshape(grid_side, grid_side, 3))

    _, quats_mins, quats_maxs = _quantize_u8_per_channel(quats_grid[:n])
    quats_u8_full = _quantize_u8_scaled(quats_grid, quats_mins, quats_maxs)
    _save_webp(output_dir / "quaternions.webp", quats_u8_full.reshape(grid_side, grid_side, 4))

    _, f_dc_mins, f_dc_maxs = _quantize_u8_per_channel(f_dc_grid[:n])
    f_dc_u8_full = _quantize_u8_scaled(f_dc_grid, f_dc_mins, f_dc_maxs)
    _save_webp(output_dir / "f_dc.webp", f_dc_u8_full.reshape(grid_side, grid_side, 3))

    # --- View-dependent color codebook -------------------------------------
    k = sh_codebook_side * sh_codebook_side
    # Only active splats are clustered; inactive cells get label 0.
    centroids, labels_active = _build_codebook(f_rest[:n], k, iterations)
    labels = np.zeros(cells, dtype=np.int64)
    labels[:n] = labels_active

    # Place centroids on a square grid (row-major), label = v*side + u.
    centroid_grid = centroids.reshape(sh_codebook_side, sh_codebook_side, 45)
    centroid_grid_u8, cen_mins, cen_maxs = _quantize_u8_per_channel(centroid_grid)
    tiled = _tile_image(centroid_grid_u8, tile_rows=3, tile_cols=5)
    _save_webp(output_dir / "f_rest_centroids.webp", tiled)

    # Label plane: channel 0 = u, channel 1 = v, channel 2 = 0.
    u = (labels % sh_codebook_side).astype(np.uint8)
    v = (labels // sh_codebook_side).astype(np.uint8)
    labels_img = np.zeros((grid_side, grid_side, 3), dtype=np.uint8)
    labels_img[:, :, 0] = u.reshape(grid_side, grid_side)
    labels_img[:, :, 1] = v.reshape(grid_side, grid_side)
    _save_webp(output_dir / "f_rest_labels.webp", labels_img)

    # --- meta.json ---------------------------------------------------------
    meta = {
        "version": 3,
        "format": "sog-xt",
        "profile": "SOG-XT",
        "count": int(n),
        "gridSide": int(grid_side),
        "mask": {"files": ["active_mask.webp"]},
        "means": {
            "mins": means_min,
            "maxs": means_max,
            "files": ["means_bytes_0.webp", "means_bytes_1.webp"],
        },
        "opacities": {
            "mins": op_min,
            "maxs": op_max,
            "files": ["opacities.webp"],
            "normalize": "observed-minmax",
        },
        "scales": {
            "mins": scales_mins.tolist(),
            "maxs": scales_maxs.tolist(),
            "files": ["scales.webp"],
            "normalize": "observed-minmax",
        },
        "quats": {
            "mins": quats_mins.tolist(),
            "maxs": quats_maxs.tolist(),
            "files": ["quaternions.webp"],
            "encoding": "direct",
            "normalize": "observed-minmax",
        },
        "sh0": {
            "mins": f_dc_mins.tolist(),
            "maxs": f_dc_maxs.tolist(),
            "files": ["f_dc.webp"],
            "normalize": "observed-minmax",
        },
        "shN": {
            "layout": "uv-codebook",
            "coeffs": 15,
            "centroidSide": int(sh_codebook_side),
            "tileRows": 3,
            "tileCols": 5,
            "centroidsMins": cen_mins.tolist(),
            "centroidsMaxs": cen_maxs.tolist(),
            "files": ["f_rest_centroids.webp", "f_rest_labels.webp"],
            "normalize": "observed-minmax",
        },
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- scene.json manifest (for web viewers) ------------------------------
    scene = {
        "version": 1,
        "metadata": {"url": "meta.json", "bytes": (output_dir / "meta.json").stat().st_size},
        "planes": {
            "mask": {
                "url": "active_mask.webp",
                "bytes": (output_dir / "active_mask.webp").stat().st_size,
            },
            "positionCoarse": {
                "url": "means_bytes_1.webp",
                "bytes": (output_dir / "means_bytes_1.webp").stat().st_size,
            },
            "positionDetail": {
                "url": "means_bytes_0.webp",
                "bytes": (output_dir / "means_bytes_0.webp").stat().st_size,
            },
            "baseColor": {"url": "f_dc.webp", "bytes": (output_dir / "f_dc.webp").stat().st_size},
            "opacity": {
                "url": "opacities.webp",
                "bytes": (output_dir / "opacities.webp").stat().st_size,
            },
            "scales": {"url": "scales.webp", "bytes": (output_dir / "scales.webp").stat().st_size},
            "rotations": {
                "url": "quaternions.webp",
                "bytes": (output_dir / "quaternions.webp").stat().st_size,
            },
            "shCentroids": {
                "url": "f_rest_centroids.webp",
                "bytes": (output_dir / "f_rest_centroids.webp").stat().st_size,
            },
            "shLabels": {
                "url": "f_rest_labels.webp",
                "bytes": (output_dir / "f_rest_labels.webp").stat().st_size,
            },
        },
    }
    with open(output_dir / "scene.json", "w") as f:
        json.dump(scene, f, indent=2)

    if verbose:
        total = sum(p.stat().st_size for p in output_dir.iterdir())
        print(f"SUCCESS: SOG-XT container written ({total} bytes total)")


# ============================================================================
# PLY reader
# ============================================================================


def read_ply(path: str) -> dict[str, np.ndarray]:
    """
    Read a 3DGS-INRIA ``.ply`` into the splat dictionary consumed by
    :func:`run_sogxt_compression`.

    Returns keys: 'means', 'opacities' (logit), 'scales' (log),
    'quats' (w,x,y,z), 'f_dc' (N,3), 'f_rest' (N,45).
    """
    import plyfile

    plydata = plyfile.PlyData.read(path)
    vd = plydata["vertex"].data

    means = np.stack([vd["x"], vd["y"], vd["z"]], axis=-1).astype(np.float32)
    opacities = np.asarray(vd["opacity"], dtype=np.float32)  # already logit
    scales = np.stack([vd[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float32)  # log
    quats = np.stack([vd[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float32)
    f_dc = np.stack([vd[f"f_dc_{i}"] for i in range(3)], axis=-1).astype(np.float32)

    if "f_rest_44" in vd.dtype.names:
        f_rest = np.stack([vd[f"f_rest_{i}"] for i in range(45)], axis=-1).astype(np.float32)
    else:
        f_rest = np.zeros((len(means), 45), dtype=np.float32)

    return {
        "means": means,
        "opacities": opacities,
        "scales": scales,
        "quats": quats,
        "f_dc": f_dc,
        "f_rest": f_rest,
    }


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input INRIA .ply file.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output SOG-XT container directory."
    )
    parser.add_argument(
        "--sh-codebook-side",
        type=int,
        default=None,
        help="Color codebook grid side (clamped to [16, 256]). Derived from count by default.",
    )
    parser.add_argument(
        "--grid-side",
        type=int,
        default=None,
        help="Attribute image grid side. Derived from count by default.",
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="K-means iterations (default: 10)."
    )
    args = parser.parse_args()

    splats = read_ply(str(args.input))
    run_sogxt_compression(
        args.output,
        splats,
        iterations=args.iterations,
        sh_codebook_side=args.sh_codebook_side,
        grid_side=args.grid_side,
    )


if __name__ == "__main__":
    main()
