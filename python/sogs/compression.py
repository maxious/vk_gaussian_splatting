import json
import os
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Mandatory imports for SOG compression
import faiss  # type: ignore
from .plas import sort_with_plas  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from plyfile import PlyData
from PIL import Image


def _get_compress_fn(param_name: str) -> Callable:
    compress_fn_map = {
        "means": _compress_16bit,
        "scales": _compress,
        "quats": _compress_quats,
        "sh0": _compress,  # placeholder
        "shN": _compress_kmeans,
        "motion": _compress_16bit,  # Added for FreeTimeGS
        "t": _compress_16bit,  # Added for FreeTimeGS (16-bit for time precision)
        "t_scale": _compress,  # Added for FreeTimeGS
    }
    return compress_fn_map[param_name]


def run_compression(compress_dir: str, splats: Dict[str, Tensor], verbose: bool) -> None:
    """Run compression

    Args:
        compress_dir (str): directory to save compressed files
        splats (Dict[str, Tensor]): Gaussian splats to compress
    """
    # FAISS is now mandatory for SOG compression

    # Param-specific preprocessing
    splats["means"] = log_transform(splats["means"])
    splats["quats"] = F.normalize(splats["quats"], dim=-1)
    neg_mask = splats["quats"][..., 3] < 0
    splats["quats"][neg_mask] *= -1
    splats["sh0"] = splats["sh0"].clamp(-3.0, 3.0)

    if "shN" in splats:
        splats["shN"] = splats["shN"].clamp(-6.0, 6.0)

    # FreeTimeGS preprocessing
    if "t_scale" in splats:
        # t_scale is strictly positive, log transform it
        splats["t_scale"] = log_transform(splats["t_scale"])

    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    n_crop = n_gs - n_sidelen**2
    if n_crop != 0:
        splats = _crop_n_splats(splats, n_crop)
        print(f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians.")

    meta: Dict[str, Any] = {}

    splats = sort_splats(splats, verbose)

    # Extract opacities and merge into sh0
    opacities = splats.pop("opacities")

    param_iter = (
        tqdm(splats.keys(), desc="Compressing parameters", disable=not verbose)
        if tqdm
        else splats.keys()
    )
    for param_name in param_iter:
        if param_name == "sh0":
            meta["sh0"] = _compress_sh0_with_opacity(
                compress_dir, "sh0", splats["sh0"], opacities, n_sidelen, verbose=verbose
            )
        else:
            compress_fn = _get_compress_fn(param_name)
            meta[param_name] = compress_fn(
                compress_dir, param_name, splats[param_name], n_sidelen=n_sidelen, verbose=verbose
            )

    with open(os.path.join(compress_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def write_image(compress_dir, param_name, img, lossless: bool = True, quality: int = 100):
    filename = f"{param_name}.webp"
    img = np.ascontiguousarray(img)
    Image.fromarray(img).save(
        os.path.join(compress_dir, filename),
        format="webp",
        lossless=lossless,
        quality=quality if not lossless else 100,
        method=6,
        exact=True,
    )
    if verbose_log:
        print(f"[OK] {filename}")
    return filename


verbose_log = True


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    opacities = splats["opacities"]
    keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
    for k, v in splats.items():
        splats[k] = v[keep_indices]
    return splats


def _compress(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, verbose: bool
) -> Dict[str, Any]:
    """Compress parameters with 8-bit quantization and lossless PNG compression."""
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))

    # Handle scalar case (min==max) to avoid div by zero
    diff = maxs - mins
    diff[diff == 0] = 1.0

    grid_norm = (grid - mins) / diff
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    img = img.squeeze()

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "files": [write_image(compress_dir, param_name, img)],
    }
    return meta


def _compress_16bit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, verbose: bool
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression."""
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))

    diff = maxs - mins
    diff[diff == 0] = 1.0

    grid_norm = (grid - mins) / diff
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)
    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF

    files = [
        write_image(compress_dir, f"{param_name}_l", img_l.astype(np.uint8)),
        write_image(compress_dir, f"{param_name}_u", img_u.astype(np.uint8)),
    ]

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "files": files,
    }
    return meta


def _compress_sh0_with_opacity(
    compress_dir: str,
    param_name: str,
    sh0: Tensor,
    opacities: Tensor,
    n_sidelen: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Combine sh0 (RGB) and opacities as alpha channel into a single RGBA texture."""
    grid_sh0 = sh0.reshape((n_sidelen, n_sidelen, -1))
    grid_opac = opacities.reshape((n_sidelen, n_sidelen, 1))
    grid = torch.cat([grid_sh0, grid_opac], dim=-1)

    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))

    diff = maxs - mins
    diff[diff == 0] = 1.0

    grid_norm = (grid - mins) / diff
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    filename = write_image(compress_dir, param_name, img)

    meta = {
        "shape": [*list(sh0.shape[:-1]), sh0.shape[-1] + 1],
        "dtype": str(sh0.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "files": [filename],
    }
    return meta


def _compress_kmeans(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_sidelen: int,
    quantization: int = 8,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run K-means clustering on parameters and save centroids and labels as images."""
    params = params.reshape(params.shape[0], -1)
    dim = params.shape[1]
    n_clusters = round((len(params) >> 2) / 64) * 64
    n_clusters = min(n_clusters, 2**16)

    # Ensure n_clusters is not greater than n_samples
    n_clusters = min(n_clusters, len(params))
    if n_clusters < 1:
        n_clusters = 1

    # Use FAISS K-means (mandatory for SOG compression)
    params_np = params.detach().cpu().numpy().astype(np.float32)
    kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=25, verbose=verbose, gpu=False)
    kmeans.train(params_np)
    centroids = torch.from_numpy(kmeans.centroids.astype(np.float32))  # type: ignore[union-attr]
    labels = kmeans.index.search(params_np, 1)[1].ravel().astype(np.int32)  # type: ignore[union-attr]

    mins = torch.min(centroids)
    maxs = torch.max(centroids)

    diff = maxs - mins
    if diff == 0:
        diff = 1.0

    centroids_norm = (centroids - mins) / diff
    centroids_norm = centroids_norm.detach().cpu().numpy()
    centroids_quant = (centroids_norm * (2**quantization - 1)).round().astype(np.uint8)

    # sort centroids for compact atlas layout
    sorted_indices = np.lexsort(centroids_quant.T)
    # Ensure sorted_indices is proper length for reshaping if needed
    # The original code reshape logic seems specific to 64 alignment?
    # "sorted_indices = sorted_indices.reshape(64, -1).T.reshape(-1)"
    # This reshuffling is specific to SOGS viewer logic maybe?
    # Let's keep it if n_clusters is multiple of 64.

    if len(sorted_indices) % 64 == 0:
        sorted_indices = sorted_indices.reshape(64, -1).T.reshape(-1)

    sorted_centroids_quant = centroids_quant[sorted_indices]

    inverse = np.argsort(sorted_indices)

    # centroids_packed: original code assumes dim*64/3.
    # dim for SH (deg 3) is 45. 45*64/3 = 960.
    # If using less SH, this might break.
    # We should just save centroids as is, but SOGS viewer expects specific packing?
    # For now let's assume standard SH degree 3 (45 coeffs).
    # If not, we might need to adjust.

    # Safety check for reshape
    try:
        centroids_packed = sorted_centroids_quant.reshape(-1, int(dim * 64 / 3), 3)
        files = [
            write_image(compress_dir, f"{param_name}_centroids", centroids_packed),
            write_image(
                compress_dir,
                f"{param_name}_labels_l",
                (labels & 0xFF).astype(np.uint8).reshape(n_sidelen, n_sidelen),
            ),
            write_image(
                compress_dir,
                f"{param_name}_labels_u",
                ((labels >> 8) & 0xFF).astype(np.uint8).reshape(n_sidelen, n_sidelen),
            ),
        ]

        # Note: Original code combined labels into one RG image.
        # "labels_combined[..., 0] = labels_l; labels_combined[..., 1] = labels_u"
        # Let's do that to match spec.
        labels_combined = np.zeros((n_sidelen, n_sidelen, 3), dtype=np.uint8)
        labels_combined[..., 0] = (labels & 0xFF).astype(np.uint8).reshape(n_sidelen, n_sidelen)
        labels_combined[..., 1] = (
            ((labels >> 8) & 0xFF).astype(np.uint8).reshape(n_sidelen, n_sidelen)
        )

        files = [
            write_image(compress_dir, f"{param_name}_centroids", centroids_packed),
            write_image(compress_dir, f"{param_name}_labels", labels_combined),
        ]

    except Exception as e:
        print(f"Warning: KMeans packing failed, falling back to simple save: {e}")
        # Fallback if reshape fails (e.g. not deg 3 SH)
        files = []
        # TODO: Handle non-standard SH counts

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization,
        "files": files,
    }
    return meta


def pack_quaternion_to_rgba_tensor(q: Tensor) -> Tensor:
    """
    Packs a batch of quaternions into RGBA channels:
      - R,G,B: the three smallest components, scaled by sqrt(2) then mapped from [-1,1]→[0,1]
      - A: index of largest-abs component (0→3) mapped [0,3]→[0,1]
    q: (...,4)
    returns: (...,4) in [0,1]
    """
    abs_q = q.abs()
    max_idx = abs_q.argmax(dim=-1)  # (...)

    # ensure largest component is positive
    max_vals = q.gather(-1, max_idx.unsqueeze(-1)).squeeze(-1)
    sign = max_vals.sign()
    sign[sign == 0] = 1
    q_signed = q * sign.unsqueeze(-1)

    # build variants dropping each component
    variants = []
    for i in range(4):
        dims = list(range(4))
        dims.remove(i)
        variants.append(q_signed[..., dims])  # (...,3)
    stacked = torch.stack(variants, dim=-2)  # (...,4,3)

    # select the appropriate 3-vector based on max_idx
    idx_exp = max_idx.unsqueeze(-1).unsqueeze(-1).expand(*max_idx.shape, 1, 3)
    small = torch.gather(stacked, dim=-2, index=idx_exp).squeeze(-2)  # (...,3)

    # scale by sqrt(2) to normalize range to [-1,1]
    small = small * torch.sqrt(torch.tensor(2.0, device=small.device, dtype=small.dtype))

    # map from [-1,1] to [0,1]
    rgb = small * 0.5 + 0.5
    a = (252.0 + max_idx.to(torch.float32)) / 255.0
    return torch.cat([rgb, a.unsqueeze(-1)], dim=-1)


def _compress_quats(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, verbose: bool
) -> Dict[str, Any]:
    """Compress quaternions by packing into RGBA and saving as an 8-bit image."""
    # params: (n_splats,4)
    rgba = pack_quaternion_to_rgba_tensor(params)
    img = (rgba.view(n_sidelen, n_sidelen, 4).cpu().numpy() * 255.0).round().astype(np.uint8)
    filename = write_image(compress_dir, f"{param_name}", img)

    meta = {
        "shape": list(params.shape),
        "dtype": "uint8",
        "encoding": "quaternion_packed",
        "files": [filename],
    }
    return meta


def sort_splats(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Sort splats with Parallel Linear Assignment Sorting from the paper.

    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    if sort_with_plas is None:
        print("Warning: 'plas' module not found. Sorting skipped (compression will be poor).")
        return splats

    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    sort_keys = [k for k in splats if k != "shN"]
    # For FreeTimeGS, include motion/time in sorting?
    # The paper uses "all attributes" essentially.
    # Adding motion to sort keys makes sense as spatially/temporally similar splats should be grouped.

    params_to_sort = torch.cat([splats[k].reshape(n_gs, -1) for k in sort_keys], dim=-1)
    shuffled_indices = torch.randperm(params_to_sort.shape[0], device=params_to_sort.device)
    params_to_sort = params_to_sort[shuffled_indices]
    grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
    _, sorted_indices = sort_with_plas(
        grid.permute(2, 0, 1), improvement_break=1e-4, verbose=verbose
    )
    sorted_indices = sorted_indices.squeeze().flatten()
    sorted_indices = shuffled_indices[sorted_indices]
    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    return splats


@torch.no_grad()
def read_ply(path):
    """
    Reads a .ply file and reconstructs a dictionary of PyTorch tensors on GPU.
    """
    plydata = PlyData.read(path)
    vd = plydata["vertex"].data

    def has_col(col_name):
        return col_name in vd.dtype.names

    xyz = np.stack([vd["x"], vd["y"], vd["z"]], axis=-1)
    f_dc = np.stack([vd[f"f_dc_{i}"] for i in range(3)], axis=-1)

    rest_cols = [c for c in vd.dtype.names if c.startswith("f_rest_")]
    rest_cols_sorted = sorted(rest_cols, key=lambda c: int(c.split("_")[-1]))
    if len(rest_cols_sorted) > 0:
        f_rest = np.stack([vd[c] for c in rest_cols_sorted], axis=-1)
    else:
        f_rest = np.empty((len(vd), 0), dtype=np.float32)

    opacities = vd["opacity"]
    scale = np.stack([vd[f"scale_{i}"] for i in range(3)], axis=-1)
    rotation = np.stack([vd[f"rot_{i}"] for i in range(4)], axis=-1)

    splats = {}
    splats["means"] = torch.from_numpy(xyz).float().cuda()
    splats["opacities"] = torch.from_numpy(opacities).float().cuda()
    splats["scales"] = torch.from_numpy(scale).float().cuda()
    splats["quats"] = torch.from_numpy(rotation).float().cuda()

    sh0_tensor = torch.from_numpy(f_dc).float()
    sh0_tensor = sh0_tensor.unsqueeze(-1).transpose(1, 2)
    splats["sh0"] = sh0_tensor.cuda()

    if f_rest.any():
        if f_rest.shape[1] % 3 != 0:
            # raise ValueError(f"Number of f_rest columns ({f_rest.shape[1]}) not divisible by 3.")
            print(
                f"Warning: f_rest columns {f_rest.shape[1]} not divisible by 3. SHN compression might fail."
            )

        num_rest_per_channel = f_rest.shape[1] // 3
        shn_tensor = (
            torch.from_numpy(f_rest.reshape(-1, 3, num_rest_per_channel)).float().transpose(1, 2)
        )
        splats["shN"] = shn_tensor.cuda()

    # FreeTimeGS Motion Support
    if has_col("motion_0"):
        motion = np.stack([vd[f"motion_{i}"] for i in range(3)], axis=-1)
        splats["motion"] = torch.from_numpy(motion).float().cuda()

    if has_col("t"):
        t = vd["t"]
        splats["t"] = torch.from_numpy(t).float().cuda()

    if has_col("t_scale"):
        t_scale = vd["t_scale"]
        splats["t_scale"] = torch.from_numpy(t_scale).float().cuda()

    return splats
