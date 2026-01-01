import numpy as np
import struct
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Optional
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    min_x: float = 0.0
    max_x: float = 0.0
    min_y: float = 0.0
    max_y: float = 0.0
    min_z: float = 0.0
    max_z: float = 0.0
    min_scale_x: float = 0.0
    max_scale_x: float = 0.0
    min_scale_y: float = 0.0
    max_scale_y: float = 0.0
    min_scale_z: float = 0.0
    max_scale_z: float = 0.0
    min_r: float = 0.0
    max_r: float = 0.0
    min_g: float = 0.0
    max_g: float = 0.0
    min_b: float = 0.0
    max_b: float = 0.0
    min_motion_x: float = 0.0
    max_motion_x: float = 0.0
    min_motion_y: float = 0.0
    max_motion_y: float = 0.0
    min_motion_z: float = 0.0
    max_motion_z: float = 0.0
    min_time_scale: float = 0.0
    max_time_scale: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0


def quantize_normalized(values, min_val, max_val, bits):
    """Quantize values to [0, 2^bits - 1] range based on min/max."""
    if min_val == max_val:
        return np.zeros_like(values, dtype=np.uint32)

    # Normalize to 0..1
    norm = (values - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0.0, 1.0)

    # Scale to integer range
    max_int = (1 << bits) - 1
    return np.round(norm * max_int).astype(np.uint32)


def pack_11_10_11(v0, v1, v2):
    """Pack three values into 32 bits (11, 10, 11 bits)."""
    # v0: 11 bits (MSB)
    # v1: 10 bits
    # v2: 11 bits (LSB)
    return ((v0 & 0x7FF) << 21) | ((v1 & 0x3FF) << 11) | (v2 & 0x7FF)


def export_4dv(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    motion: np.ndarray,
    time_center: np.ndarray,
    time_scale: np.ndarray,
    sh_rest: Optional[np.ndarray] = None,
    chunk_size: int = 256,
):
    """Export to 4DV format."""
    num_splats = len(means)
    num_chunks = (num_splats + chunk_size - 1) // chunk_size

    logger.info(f"Exporting {num_splats} splats to 4DV ({num_chunks} chunks)...")

    # Prepare data arrays
    chunks = []
    packed_positions = np.zeros(num_splats, dtype=np.uint32)
    packed_rotations = np.zeros(num_splats, dtype=np.uint32)
    packed_scales = np.zeros(num_splats, dtype=np.uint32)
    packed_colors = np.zeros(num_splats, dtype=np.uint32)
    packed_motions = np.zeros(num_splats, dtype=np.uint32)
    packed_times = np.zeros(num_splats, dtype=np.uint32)

    # Process SH if present
    has_sh = sh_rest is not None and sh_rest.shape[1] > 0
    sh_data = None
    if has_sh:
        # Quantize SH to [-1, 1] -> [0, 255]
        # C++ loader assumes: output = (val / 255.0) * 2.0 - 1.0
        # Inverse: val = (output + 1.0) / 2.0 * 255.0
        sh_data = np.clip((sh_rest + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)

    # Process chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, num_splats)
        idx = slice(start, end)

        # 1. Positions (11-10-11)
        # ----------------------------------------------------------------
        pos_chunk = means[idx]
        min_p = pos_chunk.min(axis=0)
        max_p = pos_chunk.max(axis=0)

        # Expand bounds slightly to avoid precision issues at edges
        margin = 1e-5
        min_p -= margin
        max_p += margin

        px = quantize_normalized(pos_chunk[:, 0], min_p[0], max_p[0], 11)
        py = quantize_normalized(pos_chunk[:, 1], min_p[1], max_p[1], 10)
        pz = quantize_normalized(pos_chunk[:, 2], min_p[2], max_p[2], 11)
        packed_positions[idx] = pack_11_10_11(px, py, pz)

        # 2. Scales (11-10-11)
        # ----------------------------------------------------------------
        # Scales are already log-space in input
        scale_chunk = scales[idx]

        # Use linear scale for min/max in chunk (C++ loader uses unpackLerp -> log)
        # Wait, C++ loader: output.scale[...] = std::log(unpackLerp(...))
        # So we should store linear values in the chunk bounds?
        # NO. C++: output.scale[...] = log(unpackLerp(...)) implies the stored value
        # (unpacked from bits) is LINEAR.
        # So min_scale/max_scale are linear values.
        # But our input `scales` is log-scale.
        # So we need to convert input log-scales to linear, find min/max,
        # quantize linear values, and store min/max linear.

        scale_linear = np.exp(scale_chunk)
        min_s = scale_linear.min(axis=0)
        max_s = scale_linear.max(axis=0)
        min_s *= 1.0 - margin
        max_s *= 1.0 + margin

        sx = quantize_normalized(scale_linear[:, 0], min_s[0], max_s[0], 11)
        sy = quantize_normalized(scale_linear[:, 1], min_s[1], max_s[1], 10)
        sz = quantize_normalized(scale_linear[:, 2], min_s[2], max_s[2], 11)
        packed_scales[idx] = pack_11_10_11(sx, sy, sz)

        # 3. Colors (8-8-8 + 8 Opacity)
        # ----------------------------------------------------------------
        # Colors are SH DC.
        # C++ loader: output.f_dc[...] = (unpackLerp(min_r, max_r, cr, 8) - 0.5) / SH_C0
        # So min_r/max_r are in "0..1 + offset" space?
        # Let's see: unpackLerp returns a value X. Then (X - 0.5) / 0.282.
        # This implies X = f_dc * 0.282 + 0.5.
        # So we need to transform input f_dc to this "X" space (0..1-ish RGB).

        SH_C0 = 0.28209479177387814
        rgb_chunk = colors[idx] * SH_C0 + 0.5
        # Clamp to reasonable range before finding min/max to avoid outliers blowing bits?
        # Standard RGB is 0..1.

        min_c = rgb_chunk.min(axis=0)
        max_c = rgb_chunk.max(axis=0)
        # Add slight margin
        min_c = np.maximum(0.0, min_c - margin)
        max_c = np.minimum(1.0, max_c + margin)

        cr = quantize_normalized(rgb_chunk[:, 0], min_c[0], max_c[0], 8)
        cg = quantize_normalized(rgb_chunk[:, 1], min_c[1], max_c[1], 8)
        cb = quantize_normalized(rgb_chunk[:, 2], min_c[2], max_c[2], 8)

        # Opacity: C++: op = unpackLerp(0.0, 1.0, ca, 8); output = log(op / (1-op))
        # So we need to sigmoid input opacity (which is logit) to get 0..1 prob
        op_chunk_logit = opacities[idx]
        op_chunk_prob = 1.0 / (1.0 + np.exp(-op_chunk_logit))
        ca = quantize_normalized(op_chunk_prob, 0.0, 1.0, 8)

        packed_colors[idx] = (cr << 24) | (cg << 16) | (cb << 8) | ca

        # 4. Motion (11-10-11)
        # ----------------------------------------------------------------
        mot_chunk = motion[idx]
        min_m = mot_chunk.min(axis=0)
        max_m = mot_chunk.max(axis=0)
        min_m -= margin
        max_m += margin

        mx = quantize_normalized(mot_chunk[:, 0], min_m[0], max_m[0], 11)
        my = quantize_normalized(mot_chunk[:, 1], min_m[1], max_m[1], 10)
        mz = quantize_normalized(mot_chunk[:, 2], min_m[2], max_m[2], 11)
        packed_motions[idx] = pack_11_10_11(mx, my, mz)

        # 5. Time (11 scale - 10 center - 11 unused)
        # ----------------------------------------------------------------
        # Time Center
        t_chunk = time_center[idx]
        min_t = t_chunk.min()
        max_t = t_chunk.max()
        if min_t == max_t:
            max_t += 1e-6

        tc = quantize_normalized(t_chunk, min_t, max_t, 10)

        # Time Scale
        # C++: output = exp(unpackLerp(...)) -> stored is log(time_scale)
        # Input `time_scale` is typically linear (e.g. 1.0, 0.5) or log?
        # export_gaussian_ply.py: elements["t_scale"] = time_scale
        # In motion_tracking_cpu.py: time_scale = np.ones(...) * 1.0 (linear)
        # So input is linear. We need to store LOG values in the chunk bounds.
        ts_chunk = np.log(np.maximum(time_scale[idx], 1e-6))
        min_ts = ts_chunk.min()
        max_ts = ts_chunk.max()
        if min_ts == max_ts:
            max_ts += 1e-6

        ts = quantize_normalized(ts_chunk, min_ts, max_ts, 11)

        # Pack: (scale << 21) | (center << 11) | (unused)
        packed_times[idx] = ((ts & 0x7FF) << 21) | ((tc & 0x3FF) << 11)

        # 6. Rotation (Packed 32 bit)
        # ----------------------------------------------------------------
        # Smallest 3 components
        # Find max component index
        rot_chunk = rotations[idx]  # (N, 4) w,x,y,z
        # Ensure normalized
        rot_chunk /= np.linalg.norm(rot_chunk, axis=1, keepdims=True)

        # We need to store [a, b, c] + index
        # C++ decoder:
        # a = (norm(r0) * 1.414) - 0.707
        # This maps 0..1023 -> -0.707..0.707
        # So we need to map input (-0.707..0.707) -> 0..1023

        abs_rot = np.abs(rot_chunk)
        max_idx = np.argmax(abs_rot, axis=1)  # 0=w, 1=x, 2=y, 3=z

        # Sign fix: ensure max component is positive
        signs = np.sign(rot_chunk[np.arange(end - start), max_idx])
        rot_chunk *= signs[:, np.newaxis]

        # Extract other 3 components
        # We need a robust way to select "not max"
        # Masking approach
        packed_r = np.zeros(end - start, dtype=np.uint32)

        # Helper range
        range_min = -0.70710678
        range_max = 0.70710678

        for k in range(end - start):
            r = rot_chunk[k]
            mi = max_idx[k]

            # Get the 3 other components
            others = np.delete(r, mi)

            # Quantize to 10 bits
            q = quantize_normalized(others, range_min, range_max, 10)

            # Pack: index (2 bits) | r0 (10) | r1 (10) | r2 (10)
            # C++: r0 @ 20, r1 @ 10, r2 @ 0
            # Note: C++ decoder mapping:
            # case 0 (w dropped): q_x=a, q_y=b, q_z=c -> implies order x,y,z
            # case 1 (x dropped): q_y=b, q_z=c, q_w=a -> implies order w,y,z? No.
            #   case 1: q_x=d, q_y=b, q_z=c, q_w=a
            #   So stored a,b,c map to w, y, z
            #   Wait, a corresponds to r0 (MSB of payload), b to r1, c to r2.
            #   So if index=1 (x max), we stored w,y,z in a,b,c order.

            # Let's match indices to C++ cases:
            # Case 0: w is max (rest: x,y,z) -> a=x, b=y, c=z
            # Case 1: x is max (rest: w,y,z) -> a=w, b=y, c=z
            # Case 2: y is max (rest: w,x,z) -> a=w, b=x, c=z
            # Case 3: z is max (rest: w,x,y) -> a=w, b=x, c=y

            # So we just take the remaining components in order.

            packed_r[k] = (
                (mi << 30) | ((q[0] & 0x3FF) << 20) | ((q[1] & 0x3FF) << 10) | (q[2] & 0x3FF)
            )

        packed_rotations[idx] = packed_r

        # Store Chunk Info
        chunks.append(
            ChunkInfo(
                min_x=min_p[0],
                max_x=max_p[0],
                min_y=min_p[1],
                max_y=max_p[1],
                min_z=min_p[2],
                max_z=max_p[2],
                min_scale_x=min_s[0],
                max_scale_x=max_s[0],
                min_scale_y=min_s[1],
                max_scale_y=max_s[1],
                min_scale_z=min_s[2],
                max_scale_z=max_s[2],
                min_r=min_c[0],
                max_r=max_c[0],
                min_g=min_c[1],
                max_g=max_c[1],
                min_b=min_c[2],
                max_b=max_c[2],
                min_motion_x=min_m[0],
                max_motion_x=max_m[0],
                min_motion_y=min_m[1],
                max_motion_y=max_m[1],
                min_motion_z=min_m[2],
                max_motion_z=max_m[2],
                min_time_scale=min_ts,
                max_time_scale=max_ts,
                min_time=min_t,
                max_time=max_t,
            )
        )

    # Construct PLY
    # Define elements

    # 1. Chunk Element
    chunk_dtype = [
        ("min_x", "f4"),
        ("max_x", "f4"),
        ("min_y", "f4"),
        ("max_y", "f4"),
        ("min_z", "f4"),
        ("max_z", "f4"),
        ("min_scale_x", "f4"),
        ("max_scale_x", "f4"),
        ("min_scale_y", "f4"),
        ("max_scale_y", "f4"),
        ("min_scale_z", "f4"),
        ("max_scale_z", "f4"),
        ("min_r", "f4"),
        ("max_r", "f4"),
        ("min_g", "f4"),
        ("max_g", "f4"),
        ("min_b", "f4"),
        ("max_b", "f4"),
        ("min_motion_x", "f4"),
        ("max_motion_x", "f4"),
        ("min_motion_y", "f4"),
        ("max_motion_y", "f4"),
        ("min_motion_z", "f4"),
        ("max_motion_z", "f4"),
        ("min_time_scale", "f4"),
        ("max_time_scale", "f4"),
        ("min_time", "f4"),
        ("max_time", "f4"),
    ]

    chunk_data = np.zeros(num_chunks, dtype=chunk_dtype)
    for i, c in enumerate(chunks):
        chunk_data[i] = (
            c.min_x,
            c.max_x,
            c.min_y,
            c.max_y,
            c.min_z,
            c.max_z,
            c.min_scale_x,
            c.max_scale_x,
            c.min_scale_y,
            c.max_scale_y,
            c.min_scale_z,
            c.max_scale_z,
            c.min_r,
            c.max_r,
            c.min_g,
            c.max_g,
            c.min_b,
            c.max_b,
            c.min_motion_x,
            c.max_motion_x,
            c.min_motion_y,
            c.max_motion_y,
            c.min_motion_z,
            c.max_motion_z,
            c.min_time_scale,
            c.max_time_scale,
            c.min_time,
            c.max_time,
        )

    chunk_el = PlyElement.describe(chunk_data, "chunk")

    # 2. Vertex Element
    vertex_dtype = [
        ("packed_position", "u4"),
        ("packed_rotation", "u4"),
        ("packed_scale", "u4"),
        ("packed_color", "u4"),
        ("packed_motion", "u4"),
        ("packed_time", "u4"),
    ]

    vertex_data = np.zeros(num_splats, dtype=vertex_dtype)
    vertex_data["packed_position"] = packed_positions
    vertex_data["packed_rotation"] = packed_rotations
    vertex_data["packed_scale"] = packed_scales
    vertex_data["packed_color"] = packed_colors
    vertex_data["packed_motion"] = packed_motions
    vertex_data["packed_time"] = packed_times

    vertex_el = PlyElement.describe(vertex_data, "vertex")

    elements = [chunk_el, vertex_el]

    # 3. SH Element (if any)
    if has_sh:
        # C++ loader expects property names "f_rest_0" ... "f_rest_44"
        # Type is uchar (u1)
        sh_dtype = [(f"f_rest_{i}", "u1") for i in range(45)]

        # Reshape flat array to structured array
        # This is tricky with numpy structured arrays from 2D data
        # We can construct it via a list of tuples (slow) or view casting

        # sh_data is (N, 45) uint8
        # We need a structured array of shape (N,)

        sh_struct = np.zeros(num_splats, dtype=sh_dtype)
        for i in range(45):
            sh_struct[f"f_rest_{i}"] = sh_data[:, i]

        sh_el = PlyElement.describe(sh_struct, "sh")
        elements.append(sh_el)

    PlyData(elements, text=False).write(str(path))
    logger.info(f"Saved 4DV file to {path}")
