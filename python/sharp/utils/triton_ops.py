import torch
import triton
import triton.language as tl


@triton.jit
def compose_covariance_kernel(
    q_ptr,  # pointer to quaternions (N, 4)
    s_ptr,  # pointer to scales (N, 3)
    cov_ptr,  # pointer to output covariance matrices (N, 3, 3) or (N, 9)
    stride_qn,
    stride_q4,
    stride_sn,
    stride_s3,
    stride_cn,
    stride_c3,
    stride_c3_2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load quaternion (w, x, y, z) - assuming real part is index 0
    # sharp.utils.linalg.rotation_matrices_from_quaternions uses q[..., 0] as real
    w = tl.load(q_ptr + offsets * stride_qn + 0 * stride_q4, mask=mask)
    x = tl.load(q_ptr + offsets * stride_qn + 1 * stride_q4, mask=mask)
    y = tl.load(q_ptr + offsets * stride_qn + 2 * stride_q4, mask=mask)
    z = tl.load(q_ptr + offsets * stride_qn + 3 * stride_q4, mask=mask)

    # Normalize quaternion
    # norm = sqrt(w*w + x*x + y*y + z*z)
    norm = tl.sqrt(w * w + x * x + y * y + z * z + 1e-6)
    w = w / norm
    x = x / norm
    y = y / norm
    z = z / norm

    # Compute Rotation Matrix elements
    # R = [ 1-2(y^2+z^2)   2(xy-zw)      2(xz+yw)
    #       2(xy+zw)       1-2(x^2+z^2)  2(yz-xw)
    #       2(xz-yw)       2(yz+xw)      1-2(x^2+y^2) ]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    # Load scales (s0, s1, s2)
    s0 = tl.load(s_ptr + offsets * stride_sn + 0 * stride_s3, mask=mask)
    s1 = tl.load(s_ptr + offsets * stride_sn + 1 * stride_s3, mask=mask)
    s2 = tl.load(s_ptr + offsets * stride_sn + 2 * stride_s3, mask=mask)

    # Square scales to get diagonal S^2 matrix elements
    S0 = s0 * s0
    S1 = s1 * s1
    S2 = s2 * s2

    # Compute Cov = R * S^2 * R^T
    # This means M = R * diag(S) * R^T
    # M_ij = sum_k (R_ik * S_k * R_jk)

    # Row 0 of Cov
    c00 = r00 * S0 * r00 + r01 * S1 * r01 + r02 * S2 * r02
    c01 = r00 * S0 * r10 + r01 * S1 * r11 + r02 * S2 * r12
    c02 = r00 * S0 * r20 + r01 * S1 * r21 + r02 * S2 * r22

    # Row 1 of Cov
    c10 = c01  # Symmetric
    c11 = r10 * S0 * r10 + r11 * S1 * r11 + r12 * S2 * r12
    c12 = r10 * S0 * r20 + r11 * S1 * r21 + r12 * S2 * r22

    # Row 2 of Cov
    c20 = c02  # Symmetric
    c21 = c12  # Symmetric
    c22 = r20 * S0 * r20 + r21 * S1 * r21 + r22 * S2 * r22

    # Store result (N, 3, 3)
    # Row 0
    tl.store(cov_ptr + offsets * stride_cn + 0 * stride_c3 + 0 * stride_c3_2, c00, mask=mask)
    tl.store(cov_ptr + offsets * stride_cn + 0 * stride_c3 + 1 * stride_c3_2, c01, mask=mask)
    tl.store(cov_ptr + offsets * stride_cn + 0 * stride_c3 + 2 * stride_c3_2, c02, mask=mask)

    # Row 1
    tl.store(cov_ptr + offsets * stride_cn + 1 * stride_c3 + 0 * stride_c3_2, c10, mask=mask)
    tl.store(cov_ptr + offsets * stride_cn + 1 * stride_c3 + 1 * stride_c3_2, c11, mask=mask)
    tl.store(cov_ptr + offsets * stride_cn + 1 * stride_c3 + 2 * stride_c3_2, c12, mask=mask)

    # Row 2
    tl.store(cov_ptr + offsets * stride_cn + 2 * stride_c3 + 0 * stride_c3_2, c20, mask=mask)
    tl.store(cov_ptr + offsets * stride_cn + 2 * stride_c3 + 1 * stride_c3_2, c21, mask=mask)
    tl.store(cov_ptr + offsets * stride_cn + 2 * stride_c3 + 2 * stride_c3_2, c22, mask=mask)


def compose_covariance_matrices_triton(
    quaternions: torch.Tensor, singular_values: torch.Tensor
) -> torch.Tensor:
    """
    Fused kernel to compute covariance matrices from quaternions and scales.

    Args:
        quaternions: (N, 4)
        singular_values: (N, 3)

    Returns:
        (N, 3, 3) covariance matrices
    """
    assert quaternions.is_cuda and singular_values.is_cuda
    N = quaternions.shape[0]
    assert singular_values.shape[0] == N

    output = torch.empty((N, 3, 3), device=quaternions.device, dtype=quaternions.dtype)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    compose_covariance_kernel[grid](
        quaternions,
        singular_values,
        output,
        quaternions.stride(0),
        quaternions.stride(1),
        singular_values.stride(0),
        singular_values.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        N,
        BLOCK_SIZE=1024,
    )

    return output
