import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d

try:
    from flex_gemm.ops.spconv.multi_xpu import MultiXPUSpconv

    MULTI_XPU_AVAILABLE = True
except ImportError:
    MULTI_XPU_AVAILABLE = False


def sparse_conv3d_init(
    self,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    padding=None,
    bias=True,
    indice_key=None,
):
    assert stride == 1 and (padding is None), (
        "Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)"
    )

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = (
        tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    )
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    flex_gemm.ops.spconv.set_algorithm(config.FLEX_GEMM_ALGO)
    flex_gemm.ops.spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)

    # check if neighbor map is already computed
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = f"SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}"
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    # Check if we should use Multi-XPU
    want_multi_xpu = (
        getattr(config, "FLEX_GEMM_USE_MULTI_XPU", False)
        and MULTI_XPU_AVAILABLE
    )

    # First pass: always use single-XPU to build neighbor cache
    if neighbor_cache is None:
        out, neighbor_cache_ = sparse_submanifold_conv3d(
            x.feats,
            x.coords,
            torch.Size([*x.shape, *x.spatial_shape]),
            self.weight,
            self.bias,
            neighbor_cache,
            self.dilation,
        )
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)
        
        # Pre-build MultiXPU cache for subsequent passes
        if want_multi_xpu:
            if not hasattr(self, "multi_xpu_conv"):
                algo = config.FLEX_GEMM_ALGO
                if algo == "auto":
                    algo = "igemm_mma"  # Default for MultiXPUSpconv
                num_devices = (
                    torch.xpu.device_count()
                    if hasattr(torch, "xpu") and torch.xpu.is_available()
                    else 1
                )
                self.multi_xpu_conv = MultiXPUSpconv(num_devices=num_devices, algorithm=algo)
            # Pre-build the split cache for future calls
            self.multi_xpu_conv.build_cache(neighbor_cache_)
    elif want_multi_xpu:
        # Subsequent passes: use Multi-XPU
        if not hasattr(self, "multi_xpu_conv"):
            algo = config.FLEX_GEMM_ALGO
            if algo == "auto":
                algo = "igemm_mma"
            num_devices = (
                torch.xpu.device_count()
                if hasattr(torch, "xpu") and torch.xpu.is_available()
                else 1
            )
            self.multi_xpu_conv = MultiXPUSpconv(num_devices=num_devices, algorithm=algo)
        out = self.multi_xpu_conv(x.feats, neighbor_cache, self.weight, self.bias)
    else:
        # Single-XPU path with existing cache
        out, _ = sparse_submanifold_conv3d(
            x.feats,
            x.coords,
            torch.Size([*x.shape, *x.spatial_shape]),
            self.weight,
            self.bias,
            neighbor_cache,
            self.dilation,
        )

    out = x.replace(out)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("SparseInverseConv3d with flex_gemm is not implemented yet")


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError("SparseInverseConv3d with flex_gemm is not implemented yet")
