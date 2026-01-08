# TRELLIS.2 XPU + CUDA Pipeline

This guide shows how to use TRELLIS.2 with Intel XPU GPUs for the main pipeline and NVIDIA CUDA for the final o-voxel export.

## Architecture

The two-stage pipeline separates the compute-intensive TRELLIS.2 inference from the CUDA-only o-voxel export:

```
Stage 1 (XPU venv): TRELLIS.2 Inference
  Input: Images
  Device: Intel XPU (24GB each)
  Output: .pt files (latent data)

Stage 2 (CUDA venv): o-voxel Export  
  Input: .pt files (from Stage 1)
  Device: NVIDIA CUDA (16GB)
  Output: GLB or VXZ files
```

## Prerequisites

### Hardware

- 1-2x Intel XPU GPUs (24GB each) - e.g., Intel Arc Pro B60
- 1x NVIDIA CUDA GPU (16GB) - for o-voxel C++ extensions
- Enough disk space for intermediate .pt files (~100-500MB per image)

### Software

- Python 3.12+
- PyTorch with XPU support (for Stage 1)
- PyTorch with CUDA support (for Stage 2)
- Intel oneAPI (for FlexGEMM compilation only - NOT at runtime)
- Vulkan SDK (for C++ viewer)
- FFmpeg (for video encoding)

## Setup

### Step 1: XPU Virtual Environment

Create a virtual environment with XPU-enabled PyTorch:

```bash
cd /home/maxious/vk_gaussian_splatting/python

# Create XPU venv
python3 -m venv .venv-xpu
source .venv-xpu/bin/activate

# Install XPU PyTorch (2.7+ recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# Verify XPU availability
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}'); print(f'XPU devices: {torch.xpu.device_count()}')"

# Install project dependencies
pip install -e ".[offline,inference]"
```

### Step 2: Build FlexGEMM for XPU

FlexGEMM provides sparse convolution kernels for Intel XPU. It must be compiled with oneAPI but **run WITHOUT sourcing setvars.sh**.

```bash
cd ~/FlexGEMM

# CRITICAL: Use the venv python directly, don't source setvars.sh for building
# The build script handles oneAPI environment internally
./build_xpu.sh

# Install the wheel
pip install dist/flex_gemm-*.whl
```

**Key FlexGEMM modifications made:**
- `pyproject.toml`: Removed hard torch dependency, added optional `[cuda]/[xpu]` extras
- `flex_gemm/kernels/__init__.py`: Guard triton import for XPU (pytorch-triton-xpu lacks Config class)
- `build_xpu.sh`: Uses venv python directly, doesn't source setvars.sh, sets LD_LIBRARY_PATH correctly

### Step 3: Install Aule-Attention (Optional but Recommended)

Aule-Attention provides Vulkan-based FlashAttention that works on Intel XPU:

```bash
# Install from prebuilt wheel
pip install ~/Aule-Attention/python/dist/aule_attention-*.whl

# Or install from source
cd ~/Aule-Attention/python
pip install -e .
```

### Step 4: Set Up Environment Variables

**CRITICAL**: Do NOT source `/opt/intel/oneapi/setvars.sh` when running! This conflicts with PyTorch XPU's bundled SYCL runtime.

```bash
# Required for gated models (DINOv3, RMBG-2.0)
export HF_TOKEN=your_huggingface_token

# Set attention backend (choose one)
export ATTN_BACKEND=aule    # Recommended: Aule-Attention (Vulkan)
export ATTN_BACKEND=sdpa    # Fallback: PyTorch SDPA

# Optional: Enable Multi-XPU (if you have 2+ XPU devices)
# Note: Multi-XPU only activates after first conv pass builds neighbor cache
```

### Step 5: CUDA Virtual Environment (for Stage 2)

Create a virtual environment with CUDA-enabled PyTorch:

```bash
cd /home/maxious/vk_gaussian_splatting/python

# Create CUDA venv
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate

# Install CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -e ".[offline,inference]"

# Install o-voxel (CUDA C++ extensions)
git clone --recursive https://github.com/microsoft/TRELLIS.2.git /tmp/TRELLIS.2
cd /tmp/TRELLIS.2/o-voxel
pip install -e .

# Verify CUDA and o-voxel availability
python -c "import torch; import o_voxel; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Stage 1: TRELLIS.2 Inference (XPU)

Run TRELLIS.2 inference on your images using XPU GPUs:

```bash
# Activate XPU venv (use python directly, NOT after sourcing setvars.sh)
cd /home/maxious/vk_gaussian_splatting/python

# CORRECT: Use venv python directly
export HF_TOKEN=your_token
export ATTN_BACKEND=aule
.venv-xpu/bin/python3 -m offline.cli trellis2-xpu \
  --input /path/to/image.jpg \
  --output /path/to/latents/ \
  --xpu-device xpu:0 \
  --pipeline-type 1024_cascade \
  -v

# Process directory of images
.venv-xpu/bin/python3 -m offline.cli trellis2-xpu \
  --input /path/to/images/ \
  --output /path/to/latents/ \
  --xpu-device xpu:0 \
  --pattern "*.jpg" \
  --pipeline-type 1024_cascade

# Enable Multi-XPU (both GPUs)
.venv-xpu/bin/python3 -m offline.cli trellis2-xpu \
  --input /path/to/images/ \
  --output /path/to/latents/ \
  --multi-xpu \
  --pipeline-type 1024_cascade
```

**Output:** `.pt` files containing SparseTensor latents (shape_slat, tex_slat, res)

**Options:**
- `--xpu-device`: XPU device to use (`xpu:0` or `xpu:1`)
- `--multi-xpu`: Enable Multi-XPU inference (uses all available XPUs)
- `--model`: TRELLIS.2 model ID (default: `microsoft/TRELLIS.2-4B`)
- `--pipeline-type`: `512`, `1024`, `1024_cascade`, `1536_cascade`
- `--max-num-tokens`: Maximum tokens for cascade (default: 49152)
- `--num-samples`: Samples per image (default: 1)
- `--seed`: Random seed (default: 42)

### Stage 2: o-voxel Export (CUDA)

Export the saved latent data to GLB or VXZ format using CUDA:

```bash
# Activate CUDA venv
cd /home/maxious/vk_gaussian_splatting/python
source .venv-cuda/bin/activate

# Export to GLB
python -m offline.cli trellis2-cuda \
  --input /path/to/latents/ \
  --output /path/to/output/ \
  --format glb \
  --cuda-device cuda:0 \
  --decimation-target 1000000 \
  --texture-size 4096

# Export to VXZ
python -m offline.cli trellis2-cuda \
  --input /path/to/latents/ \
  --output /path/to/output/ \
  --format vxz \
  --cuda-device cuda:0 \
  --compression lzma \
  --compression-level 9
```

**Output:** `.glb` or `.vxz` files ready for use with vk_gaussian_splatting viewer

**Options:**
- `--format`: `glb` (mesh with texture) or `vxz` (o-voxel format)
- `--cuda-device`: CUDA device to use (default: `cuda:0`)
- `--decimation-target`: Target face count for mesh simplification (GLB only, default: 1000000)
- `--texture-size`: Texture resolution for GLB export (default: 4096)
- `--compression`: Compression algorithm for VXZ (`lzma`, `zlib`, `none`)
- `--compression-level`: Compression level 0-9 for VXZ (default: 9)

## Multi-XPU Details

The `--multi-xpu` flag enables data-parallel sparse convolution across multiple Intel XPU devices.

### How It Works

1. **First convolution pass**: Uses single-XPU to build the neighbor map (spatial cache)
2. **Subsequent passes**: Uses Multi-XPU with cached neighbor map for parallel execution
3. **Cache pre-building**: After first pass, `MultiXPUSpconv.build_cache()` prepares the split configuration

### Implementation Notes

The Multi-XPU implementation in FlexGEMM (`flex_gemm/ops/spconv/multi_xpu.py`):
- Splits voxels (N dimension) evenly across devices
- Replicates weights on all devices (small relative to features)
- Handles cross-device neighbor lookups by replicating boundary voxels
- Uses concurrent stream execution with driver 25.48.36300.8+

## Attention Backend Options

TRELLIS.2 uses sparse attention which requires a compatible backend:

| Backend | XPU Support | Performance | Notes |
|---------|-------------|-------------|-------|
| `aule` | ✅ Yes | Good | Vulkan-based, recommended for XPU |
| `sdpa` | ✅ Yes | Moderate | PyTorch built-in, per-sequence processing |
| `flash_attn` | ❌ No | Best | CUDA-only, not available on XPU |
| `xformers` | ❌ No | Good | CUDA-only, not available on XPU |

Set via environment variable:
```bash
export ATTN_BACKEND=aule
```

## Troubleshooting

### XPU Issues

**"Cannot open shared object file" errors:**
```bash
# WRONG: Don't source setvars.sh at runtime
source /opt/intel/oneapi/setvars.sh  # DON'T DO THIS

# CORRECT: Just use the venv python directly
.venv-xpu/bin/python3 -m offline.cli trellis2-xpu ...
```

**XPU not available:**
```bash
# Check XPU installation
python -c "import torch; print(torch.xpu.is_available())"

# Reinstall XPU PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

**Out of memory on XPU:**
- Use lower pipeline type: `--pipeline-type 512`
- Reduce max tokens: `--max-num-tokens 25000`
- Process fewer images at once

### FlexGEMM Build Issues

**Triton import error:**
FlexGEMM's `flex_gemm/kernels/__init__.py` guards the triton import since `pytorch-triton-xpu` lacks the `Config` class. If you see triton errors, verify the guard is in place.

**SYCL library conflicts:**
The runtime SYCL from oneAPI conflicts with PyTorch XPU's bundled version. Always use the venv python directly without sourcing setvars.sh.

### CUDA Issues

**o-voxel import error:**
```bash
# Reinstall o-voxel
pip uninstall o-voxel
cd /tmp/TRELLIS.2/o-voxel
pip install -e .
```

**CUDA out of memory:**
- Reduce decimation target: `--decimation-target 500000`
- Reduce texture size: `--texture-size 2048`
- Export images one at a time

## Performance

### Tested Configuration
- 2x Intel Arc Pro B60 (24GB each)
- Intel Core Ultra 9 285K
- Ubuntu with Intel XPU drivers

### Timing (per image, 1024_cascade pipeline)
| Stage | Time | Notes |
|-------|------|-------|
| Sparse structure sampling | ~10s | |
| Shape cascade (512→1024) | ~40s | |
| Texture sampling | ~30s | |
| Save latents | <1s | |
| **Total XPU** | **~1.5 min** | |

## File Formats

| File type | Approx. size | Notes |
|----------|--------------|--------|
| .pt (latent data) | 100-500MB | SparseTensor with coords, feats |
| .glb (final) | 20-100MB | Compressed mesh + texture |
| .vxz (final) | 10-50MB | LZMA compressed o-voxel |

## Viewing Results

Load the exported GLB/VXZ files in the vk_gaussian_splatting viewer:

```bash
cd /home/maxious/vk_gaussian_splatting
./_bin/Release/vk_gaussian_splatting /path/to/output/model.glb
```

## References

- [TRELLIS.2 Paper](https://arxiv.org/abs/2411.12135)
- [TRELLIS.2 GitHub](https://github.com/microsoft/TRELLIS.2)
- [o-voxel Format](https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel)
- [FlexGEMM](https://github.com/IntelLabs/FlexGEMM) - Intel sparse GEMM kernels
- [Aule-Attention](https://github.com/AuleTechnologies/Aule-Attention) - Vulkan FlashAttention
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
