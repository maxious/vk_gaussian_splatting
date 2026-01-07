# TRELLIS.2 XPU + CUDA Pipeline

This guide shows how to use TRELLIS.2 with Intel XPU GPUs for the main pipeline and NVIDIA CUDA for the final o-voxel export.

## Architecture

The two-stage pipeline separates the compute-intensive TRELLIS.2 inference from the CUDA-only o-voxel export:

```
Stage 1 (XPU venv): TRELLIS.2 Inference
  Input: Images
  Device: Intel XPU (24GB each)
  Output: .pt files (mesh data)

Stage 2 (CUDA venv): o-voxel Export
  Input: .pt files (from Stage 1)
  Device: NVIDIA CUDA (16GB)
  Output: GLB or VXZ files
```

## Prerequisites

### Hardware

- 2x Intel XPU GPUs (24GB each) - e.g., Intel Arc A770 or Data Center GPU Max
- 1x NVIDIA CUDA GPU (16GB) - for o-voxel C++ extensions
- Enough disk space for intermediate .pt files (~100-500MB per image)

### Software

- Python 3.10+
- PyTorch with XPU support (for Stage 1)
- PyTorch with CUDA support (for Stage 2)
- Vulkan SDK (for C++ viewer)
- FFmpeg (for video encoding)

## Setup

### Step 1: XPU Virtual Environment

Create a virtual environment with XPU-enabled PyTorch:

```bash
cd /home/maxious/vk_gaussian_splatting/python

# Create XPU venv
python -m venv .venv_xpu
source .venv_xpu/bin/activate

# Install XPU PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# Install Intel Extension for PyTorch (optional optimizations)
pip install intel_extension_for_pytorch

# Install project dependencies
pip install -e .
pip install -e ".[offline,inference]"

# Verify XPU availability
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}'); print(f'XPU devices: {torch.xpu.device_count()}')"
```

### Step 2: CUDA Virtual Environment

Create a virtual environment with CUDA-enabled PyTorch:

```bash
cd /home/maxious/vk_gaussian_splatting/python

# Create CUDA venv
python -m venv .venv_cuda
source .venv_cuda/bin/activate

# Install CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -e .
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
# Activate XPU venv
cd /home/maxious/vk_gaussian_splatting/python
source .venv_xpu/bin/activate

# Process single image
python -m offline.cli trellis2-xpu \
  --input /path/to/image.jpg \
  --output /path/to/meshes/ \
  --xpu-device xpu:0 \
  --pipeline-type 1024_cascade

# Process directory of images
python -m offline.cli trellis2-xpu \
  --input /path/to/images/ \
  --output /path/to/meshes/ \
  --xpu-device xpu:0 \
  --pattern "*.jpg" \
  --pipeline-type 1024_cascade
```

**Output:** `.pt` files containing mesh data (vertices, faces, coords, attrs, etc.)

**Options:**
- `--xpu-device`: XPU device to use (`xpu:0` or `xpu:1`)
- `--model`: TRELLIS.2 model ID (default: `microsoft/TRELLIS.2-4B`)
- `--pipeline-type`: `512`, `1024`, `1024_cascade`, `1536_cascade`
- `--max-num-tokens`: Maximum tokens for cascade (default: 49152)
- `--num-samples`: Samples per image (default: 1)
- `--seed`: Random seed (default: 42)

### Stage 2: o-voxel Export (CUDA)

Export the saved mesh data to GLB or VXZ format using CUDA:

```bash
# Activate CUDA venv
cd /home/maxious/vk_gaussian_splatting/python
source .venv_cuda/bin/activate

# Export to GLB
python -m offline.cli trellis2-cuda \
  --input /path/to/meshes/ \
  --output /path/to/output/ \
  --format glb \
  --cuda-device cuda:0 \
  --decimation-target 1000000 \
  --texture-size 4096

# Export to VXZ
python -m offline.cli trellis2-cuda \
  --input /path/to/meshes/ \
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

## Parallel Processing (Single XPU)

For multiple images, the pipeline processes them sequentially on the selected XPU device:

```bash
python -m offline.cli trellis2-xpu \
  --input /path/to/images/ \
  --output /path/to/meshes/ \
  --xpu-device xpu:0 \
  --pattern "*.jpg"
```

## Parallel Processing (Two XPUs)

To use both XPUs in parallel, run two separate processes:

```bash
# Terminal 1: Process even-numbered images on xpu:0
source .venv_xpu/bin/activate
python -m offline.cli trellis2-xpu \
  --input /path/to/images/ \
  --output /path/to/meshes_xpu0/ \
  --xpu-device xpu:0 \
  --pattern "*[02468].jpg"

# Terminal 2: Process odd-numbered images on xpu:1
source .venv_xpu/bin/activate
python -m offline.cli trellis2-xpu \
  --input /path/to/images/ \
  --output /path/to/meshes_xpu1/ \
  --xpu-device xpu:1 \
  --pattern "*[13579].jpg"

# Merge results
mkdir -p /path/to/meshes/
cp /path/to/meshes_xpu0/*.pt /path/to/meshes/
cp /path/to/meshes_xpu1/*.pt /path/to/meshes/

# Stage 2: Export with CUDA
source .venv_cuda/bin/activate
python -m offline.cli trellis2-cuda \
  --input /path/to/meshes/ \
  --output /path/to/output/ \
  --format glb
```

## Performance Estimates

| Stage | Device | Time (per image) | Notes |
|--------|---------|------------------|--------|
| TRELLIS.2 inference | XPU 24GB | 30-60s | Model: 4B params |
| Save to .pt | CPU | <1s | ~100-500MB file |
| Load .pt | CPU | <1s | |
| Transfer to CUDA | CPU→GPU | 10-20ms | PCIe transfer |
| o-voxel export | CUDA 16GB | 5-10s | GLB generation |
| **Total (per image)** | | **35-70s** | |
| **2 XPUs parallel** | | **~18-35s per image** | Throughput doubled |

## File Sizes

| File type | Approx. size | Notes |
|----------|--------------|--------|
| .pt (mesh data) | 100-500MB | Uncompressed tensors |
| .glb (final) | 20-100MB | Compressed mesh + texture |
| .vxz (final) | 10-50MB | LZMA compressed |

## Troubleshooting

### XPU Issues

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
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
