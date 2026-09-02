# Python Tools for vk_gaussian_splatting

This directory contains Python tools for video-to-depth processing and Gaussian Splatting export, supporting both **real-time streaming** and **offline preprocessing** workflows.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Input Video                                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
    ┌───────────────────────────┐    ┌───────────────────────────────────────┐
    │   Real-time Streaming     │    │       Offline Preprocessing           │
    │   (backend/)              │    │       (offline/)                      │
    │                           │    │                                       │
    │  • FastAPI + HLS stream   │    │  • DA3/MoGe depth inference           │
    │  • Per-frame inference    │    │  • Gaussian Splatting export          │
    │  • Side-by-side depth     │    │  • H.265 depth video output           │
    └─────────────┬─────────────┘    └──────────────────┬────────────────────┘
                  │                                      │
                  ▼                                      ▼
    ┌───────────────────────────┐    ┌───────────────────────────────────────┐
    │  Web Viewer (HLS)         │    │        Output Files                   │
    │  Standards-compliant      │    │                                       │
    │  RGB + depth side-by-side │    │  • depth_sequence.mp4 (H.265)         │
    └───────────────────────────┘    │  • frame_*.ply (Gaussian Splatting)   │
                                     │  • metadata.json                      │
                                     └──────────────────────────────────────────┘
```

## Installation

The project uses **uv** with extras-based GPU backend selection. PyTorch and related packages are automatically sourced from the correct index based on your chosen backend.

### Quick Start

```bash
cd python
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# NVIDIA GPU (most common)
uv sync --extra cuda --extra offline --extra inference

# Intel Arc/XPU
uv sync --extra xpu --extra offline --extra inference

# CPU only (no GPU)
uv sync --extra cpu --extra offline --extra inference
```

### GPU Backend Extras (mutually exclusive)

| Extra | Hardware | PyTorch Index |
|-------|----------|---------------|
| `cuda` | NVIDIA GPUs | `download.pytorch.org/whl/cu130` |
| `xpu` | Intel Arc, Data Center GPU | `download.pytorch.org/whl/xpu` |
| `cpu` | No GPU | `download.pytorch.org/whl/cpu` |

### Feature Extras (combinable)

| Extra | Description |
|-------|-------------|
| `backend` | FastAPI + HLS streaming server |
| `inference` | Transformers, timm, einops for depth models |
| `offline` | Trimesh, plyfile for PLY generation |
| `matrix3d` | Apple Matrix3D support via pytorch3d |
| `dev` | pytest, ruff, ty for development |

### Example Configurations

```bash
# Full offline toolchain with CUDA + Matrix3D
uv sync --extra cuda --extra offline --extra inference --extra matrix3d

# Streaming backend only
uv sync --extra cuda --extra backend --extra inference

# Development setup
uv sync --extra cuda --extra offline --extra inference --extra dev
```

## Workflows

### 1. Real-time Streaming

For interactive viewing with live depth estimation:

```bash
# Start backend server with CUDA
uv run --extra cuda --extra backend uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Start backend server with Intel XPU (multi-GPU mode)
VIDEO_DEPTH_MULTI_DEVICE=1 VIDEO_DEPTH_DEVICE_SPEC=xpu:0,1 \
  uv run --extra xpu --extra backend uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The backend provides HLS streaming with RGB + depth side-by-side video.

### 2. Offline Depth Video Export

Extract depth from video and output as H.265 video:

```bash
# Using DA3 model (default)
uv run --extra cuda python -m offline.cli depth \
  --input video.mp4 --output ./depth_output/ \
  --model "depth-anything/DA3METRIC-LARGE" --device-spec cuda

# Using MoGe-3 model (produces depth + normals side-by-side)
uv run --extra cuda python -m offline.cli depth \
  --input video.mp4 --output ./depth_output/ \
  --model "Ruicheng/moge-3-vitl" --device-spec cuda
```

Output:
- `depth_sequence.mp4` - H.265 lossless depth video (with normals if MoGe)
- `metadata.json` - Processing info including z_min/z_max

### 3. Gaussian Splatting Export

Convert images or video to Gaussian Splatting PLY files:

```bash
# Export images to per-frame PLYs
uv run --extra cuda python -m offline.cli images \
  --input ./images/ --output ./ply_output/ \
  --model "Ruicheng/moge-3-vitl" --mode frames

# Export video to per-frame PLYs
uv run --extra cuda python -m offline.cli export \
  --input video.mp4 --output ./ply_output/ \
  --model "depth-anything/DA3-GIANT" --mode frames

# Postprocess to FreeTimeGS with motion vectors (standard)
uv run --extra cuda python -m offline.cli postprocess \
  --input ./ply_output/ --output scene_4d.ply \
  --fps 30.0

# Export directly to FreeTimeGS with delta compression (Int16, ~51x compression)
uv run --extra cuda python -m offline.cli images \
  --input ./images/ --output scene_4d_compressed.ply \
  --model "depth-anything/DA3-GIANT" --mode freetimegs-delta

# Export directly to FreeTimeGS with high compression (Int8)
uv run --extra cuda python -m offline.cli images \
  --input ./images/ --output scene_4d_high_compression.ply \
  --model "depth-anything/DA3-GIANT" --mode freetimegs-delta-int8
```

### 4. 4DAnyone + ZipSplat prototype

After running 4DAnyone, use its generated synchronized views to create one
static ZipSplat scene per timestamp:

```bash
uv sync --extra cuda --extra zipsplat
uv run --extra cuda --extra zipsplat python -m offline.cli zipsplat \
  --input /path/to/data/fdanyone/<clip> \
  --output ./zipsplat_frames \
  --frame-skip 1

uv run --extra cuda python -m offline.cli postprocess \
  --input ./zipsplat_frames \
  --output ./scene_4d.4dv \
  --fps 25 \
  --format 4dv
```

The input must contain `videos/dense/*.mp4`. This is a prototype: ZipSplat
does not estimate temporal motion, so the final motion is derived by the
existing FreeTimeGS matcher. The released ZipSplat weights are CC BY-NC 4.0.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_DEPTH_MULTI_DEVICE` | Enable multi-device mode | `0` |
| `VIDEO_DEPTH_DEVICE_SPEC` | Device specification (e.g., `xpu:0,1`) | `auto` |
| `VIDEO_DEPTH_MODEL_ID` | Model to use | `depth-anything/DA3METRIC-LARGE` |
| `VIDEO_DEPTH_PROCESS_RES` | Processing resolution | `640` |
| `VIDEO_DEPTH_LOG_LEVEL` | Logging level | `WARNING` |

## Development

```bash
# Run tests
pytest

# Format code
ruff format .
ruff check --fix .

# Type check
uvx ty check
```

## Test Videos

A good source for test videos (e.g., Big Buck Bunny) is:
https://download.blender.org/peach/bigbuckbunny_movies/
