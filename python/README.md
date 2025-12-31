# Python Tools for vk_gaussian_splatting

This directory contains Python tools for video-to-depth processing, supporting both **real-time streaming** and **offline preprocessing** workflows.

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
    │  • FastAPI + WebSocket    │    │  • DA3-streaming chunk processing     │
    │  • Per-frame DA3 inference│    │  • Sim3 alignment between chunks      │
    │  • VDZ packets over WS    │    │  • Global scale consistency           │
    │  • ~100ms latency         │    │  • Camera pose estimation             │
    └─────────────┬─────────────┘    └──────────────────┬────────────────────┘
                  │                                      │
                  ▼                                      ▼
    ┌───────────────────────────┐    ┌───────────────────────────────────────┐
    │  C++ Viewer (real-time)   │    │        Output Files                   │
    │  DepthStreamClient        │    │                                       │
    │  WebSocket → VDZ frames   │    │  • depth_sequence.vdz (per-frame)     │
    └───────────────────────────┘    │  • camera_poses.txt (4x4 c2w)         │
                                     │  • intrinsics.txt (fx,fy,cx,cy)       │
                                     │  • metadata.json                      │
                                     └──────────────────┬────────────────────┘
                                                        │
                                     ┌──────────────────┴────────────────────┐
                                     ▼                                        ▼
                      ┌───────────────────────────┐    ┌─────────────────────────┐
                      │  C++ Viewer (playback)    │    │   PLY Export            │
                      │  Load preprocessed seq    │    │   vkgs-export-ply       │
                      │  Use poses for 3D recon   │    │   Fused point cloud     │
                      └───────────────────────────┘    └─────────────────────────┘
```

## Installation

```bash
cd python

# Create virtual environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install base tools
uv pip install -e .

# For real-time backend
uv pip install -e ".[backend,inference]"

# For offline preprocessing
uv pip install -e ".[offline,inference]"

# For everything
uv pip install -e ".[backend,offline,inference,dev]"

# Install CUDA-enabled PyTorch (recommended for GPU acceleration)
uv pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu130

# Windows-specific: Install triton-windows for xformers optimization
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    uv pip install triton-windows
fi
```

## Workflows

### 1. Real-time Streaming (existing)

For interactive viewing with live depth estimation:

```bash
# Start backend server
vkgs-backend

# Or directly:
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The C++ viewer connects via WebSocket, uploads video, and receives VDZ depth frames in real-time.

**Pros:** Interactive, immediate feedback
**Cons:** No temporal consistency between frames, per-frame scale varies

### 2. Offline Preprocessing (new)

For high-quality depth with global consistency:

```bash
# Preprocess video with chunk-based alignment
vkgs-preprocess --input video.mp4 --output ./preprocessed/

# Options:
#   --chunk-size 60    Frames per chunk (default: 60)
#   --overlap 30       Overlap for alignment (default: 30)
#   --process-res 518  Model resolution (default: 518)
#   --no-poses         Skip camera pose estimation
```

Output structure:
```
preprocessed/
├── depth_sequence.vdz    # All frames in single VDZ container
├── camera_poses.txt      # 4x4 c2w matrices (16 values per line)
├── intrinsics.txt        # fx, fy, cx, cy per line
└── metadata.json         # Video info and config
```

### 3. PLY Export

Convert preprocessed depth to point cloud:

```bash
vkgs-export-ply --input ./preprocessed/ --output scene.ply

# Options:
#   --video video.mp4   Original video for RGB colors
#   --sample-ratio 0.01 Fraction of points per frame
#   --frame-skip 5      Use every Nth frame
#   --no-poses          Don't use camera poses (camera-space output)
```

## Implementation Steps

### Phase 1: Backend Integration ✅
- [x] Copy VideoDepthViewer3D backend to `python/backend/`
- [x] Create unified `pyproject.toml` with optional dependencies
- [x] Define output formats in `offline/formats.py`

### Phase 2: Offline Preprocessing (TODO)
- [ ] Implement chunk-based processing wrapper for DA3
- [ ] Add Sim3 alignment between chunks (ported from DA3-streaming)
- [ ] Test with sample videos
- [ ] Add loop closure support (optional)

### Phase 3: C++ Viewer Updates (TODO)
- [ ] Add "Load Preprocessed Sequence" UI option
- [ ] Create `VdzSequenceLoader` to read multi-frame VDZ files
- [ ] Load camera poses and apply to mesh reconstruction
- [ ] Support seeking within preprocessed sequence

### Phase 4: Unified CLI (TODO)
- [ ] Single entry point: `vkgs-depth` with subcommands
- [ ] Progress reporting and ETA
- [ ] GPU memory management for long videos

## Format Comparison

| Feature | Real-time VDZ | Offline VDZ Sequence | DA3-streaming PLY |
|---------|---------------|----------------------|-------------------|
| Per-frame depth | ✅ | ✅ | ❌ (fused only) |
| Camera poses | ❌ | ✅ | ✅ |
| Scale consistency | ❌ | ✅ (Sim3 aligned) | ✅ |
| Temporal coherence | ❌ | ✅ (chunk overlap) | ✅ |
| File size | Small/frame | Medium (all frames) | Large (point cloud) |
| Use case | Interactive | Playback + 3D | Static scene |

## Development

```bash
# Run tests
pytest

# Format code
ruff format .
ruff check --fix .
```

## Test Videos

A good source for test videos (e.g., Big Buck Bunny) is:
https://download.blender.org/peach/bigbuckbunny_movies/
