# Agent Guidelines for vk_gaussian_splatting

## Logging

**Do NOT use `std::cout` or `std::cerr` for logging.** Use the nvutils logger macros instead:

- `LOGI("message %s\n", arg)` - Info messages
- `LOGW("message %s\n", arg)` - Warnings  
- `LOGE("message %s\n", arg)` - Errors
- `LOGD("message %s\n", arg)` - Debug (verbose)

Include `<nvutils/logger.hpp>` to use these macros.

The logger writes to both console and a log file at `_bin/Debug/log_vk_viewer.txt` (or `_bin/Release/log_vk_viewer.txt`). This is useful for debugging as it captures all output even if the app crashes.

### Debug Logging

To enable verbose debug logging, set the environment variable:
```bash
export VK_VIEWER_DEBUG=1
```

This enables `LOGD()` debug messages which are suppressed by default.

## Code Style

- Follow existing code conventions (see `.clang-format`)
- Use `STATE_` prefix for enum values to avoid Windows macro conflicts (e.g., `VideoRenderState::STATE_RENDERING`)
- Prefer `std::filesystem::path` for file paths

## Build & Test

### Vulkan SDK Setup

Ensure the Vulkan SDK is installed and the environment variables are set.
If you installed the SDK to `/opt/vulkan` (e.g., `/opt/vulkan/1.4.335.0/`), source the setup script before building:

```bash
source /opt/vulkan/1.4.335.0/setup-env.sh
```

### Building

Build with CMake (Visual Studio or command line):
```bash
# Configure
cmake -S . -B build

# Build (Release or Debug)
cmake --build build --config Release
cmake --build build --config Debug
```

**Important Build Notes:**
- **DO NOT run `rm -rf build`** - CMake caches configuration and re-running is faster. Only delete build if you need a completely clean slate.
- **Slang comes from Vulkan SDK** at `/opt/vulkan/1.4.335.0/x86_64/` - no need to specify `Slang_ROOT`

### Running Tests

The project uses doctest for unit testing. The test suite is header-only (`tests/doctest.h`).

To build and run tests:
```bash
# Build tests
cmake --build build --target unit_tests --config Debug

# Run tests (Windows)
.\_bin\Release\Debug\unit_tests.exe

# Or use ctest
cmake --build build --target RUN_TESTS --config Debug
```

Test files are located in the `tests/` directory. Add new test files to `tests/CMakeLists.txt` as needed.

### Functional Screenshot Testing

To verify the rendering pipeline works correctly, run the viewer with a test scene and auto-screenshot:

```bash
source /opt/vulkan/1.4.335.0/setup-env.sh
cd _bin/Debug
./vk_viewer --inputFile ../../_downloaded_resources/flowers_1/flowers_1.ply \
  --screenshotDelay 3.0 --screenshot /tmp/test_render.png \
  --size 800 600 --validation 0
```

This loads the default flower scene, waits 3 seconds for rendering to stabilize, takes a screenshot, and exits. Verify the output:
```bash
file /tmp/test_render.png  # Should show: PNG image data, 800 x 600
```

**Note**: Manual testing is still required for UI features and Vulkan rendering.

## Key Subsystems

- **Rendering Core**: `vk_viewer.cpp` (main class), `vk_viewer_render.cpp` (orchestrator)
- **Frame Helpers**: `vk_viewer_frame.cpp` (beginFrame, buildViews, renderSingleView, etc.)
- **Multiview/XR**: `vk_viewer_multiview.cpp` (VK_KHR_multiview stereo rendering)
- **RTX**: `vk_viewer_rtx.cpp` (ray tracing pipeline)
- **UI**: `vk_viewer_ui.cpp` (ImGui-based)
- **Video Export**: `video_renderer.cpp`, `camera_trajectory.cpp`
- **Scene Loading**: `splat_loader_async.cpp`, `sog_loader.cpp`, `splat_set.cpp`
- **Depth Video**: `vk_viewer_video.cpp`, `depth_video_loader.cpp`

## Python Tools

For the `python/` subdirectory, we use the following tools:

- **Type Checker**: [ty](https://github.com/astral-sh/ty) (fast Python type checker by Astral)
- **Linter/Formatter**: `ruff`
- **Package Manager**: `uv`

### Python Setup

The project uses **uv** with extras-based GPU backend selection. PyTorch is automatically sourced from the correct index.

```bash
cd python
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate

# NVIDIA GPU (most common)
uv sync --extra cuda --extra offline --extra inference

# Intel Arc/XPU
uv sync --extra xpu --extra offline --extra inference

# CPU only
uv sync --extra cpu --extra offline --extra inference

# With Matrix3D/pytorch3d support (CUDA only)
uv sync --extra cuda --extra offline --extra inference --extra matrix3d
```

**GPU Backend Extras** (mutually exclusive - pick ONE):
- `cuda` - NVIDIA GPUs (cu130 index)
- `xpu` - Intel Arc/Data Center GPU
- `cpu` - No GPU acceleration

**Feature Extras** (combinable):
- `offline` - PLY generation tools
- `inference` - Depth model dependencies
- `backend` - FastAPI streaming server
- `matrix3d` - Apple Matrix3D via pytorch3d
- `dev` - Testing and linting tools

### Offline Processing Tools

The `python/offline/` directory contains tools for generating Gaussian Splats from video or images.

**Key Features:**
- **DA3-GIANT**: Default model for metric depth estimation and splat generation.
- **Apple SHARP**: Integrated support for Apple's SHARP model (vendored in `python/sharp`).
- **FreeTimeGS**: Generates 4D Gaussian Splats with motion vectors (requires `cupy` or `faiss-gpu`).
- **Pruning**: Opacity-based pruning to reduce file size.

**Usage:**

Run these commands from the `python/` directory to ensure local modules are found:

```bash
cd python

# 1. Export video frames to PLY using DA3 (Metric)
uv run python -m offline.export_gaussian_ply export \
  --input video.mp4 --output output_folder/ \
  --mode frames --model "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# 2. Export images to PLY using SHARP (auto-downloads or uses local cache)
uv run python -m offline.export_gaussian_ply images \
  --input "path/to/images" --output output_folder/ \
  --mode frames --model sharp --opacity-threshold 0.05

# 3. Export 4D FreeTimeGS (Single PLY with motion)
# Uses CuPy for GPU acceleration if FAISS is not available
uv run python -m offline.export_gaussian_ply images \
  --input "path/to/images" --output scene.ply \
  --mode freetimegs --model sharp
```

**Note on SHARP**:
The `sharp` library is vendored in `python/sharp` to provide better control and suppress noisy logging. If running from the project root, set `PYTHONPATH=python` or run via `uv run python -m ...` from inside the `python/` directory.

### Running Commands with uv run

For any Python command that requires GPU support, use `uv run --extras` to ensure the correct GPU backend is active:

```bash
# Run with CUDA backend (NVIDIA GPUs)
uv run --extra cuda python script.py

# Run with XPU backend (Intel Arc/GPU)
uv run --extra xpu python script.py

# Run with CPU only (no GPU)
uv run --extra cpu python script.py

# Combine with feature extras (e.g., backend for streaming server)
uv run --extra cuda --extra backend uvicorn backend.main:app --port 8000
```

### Backend Environment Variables

When running the streaming backend, these environment variables control behavior:

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_DEPTH_MULTI_DEVICE` | Enable multi-device mode | `0` |
| `VIDEO_DEPTH_DEVICE_SPEC` | Device specification (e.g., `xpu:0,1`) | `auto` |
| `VIDEO_DEPTH_MODEL_ID` | Model to use | `depth-anything/DA3METRIC-LARGE` |
| `VIDEO_DEPTH_PROCESS_RES` | Processing resolution | `640` |
| `VIDEO_DEPTH_LOG_LEVEL` | Logging level | `WARNING` |

Example with multi-XPU:
```bash
VIDEO_DEPTH_MULTI_DEVICE=1 VIDEO_DEPTH_DEVICE_SPEC=xpu:0,1 VIDEO_DEPTH_LOG_LEVEL=INFO \
  uv run --extra xpu --extra backend uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Running Type Checks

Use `ty` to run type checks:

```bash
uvx ty check
```

### Running the Backend Server

The backend server provides HTTP endpoints and WebSocket streaming for video depth processing. Use a PTY session for interactive debugging:

```bash
# Start the backend server in a PTY session
pty_spawn --command "bash" --title "Backend Server"

# Inside the PTY, run:
cd /home/maxious/vk_gaussian_splatting/python
source .venv/bin/activate
uv run --extra backend uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Useful curl commands for testing:**

```bash
# Create a session from a video file
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/sessions

# Start HLS stream generation (returns status URL)
curl -X POST "http://localhost:8000/api/sessions/{session_id}/hls" \
  -H "Content-Type: application/json" \
  -d '{"fps": 30, "segment_duration": 2.0, "process_res": 640}'

# Poll HLS generation status
curl http://localhost:8000/api/sessions/{session_id}/hls/status
```

**Environment Variables for Backend:**

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_DEPTH_MULTI_DEVICE` | Enable multi-device mode (for CPU/XPU fallback) | `0` |
| `VIDEO_DEPTH_DEVICE_SPEC` | Device specification (e.g., `xpu:0,1`, `cpu`) | `auto` |
| `VIDEO_DEPTH_MODEL_ID` | Model to use | `depth-anything/DA3METRIC-LARGE` |
| `VIDEO_DEPTH_PROCESS_RES` | Processing resolution | `640` |
| `VIDEO_DEPTH_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `WARNING` |

Example with CPU fallback:
```bash
VIDEO_DEPTH_MULTI_DEVICE=1 VIDEO_DEPTH_LOG_LEVEL=DEBUG \
  uv run --extra backend uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Third-party Libraries

- nvpro_core2 - NVIDIA Vulkan utilities (submodule)
- ImGui - UI framework
- GLM - Math library
- FFmpeg - Video encoding (external, must be in PATH)

### FFmpeg Setup

Download the shared build from https://github.com/GyanD/codexffmpeg/releases/tag/8.0.1 and extract to `ffmpeg-8.0.1-full_build-shared/` in the project root directory.

The CMake configuration will automatically find FFmpeg in this location. The directory is ignored by git, so each developer needs to download and extract it manually.

Example:
```bash
# Download ffmpeg-8.0.1-full_build-shared.zip from the releases page
# Extract to the project root directory
# Resulting structure:
# vk_gaussian_splatting/
# ├── ffmpeg-8.0.1-full_build-shared/
# │   ├── bin/
# │   ├── include/
# │   ├── lib/
# │   └── ...
# └── ...
```
