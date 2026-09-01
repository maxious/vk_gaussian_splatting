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
If you installed the SDK to `/opt/vulkan` (e.g., `/opt/vulkan/1.4.357.0/`), source the setup script before building:

```bash
source /opt/vulkan/1.4.357.0/setup-env.sh
```

### Building

Build with CMake (Visual Studio or command line):
```bash
# Configure (include TBB_DIR for Intel oneAPI TBB)
cmake -S . -B build -DTBB_DIR=/opt/intel/oneapi/tbb/2023.0/lib/cmake/tbb

# Build (Release or Debug)
cmake --build build --config Release
cmake --build build --config Debug
```

**Important Build Notes:**
- **DO NOT run `rm -rf build`** - CMake caches configuration and re-running is faster. Only delete build if you need a completely clean slate.
- **Slang is a separate dependency** downloaded by NvproCore2 (pinned to `2026.16.1` in `CMakeLists.txt`), independent of the Vulkan SDK. No need to specify `Slang_ROOT`.

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
source /opt/vulkan/1.4.357.0/setup-env.sh
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

### Stochastic GS Screenshot Test

To verify the compute stochastic GS pipeline (--pipeline 6) renders correctly:

```bash
# Run with default Release build
./tests/run_stochastic_screenshot_test.sh

# Or specify build config
./tests/run_stochastic_screenshot_test.sh Debug
```

The script:
1. Builds the viewer if needed
2. Runs `vk_viewer --pipeline 6 --stochasticSamplesPerPixel 16 --stochasticUseGps 0` with the default flower scene
3. Verifies the PNG output exists, is 800x600, and contains non-black pixels

## Key Subsystems

- **Rendering Core**: `vk_viewer.cpp` (main class), `vk_viewer_render.cpp` (orchestrator)
- **Frame Helpers**: `vk_viewer_frame.cpp` (beginFrame, buildViews, renderSingleView, etc.)
- **Multiview/XR**: `vk_viewer_multiview.cpp` (VK_KHR_multiview stereo rendering)
- **RTX**: `vk_viewer_rtx.cpp` (ray tracing pipeline)
- **UI**: `vk_viewer_ui.cpp` (ImGui-based)
- **Video Export**: `video_renderer.cpp`, `camera_trajectory.cpp`
- **Scene Loading**: `splat_loader_async.cpp`, `sog_loader.cpp`, `sogxt_loader.cpp`, `splat_set.cpp`
- **Depth Video**: `vk_viewer_video.cpp`, `depth_video_loader.cpp`
- **Vulkan Video Decoder** (PoC): `vulkan_video_decoder.cpp` - GPU-accelerated H.265 decoding
- **TRON PBR Pipeline**: `shaders/pbr_shading.h.slang` - Ray-traced PBR relighting (--pipeline 2 + --pbrEnabled 1)
- **Compute Stochastic GS**: `vk_viewer_stochasticgs.cpp` - Sorting-free stochastic rasterization (--pipeline 6)

### SOG-XT Container Format (KISS-GS)

SOG-XT is the container format from the KISS-GS paper (arXiv:2608.26948,
https://fraunhoferhhi.github.io/KISS-GS/). It extends SOG by storing a scene
as a directory of ordinary lossless WebP planes plus a `meta.json` (v3)
manifest. Key differences from SOG:

- Positions stored as a **coarse/high byte** plane (`means_bytes_1.webp`) and a
  **detail/low byte** plane (`means_bytes_0.webp`) after a signed-log remap.
- View-dependent color uses a **2D-sorted codebook**: `f_rest_centroids.webp`
  (a 3x5 tiled grid of 176x176 RGB tiles = 45 SH coefficients per centroid)
  indexed by a UV label plane `f_rest_labels.webp`.
- Per-attribute `mins`/`maxs` ranges carried in `meta.json` with
  `normalize: "observed-minmax"` (the reference decoder rescales each plane by
  its own observed byte range).
- The full garden sample (Mip-NeRF 360, 256k splats) is shipped in the repo at
  `tests/fixtures/sog_xt_garden/` and `_downloaded_resources/sog_xt_garden/`.

**CLI / Loading:**
```bash
# Load a SOG-XT container directory (or its meta.json / scene.json)
./vk_viewer --inputFile path/to/container_dir \
  --screenshotDelay 3.0 --screenshot /tmp/out.png --size 800 600 --validation 0
```

**Python Encoder** (`python/sogs/sogxt_encoder.py`):
```bash
cd python
uv run python -m sogs.sogxt_encoder \
  --input scene.ply --output container_dir --sh-codebook-side 176
```
Options: `--sh-codebook-side` (codebook grid side, clamped to [16,256]),
`--grid-side` (attribute image grid side), `--iterations` (k-means iters).
Reusable helpers: `morton_order_sort`, `run_sogxt_compression`, `read_ply`.
Container output round-trips through the reference `decode_sogxt.py` and the
C++ `SogXtLoader`.

**Python Tests:**
```bash
cd python
uv run --extra dev pytest tests/test_sogxt_encoder.py -v
```

**C++ Decoder** (`src/sogxt_loader.h/cpp`): `SogXtLoader::load()` accepts a
container directory, `meta.json`, or `scene.json`; parses the v3 manifest,
decodes the WebP planes (parallel), reconstructs positions (inverse signed
log), opacities (logit), scales (log), quats, base color and 45 SH coefficients
via the UV codebook lookup, applies the active mask, and converts RDF->RUB.

**C++ Tests:**
```bash
cmake --build build --target unit_tests --config Release
_bin/Release/unit_tests.exe --test-suite=SogXtLoader
```

Requires: WebP (already linked for SOG support).

### Compute Stochastic GS Mode

The Compute Stochastic GS mode (--pipeline 6) implements sorting-free stochastic
rasterization using 64-bit atomicMin on a per-pixel framebuffer SSBO. It supports
two sub-modes:
- Stochastic Transparency (ST): per-pixel AABB with stochastic acceptance
- Gaussian Point Splatting (GPS): Poisson-distributed sample points

Controls:
- `--stochasticSamplesPerPixel`: samples per pixel (1=interactive, 64=converged)
- `--stochasticUseGps`: 0=ST mode, 1=GPS mode
- `--stochasticMaxSamples`: max accumulation frames before auto-reset

Requires: NVIDIA GPU with VK_KHR_shader_atomic_int64 support

### TRON PBR Pipeline (--pipeline 2, --pbrEnabled 1)

The TRON PBR pipeline (based on TRON paper arXiv:2606.11314) adds ray-traced PBR relighting to the 3DGRT pipeline. Per-particle material attributes (basecolor, roughness, metallic) are composited into a deferred G-buffer using front-to-back Over compositing, then shaded with Cook-Torrance split-sum PBR against HDR environment maps, with optional MIS shadow rays and AgX tone mapping.

**Pipeline:** Reuses `--pipeline 2` (3DGRT) with `--pbrEnabled 1` to activate PBR mode. Not a separate pipeline number.

**CLI Arguments:**
- `--envmap <path>`: Path to HDR environment map file (.hdr, .exr)
- `--envmapRotation <float>`: Environment map rotation in radians (default: 0.0)
- `--envmapExposure <float>`: Exposure multiplier (default: 1.0)
- `--pbrEnabled <0|1>`: Enable/disable PBR rendering (default: 0)
- `--irradianceEnabled <0|1>`: Enable/disable MIS shadow rays (default: 0)
- `--toneMapEnabled <0|1>`: Enable/disable AgX tone mapping (default: 0)

**PLY Property Format:**
Extended with optional float properties:
- `basecolor_0`, `basecolor_1`, `basecolor_2`: RGB base color
- `roughness`: Surface roughness [0,1]
- `metallic`: Metallicity [0,1]
- Missing properties default to: basecolor = SH DC coefficients, roughness = 0.5, metallic = 0.0

**Default Envmap:**
Automatically downloaded from Poly Haven (CC0) to `_downloaded_resources/studio_small_07_4k.hdr` during CMake configure. Skip with `-DDISABLE_HDR_ENVMAP_DOWNLOAD=ON`.

**Build:**
No special CMake flags needed. PBR is always compiled in. The default envmap is downloaded automatically.

**Running:**
```bash
source /opt/vulkan/1.4.357.0/setup-env.sh
cd _bin/Debug
./vk_viewer --inputFile ../../_downloaded_resources/flowers_1/flowers_1.ply \
  --pipeline 2 --pbrEnabled 1 --irradianceEnabled 0 \
  --envmap ../../_downloaded_resources/studio_small_07_4k.hdr \
  --screenshotDelay 3.0 --screenshot /tmp/test_pbr.png \
  --size 800 600 --validation 0
```

**CPU Unit Tests:**
```bash
# Build and run BRDF math unit tests
cmake --build build --config Debug --target unit_tests
ctest --output-on-failure -R test_pbr_helpers
```

**Key Files:**
- `shaders/pbr_shading.h.slang` - Cook-Torrance PBR shading with split-sum IBL
- `shaders/threedgrt_raytrace.rgen.slang` - G-buffer compositing, PBR evaluation, MIS shadow rays
- `shaders/threedgrt.h.slang` - GBufferData struct, particleProcessHitGbuffer with front-to-back Over
- `shaders/shaderio.h` - BINDING_PBR_* defines (45-50), FrameInfo PBR fields
- `src/splat_set.h` - Material vectors in SplatSet
- `src/splat_loader_fast.cpp` - PLY parser for basecolor/roughness/metallic
- `src/splat_set_vk.cpp` - GPU buffer upload for material data
- `src/vk_viewer.cpp` - HdrIbl/HdrEnvDome integration, IBL descriptor binding
- `src/parameters.h` - PbrParameters struct
- `src/parameters.cpp` - CLI arg registration
- `tests/test_pbr_helpers.cpp` - CPU unit tests for GGX, MIS, AgX math

**Limitations:**
- PBR mode only works with `--pipeline 2` (3DGRT)
- No DLSS-RR integration
- Mesh shading remains Blinn-Phong (unaffected)
- No normal maps, emissive, subsurface, or anisotropy
- Single envmap, no explicit light sources
- PBR at primary surface only (no secondary bounces)
- Default envmap: `studio_small_07_4k` (CC0, Poly Haven)

Requires: NVIDIA GPU with VK_KHR_ray_tracing_pipeline support

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
- **MoGe**: Metric depth + normals model with dual XPU support (see below).
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

### Dual XPU Processing for MoGe

The MoGe model supports **dual Intel XPU device processing** for significant speedups.

**Performance Results** (640x480 resolution, 100 frames):
- Single XPU: 168.5 ms/frame (5.94 fps)
- Dual XPU: **91.6 ms/frame (10.92 fps)**
- **Speedup: 1.84x** with **92% parallel efficiency**

**Important:** Do NOT source `/opt/intel/oneapi/setvars.sh` - PyTorch XPU bundles its own Intel runtime. Sourcing oneAPI causes library version conflicts.

**Usage:**

```bash
cd python

# Benchmark single vs dual XPU
uv run --extra xpu python benchmark_moge_multi_xpu.py \
  --video video.mp4 \
  --max-frames 100 --target-width 640 --target-height 480 \
  --device-spec xpu:0,1 --batch-size 8

# Backend server with dual XPU
export VIDEO_DEPTH_MULTI_DEVICE=1
export VIDEO_DEPTH_DEVICE_SPEC=xpu:0,1
export VIDEO_DEPTH_MODEL_ID=Ruicheng/moge-2-vitl-normal
uv run --extra xpu --extra backend uvicorn backend.main:app --port 8000

# Offline processing with dual XPU
uv run --extra xpu python -m offline.cli images \
  --input ./frames/ --output ./ply_output/ \
  --model "Ruicheng/moge-2-vitl-normal" --mode frames \
  --use-multi-device --device-spec xpu:0,1
```

**Performance Tips:**
- Use batch_size ≥ 8 for better efficiency
- Process ≥ 50 frames (multi-device overhead is ~5-6 seconds for model loading)
- Lower resolutions (640x480) achieve better parallel scaling than higher resolutions
- No need to source Intel oneAPI environment

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

## Free-Splatter (Image-to-Gaussians)

`depth_server` can run in `--mode splat` to accept 2-4 photos and generate 3D Gaussian Splat scenes using `localai-org/free-splatter.cpp`. The generated `.splat` files load directly in vk_viewer.

### Building with Free-Splatter

```bash
# Configure with free-splatter enabled
cmake -S . -B build -DENABLE_FREESPLATTER=ON

# Vulkan backend is auto-detected; force CPU if Vulkan loader missing
cmake -S . -B build -DENABLE_FREESPLATTER=ON -DFREE_SPLATTER_VULKAN=OFF
```

### Starting the Splat Server

Run the splat server as a separate process (on a different port than the depth server):

```bash
source /opt/vulkan/1.4.357.0/setup-env.sh
./_bin/Debug/depth_server --mode splat --port 9001 \\
  --splat-model LocalAI-io/free-splatter.cpp:freesplatter-scene-f16.gguf \\
  --splat-workers 1 --splat-backend cpu
```

First run downloads the model (~625MB) to `~/.cache/depth_server/`.

### Using from vk_viewer

File > Generate Splats from Images... > pick 2-4 photos > wait for progress > scene auto-loads.

### Functional Test

```bash
# Build the test client
cmake --build build --target test_splat_client --config Debug

# Start server, then run test
./_bin/Debug/depth_server --mode splat --port 9001 \\
  --splat-model ~/.cache/depth_server/freesplatter-scene-f16.gguf &
./_bin/Debug/test_splat_client --port 9001 \\
  --images tests/fixtures/free_splatter/box_00.png,tests/fixtures/free_splatter/box_01.png \\
  --output /tmp/test.splat
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_FREESPLATTER` | CMake option (ON/OFF) | OFF |
| `FREE_SPLATTER_VULKAN` | Enable Vulkan backend for ggml | auto-detect |

### Key Files

- `depth_server/freesplatter_capi_buffer.{h,cpp}` — C++ wrapper around free-splatter C API
- `depth_server/splat_worker.{h,cpp}` — per-process worker (ggml context + model)
- `depth_server/splat_worker_pool.{h,cpp}` — parent-side worker pool
- `depth_server/protocol.h` — MSG_SPLAT_* message types (0x10-0x15)
- `src/free_splatter_client.{h,cpp}` — raw POSIX TCP client in vk_viewer
- `src/local_splat_server_manager.{h,cpp}` — lifecycle manager for spawning the splat server
- `tests/test_splat_client.cpp` — CLI client for functional verification
- `tests/fixtures/free_splatter/` — synthetic 2-image test fixture

### 3rdparty

- `3rdparty/free-splatter/` — git submodule at `https://github.com/localai-org/free-splatter.cpp.git`
- `ggml` (nested submodule in `3rdparty/free-splatter/ggml/`) — tensor/ML backend
- License: Apache-2.0 (both free-splatter.cpp and its model weights)

### Cloud Streaming Mode (--mode cloud)

Turns a sequence of JPEG frames into a coherent 3D point cloud (.splat) using depth-anything.cpp's `da_capi_points_stream` sliding-window Sim3 stitching. Optional de-ghosting: TSDF voxel fusion, per-seam ICP refinement, loop-closure pose-graph.

**Requires**: a pose-capable DA3 model (e.g. `depth-anything-giant-f32.gguf`). Auto-downloaded on first use via `model_downloader.cpp`.

**Build Flag**: `ENABLE_CLOUD=ON` for vk_viewer UI integration (default OFF). No extra flag needed for depth_server.

**CLI Arguments** (`depth_server --mode cloud`):
| Flag | Default | Description |
|------|---------|-------------|
| `--cloud-model` | (required) | DA3 GGUF path |
| `--cloud-backend` | `cpu` | Inference backend (`cpu`, `cuda`) |
| `--cloud-workers` | `1` | Number of worker processes |
| `--cloud-max-frames` | `64` | Max frames per job (2..200) |
| `--cloud-chunk-size` | `12` | Frames per sliding window (2..24) |
| `--cloud-overlap` | `3` | Window overlap |
| `--cloud-fuse` | off | TSDF voxel fusion |
| `--cloud-metric` | off | Absolute-metre rescale |
| `--cloud-icp` | off | Per-seam ICP refinement |
| `--cloud-loop-close` | off | Loop-closure pose-graph |
| `--cloud-conf-pct` | `55` | Confidence percentile |
| `--cloud-point-size` | `1.2` | Point radius multiplier |
| `--cloud-fuse-voxel-frac` | `0.004` | Voxel fraction of bbox diagonal |
| `--cloud-fuse-trunc-mult` | `4` | Truncation multiple of voxel |

**Frame Cap**: 2 minimum, 64 default, 200 hard maximum.

**Toggles**: All de-ghosting options are OFF by default. Enable individually: `--cloud-fuse`, `--cloud-icp`, `--cloud-loop-close`, `--cloud-metric`.

**Output**: `.splat` file in `~/.cache/depth_server/clouds/job_{id}.splat`. Loadable in vk_viewer.

**Test Command**:
```bash
source /opt/vulkan/1.4.357.0/setup-env.sh
cd _bin/Debug
# Start server on port 9102
./depth_server --mode cloud --cloud-model ~/.cache/depth_server/depth-anything-giant-f32.gguf --port 9102 &
# Run test client with 8 test frames
./test_cloud_client --port 9102 --frames /path/to/frames/frame_*.jpg --output /tmp/test.splat
# Verify
file /tmp/test.splat
```

**Limitations**:
- Hard cap of 200 frames per request
- `--mode cloud` and `--mode splat` are mutually exclusive (cannot run both in same process)
- No flythrough camera poses in this version (da_capi_stream_last_poses available but not wired yet)
- No glTF/OBJ/VDB/VOX export (point cloud only; viewer renders Minecraft cube mode)

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
