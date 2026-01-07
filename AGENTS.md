# Agent Guidelines for vk_gaussian_splatting

## Logging

**Do NOT use `std::cout` or `std::cerr` for logging.** Use the nvutils logger macros instead:

- `LOGI("message %s\n", arg)` - Info messages
- `LOGW("message %s\n", arg)` - Warnings  
- `LOGE("message %s\n", arg)` - Errors
- `LOGD("message %s\n", arg)` - Debug (verbose)

Include `<nvutils/logger.hpp>` to use these macros.

The logger writes to both console and log file, making debugging easier.

## Code Style

- Follow existing code conventions (see `.clang-format`)
- Use `STATE_` prefix for enum values to avoid Windows macro conflicts (e.g., `VideoRenderState::STATE_RENDERING`)
- Prefer `std::filesystem::path` for file paths

## Build & Test

### Vulkan SDK Setup

Ensure the Vulkan SDK is installed and the environment variables are set.
If you installed the SDK to `~/vulkan` (e.g., `~/vulkan/1.4.335.0/`), source the setup script before building:

```bash
source ~/vulkan/1.4.335.0/setup-env.sh
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

**Note**: Manual testing is still required for UI features and Vulkan rendering.

## Key Subsystems

- **Rendering**: `gaussian_splatting.cpp`, `gaussian_splatting_render.cpp`
- **UI**: `gaussian_splatting_ui.cpp` (ImGui-based)
- **Video Export**: `video_renderer.cpp`, `camera_trajectory.cpp`
- **Scene Loading**: `splat_loader_async.cpp`, `sog_loader.cpp`, `splat_set.cpp`

## Python Tools

For the `python/` subdirectory, we use the following tools:

- **Type Checker**: [ty](https://github.com/astral-sh/ty) (fast Python type checker by Astral)
- **Linter/Formatter**: `ruff`
- **Package Manager**: `uv`

### Python Setup

```bash
cd python
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate
uv pip install -e ".[dev,offline,inference,cuda]"

# Install CUDA-enabled PyTorch (recommended for GPU acceleration)
uv pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu124  # or cu118/cu121

# Windows-specific: Install triton-windows for xformers optimization
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    uv pip install triton-windows
fi
```

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

### Running Type Checks

Use `ty` to run type checks:

```bash
uvx ty check
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
