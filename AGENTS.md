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

## Third-party Libraries

- nvpro_core2 - NVIDIA Vulkan utilities (submodule)
- ImGui - UI framework
- GLM - Math library
- FFmpeg - Video encoding (external, must be in PATH)
