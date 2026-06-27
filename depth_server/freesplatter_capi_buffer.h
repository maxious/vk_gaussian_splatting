#pragma once

#include <cstdint>
#include <string>
#include <vector>

// C API forward declaration (opaque handle). The full C API is exposed in
// "free_splatter.h"; we keep it opaque here so consumers of this wrapper do
// not need to pull the upstream headers in.
struct free_splatter_ctx;

// FreeSplatterBuffer: a thin C++ wrapper around the free-splatter flat C API
// (3rdparty/free-splatter/include/free_splatter.h). It owns a model context,
// preprocesses JPEG/PNG image bytes into the NCHW float32 tensor the engine
// expects, runs inference, and converts the activated per-pixel Gaussian
// tensor into the antimatter15 .splat byte format (32 bytes/gaussian).
class FreeSplatterBuffer {
public:
    struct InitOptions {
        std::string model_path;
        std::string device = "cpu";     // "cpu" | "vulkan" | "cuda" | "gpu"
        int         n_threads = 4;
    };

    struct RunResult {
        std::vector<std::uint8_t> splat_bytes;  // antimatter15 .splat format
        int                       num_gaussians = 0;
        std::string               error_msg;
    };

    FreeSplatterBuffer() = default;
    ~FreeSplatterBuffer();

    FreeSplatterBuffer(const FreeSplatterBuffer&)            = delete;
    FreeSplatterBuffer& operator=(const FreeSplatterBuffer&) = delete;

    // Initialize model: calls free_splatter_load + free_splatter_geometry_of.
    // Returns false on failure (see error_msg via run() / last_error()).
    bool initialize(const InitOptions& opts);

    // Run inference on N images. Each entry in `images` is the raw bytes of a
    // JPEG/PNG file. Images are decoded, center-cropped to a square, resized
    // to the model's expected size, scaled to [0,1] and laid out NCHW. The
    // activated [n_views, H, W, gaussian_channels] f32 tensor is converted to
    // the antimatter15 .splat byte format and returned in `RunResult`.
    RunResult run(const std::vector<std::vector<std::uint8_t>>& images);

    // Release the model context. Safe to call multiple times; safe on an
    // uninitialized instance.
    void shutdown();

private:
    free_splatter_ctx* m_ctx = nullptr;
    int m_expected_w       = 512;
    int m_expected_h       = 512;
    int m_in_channels      = 3;
    int m_gaussian_channels = 23;

    // Preprocess: decode JPEG/PNG bytes -> center-crop to square -> resize to
    // target_size x target_size -> float32 NCHW in [0,1]. Appends to `out`.
    // Mirrors load_image_chw() in free-splatter/tools/free_splatter-cli.cpp.
    static bool loadImageCHW(const std::uint8_t* jpeg_bytes, std::size_t len,
                             int target_size, std::vector<float>& out);

    // Convert the engine's activated gaussians [n*gc] f32 tensor to the
    // antimatter15 .splat byte format (32 bytes/gaussian). Prunes gaussians
    // with opacity <= opacity_threshold, sorts by importance (opacity*volume).
    // Mirrors write_splat() in free-splatter/tools/free_splatter-cli.cpp.
    static std::vector<std::uint8_t> f32ToSplat(const float* g, std::size_t n,
                                                int gc,
                                                float opacity_threshold = 5e-3f);
};