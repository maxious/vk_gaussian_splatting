#include "freesplatter_capi_buffer.h"

#include "free_splatter.h"

#include "stb_image.h"
#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

FreeSplatterBuffer::~FreeSplatterBuffer() { shutdown(); }

bool FreeSplatterBuffer::initialize(const InitOptions& opts) {
    shutdown();

    free_splatter_options* fopts = free_splatter_options_new();
    if (!fopts) return false;
    free_splatter_options_set_device(fopts, opts.device.c_str());
    free_splatter_options_set_threads(fopts, opts.n_threads);

    m_ctx = free_splatter_load(opts.model_path.c_str(), fopts);
    free_splatter_options_free(fopts);

    if (!m_ctx) return false;
    if (const char* err = free_splatter_last_error(m_ctx)) {
        if (err && err[0]) { shutdown(); return false; }
    }

    free_splatter_geometry geo{};
    if (free_splatter_geometry_of(m_ctx, &geo) != 0) {
        shutdown();
        return false;
    }
    m_in_channels      = geo.in_channels;
    m_expected_h       = geo.image_height;
    m_expected_w       = geo.image_width;
    m_gaussian_channels = geo.gaussian_channels;
    return true;
}

void FreeSplatterBuffer::shutdown() {
    if (m_ctx) {
        free_splatter_free(m_ctx);
        m_ctx = nullptr;
    }
}

FreeSplatterBuffer::RunResult
FreeSplatterBuffer::run(const std::vector<std::vector<std::uint8_t>>& images) {
    RunResult res;
    if (!m_ctx) { res.error_msg = "model not initialized"; return res; }
    if (images.empty()) { res.error_msg = "no input images"; return res; }

    const int size = m_expected_w;
    std::vector<float> buf;
    buf.reserve(static_cast<std::size_t>(images.size()) * m_in_channels * size * size);
    for (const auto& img : images) {
        if (!loadImageCHW(img.data(), img.size(), size, buf)) {
            res.error_msg = "image decode/preprocess failed";
            return res;
        }
    }
    const int32_t n_views = static_cast<int32_t>(images.size());

    float*  out   = nullptr;
    std::size_t n_out = 0;
    if (free_splatter_run(m_ctx, buf.data(), n_views, m_expected_h, m_expected_w,
                          &out, &n_out) != 0) {
        const char* err = free_splatter_last_error(m_ctx);
        res.error_msg = err ? err : "free_splatter_run failed";
        return res;
    }

    const std::size_t n_gauss = (m_gaussian_channels > 0)
        ? n_out / static_cast<std::size_t>(m_gaussian_channels) : 0;
    res.splat_bytes   = f32ToSplat(out, n_gauss, m_gaussian_channels);
    res.num_gaussians = static_cast<int>(res.splat_bytes.size() / 32);
    free_splatter_buf_free(out);
    return res;
}

bool FreeSplatterBuffer::loadImageCHW(const std::uint8_t* jpeg_bytes,
                                      std::size_t len, int target_size,
                                      std::vector<float>& out) {
    int w = 0, h = 0, n = 0;
    unsigned char* px = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(jpeg_bytes),
        static_cast<int>(len), &w, &h, &n, 3);
    if (!px) return false;

    const int s    = std::min(w, h);
    const int left = (w - s) / 2;
    const int top  = (h - s) / 2;
    std::vector<unsigned char> sq(static_cast<std::size_t>(s) * s * 3);
    for (int y = 0; y < s; y++)
        std::memcpy(&sq[static_cast<std::size_t>(y) * s * 3],
                    &px[(static_cast<std::size_t>(top + y) * w + left) * 3],
                    static_cast<std::size_t>(s) * 3);
    stbi_image_free(px);

    std::vector<unsigned char> rz(static_cast<std::size_t>(target_size) * target_size * 3);
    stbir_resize_uint8_linear(sq.data(), s, s, 0,
                              rz.data(), target_size, target_size, 0, STBIR_RGB);

    const std::size_t base = out.size();
    out.resize(base + static_cast<std::size_t>(3) * target_size * target_size);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < target_size * target_size; i++)
            out[base + static_cast<std::size_t>(c) * target_size * target_size + i]
                = rz[static_cast<std::size_t>(i) * 3 + c] / 255.0f;
    return true;
}

std::vector<std::uint8_t>
FreeSplatterBuffer::f32ToSplat(const float* g, std::size_t n, int gc,
                               float opacity_threshold) {
    // Mirrors write_splat() in free-splatter/tools/free_splatter-cli.cpp:57-94.
    const double C0 = 0.28209479177387814;
    std::vector<std::pair<float, std::size_t>> keep;
    keep.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        const float op = g[i * gc + 15];
        if (op <= opacity_threshold) continue;
        const float vol = std::max(g[i * gc + 16], 1e-9f)
                        * std::max(g[i * gc + 17], 1e-9f)
                        * std::max(g[i * gc + 18], 1e-9f);
        keep.push_back({op * vol, i});
    }
    std::sort(keep.begin(), keep.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    auto u8 = [](float v) -> unsigned char {
        float t = v < 0 ? 0 : v > 255 ? 255 : v;
        return static_cast<unsigned char>(t);
    };

    std::vector<std::uint8_t> out;
    out.reserve(keep.size() * 32);
    for (const auto& k : keep) {
        const float* x = &g[k.second * gc];
        // FreeSplatter's reference frame is OpenCV (y down, z forward); convert
        // to the viewer's OpenGL convention (y up) via a 180deg rotation about X
        // = diag(1,-1,-1): position.yz negate, quaternion (w,x,y,z)->(-x,w,-z,y).
        float pos[3]   = { x[0], -x[1], -x[2] };
        float scale[3] = { x[16], x[17], x[18] };
        unsigned char rgba[4], rot[4];
        for (int c = 0; c < 3; c++) {
            float v = 0.5f + static_cast<float>(C0) * x[3 + c];
            rgba[c] = u8((v < 0 ? 0 : v > 1 ? 1 : v) * 255.0f);
        }
        rgba[3] = u8(std::min(std::max(x[15], 0.0f), 1.0f) * 255.0f);
        float q[4] = { -x[20], x[19], -x[22], x[21] };
        float nrm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) + 1e-12f;
        for (int c = 0; c < 4; c++) rot[c] = u8(q[c] / nrm * 128.0f + 128.0f);

        const auto* p  = reinterpret_cast<const std::uint8_t*>(pos);
        const auto* sc = reinterpret_cast<const std::uint8_t*>(scale);
        out.insert(out.end(), p,  p  + 12);
        out.insert(out.end(), sc, sc + 12);
        out.insert(out.end(), rgba, rgba + 4);
        out.insert(out.end(), rot,  rot  + 4);
    }
    return out;
}