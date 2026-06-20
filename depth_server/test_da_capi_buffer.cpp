#include "da_capi_buffer.h"

#include <nvutils/logger.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool writeRawFrame(const std::string& path, const std::vector<std::uint8_t>& rgb)
{
    std::ofstream out(path, std::ios::binary);
    if(!out)
    {
        return false;
    }
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return out.good();
}

bool readRawFrame(const std::string& path, std::vector<std::uint8_t>& rgb)
{
    std::ifstream in(path, std::ios::binary);
    if(!in)
    {
        return false;
    }
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return in.good() || in.eof();
}

bool allFinite(const float* data, std::size_t count)
{
    for(std::size_t i = 0; i < count; ++i)
    {
        if(!std::isfinite(data[i]))
        {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    constexpr int kWidth  = 640;
    constexpr int kHeight = 480;
    const std::size_t frameBytes = static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight) * 3u;

    std::vector<std::uint8_t> frame(frameBytes, 0);
    const std::string rawPath = argc > 2 ? argv[2] : "/tmp/test_da_capi_buffer.rgb";
    if(!writeRawFrame(rawPath, frame))
    {
        LOGE("test_da_capi_buffer: failed to write %s\n", rawPath.c_str());
        return 1;
    }

    std::vector<std::uint8_t> loaded(frameBytes, 0xff);
    if(!readRawFrame(rawPath, loaded))
    {
        LOGE("test_da_capi_buffer: failed to read %s\n", rawPath.c_str());
        return 1;
    }

    if(da_capi_depth_from_rgb(nullptr, loaded.data(), kWidth, kHeight, 3,
                              nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr) == 0)
    {
        LOGE("test_da_capi_buffer: null ctx should fail\n");
        return 1;
    }

    if(argc < 2)
    {
        LOGI("test_da_capi_buffer: null-ctx check passed; provide <model.gguf> to run inference\n");
        return 0;
    }

    da_ctx* ctx = da_capi_load(argv[1], 4);
    if(!ctx)
    {
        LOGE("test_da_capi_buffer: failed to load model %s\n", argv[1]);
        return 1;
    }

    int out_h = 0;
    int out_w = 0;
    int is_metric = 0;
    float* depth = nullptr;
    float* conf = nullptr;
    float* sky = nullptr;
    float ext[12] = {};
    float intr[9] = {};
    const int ret = da_capi_depth_from_rgb(ctx, loaded.data(), kWidth, kHeight, 3,
                                           &out_h, &out_w,
                                           &depth, &conf, &sky,
                                           ext, intr, &is_metric);
    if(ret != 0 || !depth)
    {
        LOGE("test_da_capi_buffer: inference failed (%d): %s\n", ret, da_capi_last_error(ctx));
        da_capi_free_floats(conf);
        da_capi_free_floats(sky);
        da_capi_free(ctx);
        return 1;
    }

    const std::size_t outCount = static_cast<std::size_t>(out_h) * static_cast<std::size_t>(out_w);
    if(out_h <= 0 || out_w <= 0 || outCount == 0 || !allFinite(depth, outCount))
    {
        LOGE("test_da_capi_buffer: invalid output dims or non-finite depth\n");
        da_capi_free_floats(depth);
        da_capi_free_floats(conf);
        da_capi_free_floats(sky);
        da_capi_free(ctx);
        return 1;
    }

    const DepthResult wrapped = processFrame(loaded.data(), kWidth, kHeight, ctx);
    if(!wrapped.depth || wrapped.w <= 0 || wrapped.h <= 0)
    {
        LOGE("test_da_capi_buffer: processFrame failed\n");
        da_capi_free_floats(depth);
        da_capi_free_floats(conf);
        da_capi_free_floats(sky);
        da_capi_free(ctx);
        return 1;
    }

    LOGI("test_da_capi_buffer: ok (%dx%d, metric=%d, conf=%f)\n", wrapped.w, wrapped.h, is_metric, wrapped.conf);

    da_capi_free_floats(wrapped.depth);
    da_capi_free_floats(depth);
    da_capi_free_floats(conf);
    da_capi_free_floats(sky);
    da_capi_free(ctx);
    return 0;
}
