#include "da_capi_buffer.h"

#include <nvutils/logger.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

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

    const std::vector<std::uint8_t> loaded(frameBytes, 0xff);

    {
        const DepthResult null_result = processFrame(loaded.data(), kWidth, kHeight, nullptr);
        if(null_result.depth != nullptr)
        {
            LOGE("test_da_capi_buffer: processFrame with null ctx should return empty result\n");
            return 1;
        }
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

    const DepthResult wrapped = processFrame(loaded.data(), kWidth, kHeight, ctx);
    if(!wrapped.depth || wrapped.w <= 0 || wrapped.h <= 0)
    {
        LOGE("test_da_capi_buffer: processFrame failed: %s\n", da_capi_last_error(ctx));
        da_capi_free(ctx);
        return 1;
    }

    const std::size_t outCount = static_cast<std::size_t>(wrapped.h) * static_cast<std::size_t>(wrapped.w);
    if(outCount == 0 || !allFinite(wrapped.depth, outCount))
    {
        LOGE("test_da_capi_buffer: invalid output dims or non-finite depth\n");
        da_capi_free_floats(wrapped.depth);
        da_capi_free(ctx);
        return 1;
    }

    LOGI("test_da_capi_buffer: ok (%dx%d, metric=%d, conf=%f)\n", wrapped.w, wrapped.h, wrapped.is_metric, wrapped.conf);

    da_capi_free_floats(wrapped.depth);
    da_capi_free(ctx);
    return 0;
}
