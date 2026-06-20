#include "da_capi_buffer.h"

#include <nvutils/logger.hpp>

#include <cstring>

DepthResult processFrame(const std::uint8_t* rgb, std::uint32_t w, std::uint32_t h, da_ctx* ctx)
{
    DepthResult result;
    if(!ctx || !rgb || w == 0 || h == 0)
    {
        LOGE("processFrame: invalid context or frame buffer\n");
        return result;
    }

    int out_w = 0;
    int out_h = 0;
    int is_metric = 0;
    float* depth = nullptr;
    float* conf = nullptr;
    float* sky = nullptr;
    float ext[12] = {};
    float intr[9] = {};

    const int ret = da_capi_depth_from_rgb(ctx, rgb, static_cast<int>(w), static_cast<int>(h), 3,
                                           &out_h, &out_w,
                                           &depth, &conf, &sky,
                                           ext, intr, &is_metric);
    if(ret != 0 || !depth)
    {
        LOGE("processFrame: inference failed (%d): %s\n", ret, da_capi_last_error(ctx));
        da_capi_free_floats(conf);
        da_capi_free_floats(sky);
        return result;
    }

    result.depth = depth;
    result.w = out_w;
    result.h = out_h;
    result.conf = conf ? *conf : 0.0f;
    result.is_metric = (is_metric != 0);
    std::memcpy(result.ext, ext, sizeof(ext));
    std::memcpy(result.intr, intr, sizeof(intr));

    da_capi_free_floats(conf);
    da_capi_free_floats(sky);

    LOGD("processFrame: %ux%u -> %dx%d (metric=%d)\n", w, h, out_w, out_h, is_metric);
    return result;
}
