#include "da_capi_buffer.h"

#include <nvutils/logger.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

DepthResult processFrame(const std::uint8_t* rgb, std::uint32_t w, std::uint32_t h, da_ctx* ctx)
{
    DepthResult result;
    if(!ctx || !rgb || w == 0 || h == 0)
    {
        LOGE("processFrame: invalid context or frame buffer\n");
        return result;
    }

    // Write RGB buffer to a temporary PPM P6 file for depth-anything.cpp's
    // file-path-based API (da_capi_depth_from_rgb was removed in ABI 6→10).
    char temp_template[] = "/tmp/da_depth_XXXXXX";
    int fd = ::mkstemp(temp_template);
    if(fd < 0)
    {
        LOGE("processFrame: mkstemp failed (errno=%d)\n", errno);
        return result;
    }
    {
        char header[64];
        const int hdr_len = std::snprintf(header, sizeof(header), "P6\n%u %u\n255\n", w, h);
        if(hdr_len < 0 || ::write(fd, header, static_cast<size_t>(hdr_len)) != hdr_len)
        {
            LOGE("processFrame: write PPM header failed\n");
            ::close(fd);
            ::unlink(temp_template);
            return result;
        }
        const size_t pixel_bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
        if(::write(fd, rgb, pixel_bytes) != static_cast<ssize_t>(pixel_bytes))
        {
            LOGE("processFrame: write PPM pixels failed\n");
            ::close(fd);
            ::unlink(temp_template);
            return result;
        }
    }
    ::close(fd);

    int out_w = 0;
    int out_h = 0;
    int is_metric = 0;
    float* depth = nullptr;
    float* conf = nullptr;
    float* sky = nullptr;
    float ext[12] = {};
    float intr[9] = {};

    const int ret = da_capi_depth_dense(ctx, temp_template,
                                        &out_h, &out_w,
                                        &depth, &conf, &sky,
                                        ext, intr, &is_metric);

    // Clean up temp file immediately — da_capi_depth_dense has already read it.
    ::unlink(temp_template);

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
