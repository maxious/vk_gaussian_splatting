#pragma once

#include <cstdint>

#include "da_capi.h"

struct DepthResult {
    float* depth = nullptr;
    std::int32_t w = 0;
    std::int32_t h = 0;
    float conf = 0.0f;
    float ext[12] = {};
    float intr[9] = {};
    bool is_metric = false;
};

DepthResult processFrame(const std::uint8_t* rgb, std::uint32_t w, std::uint32_t h, da_ctx* ctx);
