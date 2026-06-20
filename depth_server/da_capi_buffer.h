#pragma once

#include <cstdint>

struct DepthResult {
    float* depth = nullptr;
    std::int32_t w = 0;
    std::int32_t h = 0;
    float conf = 0.0f;
    float scale = 1.0f;
    float bias = 0.0f;
    float zMax = 0.0f;
};
