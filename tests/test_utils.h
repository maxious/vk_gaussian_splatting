// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#pragma once

#include <vector>
#include <cstdint>
#include <cstring>

namespace test_utils {

// Helper to construct a valid VDZ1 (uncompressed) depth header
inline std::vector<uint8_t> createVDZ1Header(
    uint32_t timestampMs = 0,
    uint32_t width = 640,
    uint32_t height = 480,
    float scale = 0.001f,
    float bias = 0.0f,
    float zMax = 10.0f)
{
    struct DepthHeader {
        char     magic[4];       // "VDZ1"
        uint16_t version;       // Always 1
        uint16_t dataType;      // Always 1 (uint16)
        uint32_t timestampMs;
        uint32_t width;
        uint32_t height;
        float     scale;
        float     bias;
        float     zMax;
    };

    DepthHeader header{};
    std::memcpy(header.magic, "VDZ1", 4);
    header.version = 1;
    header.dataType = 1;
    header.timestampMs = timestampMs;
    header.width = width;
    header.height = height;
    header.scale = scale;
    header.bias = bias;
    header.zMax = zMax;

    std::vector<uint8_t> buffer(sizeof(DepthHeader));
    std::memcpy(buffer.data(), &header, sizeof(DepthHeader));

    return buffer;
}

// Helper to construct a valid VDZ2 (compressed) depth header
inline std::vector<uint8_t> createVDZ2Header(
    uint32_t timestampMs = 0,
    uint32_t width = 640,
    uint32_t height = 480,
    float scale = 0.001f,
    float bias = 0.0f,
    float zMax = 10.0f)
{
    struct DepthHeader {
        char     magic[4];       // "VDZ2"
        uint16_t version;       // Always 1
        uint16_t dataType;      // Always 1 (uint16)
        uint32_t timestampMs;
        uint32_t width;
        uint32_t height;
        float     scale;
        float     bias;
        float     zMax;
    };

    DepthHeader header{};
    std::memcpy(header.magic, "VDZ2", 4);
    header.version = 1;
    header.dataType = 1;
    header.timestampMs = timestampMs;
    header.width = width;
    header.height = height;
    header.scale = scale;
    header.bias = bias;
    header.zMax = zMax;

    std::vector<uint8_t> buffer(sizeof(DepthHeader));
    std::memcpy(buffer.data(), &header, sizeof(DepthHeader));

    return buffer;
}

} // namespace test_utils
