#pragma once

#include <vector>
#include <cstdint>

struct DepthHeader {
    char     magic[4];       // "VDZ1" (raw) or "VDZ2" (compressed)
    uint16_t version;       // Always 1
    uint16_t dataType;      // Always 1 (uint16)
    uint32_t timestampMs;   // Frame timestamp in ms
    uint32_t width;        // Depth map width
    uint32_t height;       // Depth map height
    float     scale;         // Quantization scale
    float     bias;          // Quantization bias
    float     zMax;         // Maximum depth value

    // Total: 32 bytes
};

// Ensure struct is packed to 32 bytes
static_assert(sizeof(DepthHeader) == 32, "DepthHeader must be 32 bytes");

struct DepthFrame {
    uint32_t timestampMs;
    uint32_t width;
    uint32_t height;
    std::vector<float> data;  // Depth values in meters
    float scale;
    float bias;
    float zMax;
};

bool parseDepthFrame(const std::vector<uint8_t>& buffer, DepthFrame& outFrame);
