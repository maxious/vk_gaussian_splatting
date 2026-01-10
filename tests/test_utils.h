// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#pragma once

#include <vector>
#include <cstdint>
#include <cstring>

namespace test_utils {

// Helper to construct a valid VDZ1 (uncompressed) depth header with optional depth data
// Based on real VDZ1 file format (320x180 @ 0.000001 scale)
inline std::vector<uint8_t> createVDZ1Header(
    uint32_t timestampMs = 125,
    uint32_t width = 320,
    uint32_t height = 180,
    float scale = 0.000001f,
    float bias = 0.646956f,
    float zMax = 0.734110f,
    bool includeDepthData = false)
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

    // Add synthetic depth data if requested
    if (includeDepthData) {
        uint32_t pixelCount = width * height;
        std::vector<uint16_t> depthData(pixelCount);
        
        // Generate realistic depth values based on scale/bias/zMax
        // Depth values typically range from ~0 to zMax when scaled
        for (uint32_t i = 0; i < pixelCount; ++i) {
            // Create a gradient pattern for testing
            uint16_t value = static_cast<uint16_t>((i % 256) * 256 + (i / 256) % 256);
            depthData[i] = value;
        }
        
        buffer.insert(buffer.end(),
            reinterpret_cast<const uint8_t*>(depthData.data()),
            reinterpret_cast<const uint8_t*>(depthData.data()) + pixelCount * sizeof(uint16_t));
    }

    return buffer;
}

// Helper to construct a valid VDZ2 (compressed) depth header with optional depth data
// Uses realistic parameters from real depth streams
inline std::vector<uint8_t> createVDZ2Header(
    uint32_t timestampMs = 166,
    uint32_t width = 320,
    uint32_t height = 180,
    float scale = 0.000001f,
    float bias = 0.657012f,
    float zMax = 0.752488f,
    bool includeDepthData = false)
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

    // Add synthetic depth data if requested
    if (includeDepthData) {
        uint32_t pixelCount = width * height;
        std::vector<uint16_t> depthData(pixelCount);
        
        // Generate depth values with realistic variance
        for (uint32_t i = 0; i < pixelCount; ++i) {
            // Create varied depth values for better compression testing
            uint32_t x = i % width;
            uint32_t y = i / width;
            uint16_t value = static_cast<uint16_t>((x + y * 2) % 65536);
            depthData[i] = value;
        }
        
        buffer.insert(buffer.end(),
            reinterpret_cast<const uint8_t*>(depthData.data()),
            reinterpret_cast<const uint8_t*>(depthData.data()) + pixelCount * sizeof(uint16_t));
    }

    return buffer;
}

// Helper to create a minimal VDZ1 header (header only, no depth data)
inline std::vector<uint8_t> createMinimalVDZ1()
{
    return createVDZ1Header(125, 320, 180, 0.000001f, 0.646956f, 0.734110f, false);
}

// Helper to create a VDZ1 header with full depth data (matches real file structure)
inline std::vector<uint8_t> createVDZ1WithData()
{
    return createVDZ1Header(125, 320, 180, 0.000001f, 0.646956f, 0.734110f, true);
}

// Helper to create a small test frame (64x48 for fast tests)
inline std::vector<uint8_t> createSmallVDZ1()
{
    return createVDZ1Header(0, 64, 48, 0.000001f, 0.5f, 1.0f, true);
}

// Helper to create a high-resolution test frame
inline std::vector<uint8_t> createHighResVDZ1()
{
    return createVDZ1Header(1000, 1280, 720, 0.000001f, 0.5f, 1.0f, true);
}

// Helper to create varying scale parameters (test case from frame 2)
inline std::vector<uint8_t> createVDZ1LargeScale()
{
    return createVDZ1Header(208, 320, 180, 0.000002f, 0.902846f, 1.004264f, true);
}



} // namespace test_utils
