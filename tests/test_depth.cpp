// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#include "doctest.h"
#include "../src/depth_parser.h"
#include "test_utils.h"
#include <cstring>

TEST_CASE("Depth Header Parsing - VDZ1 (uncompressed)")
{
    // Create a minimal valid VDZ1 header
    auto headerBuffer = test_utils::createVDZ1Header(1000, 640, 480, 0.001f, 0.0f, 10.0f);

    // Add minimal depth data (2 pixels)
    uint16_t depthData[] = {1000, 2000}; // 1.0m and 2.0m at scale 0.001
    headerBuffer.insert(headerBuffer.end(),
        reinterpret_cast<const uint8_t*>(depthData),
        reinterpret_cast<const uint8_t*>(depthData) + sizeof(depthData));

    DepthFrame frame;
    bool result = parseDepthFrame(headerBuffer, frame);

    CHECK(result == true);
    CHECK(frame.timestampMs == 1000);
    CHECK(frame.width == 640);
    CHECK(frame.height == 480);
    CHECK(frame.scale == doctest::Approx(0.001f));
    CHECK(frame.bias == doctest::Approx(0.0f));
    CHECK(frame.zMax == doctest::Approx(10.0f));
    CHECK(frame.data.size() == 2);
    CHECK(frame.data[0] == doctest::Approx(1.0f));  // 1000 * 0.001 + 0.0
    CHECK(frame.data[1] == doctest::Approx(2.0f));  // 2000 * 0.001 + 0.0
}

TEST_CASE("Depth Header Parsing - VDZ2 (compressed)")
{
    // Create a minimal valid VDZ2 header
    auto headerBuffer = test_utils::createVDZ2Header(500, 320, 240, 0.002f, 0.5f, 5.0f);

    // Add compressed depth data (raw uncompressed: {500, 1500})
    uint16_t depthData[] = {500, 1500};
    headerBuffer.insert(headerBuffer.end(),
        reinterpret_cast<const uint8_t*>(depthData),
        reinterpret_cast<const uint8_t*>(depthData) + sizeof(depthData));

    DepthFrame frame;
    bool result = parseDepthFrame(headerBuffer, frame);

    CHECK(result == true);
    CHECK(frame.timestampMs == 500);
    CHECK(frame.width == 320);
    CHECK(frame.height == 240);
    CHECK(frame.scale == doctest::Approx(0.002f));
    CHECK(frame.bias == doctest::Approx(0.5f));
    CHECK(frame.zMax == doctest::Approx(5.0f));
}

TEST_CASE("Depth Header - Invalid Magic Bytes")
{
    // Create header with invalid magic bytes
    auto buffer = test_utils::createVDZ1Header();

    // Corrupt the magic bytes
    std::memcpy(buffer.data(), "XXXX", 4);

    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == false);
}

TEST_CASE("Depth Header - Buffer Too Small")
{
    // Buffer smaller than header size
    std::vector<uint8_t> buffer(10); // Less than 32 bytes

    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == false);
}

TEST_CASE("Depth Header - Unsupported Version")
{
    auto buffer = test_utils::createVDZ1Header();

    // Modify version to unsupported value
    buffer[4] = 2; // Version field is at offset 4 (after magic bytes)

    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == false);
}
