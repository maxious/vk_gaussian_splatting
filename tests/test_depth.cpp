// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#include "doctest.h"
#include "../src/depth_parser.h"
#include "test_utils.h"
#include <cstring>

// ===== SYNTHETIC TESTS BASED ON REAL VDZ FILE FORMAT =====

TEST_CASE("Synthetic VDZ1 - Minimal Header Only")
{
    // Test header-only parsing (no depth data)
    auto buffer = test_utils::createMinimalVDZ1();
    
    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == true);
    CHECK(frame.timestampMs == 125);
    CHECK(frame.width == 320);
    CHECK(frame.height == 180);
    CHECK(frame.scale == doctest::Approx(0.000001f).epsilon(1e-6f));
    CHECK(frame.bias == doctest::Approx(0.646956f).epsilon(1e-6f));
    CHECK(frame.zMax == doctest::Approx(0.734110f).epsilon(1e-6f));
}

TEST_CASE("Synthetic VDZ1 - With Depth Data")
{
    // Test full frame with synthetic depth data
    auto buffer = test_utils::createVDZ1WithData();
    
    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == true);
    CHECK(frame.width == 320);
    CHECK(frame.height == 180);
    CHECK(frame.data.size() == 320 * 180);
    
    // Verify depth data was parsed (just check it exists)
    CHECK(frame.data.size() > 0);
}

TEST_CASE("Synthetic VDZ1 - Small Resolution")
{
    // Test with smaller resolution for faster unit tests
    auto buffer = test_utils::createSmallVDZ1();
    
    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == true);
    CHECK(frame.width == 64);
    CHECK(frame.height == 48);
    CHECK(frame.data.size() == 64 * 48);
}

TEST_CASE("Synthetic VDZ1 - High Resolution")
{
    // Test with higher resolution
    auto buffer = test_utils::createHighResVDZ1();
    
    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == true);
    CHECK(frame.width == 1280);
    CHECK(frame.height == 720);
    CHECK(frame.data.size() == 1280 * 720);
}

TEST_CASE("Synthetic VDZ1 - Varying Scale (Large)")
{
    // Test with larger scale value (from real file test_frame_2.vdz)
    auto buffer = test_utils::createVDZ1LargeScale();
    
    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == true);
    CHECK(frame.timestampMs == 208);
    CHECK(frame.scale == doctest::Approx(0.000002f).epsilon(1e-6f));
    CHECK(frame.bias == doctest::Approx(0.902846f).epsilon(1e-6f));
    CHECK(frame.zMax == doctest::Approx(1.004264f).epsilon(1e-6f));
}

// ===== ERROR CASES =====

TEST_CASE("Error - Invalid Magic Bytes")
{
    auto buffer = test_utils::createMinimalVDZ1();
    
    // Corrupt the magic bytes
    std::memcpy(buffer.data(), "XXXX", 4);

    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == false);
}

TEST_CASE("Error - Buffer Too Small")
{
    // Buffer smaller than header size (32 bytes)
    std::vector<uint8_t> buffer(10);

    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == false);
}

TEST_CASE("Error - Unsupported Version")
{
    auto buffer = test_utils::createMinimalVDZ1();

    // Modify version to unsupported value (version at offset 4-5)
    buffer[4] = 2; // Version = 2 (unsupported)

    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == false);
}

TEST_CASE("Error - Unsupported DataType")
{
    auto buffer = test_utils::createMinimalVDZ1();

    // Modify dataType to unsupported value (dataType at offset 6-7)
    buffer[6] = 2; // DataType = 2 (unsupported)

    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == false);
}

TEST_CASE("Error - Invalid Magic with Valid Size")
{
    // Edge case: buffer correct size but invalid magic
    std::vector<uint8_t> buffer(32, 0);
    std::memcpy(buffer.data(), "BADX", 4);

    DepthFrame frame;
    bool result = parseDepthFrame(buffer, frame);

    CHECK(result == false);
}
