// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#include "doctest.h"
#include "../src/vdz_sequence_loader.h"
#include "test_utils.h"
#include <fstream>
#include <filesystem>

using namespace vk_gaussian_splatting;

// Helper to write a buffer to a temp file
static std::filesystem::path writeTempVdz(const std::vector<uint8_t>& data)
{
    auto tempPath = std::filesystem::temp_directory_path() / "test_sequence.vdz";
    std::ofstream file(tempPath, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    file.close();
    return tempPath;
}

TEST_CASE("VDZSequenceLoader - Single Frame")
{
    auto buffer = test_utils::createVDZ1Header(100, 64, 48, 0.000001f, 0.5f, 1.0f, true);
    auto tempPath = writeTempVdz(buffer);
    
    VDZSequenceLoader loader;
    CHECK(loader.open(tempPath));
    
    CHECK(loader.getFrameCount() == 1);
    CHECK(loader.getWidth() == 64);
    CHECK(loader.getHeight() == 48);
    CHECK(loader.getDurationMs() == 100);
    
    DepthFrame frame;
    CHECK(loader.getFrame(0, frame));
    CHECK(frame.timestampMs == 100);
    CHECK(frame.width == 64);
    CHECK(frame.height == 48);
    CHECK(frame.data.size() == 64 * 48);
    
    loader.close();
    std::filesystem::remove(tempPath);
}

TEST_CASE("VDZSequenceLoader - Multiple Frames")
{
    auto buffer = test_utils::createVDZ1Sequence(10);
    auto tempPath = writeTempVdz(buffer);
    
    VDZSequenceLoader loader;
    CHECK(loader.open(tempPath));
    
    CHECK(loader.getFrameCount() == 10);
    CHECK(loader.getDurationMs() == 9 * 33);  // Last frame timestamp
    
    // Check first frame
    DepthFrame frame0;
    CHECK(loader.getFrame(0, frame0));
    CHECK(frame0.timestampMs == 0);
    
    // Check middle frame
    DepthFrame frame5;
    CHECK(loader.getFrame(5, frame5));
    CHECK(frame5.timestampMs == 5 * 33);
    
    // Check last frame
    DepthFrame frame9;
    CHECK(loader.getFrame(9, frame9));
    CHECK(frame9.timestampMs == 9 * 33);
    
    loader.close();
    std::filesystem::remove(tempPath);
}

TEST_CASE("VDZSequenceLoader - Timestamp Lookup")
{
    auto buffer = test_utils::createVDZ1Sequence(10);
    auto tempPath = writeTempVdz(buffer);
    
    VDZSequenceLoader loader;
    CHECK(loader.open(tempPath));
    
    // Exact match
    CHECK(loader.getFrameIndexForTimestamp(0) == 0);
    CHECK(loader.getFrameIndexForTimestamp(33) == 1);
    CHECK(loader.getFrameIndexForTimestamp(66) == 2);
    
    // Between frames - should find closest
    CHECK(loader.getFrameIndexForTimestamp(16) == 0);  // Closer to 0 than 33
    CHECK(loader.getFrameIndexForTimestamp(17) == 1);  // Closer to 33 than 0
    CHECK(loader.getFrameIndexForTimestamp(50) == 2);  // Closer to 66 than 33
    
    // Beyond last frame
    CHECK(loader.getFrameIndexForTimestamp(1000) == 9);
    
    loader.close();
    std::filesystem::remove(tempPath);
}

TEST_CASE("VDZSequenceLoader - Get Frame By Timestamp")
{
    auto buffer = test_utils::createVDZ1Sequence(5);
    auto tempPath = writeTempVdz(buffer);
    
    VDZSequenceLoader loader;
    CHECK(loader.open(tempPath));
    
    DepthFrame frame;
    CHECK(loader.getFrameByTimestamp(100, frame));  // Should get frame 3 (99ms)
    CHECK(frame.timestampMs == 99);  // 3 * 33 = 99
    
    loader.close();
    std::filesystem::remove(tempPath);
}

TEST_CASE("VDZSequenceLoader - File Not Found")
{
    VDZSequenceLoader loader;
    CHECK_FALSE(loader.open("nonexistent_file.vdz"));
    CHECK(loader.getFrameCount() == 0);
}

TEST_CASE("VDZSequenceLoader - Invalid Frame Index")
{
    auto buffer = test_utils::createVDZ1Sequence(3);
    auto tempPath = writeTempVdz(buffer);
    
    VDZSequenceLoader loader;
    CHECK(loader.open(tempPath));
    
    DepthFrame frame;
    CHECK_FALSE(loader.getFrame(10, frame));  // Index out of range
    
    loader.close();
    std::filesystem::remove(tempPath);
}
