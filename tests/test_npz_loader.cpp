// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#include "doctest.h"
#include "../src/npz_loader.h"
#include <filesystem>

// Define TEST_RESOURCE_DIR if not already defined
#ifndef TEST_RESOURCE_DIR
#define TEST_RESOURCE_DIR "_downloaded_resources"
#endif

TEST_CASE("NPZ Loader - Simple Format (means3d + colors)")
{
  std::filesystem::path testFile = std::filesystem::path(TEST_RESOURCE_DIR) / "test_garden.npz";

  if(!std::filesystem::exists(testFile))
  {
    return;  // Skip if file doesn't exist
  }

  SUBCASE("canLoad detects valid NPZ file")
  {
    CHECK(vk_viewer::NpzLoader::canLoad(testFile));
  }

  SUBCASE("getTimestepCount returns 1 for static file")
  {
    CHECK_EQ(vk_viewer::NpzLoader::getTimestepCount(testFile), 1);
  }

  SUBCASE("hasMultipleTimesteps returns false for static file")
  {
    CHECK_FALSE(vk_viewer::NpzLoader::hasMultipleTimesteps(testFile));
  }

  SUBCASE("load succeeds and populates SplatSet")
  {
    vk_viewer::SplatSet splats;
    bool success = vk_viewer::NpzLoader::load(testFile, splats);

    CHECK(success);
    CHECK_GT(splats.size(), 0);

    // Check that simple format generated default values
    CHECK_EQ(splats.opacity.size(), splats.size());
    CHECK_EQ(splats.scale.size(), splats.size() * 3);
    CHECK_EQ(splats.rotation.size(), splats.size() * 4);
  }

  SUBCASE("Loaded splats have valid data")
  {
    vk_viewer::SplatSet splats;
    bool success = vk_viewer::NpzLoader::load(testFile, splats);

    CHECK(success);
    CHECK(splats.size() > 0);

    // Verify positions are reasonable (not all zeros)
    bool hasNonZeroPosition = false;
    for(size_t i = 0; i < std::min(splats.size(), size_t(100)); ++i)
    {
      if(splats.positions[i * 3] != 0.0f ||
         splats.positions[i * 3 + 1] != 0.0f ||
         splats.positions[i * 3 + 2] != 0.0f)
      {
        hasNonZeroPosition = true;
        break;
      }
    }
    CHECK(hasNonZeroPosition);

    // Verify colors converted to SH DC coefficients (should be in range [-2, 2])
    for(size_t i = 0; i < std::min(splats.f_dc.size(), size_t(100)); ++i)
    {
      CHECK_GE(splats.f_dc[i], -2.0f);
      CHECK_LE(splats.f_dc[i], 2.0f);
    }
  }
}

TEST_CASE("NPZ Loader - File Extension Detection")
{
  CHECK_FALSE(vk_viewer::NpzLoader::canLoad(std::filesystem::path("test.ply")));
  CHECK_FALSE(vk_viewer::NpzLoader::canLoad(std::filesystem::path("test.bin")));
  CHECK_FALSE(vk_viewer::NpzLoader::canLoad(std::filesystem::path("test.txt")));
}

TEST_CASE("NPZ Loader - Non-existent File")
{
  std::filesystem::path nonExistent = std::filesystem::path("/tmp/non_existent_file_12345.npz");
  CHECK_FALSE(std::filesystem::exists(nonExistent));
  CHECK_FALSE(vk_viewer::NpzLoader::canLoad(nonExistent));
}
