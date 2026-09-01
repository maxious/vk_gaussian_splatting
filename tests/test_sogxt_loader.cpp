/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "doctest.h"
#include "sogxt_loader.h"

#include <filesystem>
#include <fstream>
#include <limits>

using namespace vk_viewer;

#ifndef SOGXT_TEST_FIXTURE_DIR
#define SOGXT_TEST_FIXTURE_DIR "fixtures"
#endif

TEST_SUITE("SogXtLoader")
{
  TEST_CASE("parseMeta rejects non-SOG-XT json")
  {
    const std::string bogus = R"({"version": 2, "count": 10, "format": "sog"})";
    std::vector<uint8_t> data(bogus.begin(), bogus.end());
    SogXtLoader::SogXtMeta meta;
    CHECK_FALSE(SogXtLoader::parseMeta(data, meta));
  }

  TEST_CASE("parseMeta accepts v3 container")
  {
    const std::string json = R"({
      "version": 3, "format": "sog-xt", "profile": "SOG-XT",
      "count": 256000, "gridSide": 512,
      "mask": {"files": ["active_mask.webp"]},
      "means": {"mins": -7.26, "maxs": 7.34, "files": ["means_bytes_0.webp", "means_bytes_1.webp"]},
      "opacities": {"mins": -13.8, "maxs": 13.8, "files": ["opacities.webp"], "normalize": "observed-minmax"},
      "scales": {"mins": [-13.9, -14.0, -13.9], "maxs": [2.0, 2.1, 2.1], "files": ["scales.webp"]},
      "quats": {"mins": [-0.7, -0.7, -0.7, -0.6], "maxs": [1.0, 0.9, 0.9, 0.9], "files": ["quaternions.webp"]},
      "sh0": {"mins": [-2.5, -2.9, -3.1], "maxs": [9.6, 10.4, 10.8], "files": ["f_dc.webp"]},
      "shN": {"layout": "uv-codebook", "coeffs": 15, "centroidSide": 176, "tileRows": 3, "tileCols": 5,
              "centroidsMins": [], "centroidsMaxs": [],
              "files": ["f_rest_centroids.webp", "f_rest_labels.webp"]}
    })";
    std::vector<uint8_t> data(json.begin(), json.end());
    SogXtLoader::SogXtMeta meta;
    CHECK(SogXtLoader::parseMeta(data, meta));
    CHECK(meta.count == 256000);
    CHECK(meta.gridSide == 512);
    CHECK(meta.means.files.size() == 2);
    CHECK(meta.shN.centroidSide == 176);
    CHECK(meta.shN.tileRows == 3);
    CHECK(meta.shN.tileCols == 5);
  }

  TEST_CASE("load extracted KISS-GS garden container")
  {
    // The sample container extracted from the KISS-GS HAR export.
    std::filesystem::path container = std::filesystem::path(SOGXT_TEST_FIXTURE_DIR) / "sog_xt_garden";
    if(!std::filesystem::is_directory(container))
    {
      MESSAGE("SOG-XT fixture not present, skipping: ", container.string());
      return;
    }

    SplatSet output;
    bool ok = SogXtLoader::load(container, output);
    CHECK(ok);
    if(!ok)
      return;

    // Garden scene has 256000 active splats on a 512x512 grid.
    CHECK(output.size() == 256000);

    // All attribute arrays sized correctly.
    CHECK(output.positions.size() == output.size() * 3);
    CHECK(output.f_dc.size() == output.size() * 3);
    CHECK(output.f_rest.size() == output.size() * 45);
    CHECK(output.opacity.size() == output.size());
    CHECK(output.scale.size() == output.size() * 3);
    CHECK(output.rotation.size() == output.size() * 4);

    // Positions span a plausible range (RDF scene, after RUB conversion).
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    for(size_t i = 0; i < output.size(); ++i)
    {
      minX = std::min(minX, output.positions[i * 3 + 0]);
      maxX = std::max(maxX, output.positions[i * 3 + 0]);
    }
    CHECK(minX < maxX);
    CHECK(maxX - minX > 1.0f);  // not degenerate

    // No NaNs in positions.
    for(size_t i = 0; i < output.positions.size(); ++i)
    {
      CHECK_FALSE(std::isnan(output.positions[i]));
    }
  }

  TEST_CASE("isSogXtManifest detects container and rejects plain JSON")
  {
    // Garden container directory / meta.json / scene.json
    std::filesystem::path container = std::filesystem::path(SOGXT_TEST_FIXTURE_DIR) / "sog_xt_garden";
    if(std::filesystem::is_directory(container))
    {
      CHECK(SogXtLoader::isSogXtManifest(container));
      CHECK(SogXtLoader::isSogXtManifest(container / "meta.json"));
      CHECK(SogXtLoader::isSogXtManifest(container / "scene.json"));
    }

    // Non-SOG-XT JSON (e.g. depth-video metadata) is rejected
    std::filesystem::path tempDir = std::filesystem::temp_directory_path();
    std::filesystem::path bogusJson = tempDir / "sogxt_bogus_meta.json";
    {
      std::ofstream ofs(bogusJson);
      ofs << R"({"version": 2, "count": 10, "format": "sog"})";
    }
    CHECK_FALSE(SogXtLoader::isSogXtManifest(bogusJson));
    std::filesystem::remove(bogusJson);
  }
}