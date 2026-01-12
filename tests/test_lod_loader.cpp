/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "doctest.h"
#include "lod_loader.h"

#include <fstream>
#include <filesystem>

using namespace vk_viewer;

TEST_SUITE("LodLoader")
{
  TEST_CASE("LodAabb basic operations")
  {
    LodAabb box;
    box.min = {-1.0f, -2.0f, -3.0f};
    box.max = {1.0f, 2.0f, 3.0f};

    SUBCASE("center")
    {
      glm::vec3 c = box.center();
      CHECK(c.x == doctest::Approx(0.0f));
      CHECK(c.y == doctest::Approx(0.0f));
      CHECK(c.z == doctest::Approx(0.0f));
    }

    SUBCASE("extent")
    {
      glm::vec3 e = box.extent();
      CHECK(e.x == doctest::Approx(2.0f));
      CHECK(e.y == doctest::Approx(4.0f));
      CHECK(e.z == doctest::Approx(6.0f));
    }

    SUBCASE("radius")
    {
      float r = box.radius();
      CHECK(r == doctest::Approx(glm::length(glm::vec3(2.0f, 4.0f, 6.0f)) * 0.5f));
    }

    SUBCASE("contains - inside")
    {
      CHECK(box.contains({0.0f, 0.0f, 0.0f}));
      CHECK(box.contains({0.5f, 1.0f, 2.0f}));
    }

    SUBCASE("contains - on boundary")
    {
      CHECK(box.contains({1.0f, 2.0f, 3.0f}));
      CHECK(box.contains({-1.0f, -2.0f, -3.0f}));
    }

    SUBCASE("contains - outside")
    {
      CHECK_FALSE(box.contains({2.0f, 0.0f, 0.0f}));
      CHECK_FALSE(box.contains({0.0f, 3.0f, 0.0f}));
      CHECK_FALSE(box.contains({0.0f, 0.0f, 4.0f}));
    }
  }

  TEST_CASE("LodAabb frustum intersection")
  {
    LodAabb box;
    box.min = {-1.0f, -1.0f, -1.0f};
    box.max = {1.0f, 1.0f, 1.0f};

    SUBCASE("identity frustum - box at origin visible")
    {
      glm::mat4 viewProj = glm::mat4(1.0f);
      CHECK(box.intersectsFrustum(viewProj));
    }

    SUBCASE("perspective frustum - box at origin visible")
    {
      glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 100.0f);
      glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
      glm::mat4 viewProj = proj * view;
      CHECK(box.intersectsFrustum(viewProj));
    }

    SUBCASE("box behind camera - not visible")
    {
      glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 100.0f);
      glm::mat4 view = glm::lookAt(glm::vec3(0, 0, -5), glm::vec3(0, 0, -10), glm::vec3(0, 1, 0));
      glm::mat4 viewProj = proj * view;
      CHECK_FALSE(box.intersectsFrustum(viewProj));
    }
  }

  TEST_CASE("LodDataRef structure")
  {
    LodDataRef ref;
    ref.fileIndex = 2;
    ref.offset    = 1000;
    ref.count     = 500;

    CHECK(ref.fileIndex == 2);
    CHECK(ref.offset == 1000);
    CHECK(ref.count == 500);
  }

  TEST_CASE("LodNode leaf detection")
  {
    LodNode node;
    
    SUBCASE("empty node is leaf")
    {
      CHECK(node.isLeaf());
    }

    SUBCASE("node with children is not leaf")
    {
      node.children.push_back(LodNode{});
      CHECK_FALSE(node.isLeaf());
    }

    SUBCASE("node with lods but no children is still leaf")
    {
      node.lods[0] = LodDataRef{0, 0, 100};
      node.lods[1] = LodDataRef{0, 100, 50};
      CHECK(node.isLeaf());
    }
  }

  TEST_CASE("LodSelection policy - FIXED_LEVEL")
  {
    LodMeta meta;
    meta.lodLevels = 3;
    meta.tree.bound.min = {-10.0f, -10.0f, -10.0f};
    meta.tree.bound.max = {10.0f, 10.0f, 10.0f};
    meta.tree.lods[0] = LodDataRef{0, 0, 1000};
    meta.tree.lods[1] = LodDataRef{0, 1000, 500};
    meta.tree.lods[2] = LodDataRef{0, 1500, 250};
    meta.filenames.push_back("0_0/meta.json");

    glm::vec3 cameraPos = {0.0f, 0.0f, 50.0f};
    glm::mat4 proj      = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
    glm::mat4 view      = glm::lookAt(cameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 viewProj  = proj * view;

    SUBCASE("select level 0")
    {
      auto selections = LodScene::selectLods(meta, cameraPos, viewProj, LodSelectionPolicy::FIXED_LEVEL, 0);
      REQUIRE(selections.size() == 1);
      CHECK(selections[0].lodLevel == 0);
      CHECK(selections[0].count == 1000);
    }

    SUBCASE("select level 1")
    {
      auto selections = LodScene::selectLods(meta, cameraPos, viewProj, LodSelectionPolicy::FIXED_LEVEL, 1);
      REQUIRE(selections.size() == 1);
      CHECK(selections[0].lodLevel == 1);
      CHECK(selections[0].count == 500);
    }

    SUBCASE("select level 2")
    {
      auto selections = LodScene::selectLods(meta, cameraPos, viewProj, LodSelectionPolicy::FIXED_LEVEL, 2);
      REQUIRE(selections.size() == 1);
      CHECK(selections[0].lodLevel == 2);
      CHECK(selections[0].count == 250);
    }
  }

  TEST_CASE("LodSelection with BSP tree")
  {
    LodMeta meta;
    meta.lodLevels = 2;
    meta.tree.bound.min = {-20.0f, -10.0f, -10.0f};
    meta.tree.bound.max = {20.0f, 10.0f, 10.0f};

    LodNode leftChild;
    leftChild.bound.min = {-20.0f, -10.0f, -10.0f};
    leftChild.bound.max = {0.0f, 10.0f, 10.0f};
    leftChild.lods[0] = LodDataRef{0, 0, 500};
    leftChild.lods[1] = LodDataRef{0, 500, 200};

    LodNode rightChild;
    rightChild.bound.min = {0.0f, -10.0f, -10.0f};
    rightChild.bound.max = {20.0f, 10.0f, 10.0f};
    rightChild.lods[0] = LodDataRef{1, 0, 600};
    rightChild.lods[1] = LodDataRef{1, 600, 300};

    meta.tree.children.push_back(leftChild);
    meta.tree.children.push_back(rightChild);
    meta.filenames.push_back("0_0/meta.json");
    meta.filenames.push_back("0_1/meta.json");

    glm::vec3 cameraPos = {0.0f, 0.0f, 50.0f};
    glm::mat4 proj      = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
    glm::mat4 view      = glm::lookAt(cameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 viewProj  = proj * view;

    SUBCASE("both children visible at level 0")
    {
      auto selections = LodScene::selectLods(meta, cameraPos, viewProj, LodSelectionPolicy::FIXED_LEVEL, 0);
      REQUIRE(selections.size() == 2);
      CHECK(selections[0].lodLevel == 0);
      CHECK(selections[1].lodLevel == 0);
      CHECK(selections[0].count + selections[1].count == 1100);
    }

    SUBCASE("both children visible at level 1")
    {
      auto selections = LodScene::selectLods(meta, cameraPos, viewProj, LodSelectionPolicy::FIXED_LEVEL, 1);
      REQUIRE(selections.size() == 2);
      CHECK(selections[0].lodLevel == 1);
      CHECK(selections[1].lodLevel == 1);
      CHECK(selections[0].count + selections[1].count == 500);
    }
  }

  TEST_CASE("LodMeta parsing from JSON")
  {
    const char* jsonStr = R"({
      "lodLevels": 3,
      "environment": "env/meta.json",
      "filenames": ["0_0/meta.json", "0_1/meta.json", "1_0/meta.json"],
      "tree": {
        "bound": {
          "min": [-100.5, -50.2, -75.3],
          "max": [100.5, 50.2, 75.3]
        },
        "lods": {
          "0": {"file": 0, "offset": 0, "count": 10000},
          "1": {"file": 1, "offset": 0, "count": 5000},
          "2": {"file": 2, "offset": 0, "count": 2500}
        }
      }
    })";

    std::filesystem::path testFile = std::filesystem::temp_directory_path() / "test_lod_meta.json";
    {
      std::ofstream ofs(testFile);
      ofs << jsonStr;
    }

    LodMeta meta;
    bool    loaded = LodScene::load(testFile, meta);

    REQUIRE(loaded);
    CHECK(meta.lodLevels == 3);
    CHECK(meta.environment == "env/meta.json");
    CHECK(meta.filenames.size() == 3);
    CHECK(meta.filenames[0] == "0_0/meta.json");
    CHECK(meta.filenames[1] == "0_1/meta.json");
    CHECK(meta.filenames[2] == "1_0/meta.json");

    CHECK(meta.tree.bound.min.x == doctest::Approx(-100.5f));
    CHECK(meta.tree.bound.min.y == doctest::Approx(-50.2f));
    CHECK(meta.tree.bound.min.z == doctest::Approx(-75.3f));
    CHECK(meta.tree.bound.max.x == doctest::Approx(100.5f));
    CHECK(meta.tree.bound.max.y == doctest::Approx(50.2f));
    CHECK(meta.tree.bound.max.z == doctest::Approx(75.3f));

    REQUIRE(meta.tree.lods.size() == 3);
    CHECK(meta.tree.lods.at(0).fileIndex == 0);
    CHECK(meta.tree.lods.at(0).count == 10000);
    CHECK(meta.tree.lods.at(1).fileIndex == 1);
    CHECK(meta.tree.lods.at(1).count == 5000);
    CHECK(meta.tree.lods.at(2).fileIndex == 2);
    CHECK(meta.tree.lods.at(2).count == 2500);

    std::filesystem::remove(testFile);
  }

  TEST_CASE("LodMeta parsing with nested BSP tree")
  {
    const char* jsonStr = R"({
      "lodLevels": 2,
      "filenames": ["0_0/meta.json"],
      "tree": {
        "bound": {
          "min": [-10, -10, -10],
          "max": [10, 10, 10]
        },
        "children": [
          {
            "bound": {
              "min": [-10, -10, -10],
              "max": [0, 10, 10]
            },
            "lods": {
              "0": {"file": 0, "offset": 0, "count": 500},
              "1": {"file": 0, "offset": 500, "count": 200}
            }
          },
          {
            "bound": {
              "min": [0, -10, -10],
              "max": [10, 10, 10]
            },
            "lods": {
              "0": {"file": 0, "offset": 700, "count": 500},
              "1": {"file": 0, "offset": 1200, "count": 200}
            }
          }
        ]
      }
    })";

    std::filesystem::path testFile = std::filesystem::temp_directory_path() / "test_lod_meta_bsp.json";
    {
      std::ofstream ofs(testFile);
      ofs << jsonStr;
    }

    LodMeta meta;
    bool    loaded = LodScene::load(testFile, meta);

    REQUIRE(loaded);
    CHECK(meta.lodLevels == 2);
    CHECK(meta.tree.children.size() == 2);
    CHECK(meta.tree.isLeaf() == false);
    CHECK(meta.tree.children[0].isLeaf() == true);
    CHECK(meta.tree.children[1].isLeaf() == true);

    CHECK(meta.tree.children[0].lods.at(0).count == 500);
    CHECK(meta.tree.children[0].lods.at(1).count == 200);
    CHECK(meta.tree.children[1].lods.at(0).count == 500);
    CHECK(meta.tree.children[1].lods.at(1).count == 200);

    std::filesystem::remove(testFile);
  }
}
