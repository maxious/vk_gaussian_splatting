/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "doctest.h"
#include "morton_order.hpp"
#include "splat_set.h"
#include <cmath>

using namespace vk_viewer;

TEST_SUITE("Morton Order")
{
  TEST_CASE("expandBits10 basic values")
  {
    CHECK(expandBits10(0) == 0);
    CHECK(expandBits10(1) == 1);
    CHECK(expandBits10(2) == 8);
    CHECK(expandBits10(3) == 9);
    CHECK(expandBits10(1023) == 0x09249249);
  }

  TEST_CASE("morton3D origin")
  {
    CHECK(morton3D(0, 0, 0) == 0);
  }

  TEST_CASE("morton3D single axis")
  {
    CHECK(morton3D(1, 0, 0) == 1);
    CHECK(morton3D(0, 1, 0) == 2);
    CHECK(morton3D(0, 0, 1) == 4);
  }

  TEST_CASE("morton3D combined")
  {
    CHECK(morton3D(1, 1, 1) == 7);
    CHECK(morton3D(2, 2, 2) == 7 * 8);  // shifted by 3 bits
  }

  TEST_CASE("morton3D max values")
  {
    uint32_t maxMorton = morton3D(1023, 1023, 1023);
    CHECK(maxMorton == 0x3FFFFFFF);
  }

  TEST_CASE("mortonFromPosition normalizes correctly")
  {
    float minX = 0.0f, minY = 0.0f, minZ = 0.0f;
    float invRangeX = 1.0f, invRangeY = 1.0f, invRangeZ = 1.0f;

    CHECK(mortonFromPosition(0.0f, 0.0f, 0.0f, minX, minY, minZ, invRangeX, invRangeY, invRangeZ) == 0);
    CHECK(mortonFromPosition(1.0f, 1.0f, 1.0f, minX, minY, minZ, invRangeX, invRangeY, invRangeZ) == morton3D(1023, 1023, 1023));
    CHECK(mortonFromPosition(0.5f, 0.5f, 0.5f, minX, minY, minZ, invRangeX, invRangeY, invRangeZ) == morton3D(511, 511, 511));
  }

  TEST_CASE("mortonFromPosition with offset bounding box")
  {
    float minX = -10.0f, minY = -10.0f, minZ = -10.0f;
    float invRangeX = 1.0f / 20.0f, invRangeY = 1.0f / 20.0f, invRangeZ = 1.0f / 20.0f;

    CHECK(mortonFromPosition(-10.0f, -10.0f, -10.0f, minX, minY, minZ, invRangeX, invRangeY, invRangeZ) == 0);
    CHECK(mortonFromPosition(10.0f, 10.0f, 10.0f, minX, minY, minZ, invRangeX, invRangeY, invRangeZ) == morton3D(1023, 1023, 1023));
  }

  TEST_CASE("SplatSet::reorderByMortonCode empty set")
  {
    SplatSet set;
    set.reorderByMortonCode();
    CHECK(set.size() == 0);
  }

  TEST_CASE("SplatSet::reorderByMortonCode single splat")
  {
    SplatSet set;
    set.positions = {1.0f, 2.0f, 3.0f};
    set.f_dc = {0.1f, 0.2f, 0.3f};
    set.opacity = {0.5f};
    set.scale = {1.0f, 1.0f, 1.0f};
    set.rotation = {1.0f, 0.0f, 0.0f, 0.0f};

    set.reorderByMortonCode();

    CHECK(set.size() == 1);
    CHECK(set.positions[0] == 1.0f);
    CHECK(set.positions[1] == 2.0f);
    CHECK(set.positions[2] == 3.0f);
  }

  TEST_CASE("SplatSet::reorderByMortonCode reorders by spatial locality")
  {
    SplatSet set;
    
    // Create 4 splats in deliberately wrong order
    // Splat 0: far corner (1, 1, 1)
    // Splat 1: origin (0, 0, 0) - should come first
    // Splat 2: middle (0.5, 0.5, 0.5)
    // Splat 3: near origin (0.1, 0.1, 0.1)
    set.positions = {
      1.0f, 1.0f, 1.0f,     // splat 0
      0.0f, 0.0f, 0.0f,     // splat 1
      0.5f, 0.5f, 0.5f,     // splat 2
      0.1f, 0.1f, 0.1f      // splat 3
    };
    set.f_dc = {
      0.0f, 0.0f, 0.0f,
      0.1f, 0.1f, 0.1f,
      0.2f, 0.2f, 0.2f,
      0.3f, 0.3f, 0.3f
    };
    set.opacity = {0.0f, 0.1f, 0.2f, 0.3f};
    set.scale = {
      1.0f, 1.0f, 1.0f,
      1.1f, 1.1f, 1.1f,
      1.2f, 1.2f, 1.2f,
      1.3f, 1.3f, 1.3f
    };
    set.rotation = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.9f, 0.1f, 0.0f, 0.0f,
      0.8f, 0.2f, 0.0f, 0.0f,
      0.7f, 0.3f, 0.0f, 0.0f
    };

    set.reorderByMortonCode();

    CHECK(set.size() == 4);

    // After Morton reordering:
    // Origin (0,0,0) should come first - Morton code 0
    // Near origin (0.1,0.1,0.1) should be second
    // Middle (0.5,0.5,0.5) should be third
    // Far corner (1,1,1) should be last - Morton code max

    // Check first splat is the origin (was splat 1)
    CHECK(set.positions[0] == 0.0f);
    CHECK(set.positions[1] == 0.0f);
    CHECK(set.positions[2] == 0.0f);
    CHECK(set.opacity[0] == 0.1f);

    // Check last splat is the far corner (was splat 0)
    CHECK(set.positions[9] == 1.0f);
    CHECK(set.positions[10] == 1.0f);
    CHECK(set.positions[11] == 1.0f);
    CHECK(set.opacity[3] == 0.0f);
  }

  TEST_CASE("SplatSet::reorderByMortonCode preserves all attributes")
  {
    SplatSet set;
    
    // Two splats to swap
    set.positions = {10.0f, 10.0f, 10.0f, 0.0f, 0.0f, 0.0f};
    set.f_dc = {1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f};
    set.f_rest = {10.0f, 11.0f, 12.0f, 0.0f, 1.0f, 2.0f};  // 3 SH coeffs each
    set.opacity = {0.9f, 0.1f};
    set.scale = {2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f};
    set.rotation = {0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f};

    set.reorderByMortonCode();

    // After reordering, origin (splat 1) should be first
    CHECK(set.positions[0] == 0.0f);
    CHECK(set.f_dc[0] == 0.0f);
    CHECK(set.f_dc[1] == 0.1f);
    CHECK(set.f_dc[2] == 0.2f);
    CHECK(set.f_rest[0] == 0.0f);
    CHECK(set.f_rest[1] == 1.0f);
    CHECK(set.f_rest[2] == 2.0f);
    CHECK(set.opacity[0] == 0.1f);
    CHECK(set.scale[0] == 1.0f);
    CHECK(set.scale[1] == 1.1f);
    CHECK(set.scale[2] == 1.2f);
    CHECK(set.rotation[0] == 1.0f);
    CHECK(set.rotation[1] == 0.0f);
    CHECK(set.rotation[2] == 0.0f);
    CHECK(set.rotation[3] == 0.0f);

    // Second splat should be the far one
    CHECK(set.positions[3] == 10.0f);
    CHECK(set.opacity[1] == 0.9f);
  }

  TEST_CASE("SplatSet::reorderByMortonCode handles temporal data")
  {
    SplatSet set;
    
    set.positions = {10.0f, 10.0f, 10.0f, 0.0f, 0.0f, 0.0f};
    set.f_dc = {1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f};
    set.opacity = {0.9f, 0.1f};
    set.scale = {2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f};
    set.rotation = {0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f};
    
    set.has_time_data = true;
    set.motion = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
    set.time = {0.8f, 0.2f};
    set.time_scale = {0.5f, 0.1f};

    set.reorderByMortonCode();

    // Origin splat should be first with its temporal data
    CHECK(set.motion[0] == 0.1f);
    CHECK(set.motion[1] == 0.2f);
    CHECK(set.motion[2] == 0.3f);
    CHECK(set.time[0] == 0.2f);
    CHECK(set.time_scale[0] == 0.1f);

    // Far splat's temporal data should be second
    CHECK(set.motion[3] == 1.0f);
    CHECK(set.time[1] == 0.8f);
    CHECK(set.time_scale[1] == 0.5f);
  }

  TEST_CASE("SplatSet::reorderByMortonCode handles negative coordinates")
  {
    SplatSet set;
    
    set.positions = {5.0f, 5.0f, 5.0f, -5.0f, -5.0f, -5.0f};
    set.f_dc = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    set.opacity = {1.0f, 0.0f};
    set.scale = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    set.rotation = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};

    set.reorderByMortonCode();

    // Negative coordinates should come first (lower Morton code)
    CHECK(set.positions[0] == -5.0f);
    CHECK(set.positions[1] == -5.0f);
    CHECK(set.positions[2] == -5.0f);
    CHECK(set.opacity[0] == 0.0f);
  }
}
