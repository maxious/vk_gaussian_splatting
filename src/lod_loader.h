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

#ifndef _LOD_LOADER_H_
#define _LOD_LOADER_H_

#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

#include <glm/glm.hpp>

#include "splat_set.h"

namespace vk_viewer {

// Axis-aligned bounding box
struct LodAabb
{
  glm::vec3 min{0.0f};
  glm::vec3 max{0.0f};

  glm::vec3 center() const { return (min + max) * 0.5f; }
  glm::vec3 extent() const { return max - min; }
  float     radius() const { return glm::length(extent()) * 0.5f; }

  bool contains(const glm::vec3& point) const
  {
    return point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y && point.z >= min.z
           && point.z <= max.z;
  }

  bool intersectsFrustum(const glm::mat4& viewProj) const;
};

// Reference to Gaussian data stored in a file
struct LodDataRef
{
  uint32_t fileIndex = 0;  // Index into the filenames array
  uint32_t offset    = 0;  // Offset within the file (in Gaussians)
  uint32_t count     = 0;  // Number of Gaussians at this LOD level
};

// Hierarchical octree node for LOD
struct LodNode
{
  LodAabb                        bound;                // Bounding box for this node
  std::vector<LodNode>           children;             // Child nodes (BSP: 2 children)
  std::map<uint32_t, LodDataRef> lods;                 // Map of LOD level -> file reference
  bool                           isLeaf() const { return children.empty(); }
};

// LOD Scene metadata (parsed from lod-meta.json)
struct LodMeta
{
  uint32_t                 lodLevels = 0;      // Total number of LOD levels
  std::string              environment;        // Optional environment splat path
  std::vector<std::string> filenames;          // List of data file paths (SOG meta.json files)
  LodNode                  tree;               // Root node of the hierarchical tree
  LodAabb                  sceneBounds;        // Computed scene bounding box
};

// LOD selection result for a single node
struct LodSelection
{
  uint32_t lodLevel  = 0;
  uint32_t fileIndex = 0;
  uint32_t offset    = 0;
  uint32_t count     = 0;
};

// LOD selection policy
enum class LodSelectionPolicy
{
  FIXED_LEVEL,      // Use a specific LOD level for all nodes
  DISTANCE_BASED,   // Select LOD based on camera distance
  SCREEN_SIZE       // Select LOD based on projected screen size
};

// LOD Scene - manages multi-level Gaussian splat data with spatial hierarchy
class LodScene
{
public:
  // Load LOD scene from lod-meta.json
  // progressCallback: optional callback for progress updates (0.0 to 1.0)
  static bool load(const std::filesystem::path&   metaPath,
                   LodMeta&                        meta,
                   std::function<void(float)>      progressCallback = nullptr);

  // Select visible nodes and appropriate LOD levels based on camera
  static std::vector<LodSelection> selectLods(const LodMeta&     meta,
                                              const glm::vec3&   cameraPos,
                                              const glm::mat4&   viewProj,
                                              LodSelectionPolicy policy,
                                              uint32_t           fixedLevel     = 0,
                                              float              screenSizeThreshold = 1.0f);

  // Load Gaussians for specific selections into a SplatSet
  // This loads the SOG data for the selected LOD regions
  static bool loadSelections(const std::filesystem::path&        basePath,
                             const LodMeta&                      meta,
                             const std::vector<LodSelection>&    selections,
                             SplatSet&                           output,
                             std::function<void(float)>          progressCallback = nullptr);

  // Convenience: load entire scene at a specific LOD level
  static bool loadAtLevel(const std::filesystem::path&   metaPath,
                          uint32_t                       lodLevel,
                          SplatSet&                      output,
                          std::function<void(float)>     progressCallback = nullptr);

private:
  // Parse lod-meta.json content
  static bool parseMeta(const std::vector<uint8_t>& jsonData, LodMeta& meta);

  // Recursively parse a node from JSON
  static bool parseNode(const void* jsonNode, LodNode& node);

  // Recursively collect all leaf nodes
  static void collectLeafNodes(const LodNode& node, std::vector<const LodNode*>& leaves);

  // Recursively select LODs for visible nodes
  static void selectLodsRecursive(const LodNode&             node,
                                  const glm::vec3&           cameraPos,
                                  const glm::mat4&           viewProj,
                                  LodSelectionPolicy         policy,
                                  uint32_t                   fixedLevel,
                                  float                      screenSizeThreshold,
                                  std::vector<LodSelection>& selections);

  // Compute LOD level based on distance
  static uint32_t computeLodForDistance(float distance, uint32_t maxLod, float baseDistance = 10.0f);

  // Compute LOD level based on screen size
  static uint32_t computeLodForScreenSize(float screenSize, uint32_t maxLod, float threshold);
};

}  // namespace vk_viewer

#endif
