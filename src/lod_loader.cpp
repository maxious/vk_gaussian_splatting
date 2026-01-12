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

#include "lod_loader.h"
#include "sog_loader.h"

#include <fstream>
#include <cmath>
#include <algorithm>
#include <set>

#include <nvutils/logger.hpp>
#include <tinygltf/json.hpp>

using nlohmann::json;
using namespace vk_viewer;

namespace {

std::vector<uint8_t> readFile(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file.is_open())
    return {};

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    return {};

  return buffer;
}

}  // namespace

bool LodAabb::intersectsFrustum(const glm::mat4& viewProj) const
{
  glm::vec3 corners[8] = {
      {min.x, min.y, min.z}, {max.x, min.y, min.z}, {min.x, max.y, min.z}, {max.x, max.y, min.z},
      {min.x, min.y, max.z}, {max.x, min.y, max.z}, {min.x, max.y, max.z}, {max.x, max.y, max.z},
  };

  for(int plane = 0; plane < 6; ++plane)
  {
    int outside = 0;
    for(int c = 0; c < 8; ++c)
    {
      glm::vec4 p   = viewProj * glm::vec4(corners[c], 1.0f);
      bool      out = false;
      switch(plane)
      {
        case 0:
          out = p.x < -p.w;
          break;  // left
        case 1:
          out = p.x > p.w;
          break;  // right
        case 2:
          out = p.y < -p.w;
          break;  // bottom
        case 3:
          out = p.y > p.w;
          break;  // top
        case 4:
          out = p.z < 0;
          break;  // near
        case 5:
          out = p.z > p.w;
          break;  // far
      }
      if(out)
        outside++;
    }
    if(outside == 8)
      return false;
  }
  return true;
}

bool LodScene::parseMeta(const std::vector<uint8_t>& jsonData, LodMeta& meta)
{
  try
  {
    json j = json::parse(jsonData.begin(), jsonData.end());

    meta.lodLevels = j.value("lodLevels", 0u);
    if(meta.lodLevels == 0)
    {
      LOGE("LOD meta: lodLevels must be > 0\n");
      return false;
    }

    meta.environment = j.value("environment", "");

    if(j.contains("filenames") && j["filenames"].is_array())
    {
      for(const auto& f : j["filenames"])
      {
        meta.filenames.push_back(f.get<std::string>());
      }
    }
    else
    {
      LOGE("LOD meta: filenames array is required\n");
      return false;
    }

    if(j.contains("tree"))
    {
      if(!parseNode(&j["tree"], meta.tree))
      {
        LOGE("LOD meta: failed to parse tree\n");
        return false;
      }
      meta.sceneBounds = meta.tree.bound;
    }
    else
    {
      LOGE("LOD meta: tree is required\n");
      return false;
    }

    LOGI("LOD meta: %u levels, %zu files, bounds: (%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)\n", meta.lodLevels,
         meta.filenames.size(), meta.sceneBounds.min.x, meta.sceneBounds.min.y, meta.sceneBounds.min.z,
         meta.sceneBounds.max.x, meta.sceneBounds.max.y, meta.sceneBounds.max.z);

    return true;
  }
  catch(const std::exception& e)
  {
    LOGE("Failed to parse LOD meta.json: %s\n", e.what());
    return false;
  }
}

bool LodScene::parseNode(const void* jsonNodePtr, LodNode& node)
{
  const json& j = *static_cast<const json*>(jsonNodePtr);

  if(j.contains("bound"))
  {
    const auto& b = j["bound"];
    if(b.contains("min") && b["min"].is_array() && b["min"].size() >= 3)
    {
      node.bound.min.x = b["min"][0].get<float>();
      node.bound.min.y = b["min"][1].get<float>();
      node.bound.min.z = b["min"][2].get<float>();
    }
    if(b.contains("max") && b["max"].is_array() && b["max"].size() >= 3)
    {
      node.bound.max.x = b["max"][0].get<float>();
      node.bound.max.y = b["max"][1].get<float>();
      node.bound.max.z = b["max"][2].get<float>();
    }
  }

  if(j.contains("children") && j["children"].is_array())
  {
    for(const auto& child : j["children"])
    {
      LodNode childNode;
      if(parseNode(&child, childNode))
      {
        node.children.push_back(std::move(childNode));
      }
    }
  }

  if(j.contains("lods") && j["lods"].is_object())
  {
    for(auto& [key, value] : j["lods"].items())
    {
      uint32_t   level = static_cast<uint32_t>(std::stoul(key));
      LodDataRef ref;
      ref.fileIndex = value.value("file", 0u);
      ref.offset    = value.value("offset", 0u);
      ref.count     = value.value("count", 0u);
      node.lods[level] = ref;
    }
  }

  return true;
}

void LodScene::collectLeafNodes(const LodNode& node, std::vector<const LodNode*>& leaves)
{
  if(node.isLeaf())
  {
    leaves.push_back(&node);
  }
  else
  {
    for(const auto& child : node.children)
    {
      collectLeafNodes(child, leaves);
    }
  }
}

uint32_t LodScene::computeLodForDistance(float distance, uint32_t maxLod, float baseDistance)
{
  if(distance <= 0.0f || maxLod == 0)
    return 0;

  float ratio = distance / baseDistance;
  uint32_t lod = static_cast<uint32_t>(std::log2(std::max(1.0f, ratio)));
  return std::min(lod, maxLod - 1);
}

uint32_t LodScene::computeLodForScreenSize(float screenSize, uint32_t maxLod, float threshold)
{
  if(screenSize >= threshold || maxLod == 0)
    return 0;

  float ratio = threshold / std::max(0.001f, screenSize);
  uint32_t lod = static_cast<uint32_t>(std::log2(std::max(1.0f, ratio)));
  return std::min(lod, maxLod - 1);
}

void LodScene::selectLodsRecursive(const LodNode&             node,
                                   const glm::vec3&           cameraPos,
                                   const glm::mat4&           viewProj,
                                   LodSelectionPolicy         policy,
                                   uint32_t                   fixedLevel,
                                   float                      screenSizeThreshold,
                                   std::vector<LodSelection>& selections)
{
  if(!node.bound.intersectsFrustum(viewProj))
    return;

  if(node.isLeaf())
  {
    if(node.lods.empty())
      return;

    uint32_t selectedLod = 0;
    float    distance    = glm::length(node.bound.center() - cameraPos);

    switch(policy)
    {
      case LodSelectionPolicy::FIXED_LEVEL:
        selectedLod = fixedLevel;
        break;
      case LodSelectionPolicy::DISTANCE_BASED: {
        uint32_t maxLod = 0;
        for(const auto& [lod, ref] : node.lods)
        {
          maxLod = std::max(maxLod, lod + 1);
        }
        selectedLod = computeLodForDistance(distance, maxLod, node.bound.radius() * 4.0f);
        break;
      }
      case LodSelectionPolicy::SCREEN_SIZE: {
        uint32_t maxLod = 0;
        for(const auto& [lod, ref] : node.lods)
        {
          maxLod = std::max(maxLod, lod + 1);
        }
        float screenSize = node.bound.radius() / std::max(0.001f, distance);
        selectedLod      = computeLodForScreenSize(screenSize, maxLod, screenSizeThreshold);
        break;
      }
    }

    auto it = node.lods.find(selectedLod);
    if(it == node.lods.end())
    {
      it = node.lods.begin();
      uint32_t bestDiff = std::abs(static_cast<int>(it->first) - static_cast<int>(selectedLod));
      for(auto jt = node.lods.begin(); jt != node.lods.end(); ++jt)
      {
        uint32_t diff = std::abs(static_cast<int>(jt->first) - static_cast<int>(selectedLod));
        if(diff < bestDiff)
        {
          bestDiff = diff;
          it       = jt;
        }
      }
    }

    if(it != node.lods.end())
    {
      LodSelection sel;
      sel.lodLevel  = it->first;
      sel.fileIndex = it->second.fileIndex;
      sel.offset    = it->second.offset;
      sel.count     = it->second.count;
      selections.push_back(sel);
    }
  }
  else
  {
    for(const auto& child : node.children)
    {
      selectLodsRecursive(child, cameraPos, viewProj, policy, fixedLevel, screenSizeThreshold, selections);
    }
  }
}

bool LodScene::load(const std::filesystem::path& metaPath, LodMeta& meta, std::function<void(float)> progressCallback)
{
  std::vector<uint8_t> jsonData = readFile(metaPath);
  if(jsonData.empty())
  {
    LOGE("Failed to read LOD meta file: %s\n", metaPath.string().c_str());
    return false;
  }

  if(progressCallback)
    progressCallback(0.1f);

  if(!parseMeta(jsonData, meta))
  {
    return false;
  }

  if(progressCallback)
    progressCallback(1.0f);

  return true;
}

std::vector<LodSelection> LodScene::selectLods(const LodMeta&     meta,
                                               const glm::vec3&   cameraPos,
                                               const glm::mat4&   viewProj,
                                               LodSelectionPolicy policy,
                                               uint32_t           fixedLevel,
                                               float              screenSizeThreshold)
{
  std::vector<LodSelection> selections;
  selectLodsRecursive(meta.tree, cameraPos, viewProj, policy, fixedLevel, screenSizeThreshold, selections);
  return selections;
}

bool LodScene::loadSelections(const std::filesystem::path&     basePath,
                              const LodMeta&                   meta,
                              const std::vector<LodSelection>& selections,
                              SplatSet&                        output,
                              std::function<void(float)>       progressCallback)
{
  if(selections.empty())
  {
    LOGW("No LOD selections to load\n");
    return true;
  }

  std::set<uint32_t> uniqueFiles;
  for(const auto& sel : selections)
  {
    uniqueFiles.insert(sel.fileIndex);
  }

  std::map<uint32_t, SplatSet> loadedFiles;
  size_t                       filesLoaded = 0;

  for(uint32_t fileIdx : uniqueFiles)
  {
    if(fileIdx >= meta.filenames.size())
    {
      LOGE("Invalid file index %u in LOD selection\n", fileIdx);
      continue;
    }

    std::filesystem::path filePath = basePath / meta.filenames[fileIdx];
    SplatSet              fileSplats;

    bool success = SogLoader::load(filePath, fileSplats, nullptr);
    if(!success)
    {
      LOGE("Failed to load LOD file: %s\n", filePath.string().c_str());
      continue;
    }

    loadedFiles[fileIdx] = std::move(fileSplats);
    filesLoaded++;

    if(progressCallback)
    {
      progressCallback(static_cast<float>(filesLoaded) / static_cast<float>(uniqueFiles.size()) * 0.8f);
    }
  }

  output.clear();

  for(const auto& sel : selections)
  {
    auto it = loadedFiles.find(sel.fileIndex);
    if(it == loadedFiles.end())
      continue;

    const SplatSet& src       = it->second;
    uint32_t        srcOffset = sel.offset;
    uint32_t        srcCount  = sel.count;

    if(srcOffset + srcCount > src.size())
    {
      LOGW("LOD selection exceeds file bounds: offset=%u, count=%u, size=%zu\n", srcOffset, srcCount, src.size());
      srcCount = static_cast<uint32_t>(src.size()) - srcOffset;
    }

    SplatSet subset;
    subset.positions.insert(subset.positions.end(), src.positions.begin() + srcOffset * 3,
                            src.positions.begin() + (srcOffset + srcCount) * 3);
    subset.f_dc.insert(subset.f_dc.end(), src.f_dc.begin() + srcOffset * 3, src.f_dc.begin() + (srcOffset + srcCount) * 3);

    size_t shPerSplat = src.f_rest.size() / src.size();
    subset.f_rest.insert(subset.f_rest.end(), src.f_rest.begin() + srcOffset * shPerSplat,
                         src.f_rest.begin() + (srcOffset + srcCount) * shPerSplat);

    subset.opacity.insert(subset.opacity.end(), src.opacity.begin() + srcOffset,
                          src.opacity.begin() + srcOffset + srcCount);
    subset.scale.insert(subset.scale.end(), src.scale.begin() + srcOffset * 3,
                        src.scale.begin() + (srcOffset + srcCount) * 3);
    subset.rotation.insert(subset.rotation.end(), src.rotation.begin() + srcOffset * 4,
                           src.rotation.begin() + (srcOffset + srcCount) * 4);

    if(src.has_time_data)
    {
      subset.has_time_data = true;
      subset.motion.insert(subset.motion.end(), src.motion.begin() + srcOffset * 3,
                           src.motion.begin() + (srcOffset + srcCount) * 3);
      subset.time.insert(subset.time.end(), src.time.begin() + srcOffset, src.time.begin() + srcOffset + srcCount);
      subset.time_scale.insert(subset.time_scale.end(), src.time_scale.begin() + srcOffset,
                               src.time_scale.begin() + srcOffset + srcCount);
    }

    output.merge(subset);
  }

  if(progressCallback)
    progressCallback(1.0f);

  LOGI("Loaded LOD scene: %zu splats from %zu selections\n", output.size(), selections.size());
  return true;
}

bool LodScene::loadAtLevel(const std::filesystem::path& metaPath,
                           uint32_t                     lodLevel,
                           SplatSet&                    output,
                           std::function<void(float)>   progressCallback)
{
  LodMeta meta;
  if(!load(metaPath, meta, nullptr))
  {
    return false;
  }

  if(progressCallback)
    progressCallback(0.1f);

  std::vector<const LodNode*> leaves;
  collectLeafNodes(meta.tree, leaves);

  std::vector<LodSelection> selections;
  for(const LodNode* leaf : leaves)
  {
    auto it = leaf->lods.find(lodLevel);
    if(it == leaf->lods.end() && !leaf->lods.empty())
    {
      it = leaf->lods.begin();
      for(auto jt = leaf->lods.begin(); jt != leaf->lods.end(); ++jt)
      {
        if(jt->first <= lodLevel)
          it = jt;
      }
    }

    if(it != leaf->lods.end())
    {
      LodSelection sel;
      sel.lodLevel  = it->first;
      sel.fileIndex = it->second.fileIndex;
      sel.offset    = it->second.offset;
      sel.count     = it->second.count;
      selections.push_back(sel);
    }
  }

  if(progressCallback)
    progressCallback(0.2f);

  return loadSelections(metaPath.parent_path(), meta, selections, output,
                        [&](float p) {
                          if(progressCallback)
                            progressCallback(0.2f + p * 0.8f);
                        });
}
