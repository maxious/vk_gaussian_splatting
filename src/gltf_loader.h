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
 */

#pragma once

#include "obj_loader.h"
#include <filesystem>

class GltfLoader
{
public:
  bool load(const std::filesystem::path& filename);

  std::filesystem::path    filename;
  std::vector<ObjVertex>   m_vertices;
  std::vector<uint32_t>    m_indices;
  std::vector<ObjMaterial> m_materials;
  std::vector<std::string> m_matNames;
  std::vector<std::string> m_textures;
  std::vector<int32_t>     m_matIndices;
};
