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

#include "gltf_loader.h"
#include <nvutils/logger.hpp>
#include <nvvkgltf/tinygltf_utils.hpp>

// Helper to convert GLTF to our ObjVertex/ObjMaterial
bool GltfLoader::load(const std::filesystem::path& filename)
{
  tinygltf::Model    model;
  tinygltf::TinyGLTF loader;
  std::string        err;
  std::string        warn;

  bool ret = false;
  if(filename.extension() == ".glb")
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.string());
  else
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.string());

  if(!warn.empty())
    LOGW("GLTF Warn: %s\n", warn.c_str());

  if(!err.empty())
    LOGE("GLTF Error: %s\n", err.c_str());

  if(!ret)
    return false;

  // Convert Materials
  m_materials.reserve(model.materials.size());
  m_matNames.reserve(model.materials.size());
  for(const auto& mat : model.materials)
  {
    ObjMaterial m;
    m.ambient       = glm::vec3(0.1f);
    m.diffuse       = glm::vec3(mat.pbrMetallicRoughness.baseColorFactor[0],
                          mat.pbrMetallicRoughness.baseColorFactor[1],
                          mat.pbrMetallicRoughness.baseColorFactor[2]);
    m.specular      = glm::vec3(0.5f); // Approximation
    m.emission      = glm::vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
    m.transmittance = glm::vec3(0.0f);
    m.dissolve      = 1.0f; // Opacity
    m.ior           = 1.5f;
    m.shininess     = 10.0f; // Roughness approximation?
    m.illum         = 1; 

    // Textures
    int baseColorIndex = mat.pbrMetallicRoughness.baseColorTexture.index;
    if(baseColorIndex >= 0)
    {
      const auto& tex = model.textures[baseColorIndex];
      const auto& img = model.images[tex.source];
      if(!img.uri.empty())
      {
          m_textures.push_back(img.uri);
          m.textureID = static_cast<int>(m_textures.size()) - 1;
      }
    }

    m_materials.push_back(m);
    m_matNames.push_back(mat.name);
  }

  if(m_materials.empty())
  {
    m_materials.push_back(ObjMaterial());
    m_matNames.push_back("Default");
  }

  // Iterate over meshes and primitives
  for(const auto& mesh : model.meshes)
  {
    for(const auto& primitive : mesh.primitives)
    {
        // Material index
        int matId = primitive.material;
        if (matId < 0) matId = 0;

        // Accessors
        const float* positionBuffer = nullptr;
        const float* normalBuffer = nullptr;
        // const float* texCoordBuffer = nullptr;
        size_t vertexCount = 0;

        // Position
        if (primitive.attributes.find("POSITION") != primitive.attributes.end())
        {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.find("POSITION")->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            positionBuffer = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            vertexCount = accessor.count;
        }

        // Normal
        if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
        {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.find("NORMAL")->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            normalBuffer = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
        }

        size_t indexOffset = m_vertices.size();

        // Add vertices
        for (size_t v = 0; v < vertexCount; v++)
        {
            ObjVertex vertex = {};
            vertex.pos = {positionBuffer[v * 3 + 0], positionBuffer[v * 3 + 1], positionBuffer[v * 3 + 2]};
            
            if (normalBuffer)
                vertex.nrm = {normalBuffer[v * 3 + 0], normalBuffer[v * 3 + 1], normalBuffer[v * 3 + 2]};
            
            m_vertices.push_back(vertex);
        }

        // Indices
        if (primitive.indices >= 0)
        {
            const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            
            if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
            {
                const uint16_t* buf = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; i++)
                {
                    m_indices.push_back(indexOffset + buf[i]);
                }
            }
            else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
            {
                const uint32_t* buf = reinterpret_cast<const uint32_t*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; i++)
                {
                    m_indices.push_back(indexOffset + buf[i]);
                }
            }
            
            // For primitive based materials, we need to replicate matId for each Triangle
            for(size_t i=0; i < accessor.count / 3; i++)
            {
                m_matIndices.push_back(matId);
            }
        }
    }
  }

  return true;
}
