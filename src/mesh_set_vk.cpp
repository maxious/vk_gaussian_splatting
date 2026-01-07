/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mesh_set_vk.h"
#include "utilities.h"
#include "gltf_loader.h"
#include <algorithm>

//#define STB_IMAGE_IMPLEMENTATION
//#include <stb/stb_image.h>

#include <nvvk/debug_util.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

#include <meshoptimizer.h>

namespace vk_gaussian_splatting {

bool MeshSetVk::loadModel(const std::filesystem::path& filename)
{
  LOGI("Loading File:  %s \n", filename.string().c_str());
  
  struct ModelData {
      std::filesystem::path    filename;
      std::vector<ObjVertex>   m_vertices;
      std::vector<uint32_t>    m_indices;
      std::vector<ObjMaterial> m_materials;
      std::vector<std::string> m_matNames;
      std::vector<std::string> m_textures;
      std::vector<int32_t>     m_matIndices;
  };

  ModelData loadedData;
  bool loaded = false;

  std::string ext = filename.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  if (ext == ".gltf" || ext == ".glb")
  {
      GltfLoader loader;
      loaded = loader.load(filename);
      if(loaded) {
          loadedData.filename = loader.filename;
          loadedData.m_vertices = std::move(loader.m_vertices);
          loadedData.m_indices = std::move(loader.m_indices);
          loadedData.m_materials = std::move(loader.m_materials);
          loadedData.m_matNames = std::move(loader.m_matNames);
          loadedData.m_textures = std::move(loader.m_textures);
          loadedData.m_matIndices = std::move(loader.m_matIndices);
      }
  }
  else
  {
      ObjLoader loader;
      loaded = loader.load(filename);
      if(loaded) {
          loadedData.filename = loader.filename;
          loadedData.m_vertices = std::move(loader.m_vertices);
          loadedData.m_indices = std::move(loader.m_indices); 
          loadedData.m_materials = std::move(loader.m_materials);
          loadedData.m_matNames = std::move(loader.m_matNames);
          loadedData.m_textures = std::move(loader.m_textures);
          loadedData.m_matIndices = std::move(loader.m_matIndices);
      }
  }

  if(!loaded)
    return false;

  for(auto& m : loadedData.m_materials)
  {
    m.ambient  = glm::pow(m.ambient, glm::vec3(2.2f));
    m.diffuse  = glm::pow(m.diffuse, glm::vec3(2.2f));
    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
  }

  {
    SCOPED_TIMER("Mesh optimization");
    
    const size_t vertexCount = loadedData.m_vertices.size();
    const size_t indexCount  = loadedData.m_indices.size();
    
    std::vector<uint32_t> optimizedIndices(indexCount);
    meshopt_optimizeVertexCache(optimizedIndices.data(), loadedData.m_indices.data(), indexCount, vertexCount);
    
    const float* positions = reinterpret_cast<const float*>(loadedData.m_vertices.data());
    meshopt_optimizeOverdraw(optimizedIndices.data(), optimizedIndices.data(), indexCount, 
                             positions, vertexCount, sizeof(ObjVertex), 1.05f);
    
    std::vector<ObjVertex> optimizedVertices(vertexCount);
    size_t uniqueVertices = meshopt_optimizeVertexFetch(optimizedVertices.data(), optimizedIndices.data(), 
                                                        indexCount, loadedData.m_vertices.data(), 
                                                        vertexCount, sizeof(ObjVertex));
    
    loadedData.m_vertices = std::move(optimizedVertices);
    loadedData.m_indices  = std::move(optimizedIndices);
    
    LOGI("  Mesh optimized: %zu vertices (%zu unique), %zu indices\n", vertexCount, uniqueVertices, indexCount);
  }

  Mesh model;
  model.path       = loadedData.filename.string();
  model.name       = loadedData.filename.filename().string();
  model.nbIndices  = static_cast<uint32_t>(loadedData.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loadedData.m_vertices.size());
  model.materials  = loadedData.m_materials;
  model.matNames   = loadedData.m_matNames;

  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags = 
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  NVVK_CHECK(m_alloc->createBuffer(model.vertexBuffer, loadedData.m_vertices.size() * sizeof(ObjVertex),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(model.vertexBuffer.buffer);

  NVVK_CHECK(m_alloc->createBuffer(model.indexBuffer, loadedData.m_indices.size() * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(model.indexBuffer.buffer);

  NVVK_CHECK(m_alloc->createBuffer(model.materialsBuffer, loadedData.m_materials.size() * sizeof(ObjMaterial),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(model.materialsBuffer.buffer);

  NVVK_CHECK(m_alloc->createBuffer(model.matIndexBuffer, loadedData.m_matIndices.size() * sizeof(uint32_t),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(model.matIndexBuffer.buffer);


  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(model.vertexBuffer, 0, std::span(loadedData.m_vertices)));
  NVVK_CHECK(m_uploader->appendBuffer(model.indexBuffer, 0, std::span(loadedData.m_indices)));
  NVVK_CHECK(m_uploader->appendBuffer(model.materialsBuffer, 0, std::span(loadedData.m_materials)));
  NVVK_CHECK(m_uploader->appendBuffer(model.matIndexBuffer, 0, std::span(loadedData.m_matIndices)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();

  Instance instance;
  instance.transform = glm::mat4(1);
  instance.objIndex  = static_cast<uint32_t>(meshes.size());
  computeTransform(instance.scale, instance.rotation, instance.translation, instance.transform, instance.transformInverse);
  instances.push_back(instance);

  shaderio::ObjDesc desc;
  desc.vertexAddress        = (shaderio::ObjVertex*)model.vertexBuffer.address;
  desc.indexAddress         = (uint32_t*)model.indexBuffer.address;
  desc.materialAddress      = (shaderio::ObjMaterial*)model.materialsBuffer.address;
  desc.materialIndexAddress = (uint32_t*)model.matIndexBuffer.address;

  meshes.emplace_back(model);
  objectDescriptions.emplace_back(desc);

  return true;
}

void MeshSetVk::updateObjDescriptionBuffer()
{

  if(objectDescriptionsBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(objectDescriptionsBuffer);

  if(objectDescriptions.empty())
    return;


  NVVK_CHECK(m_alloc->createBuffer(objectDescriptionsBuffer, objectDescriptions.size() * sizeof(shaderio::ObjDesc),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
  NVVK_DBG_NAME(objectDescriptionsBuffer.buffer);


  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(objectDescriptionsBuffer, 0, std::span(objectDescriptions)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();
}

void MeshSetVk::updateObjMaterialsBuffer(int modelIndex)
{

  const auto& model = meshes[modelIndex];

  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(model.materialsBuffer, 0, std::span(model.materials)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();
}

nvvk::AccelerationStructureGeometryInfo MeshSetVk::rtxCreateMeshVkKHR(const Mesh& model)
{
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
  triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = model.vertexBuffer.address;
  triangles.vertexStride             = sizeof(ObjVertex);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = model.indexBuffer.address;
  triangles.transformData            = {};  
  triangles.maxVertex                = model.nbVertices - 1;

  VkAccelerationStructureGeometryKHR geometry{};
  geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
  geometry.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
  rangeInfo.firstVertex     = 0;
  rangeInfo.primitiveCount  = model.nbIndices / 3;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.transformOffset = 0;

  return nvvk::AccelerationStructureGeometryInfo{.geometry = geometry, .rangeInfo = rangeInfo};
}

void MeshSetVk::rtxInitAccelerationStructures()
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  if(!meshes.empty())
  {
    std::vector<nvvk::AccelerationStructureGeometryInfo> asGeoInfo;
    asGeoInfo.reserve(meshes.size());

    for(const auto& obj : meshes)
    {
      asGeoInfo.emplace_back(rtxCreateMeshVkKHR(obj));
    }
    rtAccelerationStructures.blasSubmitBuildAndWait(asGeoInfo, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                   | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR
                                                                   | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR);

    LOGI("%s%s\n", nvutils::ScopedTimer::indent().c_str(), rtAccelerationStructures.blasBuildStatistics.toString().c_str());
  }

  if(!instances.empty())
  {
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(instances.size());
    for(const Instance& objInst : instances)
    {
      VkAccelerationStructureInstanceKHR asInst{};
      asInst.transform           = nvvk::toTransformMatrixKHR(objInst.transform);  
      asInst.instanceCustomIndex = objInst.objIndex;                               
      asInst.accelerationStructureReference         = rtAccelerationStructures.blasSet[objInst.objIndex].address;
      asInst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      asInst.mask                                   = 0xFF;  
      asInst.instanceShaderBindingTableRecordOffset = 1;  
      tlasInstances.emplace_back(asInst);
    }
    rtAccelerationStructures.tlasSubmitBuildAndWait(tlasInstances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                       | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
  }
}

void MeshSetVk::rtxUpdateTopLevelAccelerationStructure()
{
  if(!instances.empty())
  {
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(instances.size());
    for(const Instance& objInst : instances)
    {
      VkAccelerationStructureInstanceKHR asInst{};
      asInst.transform           = nvvk::toTransformMatrixKHR(objInst.transform);  
      asInst.instanceCustomIndex = objInst.objIndex;                               
      asInst.accelerationStructureReference         = rtAccelerationStructures.blasSet[objInst.objIndex].address;
      asInst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      asInst.mask                                   = 0xFF;  
      asInst.instanceShaderBindingTableRecordOffset = 1;  
      tlasInstances.emplace_back(asInst);
    }
    rtAccelerationStructures.tlasSubmitUpdateAndWait(tlasInstances);
  }
}

void MeshSetVk::deleteInstance(uint32_t instanceId)
{
  bool       lastInstanceUsingMesh = true;
  const auto meshIndex             = instances[instanceId].objIndex;

  for(uint32_t i = 0; i < instances.size(); ++i)
  {
    const auto& instance = instances[i];
    if(instanceId != i && instance.objIndex == meshIndex)
    {
      lastInstanceUsingMesh = false;
      break;
    }
  }

  instances.erase(instances.begin() + instanceId);
  objectDescriptions.erase(objectDescriptions.begin() + instanceId);

  if(lastInstanceUsingMesh)
  {
    deinitMeshBuffers(meshes[meshIndex]);
    meshes.erase(meshes.begin() + meshIndex);
    for(auto& instance : instances)
    {
      if(instance.objIndex >= meshIndex)
      {
        --instance.objIndex;
      }
    }
  }
}

}  // namespace vk_gaussian_splatting
