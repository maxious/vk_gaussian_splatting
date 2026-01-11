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

namespace vk_viewer {

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
      std::vector<TextureData> m_textureData;
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
          loadedData.m_textureData = std::move(loader.m_textureData);
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

  LOGI("Mesh loaded: %zu vertices, %zu indices, %zu materials, %zu textures\n",
       loadedData.m_vertices.size(), loadedData.m_indices.size(),
       loadedData.m_materials.size(), loadedData.m_textures.size());

  for(auto& m : loadedData.m_materials)
  {
    m.ambient  = glm::vec4(glm::pow(glm::vec3(m.ambient), glm::vec3(2.2f)), 0.0f);
    m.diffuse  = glm::vec4(glm::pow(glm::vec3(m.diffuse), glm::vec3(2.2f)), 0.0f);
    m.specular = glm::vec4(glm::pow(glm::vec3(m.specular), glm::vec3(2.2f)), 0.0f);
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

  // Compute bounding box
  glm::vec3 bboxMin(FLT_MAX), bboxMax(-FLT_MAX);
  for(const auto& v : loadedData.m_vertices)
  {
    bboxMin = glm::min(bboxMin, v.pos);
    bboxMax = glm::max(bboxMax, v.pos);
  }
  glm::vec3 center = (bboxMin + bboxMax) * 0.5f;
  glm::vec3 extent = bboxMax - bboxMin;
  LOGI("  Mesh bounds: min(%.3f, %.3f, %.3f) max(%.3f, %.3f, %.3f)\n",
       bboxMin.x, bboxMin.y, bboxMin.z, bboxMax.x, bboxMax.y, bboxMax.z);
  LOGI("  Mesh center: (%.3f, %.3f, %.3f), extent: (%.3f, %.3f, %.3f)\n",
       center.x, center.y, center.z, extent.x, extent.y, extent.z);

  Mesh model;
  model.path       = loadedData.filename.string();
  model.name       = loadedData.filename.filename().string();
  model.nbIndices  = static_cast<uint32_t>(loadedData.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loadedData.m_vertices.size());
  model.materials  = loadedData.m_materials;
  model.matNames   = loadedData.m_matNames;
  model.bboxMin    = bboxMin;
  model.bboxMax    = bboxMax;

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

  // Load textures from GLTF
  for(const auto& texData : loadedData.m_textureData)
  {
    if(texData.pixels.empty())
    {
      model.textures.push_back(MeshTexture{});
      continue;
    }

    MeshTexture meshTex;
    meshTex.width = texData.width;
    meshTex.height = texData.height;

    // Create Vulkan image
    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.extent.width = texData.width;
    imageInfo.extent.height = texData.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    NVVK_CHECK(m_alloc->createImage(meshTex.image, imageInfo));
    NVVK_DBG_NAME(meshTex.image.image);

    // Create image view
    VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = meshTex.image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    NVVK_CHECK(vkCreateImageView(m_app->getDevice(), &viewInfo, nullptr, &meshTex.view));
    NVVK_DBG_NAME(meshTex.view);

    // Create sampler
    VkSamplerCreateInfo samplerInfo = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    NVVK_CHECK(vkCreateSampler(m_app->getDevice(), &samplerInfo, nullptr, &meshTex.sampler));
    NVVK_DBG_NAME(meshTex.sampler);

    // Upload texture data
    VkCommandBuffer texCmdBuf = m_app->createTempCmdBuffer();

    // Transition to transfer dst
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = meshTex.image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(texCmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Create staging buffer and copy
    nvvk::Buffer staging;
    NVVK_CHECK(m_alloc->createBuffer(staging, texData.pixels.size(),
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     VMA_MEMORY_USAGE_CPU_TO_GPU,
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
    memcpy(staging.mapping, texData.pixels.data(), texData.pixels.size());

    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = texData.width;
    region.imageExtent.height = texData.height;
    region.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(texCmdBuf, staging.buffer, meshTex.image.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to shader read
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(texCmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    m_app->submitAndWaitTempCmdBuffer(texCmdBuf);
    m_alloc->destroyBuffer(staging);

    model.textures.push_back(meshTex);
    LOGI("  Created GPU texture %zu: %dx%d\n", model.textures.size() - 1, texData.width, texData.height);
  }

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

}  // namespace vk_viewer
